//
// Copyright (C) 2011-12, Dynamic NDArray Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <stdexcept>
#include <sstream>

#include <dnd/dtype.hpp>
#include <dnd/diagnostics.hpp>
#include <dnd/kernels/string_assignment_kernels.hpp>

using namespace std;
using namespace dnd;

/////////////////////////////////////////
// fixedstring to fixedstring assignment

namespace {
    struct fixedstring_assign_kernel_auxdata {
        next_unicode_codepoint_t next_fn;
        append_unicode_codepoint_t append_fn;
        intptr_t dst_element_size, src_element_size;
        bool overflow_check;
    };

    /** Does a single fixed-string copy */
    static void fixedstring_assign(char *dst, const char *src,
            const fixedstring_assign_kernel_auxdata& ad)
    {
        char *dst_end = dst + ad.dst_element_size;
        const char *src_end = src + ad.src_element_size;
        next_unicode_codepoint_t next_fn = ad.next_fn;
        append_unicode_codepoint_t append_fn = ad.append_fn;
        uint32_t cp;

        while (src < src_end && dst < dst_end) {
            cp = next_fn(src, src_end);
            // The fixedstring dtype uses null-terminated strings
            if (cp == 0) {
                // Null-terminate the destination string, and we're done
                memset(dst, 0, dst_end - dst);
                return;
            } else {
                append_fn(cp, dst, dst_end);
            }
        }
        if (src < src_end) {
            if (ad.overflow_check) {
                throw std::runtime_error("Input string is too large to convert to destination fixed-size string");
            }
        } else if (dst < dst_end) {
            memset(dst, 0, dst_end - dst);
        }
   }

    struct fixedstring_assign_kernel {
        static void general_kernel(char *dst, intptr_t dst_stride, const char *src, intptr_t src_stride,
                            intptr_t count, const AuxDataBase *auxdata)
        {
            const fixedstring_assign_kernel_auxdata& ad = get_auxiliary_data<fixedstring_assign_kernel_auxdata>(auxdata);
            for (intptr_t i = 0; i < count; ++i) {
                fixedstring_assign(dst, src, ad);

                dst += dst_stride;
                src += src_stride;
            }
        }

        static void scalar_kernel(char *dst, intptr_t DND_UNUSED(dst_stride), const char *src, intptr_t DND_UNUSED(src_stride),
                            intptr_t, const AuxDataBase *auxdata)
        {
            const fixedstring_assign_kernel_auxdata& ad = get_auxiliary_data<fixedstring_assign_kernel_auxdata>(auxdata);
            fixedstring_assign(dst, src, ad);
        }

        static void scalar_to_contiguous_kernel(char *dst, intptr_t dst_stride, const char *src, intptr_t DND_UNUSED(src_stride),
                            intptr_t count, const AuxDataBase *auxdata)
        {
            const fixedstring_assign_kernel_auxdata& ad = get_auxiliary_data<fixedstring_assign_kernel_auxdata>(auxdata);
            intptr_t dst_element_size = ad.dst_element_size;

            // Convert the encoding once, then use memcpy calls for the rest.
            fixedstring_assign(dst, src, ad);
            const char *dst_first = dst;

            for (intptr_t i = 0; i < count; ++i) {
                memcpy(dst, dst_first, dst_element_size);

                dst += dst_stride;
            }
        }
    };
} // anonymous namespace

void dnd::get_fixedstring_assignment_kernel(intptr_t dst_element_size, string_encoding_t dst_encoding,
                intptr_t src_element_size, string_encoding_t src_encoding,
                assign_error_mode errmode,
                unary_specialization_kernel_instance& out_kernel)
{
    static specialized_unary_operation_table_t optable = {
        fixedstring_assign_kernel::general_kernel,
        fixedstring_assign_kernel::scalar_kernel,
        fixedstring_assign_kernel::general_kernel,
        fixedstring_assign_kernel::scalar_to_contiguous_kernel};
    out_kernel.specializations = optable;

    make_auxiliary_data<fixedstring_assign_kernel_auxdata>(out_kernel.auxdata);
    fixedstring_assign_kernel_auxdata& ad = out_kernel.auxdata.get<fixedstring_assign_kernel_auxdata>();
    ad.dst_element_size = dst_element_size;
    ad.src_element_size = src_element_size;
    ad.overflow_check = (errmode != assign_error_none);
    ad.append_fn = get_append_unicode_codepoint_function(dst_encoding, errmode);
    ad.next_fn = get_next_unicode_codepoint_function(src_encoding, errmode);
}

/////////////////////////////////////////
// blockref string to blockref string assignment

namespace {
    struct blockref_string_assign_kernel_auxdata {
        string_encoding_t dst_encoding, src_encoding;
        next_unicode_codepoint_t next_fn;
        append_unicode_codepoint_t append_fn;
        memory_block_ptr dst_memblock;
    };

    /** Does a single blockref-string copy */
    static void blockref_string_assign(char *dst, const char *src,
            const blockref_string_assign_kernel_auxdata& ad)
    {
        intptr_t src_charsize = string_encoding_char_size_table[ad.src_encoding];
        intptr_t dst_charsize = string_encoding_char_size_table[ad.dst_encoding];

        if (ad.dst_memblock.get() != NULL) {
            char *dst_begin = NULL, *dst_current, *dst_end = NULL;
            const char *src_begin = reinterpret_cast<const char * const *>(src)[0];
            const char *src_end = reinterpret_cast<const char * const *>(src)[1];
            next_unicode_codepoint_t next_fn = ad.next_fn;
            append_unicode_codepoint_t append_fn = ad.append_fn;
            uint32_t cp;

            memory_block_pod_allocator_api *allocator = get_memory_block_pod_allocator_api(ad.dst_memblock.get());

            // Allocate the initial output as the src number of characters + some padding
            // TODO: Don't add padding if the output is not a multi-character encoding
            allocator->allocate(ad.dst_memblock.get(), ((src_end - src_begin) / src_charsize + 16) * dst_charsize * 1124 / 1024,
                            dst_charsize, &dst_begin, &dst_end);

            dst_current = dst_begin;
            while (src_begin < src_end) {
                cp = next_fn(src_begin, src_end);
                // Append the codepoint, or increase the allocated memory as necessary
                if (dst_end - dst_current >= 8) {
                    append_fn(cp, dst_current, dst_end);
                } else {
                    char *dst_begin_saved = dst_begin;
                    allocator->resize(ad.dst_memblock.get(), 2 * (dst_end - dst_begin), &dst_begin, &dst_end);
                    dst_current = dst_begin + (dst_current - dst_begin_saved);

                    append_fn(cp, dst_current, dst_end);
                }
            }

            // Shrink-wrap the memory to just fit the string
            allocator->resize(ad.dst_memblock.get(), dst_current - dst_begin, &dst_begin, &dst_end);

            // Set the output
            reinterpret_cast<char **>(dst)[0] = dst_begin;
            reinterpret_cast<char **>(dst)[1] = dst_end;
        } else if (ad.dst_encoding == ad.src_encoding) {
            // Copy the pointers from the source string
            reinterpret_cast<char **>(dst)[0] = reinterpret_cast<char * const *>(src)[0];
            reinterpret_cast<char **>(dst)[1] = reinterpret_cast<char * const *>(src)[1];
        } else {
            throw runtime_error("Attempted to reference source data when changing string encoding");
        }
    }

    struct blockref_string_assign_kernel {
        static auxdata_kernel_api kernel_api;

        static auxdata_kernel_api *get_child_api(const AuxDataBase *DND_UNUSED(auxdata), int DND_UNUSED(index))
        {
            return NULL;
        }

        static int supports_referencing_src_memory_blocks(const AuxDataBase *auxdata)
        {
            const blockref_string_assign_kernel_auxdata& ad = get_auxiliary_data<blockref_string_assign_kernel_auxdata>(auxdata);
            // Can reference the src memory block when the encoding matches.
            return ad.dst_encoding == ad.src_encoding;
        }

        static void set_dst_memory_block(AuxDataBase *auxdata, memory_block_data *memblock)
        {
            blockref_string_assign_kernel_auxdata& ad = get_auxiliary_data<blockref_string_assign_kernel_auxdata>(auxdata);
            ad.dst_memblock = memblock;
        }

        static void general_kernel(char *dst, intptr_t dst_stride, const char *src, intptr_t src_stride,
                            intptr_t count, const AuxDataBase *auxdata)
        {
            const blockref_string_assign_kernel_auxdata& ad = get_auxiliary_data<blockref_string_assign_kernel_auxdata>(auxdata);
            for (intptr_t i = 0; i < count; ++i) {
                blockref_string_assign(dst, src, ad);

                dst += dst_stride;
                src += src_stride;
            }
        }

        static void scalar_kernel(char *dst, intptr_t DND_UNUSED(dst_stride), const char *src, intptr_t DND_UNUSED(src_stride),
                            intptr_t, const AuxDataBase *auxdata)
        {
            const blockref_string_assign_kernel_auxdata& ad = get_auxiliary_data<blockref_string_assign_kernel_auxdata>(auxdata);
            blockref_string_assign(dst, src, ad);
        }

        static void scalar_to_contiguous_kernel(char *dst, intptr_t dst_stride, const char *src, intptr_t DND_UNUSED(src_stride),
                            intptr_t count, const AuxDataBase *auxdata)
        {
            const blockref_string_assign_kernel_auxdata& ad = get_auxiliary_data<blockref_string_assign_kernel_auxdata>(auxdata);

            // Convert the encoding once, then use memcpy calls for the rest.
            blockref_string_assign(dst, src, ad);
            const char *dst_first = dst;

            for (intptr_t i = 0; i < count; ++i) {
                memcpy(dst, dst_first, 2 * sizeof(char *));

                dst += dst_stride;
            }
        }
    };

    auxdata_kernel_api blockref_string_assign_kernel::kernel_api = {
            &blockref_string_assign_kernel::get_child_api,
            &blockref_string_assign_kernel::supports_referencing_src_memory_blocks,
            &blockref_string_assign_kernel::set_dst_memory_block
        };
} // anonymous namespace

void dnd::get_blockref_string_assignment_kernel(string_encoding_t dst_encoding,
                string_encoding_t src_encoding,
                assign_error_mode errmode,
                unary_specialization_kernel_instance& out_kernel)
{
    static specialized_unary_operation_table_t optable = {
        blockref_string_assign_kernel::general_kernel,
        blockref_string_assign_kernel::scalar_kernel,
        blockref_string_assign_kernel::general_kernel,
        blockref_string_assign_kernel::scalar_to_contiguous_kernel};
    out_kernel.specializations = optable;

    make_auxiliary_data<blockref_string_assign_kernel_auxdata>(out_kernel.auxdata);
    blockref_string_assign_kernel_auxdata& ad = out_kernel.auxdata.get<blockref_string_assign_kernel_auxdata>();
    const_cast<AuxDataBase *>((const AuxDataBase *)out_kernel.auxdata)->kernel_api = &blockref_string_assign_kernel::kernel_api;
    ad.dst_encoding = dst_encoding;
    ad.src_encoding = src_encoding;
    ad.append_fn = get_append_unicode_codepoint_function(dst_encoding, errmode);
    ad.next_fn = get_next_unicode_codepoint_function(src_encoding, errmode);
}

/////////////////////////////////////////
// fixedstring to blockref string assignment

namespace {
    struct fixedstring_to_blockref_string_assign_kernel_auxdata {
        string_encoding_t dst_encoding, src_encoding;
        intptr_t src_element_size;
        next_unicode_codepoint_t next_fn;
        append_unicode_codepoint_t append_fn;
        memory_block_ptr dst_memblock;
    };

    /** Does a single fixed-string copy */
    static void fixedstring_to_blockref_string_assign(char *dst, const char *src,
            const fixedstring_to_blockref_string_assign_kernel_auxdata& ad)
    {
        intptr_t src_charsize = string_encoding_char_size_table[ad.src_encoding];
        intptr_t dst_charsize = string_encoding_char_size_table[ad.dst_encoding];

        if (ad.dst_memblock.get() != NULL) {
            char *dst_begin = NULL, *dst_current, *dst_end = NULL;
            const char *src_begin = src;
            const char *src_end = src + ad.src_element_size;
            next_unicode_codepoint_t next_fn = ad.next_fn;
            append_unicode_codepoint_t append_fn = ad.append_fn;
            uint32_t cp;

            memory_block_pod_allocator_api *allocator = get_memory_block_pod_allocator_api(ad.dst_memblock.get());

            // Allocate the initial output as the src number of characters + some padding
            // TODO: Don't add padding if the output is not a multi-character encoding
            allocator->allocate(ad.dst_memblock.get(), ((src_end - src_begin) / src_charsize + 16) * dst_charsize * 1124 / 1024,
                            dst_charsize, &dst_begin, &dst_end);

            dst_current = dst_begin;
            while (src_begin < src_end) {
                cp = next_fn(src_begin, src_end);
                // Append the codepoint, or increase the allocated memory as necessary
                if (cp != 0) {
                    if (dst_end - dst_current >= 8) {
                        append_fn(cp, dst_current, dst_end);
                    } else {
                        char *dst_begin_saved = dst_begin;
                        allocator->resize(ad.dst_memblock.get(), 2 * (dst_end - dst_begin), &dst_begin, &dst_end);
                        dst_current = dst_begin + (dst_current - dst_begin_saved);

                        append_fn(cp, dst_current, dst_end);
                    }
                } else {
                    break;
                }
            }

            // Shrink-wrap the memory to just fit the string
            allocator->resize(ad.dst_memblock.get(), dst_current - dst_begin, &dst_begin, &dst_end);

            // Set the output
            reinterpret_cast<char **>(dst)[0] = dst_begin;
            reinterpret_cast<char **>(dst)[1] = dst_end;
        } else if (ad.dst_encoding == ad.src_encoding) {
            // Make the blockref string point into the fixedstring
            // NOTE: It's the responsibility of the caller to ensure immutability
            //       and readonly are propagated properly.
            reinterpret_cast<char **>(dst)[0] = const_cast<char *>(src);
            reinterpret_cast<char **>(dst)[1] = const_cast<char *>(src) + strnlen(src, ad.src_element_size);
        } else {
            throw runtime_error("Attempted to reference source data when changing string encoding");
        }
    }

    struct fixedstring_to_blockref_string_assign_kernel {
        static auxdata_kernel_api kernel_api;

        static auxdata_kernel_api *get_child_api(const AuxDataBase *DND_UNUSED(auxdata), int DND_UNUSED(index))
        {
            return NULL;
        }

        static int supports_referencing_src_memory_blocks(const AuxDataBase *auxdata)
        {
            const fixedstring_to_blockref_string_assign_kernel_auxdata& ad =
                        get_auxiliary_data<fixedstring_to_blockref_string_assign_kernel_auxdata>(auxdata);
            // Can reference the src memory block when the encoding matches.
            return ad.dst_encoding == ad.src_encoding;
        }

        static void set_dst_memory_block(AuxDataBase *auxdata, memory_block_data *memblock)
        {
            fixedstring_to_blockref_string_assign_kernel_auxdata& ad =
                        get_auxiliary_data<fixedstring_to_blockref_string_assign_kernel_auxdata>(auxdata);
            ad.dst_memblock = memblock;
        }

        static void general_kernel(char *dst, intptr_t dst_stride, const char *src, intptr_t src_stride,
                            intptr_t count, const AuxDataBase *auxdata)
        {
            const fixedstring_to_blockref_string_assign_kernel_auxdata& ad =
                        get_auxiliary_data<fixedstring_to_blockref_string_assign_kernel_auxdata>(auxdata);
            for (intptr_t i = 0; i < count; ++i) {
                fixedstring_to_blockref_string_assign(dst, src, ad);

                dst += dst_stride;
                src += src_stride;
            }
        }

        static void scalar_kernel(char *dst, intptr_t DND_UNUSED(dst_stride), const char *src, intptr_t DND_UNUSED(src_stride),
                            intptr_t, const AuxDataBase *auxdata)
        {
            const fixedstring_to_blockref_string_assign_kernel_auxdata& ad =
                        get_auxiliary_data<fixedstring_to_blockref_string_assign_kernel_auxdata>(auxdata);
            fixedstring_to_blockref_string_assign(dst, src, ad);
        }

        static void scalar_to_contiguous_kernel(char *dst, intptr_t dst_stride, const char *src, intptr_t DND_UNUSED(src_stride),
                            intptr_t count, const AuxDataBase *auxdata)
        {
            const fixedstring_to_blockref_string_assign_kernel_auxdata& ad =
                        get_auxiliary_data<fixedstring_to_blockref_string_assign_kernel_auxdata>(auxdata);

            // Convert the encoding once, then use memcpy calls for the rest.
            fixedstring_to_blockref_string_assign(dst, src, ad);
            const char *dst_first = dst;

            for (intptr_t i = 0; i < count; ++i) {
                memcpy(dst, dst_first, 2 * sizeof(char *));

                dst += dst_stride;
            }
        }
    };

    auxdata_kernel_api fixedstring_to_blockref_string_assign_kernel::kernel_api = {
            &fixedstring_to_blockref_string_assign_kernel::get_child_api,
            &fixedstring_to_blockref_string_assign_kernel::supports_referencing_src_memory_blocks,
            &fixedstring_to_blockref_string_assign_kernel::set_dst_memory_block
        };
} // anonymous namespace

void dnd::get_fixedstring_to_blockref_string_assignment_kernel(string_encoding_t dst_encoding,
                intptr_t src_element_size, string_encoding_t src_encoding,
                assign_error_mode errmode,
                unary_specialization_kernel_instance& out_kernel)
{
    static specialized_unary_operation_table_t optable = {
        fixedstring_to_blockref_string_assign_kernel::general_kernel,
        fixedstring_to_blockref_string_assign_kernel::scalar_kernel,
        fixedstring_to_blockref_string_assign_kernel::general_kernel,
        fixedstring_to_blockref_string_assign_kernel::scalar_to_contiguous_kernel};
    out_kernel.specializations = optable;

    make_auxiliary_data<fixedstring_to_blockref_string_assign_kernel_auxdata>(out_kernel.auxdata);
    fixedstring_to_blockref_string_assign_kernel_auxdata& ad =
                out_kernel.auxdata.get<fixedstring_to_blockref_string_assign_kernel_auxdata>();
    const_cast<AuxDataBase *>((const AuxDataBase *)out_kernel.auxdata)->kernel_api = &fixedstring_to_blockref_string_assign_kernel::kernel_api;
    ad.dst_encoding = dst_encoding;
    ad.src_encoding = src_encoding;
    ad.src_element_size = src_element_size;
    ad.append_fn = get_append_unicode_codepoint_function(dst_encoding, errmode);
    ad.next_fn = get_next_unicode_codepoint_function(src_encoding, errmode);
}

/////////////////////////////////////////
// blockref string to fixedstring assignment

namespace {
    struct blockref_string_to_fixedstring_assign_kernel_auxdata {
        next_unicode_codepoint_t next_fn;
        append_unicode_codepoint_t append_fn;
        intptr_t dst_element_size, src_element_size;
        bool overflow_check;
    };

    /** Does a single fixed-string copy */
    static void blockref_string_to_fixedstring_assign(char *dst, const char *src,
            const blockref_string_to_fixedstring_assign_kernel_auxdata& ad)
    {
        char *dst_end = dst + ad.dst_element_size;
        const char *src_begin = reinterpret_cast<const char * const *>(src)[0];
        const char *src_end = reinterpret_cast<const char * const *>(src)[1];
        next_unicode_codepoint_t next_fn = ad.next_fn;
        append_unicode_codepoint_t append_fn = ad.append_fn;
        uint32_t cp;

        while (src_begin < src_end && dst < dst_end) {
            cp = next_fn(src_begin, src_end);
            append_fn(cp, dst, dst_end);
        }
        if (src_begin < src_end) {
            if (ad.overflow_check) {
                throw std::runtime_error("Input string is too large to convert to destination fixed-size string");
            }
        } else if (dst < dst_end) {
            memset(dst, 0, dst_end - dst);
        }
   }

    struct blockref_string_to_fixedstring_assign_kernel {
        static void general_kernel(char *dst, intptr_t dst_stride, const char *src, intptr_t src_stride,
                            intptr_t count, const AuxDataBase *auxdata)
        {
            const blockref_string_to_fixedstring_assign_kernel_auxdata& ad =
                        get_auxiliary_data<blockref_string_to_fixedstring_assign_kernel_auxdata>(auxdata);
            for (intptr_t i = 0; i < count; ++i) {
                blockref_string_to_fixedstring_assign(dst, src, ad);

                dst += dst_stride;
                src += src_stride;
            }
        }

        static void scalar_kernel(char *dst, intptr_t DND_UNUSED(dst_stride), const char *src, intptr_t DND_UNUSED(src_stride),
                            intptr_t, const AuxDataBase *auxdata)
        {
            const blockref_string_to_fixedstring_assign_kernel_auxdata& ad =
                        get_auxiliary_data<blockref_string_to_fixedstring_assign_kernel_auxdata>(auxdata);
            blockref_string_to_fixedstring_assign(dst, src, ad);
        }

        static void scalar_to_contiguous_kernel(char *dst, intptr_t dst_stride, const char *src, intptr_t DND_UNUSED(src_stride),
                            intptr_t count, const AuxDataBase *auxdata)
        {
            const blockref_string_to_fixedstring_assign_kernel_auxdata& ad =
                        get_auxiliary_data<blockref_string_to_fixedstring_assign_kernel_auxdata>(auxdata);
            intptr_t dst_element_size = ad.dst_element_size;

            // Convert the encoding once, then use memcpy calls for the rest.
            blockref_string_to_fixedstring_assign(dst, src, ad);
            const char *dst_first = dst;

            for (intptr_t i = 0; i < count; ++i) {
                memcpy(dst, dst_first, dst_element_size);

                dst += dst_stride;
            }
        }
    };
} // anonymous namespace

void dnd::get_blockref_string_to_fixedstring_assignment_kernel(intptr_t dst_element_size, string_encoding_t dst_encoding,
                string_encoding_t src_encoding,
                assign_error_mode errmode,
                unary_specialization_kernel_instance& out_kernel)
{
    static specialized_unary_operation_table_t optable = {
        blockref_string_to_fixedstring_assign_kernel::general_kernel,
        blockref_string_to_fixedstring_assign_kernel::scalar_kernel,
        blockref_string_to_fixedstring_assign_kernel::general_kernel,
        blockref_string_to_fixedstring_assign_kernel::scalar_to_contiguous_kernel};
    out_kernel.specializations = optable;

    make_auxiliary_data<fixedstring_assign_kernel_auxdata>(out_kernel.auxdata);
    blockref_string_to_fixedstring_assign_kernel_auxdata& ad =
                out_kernel.auxdata.get<blockref_string_to_fixedstring_assign_kernel_auxdata>();
    ad.dst_element_size = dst_element_size;
    ad.overflow_check = (errmode != assign_error_none);
    ad.append_fn = get_append_unicode_codepoint_function(dst_encoding, errmode);
    ad.next_fn = get_next_unicode_codepoint_function(src_encoding, errmode);
}
