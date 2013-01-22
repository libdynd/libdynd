//
// Copyright (C) 2011-13, DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <stdexcept>
#include <sstream>

#include <dynd/dtype.hpp>
#include <dynd/diagnostics.hpp>
#include <dynd/kernels/string_assignment_kernels.hpp>
#include <dynd/dtypes/string_dtype.hpp>

using namespace std;
using namespace dynd;

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
    static void fixedstring_assign_single(char *dst, const char *src, unary_kernel_static_data *extra)
    {
        const fixedstring_assign_kernel_auxdata& ad = get_auxiliary_data<fixedstring_assign_kernel_auxdata>(extra->auxdata);
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
} // anonymous namespace

void dynd::get_fixedstring_assignment_kernel(intptr_t dst_element_size, string_encoding_t dst_encoding,
                intptr_t src_element_size, string_encoding_t src_encoding,
                assign_error_mode errmode,
                kernel_instance<unary_operation_pair_t>& out_kernel)
{
    out_kernel.kernel.single = &fixedstring_assign_single;
    out_kernel.kernel.strided = NULL;

    make_auxiliary_data<fixedstring_assign_kernel_auxdata>(out_kernel.extra.auxdata);
    fixedstring_assign_kernel_auxdata& ad = out_kernel.extra.auxdata.get<fixedstring_assign_kernel_auxdata>();
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
    };

    /** Does a single blockref-string copy */
    static void blockref_string_assign_single(char *dst, const char *src, unary_kernel_static_data *extra)
    {
        const blockref_string_assign_kernel_auxdata& ad = get_auxiliary_data<blockref_string_assign_kernel_auxdata>(extra->auxdata);
        const string_dtype_metadata *dst_md = reinterpret_cast<const string_dtype_metadata *>(extra->dst_metadata);
        const string_dtype_metadata *src_md = reinterpret_cast<const string_dtype_metadata *>(extra->src_metadata);
        intptr_t src_charsize = string_encoding_char_size_table[ad.src_encoding];
        intptr_t dst_charsize = string_encoding_char_size_table[ad.dst_encoding];

        // If the blockrefs are different, require a copy operation
        if (dst_md->blockref != src_md->blockref) {
            char *dst_begin = NULL, *dst_current, *dst_end = NULL;
            const char *src_begin = reinterpret_cast<const string_dtype_data *>(src)->begin;
            const char *src_end = reinterpret_cast<const string_dtype_data *>(src)->end;
            next_unicode_codepoint_t next_fn = ad.next_fn;
            append_unicode_codepoint_t append_fn = ad.append_fn;
            uint32_t cp;

            memory_block_pod_allocator_api *allocator = get_memory_block_pod_allocator_api(dst_md->blockref);

            // Allocate the initial output as the src number of characters + some padding
            // TODO: Don't add padding if the output is not a multi-character encoding
            allocator->allocate(dst_md->blockref, ((src_end - src_begin) / src_charsize + 16) * dst_charsize * 1124 / 1024,
                            dst_charsize, &dst_begin, &dst_end);

            dst_current = dst_begin;
            while (src_begin < src_end) {
                cp = next_fn(src_begin, src_end);
                // Append the codepoint, or increase the allocated memory as necessary
                if (dst_end - dst_current >= 8) {
                    append_fn(cp, dst_current, dst_end);
                } else {
                    char *dst_begin_saved = dst_begin;
                    allocator->resize(dst_md->blockref, 2 * (dst_end - dst_begin), &dst_begin, &dst_end);
                    dst_current = dst_begin + (dst_current - dst_begin_saved);

                    append_fn(cp, dst_current, dst_end);
                }
            }

            // Shrink-wrap the memory to just fit the string
            allocator->resize(dst_md->blockref, dst_current - dst_begin, &dst_begin, &dst_end);

            // Set the output
            reinterpret_cast<string_dtype_data *>(dst)->begin = dst_begin;
            reinterpret_cast<string_dtype_data*>(dst)->end = dst_end;
        } else if (ad.dst_encoding == ad.src_encoding) {
            // Copy the pointers from the source string
            *reinterpret_cast<string_dtype_data *>(dst) = *reinterpret_cast<const string_dtype_data *>(src);
        } else {
            throw runtime_error("Attempted to reference source data when changing string encoding");
        }
    }
} // anonymous namespace

void dynd::get_blockref_string_assignment_kernel(string_encoding_t dst_encoding,
                string_encoding_t src_encoding,
                assign_error_mode errmode,
                kernel_instance<unary_operation_pair_t>& out_kernel)
{
    out_kernel.kernel.single = &blockref_string_assign_single;
    out_kernel.kernel.strided = NULL;

    make_auxiliary_data<blockref_string_assign_kernel_auxdata>(out_kernel.extra.auxdata);
    blockref_string_assign_kernel_auxdata& ad = out_kernel.extra.auxdata.get<blockref_string_assign_kernel_auxdata>();
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
    };

    /** Does a single fixed-string copy */
    static void fixedstring_to_blockref_string_assign(char *dst, const char *src, unary_kernel_static_data *extra)
    {
        const fixedstring_to_blockref_string_assign_kernel_auxdata& ad = get_auxiliary_data<fixedstring_to_blockref_string_assign_kernel_auxdata>(extra->auxdata);
        const string_dtype_metadata *dst_md = reinterpret_cast<const string_dtype_metadata *>(extra->dst_metadata);
        intptr_t src_charsize = string_encoding_char_size_table[ad.src_encoding];
        intptr_t dst_charsize = string_encoding_char_size_table[ad.dst_encoding];

        // TODO: With some additional mechanism to track the source memory block, could
        //       avoid copying the bytes data.
        char *dst_begin = NULL, *dst_current, *dst_end = NULL;
        const char *src_begin = src;
        const char *src_end = src + ad.src_element_size;
        next_unicode_codepoint_t next_fn = ad.next_fn;
        append_unicode_codepoint_t append_fn = ad.append_fn;
        uint32_t cp;

        memory_block_pod_allocator_api *allocator = get_memory_block_pod_allocator_api(dst_md->blockref);

        // Allocate the initial output as the src number of characters + some padding
        // TODO: Don't add padding if the output is not a multi-character encoding
        allocator->allocate(dst_md->blockref, ((src_end - src_begin) / src_charsize + 16) * dst_charsize * 1124 / 1024,
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
                    allocator->resize(dst_md->blockref, 2 * (dst_end - dst_begin), &dst_begin, &dst_end);
                    dst_current = dst_begin + (dst_current - dst_begin_saved);

                    append_fn(cp, dst_current, dst_end);
                }
            } else {
                break;
            }
        }

        // Shrink-wrap the memory to just fit the string
        allocator->resize(dst_md->blockref, dst_current - dst_begin, &dst_begin, &dst_end);

        // Set the output
        reinterpret_cast<string_dtype_data *>(dst)->begin = dst_begin;
        reinterpret_cast<string_dtype_data*>(dst)->end = dst_end;
    }
} // anonymous namespace

void dynd::get_fixedstring_to_blockref_string_assignment_kernel(string_encoding_t dst_encoding,
                intptr_t src_element_size, string_encoding_t src_encoding,
                assign_error_mode errmode,
                kernel_instance<unary_operation_pair_t>& out_kernel)
{
    out_kernel.kernel.single = &fixedstring_to_blockref_string_assign;
    out_kernel.kernel.strided = NULL;

    make_auxiliary_data<fixedstring_to_blockref_string_assign_kernel_auxdata>(out_kernel.extra.auxdata);
    fixedstring_to_blockref_string_assign_kernel_auxdata& ad =
                out_kernel.extra.auxdata.get<fixedstring_to_blockref_string_assign_kernel_auxdata>();
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
    static void blockref_string_to_fixedstring_assign(char *dst, const char *src, unary_kernel_static_data *extra)
    {
        const blockref_string_to_fixedstring_assign_kernel_auxdata& ad =
                    get_auxiliary_data<blockref_string_to_fixedstring_assign_kernel_auxdata>(extra->auxdata);
        char *dst_end = dst + ad.dst_element_size;
        const char *src_begin = reinterpret_cast<const string_dtype_data *>(src)->begin;
        const char *src_end = reinterpret_cast<const string_dtype_data *>(src)->end;
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
} // anonymous namespace

void dynd::get_blockref_string_to_fixedstring_assignment_kernel(intptr_t dst_element_size, string_encoding_t dst_encoding,
                string_encoding_t src_encoding,
                assign_error_mode errmode,
                kernel_instance<unary_operation_pair_t>& out_kernel)
{
    out_kernel.kernel.single = &blockref_string_to_fixedstring_assign;
    out_kernel.kernel.strided = NULL;

    make_auxiliary_data<fixedstring_assign_kernel_auxdata>(out_kernel.extra.auxdata);
    blockref_string_to_fixedstring_assign_kernel_auxdata& ad =
                out_kernel.extra.auxdata.get<blockref_string_to_fixedstring_assign_kernel_auxdata>();
    ad.dst_element_size = dst_element_size;
    ad.overflow_check = (errmode != assign_error_none);
    ad.append_fn = get_append_unicode_codepoint_function(dst_encoding, errmode);
    ad.next_fn = get_next_unicode_codepoint_function(src_encoding, errmode);
}
