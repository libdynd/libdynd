//
// Copyright (C) 2011-12, Dynamic NDArray Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <stdexcept>
#include <sstream>

#include <dynd/dtype.hpp>
#include <dynd/diagnostics.hpp>
#include <dynd/kernels/bytes_assignment_kernels.hpp>

using namespace std;
using namespace dynd;

/////////////////////////////////////////
// blockref string to blockref bytes assignment

namespace {
    struct blockref_bytes_assign_kernel_auxdata {
        size_t dst_alignment, src_alignment;
        memory_block_ptr dst_memblock;
    };

    /** Does a single blockref-string copy */
    static void blockref_bytes_assign(char *dst, const char *src,
            const blockref_bytes_assign_kernel_auxdata& ad)
    {
        if (ad.dst_memblock.get() != NULL) {
            char *dst_begin = NULL, *dst_end = NULL;
            const char *src_begin = reinterpret_cast<const char * const *>(src)[0];
            const char *src_end = reinterpret_cast<const char * const *>(src)[1];

            memory_block_pod_allocator_api *allocator = get_memory_block_pod_allocator_api(ad.dst_memblock.get());

            allocator->allocate(ad.dst_memblock.get(), src_end - src_begin,
                            ad.dst_alignment, &dst_begin, &dst_end);
            memcpy(dst_begin, src_begin, src_end - src_begin);

            // Set the output
            reinterpret_cast<char **>(dst)[0] = dst_begin;
            reinterpret_cast<char **>(dst)[1] = dst_end;
        } else if (ad.dst_alignment <= ad.src_alignment) {
            // Copy the pointers from the source bytes
            reinterpret_cast<char **>(dst)[0] = reinterpret_cast<char * const *>(src)[0];
            reinterpret_cast<char **>(dst)[1] = reinterpret_cast<char * const *>(src)[1];
        } else {
            throw runtime_error("Attempted to reference source data when increasing bytes alignment");
        }
    }

    struct blockref_bytes_assign_kernel {
        static auxdata_kernel_api kernel_api;

        static auxdata_kernel_api *get_child_api(const AuxDataBase *DYND_UNUSED(auxdata), int DYND_UNUSED(index))
        {
            return NULL;
        }

        static int supports_referencing_src_memory_blocks(const AuxDataBase *auxdata)
        {
            const blockref_bytes_assign_kernel_auxdata& ad = get_auxiliary_data<blockref_bytes_assign_kernel_auxdata>(auxdata);
            // Can reference the src memory block when the alignment is as good or better.
            return ad.dst_alignment <= ad.src_alignment;
        }

        static void set_dst_memory_block(AuxDataBase *auxdata, memory_block_data *memblock)
        {
            blockref_bytes_assign_kernel_auxdata& ad = get_auxiliary_data<blockref_bytes_assign_kernel_auxdata>(auxdata);
            ad.dst_memblock = memblock;
        }

        static void general_kernel(char *dst, intptr_t dst_stride, const char *src, intptr_t src_stride,
                            intptr_t count, const AuxDataBase *auxdata)
        {
            const blockref_bytes_assign_kernel_auxdata& ad = get_auxiliary_data<blockref_bytes_assign_kernel_auxdata>(auxdata);
            for (intptr_t i = 0; i < count; ++i) {
                blockref_bytes_assign(dst, src, ad);

                dst += dst_stride;
                src += src_stride;
            }
        }

        static void scalar_kernel(char *dst, intptr_t DYND_UNUSED(dst_stride), const char *src, intptr_t DYND_UNUSED(src_stride),
                            intptr_t, const AuxDataBase *auxdata)
        {
            const blockref_bytes_assign_kernel_auxdata& ad = get_auxiliary_data<blockref_bytes_assign_kernel_auxdata>(auxdata);
            blockref_bytes_assign(dst, src, ad);
        }

        static void scalar_to_contiguous_kernel(char *dst, intptr_t dst_stride, const char *src, intptr_t DYND_UNUSED(src_stride),
                            intptr_t count, const AuxDataBase *auxdata)
        {
            const blockref_bytes_assign_kernel_auxdata& ad = get_auxiliary_data<blockref_bytes_assign_kernel_auxdata>(auxdata);

            // Convert the encoding once, then use memcpy calls for the rest.
            blockref_bytes_assign(dst, src, ad);
            const char *dst_first = dst;

            for (intptr_t i = 0; i < count; ++i) {
                memcpy(dst, dst_first, 2 * sizeof(char *));

                dst += dst_stride;
            }
        }
    };

    auxdata_kernel_api blockref_bytes_assign_kernel::kernel_api = {
            &blockref_bytes_assign_kernel::get_child_api,
            &blockref_bytes_assign_kernel::supports_referencing_src_memory_blocks,
            &blockref_bytes_assign_kernel::set_dst_memory_block
        };
} // anonymous namespace

void dynd::get_blockref_bytes_assignment_kernel(size_t dst_alignment,
                size_t src_alignment,
                unary_specialization_kernel_instance& out_kernel)
{
    static specialized_unary_operation_table_t optable = {
        blockref_bytes_assign_kernel::general_kernel,
        blockref_bytes_assign_kernel::scalar_kernel,
        blockref_bytes_assign_kernel::general_kernel,
        blockref_bytes_assign_kernel::scalar_to_contiguous_kernel};
    out_kernel.specializations = optable;

    make_auxiliary_data<blockref_bytes_assign_kernel_auxdata>(out_kernel.auxdata);
    blockref_bytes_assign_kernel_auxdata& ad = out_kernel.auxdata.get<blockref_bytes_assign_kernel_auxdata>();
    const_cast<AuxDataBase *>((const AuxDataBase *)out_kernel.auxdata)->kernel_api = &blockref_bytes_assign_kernel::kernel_api;
    ad.dst_alignment = dst_alignment;
    ad.src_alignment = src_alignment;
}

/////////////////////////////////////////
// fixedbytes to blockref string assignment

namespace {
    struct fixedbytes_to_blockref_bytes_assign_kernel_auxdata {
        size_t dst_alignment, src_alignment;
        intptr_t src_element_size;
        memory_block_ptr dst_memblock;
    };

    /** Does a single fixed-string copy */
    static void fixedbytes_to_blockref_bytes_assign(char *dst, const char *src,
            const fixedbytes_to_blockref_bytes_assign_kernel_auxdata& ad)
    {
        if (ad.dst_memblock.get() != NULL) {
            char *dst_begin = NULL, *dst_end = NULL;
            const char *src_begin = src;
            const char *src_end = src + ad.src_element_size;

            memory_block_pod_allocator_api *allocator = get_memory_block_pod_allocator_api(ad.dst_memblock.get());

            allocator->allocate(ad.dst_memblock.get(), src_end - src_begin,
                            ad.dst_alignment, &dst_begin, &dst_end);
            memcpy(dst_begin, src_begin, src_end - src_begin);

            // Set the output
            reinterpret_cast<char **>(dst)[0] = dst_begin;
            reinterpret_cast<char **>(dst)[1] = dst_end;

        } else if (ad.dst_alignment <= ad.src_alignment) {
            // Make the blockref bytes point into the fixedbytes
            // NOTE: It's the responsibility of the caller to ensure immutability
            //       and readonly are propagated properly.
            reinterpret_cast<char **>(dst)[0] = reinterpret_cast<char * const *>(src)[0];
            reinterpret_cast<char **>(dst)[1] = reinterpret_cast<char * const *>(src)[1];
        } else {
            throw runtime_error("Attempted to reference source data when increasing bytes alignment");
        }
    }

    struct fixedbytes_to_blockref_bytes_assign_kernel {
        static auxdata_kernel_api kernel_api;

        static auxdata_kernel_api *get_child_api(const AuxDataBase *DYND_UNUSED(auxdata), int DYND_UNUSED(index))
        {
            return NULL;
        }

        static int supports_referencing_src_memory_blocks(const AuxDataBase *auxdata)
        {
            const fixedbytes_to_blockref_bytes_assign_kernel_auxdata& ad =
                        get_auxiliary_data<fixedbytes_to_blockref_bytes_assign_kernel_auxdata>(auxdata);
            // Can reference the src memory block when the alignment is as good or better.
            return ad.dst_alignment <= ad.src_alignment;
        }

        static void set_dst_memory_block(AuxDataBase *auxdata, memory_block_data *memblock)
        {
            fixedbytes_to_blockref_bytes_assign_kernel_auxdata& ad =
                        get_auxiliary_data<fixedbytes_to_blockref_bytes_assign_kernel_auxdata>(auxdata);
            ad.dst_memblock = memblock;
        }

        static void general_kernel(char *dst, intptr_t dst_stride, const char *src, intptr_t src_stride,
                            intptr_t count, const AuxDataBase *auxdata)
        {
            const fixedbytes_to_blockref_bytes_assign_kernel_auxdata& ad =
                        get_auxiliary_data<fixedbytes_to_blockref_bytes_assign_kernel_auxdata>(auxdata);
            for (intptr_t i = 0; i < count; ++i) {
                fixedbytes_to_blockref_bytes_assign(dst, src, ad);

                dst += dst_stride;
                src += src_stride;
            }
        }

        static void scalar_kernel(char *dst, intptr_t DYND_UNUSED(dst_stride), const char *src, intptr_t DYND_UNUSED(src_stride),
                            intptr_t, const AuxDataBase *auxdata)
        {
            const fixedbytes_to_blockref_bytes_assign_kernel_auxdata& ad =
                        get_auxiliary_data<fixedbytes_to_blockref_bytes_assign_kernel_auxdata>(auxdata);
            fixedbytes_to_blockref_bytes_assign(dst, src, ad);
        }

        static void scalar_to_contiguous_kernel(char *dst, intptr_t dst_stride, const char *src, intptr_t DYND_UNUSED(src_stride),
                            intptr_t count, const AuxDataBase *auxdata)
        {
            const fixedbytes_to_blockref_bytes_assign_kernel_auxdata& ad =
                        get_auxiliary_data<fixedbytes_to_blockref_bytes_assign_kernel_auxdata>(auxdata);

            // Convert the encoding once, then use memcpy calls for the rest.
            fixedbytes_to_blockref_bytes_assign(dst, src, ad);
            const char *dst_first = dst;

            for (intptr_t i = 0; i < count; ++i) {
                memcpy(dst, dst_first, 2 * sizeof(char *));

                dst += dst_stride;
            }
        }
    };

    auxdata_kernel_api fixedbytes_to_blockref_bytes_assign_kernel::kernel_api = {
            &fixedbytes_to_blockref_bytes_assign_kernel::get_child_api,
            &fixedbytes_to_blockref_bytes_assign_kernel::supports_referencing_src_memory_blocks,
            &fixedbytes_to_blockref_bytes_assign_kernel::set_dst_memory_block
        };
} // anonymous namespace

void dynd::get_fixedbytes_to_blockref_bytes_assignment_kernel(size_t dst_alignment,
                intptr_t src_element_size, size_t src_alignment,
                unary_specialization_kernel_instance& out_kernel)
{
    static specialized_unary_operation_table_t optable = {
        fixedbytes_to_blockref_bytes_assign_kernel::general_kernel,
        fixedbytes_to_blockref_bytes_assign_kernel::scalar_kernel,
        fixedbytes_to_blockref_bytes_assign_kernel::general_kernel,
        fixedbytes_to_blockref_bytes_assign_kernel::scalar_to_contiguous_kernel};
    out_kernel.specializations = optable;

    make_auxiliary_data<fixedbytes_to_blockref_bytes_assign_kernel_auxdata>(out_kernel.auxdata);
    fixedbytes_to_blockref_bytes_assign_kernel_auxdata& ad =
                out_kernel.auxdata.get<fixedbytes_to_blockref_bytes_assign_kernel_auxdata>();
    const_cast<AuxDataBase *>((const AuxDataBase *)out_kernel.auxdata)->kernel_api = &fixedbytes_to_blockref_bytes_assign_kernel::kernel_api;
    ad.dst_alignment = dst_alignment;
    ad.src_alignment = src_alignment;
    ad.src_element_size = src_element_size;
}
