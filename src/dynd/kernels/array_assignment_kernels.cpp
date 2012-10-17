//
// Copyright (C) 2011-12, Dynamic NDArray Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <stdexcept>
#include <sstream>

#include <dynd/dtype.hpp>
#include <dynd/diagnostics.hpp>
#include <dynd/kernels/assignment_kernels.hpp>
#include <dynd/kernels/array_assignment_kernels.hpp>

using namespace std;
using namespace dynd;

/////////////////////////////////////////
// blockref array to blockref array assignment

namespace {
    struct blockref_array_assign_kernel_auxdata {
        unary_specialization_kernel_instance assign_elements;
        dtype dst_dtype, src_dtype;
        memory_block_ptr dst_memblock;
    };

    /** Does a single blockref-string copy */
    static void blockref_array_assign(char *dst, const char *src,
            const blockref_array_assign_kernel_auxdata& ad)
    {
        if (ad.dst_memblock.get() != NULL) {
            char *dst_begin = NULL, *dst_end = NULL;
            const char *src_begin = reinterpret_cast<const char * const *>(src)[0];
            const char *src_end = reinterpret_cast<const char * const *>(src)[1];
            memory_block_pod_allocator_api *allocator = get_memory_block_pod_allocator_api(ad.dst_memblock.get());

            // Allocate the output array data
            intptr_t count = (src_end - src_begin) / ad.src_dtype.element_size();
            allocator->allocate(ad.dst_memblock.get(),
                        ad.dst_dtype.element_size() * count,
                        ad.dst_dtype.alignment(), &dst_begin, &dst_end);

            ad.assign_elements.specializations[contiguous_unary_specialization](
                        dst_begin, ad.dst_dtype.element_size(), src_begin, ad.src_dtype.element_size(),
                        count, ad.assign_elements.auxdata);

            // Set the output
            reinterpret_cast<char **>(dst)[0] = dst_begin;
            reinterpret_cast<char **>(dst)[1] = dst_end;
        } else {
            // Copy the pointers directly, reuse the same data
            reinterpret_cast<char **>(dst)[0] = reinterpret_cast<char * const *>(src)[0];
            reinterpret_cast<char **>(dst)[1] = reinterpret_cast<char * const *>(src)[1];
        }
    }

    struct blockref_array_assign_kernel {
        static auxdata_kernel_api kernel_api;

        static auxdata_kernel_api *get_child_api(const AuxDataBase *DYND_UNUSED(auxdata), int DYND_UNUSED(index))
        {
            return NULL;
        }

        static int supports_referencing_src_memory_blocks(const AuxDataBase *auxdata)
        {
            const blockref_array_assign_kernel_auxdata& ad = get_auxiliary_data<blockref_array_assign_kernel_auxdata>(auxdata);
            // Can reference the src memory block when there is no dtype change
            return ad.dst_dtype == ad.src_dtype;
        }

        static void set_dst_memory_block(AuxDataBase *auxdata, memory_block_data *memblock)
        {
            blockref_array_assign_kernel_auxdata& ad = get_auxiliary_data<blockref_array_assign_kernel_auxdata>(auxdata);
            ad.dst_memblock = memblock;
        }

        static void general_kernel(char *dst, intptr_t dst_stride, const char *src, intptr_t src_stride,
                            intptr_t count, const AuxDataBase *auxdata)
        {
            const blockref_array_assign_kernel_auxdata& ad = get_auxiliary_data<blockref_array_assign_kernel_auxdata>(auxdata);
            for (intptr_t i = 0; i < count; ++i) {
                blockref_array_assign(dst, src, ad);

                dst += dst_stride;
                src += src_stride;
            }
        }

        static void scalar_kernel(char *dst, intptr_t DYND_UNUSED(dst_stride), const char *src, intptr_t DYND_UNUSED(src_stride),
                            intptr_t, const AuxDataBase *auxdata)
        {
            const blockref_array_assign_kernel_auxdata& ad = get_auxiliary_data<blockref_array_assign_kernel_auxdata>(auxdata);
            blockref_array_assign(dst, src, ad);
        }

        static void scalar_to_contiguous_kernel(char *dst, intptr_t dst_stride, const char *src, intptr_t DYND_UNUSED(src_stride),
                            intptr_t count, const AuxDataBase *auxdata)
        {
            const blockref_array_assign_kernel_auxdata& ad = get_auxiliary_data<blockref_array_assign_kernel_auxdata>(auxdata);

            // Convert the encoding once, then use memcpy calls for the rest.
            // TODO: Should only do this when the destination is immutable
            blockref_array_assign(dst, src, ad);
            const char *dst_first = dst;

            for (intptr_t i = 0; i < count; ++i) {
                memcpy(dst, dst_first, 2 * sizeof(char *));

                dst += dst_stride;
            }
        }
    };

    auxdata_kernel_api blockref_array_assign_kernel::kernel_api = {
            &blockref_array_assign_kernel::get_child_api,
            &blockref_array_assign_kernel::supports_referencing_src_memory_blocks,
            &blockref_array_assign_kernel::set_dst_memory_block
        };
} // anonymous namespace

void dynd::get_blockref_array_assignment_kernel(const dtype& dst_element_type,
                const dtype& src_element_type,
                assign_error_mode errmode,
                unary_specialization_kernel_instance& out_kernel)
{
    static specialized_unary_operation_table_t optable = {
        blockref_array_assign_kernel::general_kernel,
        blockref_array_assign_kernel::scalar_kernel,
        blockref_array_assign_kernel::general_kernel,
        blockref_array_assign_kernel::scalar_to_contiguous_kernel};
    out_kernel.specializations = optable;

    make_auxiliary_data<blockref_array_assign_kernel_auxdata>(out_kernel.auxdata);
    blockref_array_assign_kernel_auxdata& ad = out_kernel.auxdata.get<blockref_array_assign_kernel_auxdata>();
    const_cast<AuxDataBase *>((const AuxDataBase *)out_kernel.auxdata)->kernel_api = &blockref_array_assign_kernel::kernel_api;
    ad.dst_dtype = dst_element_type;
    ad.src_dtype = src_element_type;
    ::dynd::get_dtype_assignment_kernel(dst_element_type, src_element_type, errmode, NULL, ad.assign_elements);
}
