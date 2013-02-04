//
// Copyright (C) 2011-13, DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <stdexcept>
#include <sstream>

#include <dynd/dtype.hpp>
#include <dynd/diagnostics.hpp>
#include <dynd/kernels/assignment_kernels.hpp>
#include <dynd/kernels/array_assignment_kernels.hpp>
#include <dynd/dtypes/var_array_dtype.hpp>

using namespace std;
using namespace dynd;

/////////////////////////////////////////
// blockref array to blockref array assignment

namespace {
    struct blockref_array_assign_kernel_auxdata {
        kernel_instance<unary_operation_pair_t> assign_elements;
        dtype dst_dtype, src_dtype;
    };

    /** Does a single blockref-array copy */
    static void blockref_array_assign_single(char *dst, const char *src, unary_kernel_static_data *extra)
    {
        blockref_array_assign_kernel_auxdata& ad = get_auxiliary_data<blockref_array_assign_kernel_auxdata>(extra->auxdata);
        const var_array_dtype_metadata *dst_md = reinterpret_cast<const var_array_dtype_metadata *>(extra->dst_metadata);
        const var_array_dtype_metadata *src_md = reinterpret_cast<const var_array_dtype_metadata *>(extra->src_metadata);
        if (dst_md->offset != 0) {
            throw runtime_error("Cannot assign to a dynd var_array which has a non-zero offset");
        }
        // If the blockrefs are different, require a copy operation
        if (dst_md->blockref != src_md->blockref) {
            char *dst_begin = NULL, *dst_end = NULL;
            const char *src_begin = reinterpret_cast<const var_array_dtype_data *>(src)->begin + src_md->offset;
            size_t src_size = reinterpret_cast<const var_array_dtype_data *>(src)->size;
            memory_block_pod_allocator_api *allocator = get_memory_block_pod_allocator_api(dst_md->blockref);

            // Allocate the output array data
            allocator->allocate(dst_md->blockref,
                        ad.dst_dtype.get_data_size() * src_size,
                        ad.dst_dtype.get_alignment(), &dst_begin, &dst_end);

            // Set the output
            reinterpret_cast<var_array_dtype_data *>(dst)->begin = dst_begin;
            reinterpret_cast<var_array_dtype_data *>(dst)->size = src_size;

            intptr_t dst_stride = dst_md->stride, src_stride = src_md->stride;
            if (ad.assign_elements.kernel.strided == NULL) {
                unary_single_operation_deprecated_t assign_single = ad.assign_elements.kernel.single;
                for (size_t i = 0; i != src_size; ++i) {
                    assign_single(dst_begin, src_begin, &ad.assign_elements.extra);
                    dst_begin += dst_stride;
                    src_begin += src_stride;
                }
            } else {
                ad.assign_elements.kernel.strided(dst_begin, dst_stride, src_begin, src_stride,
                                src_size, &ad.assign_elements.extra);
            }
        } else {
            // Copy the pointers directly, reuse the same data
            memcpy(dst, src, sizeof(var_array_dtype_data));
        }
    }
} // anonymous namespace

void dynd::get_blockref_array_assignment_kernel(const dtype& dst_element_type,
                const dtype& src_element_type,
                assign_error_mode errmode,
                kernel_instance<unary_operation_pair_t>& out_kernel)
{
    out_kernel.kernel = unary_operation_pair_t(blockref_array_assign_single, NULL);

    make_auxiliary_data<blockref_array_assign_kernel_auxdata>(out_kernel.extra.auxdata);
    blockref_array_assign_kernel_auxdata& ad = out_kernel.extra.auxdata.get<blockref_array_assign_kernel_auxdata>();
    ad.dst_dtype = dst_element_type;
    ad.src_dtype = src_element_type;
    ::dynd::get_dtype_assignment_kernel(dst_element_type, src_element_type, errmode, NULL, ad.assign_elements);
}
