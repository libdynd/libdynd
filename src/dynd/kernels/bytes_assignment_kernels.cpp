//
// Copyright (C) 2011-12, Dynamic NDArray Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <stdexcept>
#include <sstream>

#include <dynd/dtype.hpp>
#include <dynd/diagnostics.hpp>
#include <dynd/kernels/bytes_assignment_kernels.hpp>
#include <dynd/dtypes/bytes_dtype.hpp>

using namespace std;
using namespace dynd;

/////////////////////////////////////////
// blockref bytes to blockref bytes assignment

namespace {
    struct blockref_bytes_assign_kernel_auxdata {
        size_t dst_alignment, src_alignment;
    };

    /** Does a single blockref-string copy */
    static void blockref_bytes_assign_single(char *dst, const char *src, unary_kernel_static_data *extra)
    {
        blockref_bytes_assign_kernel_auxdata& ad = get_auxiliary_data<blockref_bytes_assign_kernel_auxdata>(extra->auxdata);
        const bytes_dtype_metadata *dst_md = reinterpret_cast<const bytes_dtype_metadata *>(extra->dst_metadata);
        const bytes_dtype_metadata *src_md = reinterpret_cast<const bytes_dtype_metadata *>(extra->src_metadata);
        // If the blockrefs are different, require a copy operation
        if (dst_md->blockref != src_md->blockref) {
            char *dst_begin = NULL, *dst_end = NULL;
            const char *src_begin = reinterpret_cast<const bytes_dtype_data *>(src)->begin;
            const char *src_end = reinterpret_cast<const bytes_dtype_data *>(src)->end;
            memory_block_pod_allocator_api *allocator = get_memory_block_pod_allocator_api(dst_md->blockref);

            allocator->allocate(dst_md->blockref, src_end - src_begin,
                            ad.dst_alignment, &dst_begin, &dst_end);
            memcpy(dst_begin, src_begin, src_end - src_begin);

            // Set the output
            reinterpret_cast<bytes_dtype_data *>(dst)->begin = dst_begin;
            reinterpret_cast<bytes_dtype_data *>(dst)->end = dst_end;
        } else if (ad.dst_alignment <= ad.src_alignment) {
            // Copy the pointers from the source bytes
            *reinterpret_cast<bytes_dtype_data *>(dst) = *reinterpret_cast<const bytes_dtype_data *>(src);
        } else {
            throw runtime_error("Attempted to reference source data when increasing bytes alignment");
        }
    }
} // anonymous namespace

void dynd::get_blockref_bytes_assignment_kernel(size_t dst_alignment,
                size_t src_alignment,
                kernel_instance<unary_operation_pair_t>& out_kernel)
{
    out_kernel.kernel.single = &blockref_bytes_assign_single;
    out_kernel.kernel.strided = NULL;

    make_auxiliary_data<blockref_bytes_assign_kernel_auxdata>(out_kernel.extra.auxdata);
    blockref_bytes_assign_kernel_auxdata& ad = out_kernel.extra.auxdata.get<blockref_bytes_assign_kernel_auxdata>();
    ad.dst_alignment = dst_alignment;
    ad.src_alignment = src_alignment;
}

/////////////////////////////////////////
// fixedbytes to blockref bytes assignment

namespace {
    struct fixedbytes_to_blockref_bytes_assign_kernel_auxdata {
        size_t dst_alignment, src_alignment;
        intptr_t src_element_size;
    };

    /** Does a single fixed-bytes copy */
    static void fixedbytes_to_blockref_bytes_assign_single(char *dst, const char *src, unary_kernel_static_data *extra)
    {
        fixedbytes_to_blockref_bytes_assign_kernel_auxdata& ad = get_auxiliary_data<fixedbytes_to_blockref_bytes_assign_kernel_auxdata>(extra->auxdata);
        const bytes_dtype_metadata *dst_md = reinterpret_cast<const bytes_dtype_metadata *>(extra->dst_metadata);
        // TODO: With some additional mechanism to track the source memory block, could
        //       avoid copying the bytes data.
        char *dst_begin = NULL, *dst_end = NULL;
        const char *src_begin = src;
        const char *src_end = src + ad.src_element_size;

        memory_block_pod_allocator_api *allocator = get_memory_block_pod_allocator_api(dst_md->blockref);

        allocator->allocate(dst_md->blockref, src_end - src_begin,
                        ad.dst_alignment, &dst_begin, &dst_end);
        memcpy(dst_begin, src_begin, src_end - src_begin);

        // Set the output
        reinterpret_cast<bytes_dtype_data *>(dst)->begin = dst_begin;
        reinterpret_cast<bytes_dtype_data *>(dst)->end = dst_end;
    }
} // anonymous namespace

void dynd::get_fixedbytes_to_blockref_bytes_assignment_kernel(size_t dst_alignment,
                intptr_t src_element_size, size_t src_alignment,
                kernel_instance<unary_operation_pair_t>& out_kernel)
{
    out_kernel.kernel.single = &fixedbytes_to_blockref_bytes_assign_single;
    out_kernel.kernel.strided = NULL;

    make_auxiliary_data<fixedbytes_to_blockref_bytes_assign_kernel_auxdata>(out_kernel.extra.auxdata);
    fixedbytes_to_blockref_bytes_assign_kernel_auxdata& ad =
                out_kernel.extra.auxdata.get<fixedbytes_to_blockref_bytes_assign_kernel_auxdata>();
    ad.dst_alignment = dst_alignment;
    ad.src_alignment = src_alignment;
    ad.src_element_size = src_element_size;
}
