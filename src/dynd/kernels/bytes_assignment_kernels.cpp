//
// Copyright (C) 2011-14 Mark Wiebe, DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <stdexcept>
#include <sstream>

#include <dynd/type.hpp>
#include <dynd/diagnostics.hpp>
#include <dynd/kernels/bytes_assignment_kernels.hpp>
#include <dynd/types/bytes_type.hpp>

using namespace std;
using namespace dynd;

/////////////////////////////////////////
// blockref bytes to blockref bytes assignment

namespace {
    struct blockref_bytes_kernel_extra {
        typedef blockref_bytes_kernel_extra extra_type;

        ckernel_prefix base;
        size_t dst_alignment, src_alignment;
        const bytes_type_arrmeta *dst_arrmeta, *src_arrmeta;

        /** Does a single blockref-string copy */
        static void single(char *dst, const char *const *src,
                           ckernel_prefix *extra)
        {
            extra_type *e = reinterpret_cast<extra_type *>(extra);
            const bytes_type_arrmeta *dst_md = e->dst_arrmeta;
            const bytes_type_arrmeta *src_md = e->src_arrmeta;
            bytes_type_data *dst_d = reinterpret_cast<bytes_type_data *>(dst);
            const bytes_type_data *src_d =
                *reinterpret_cast<const bytes_type_data *const *>(src);

            if (dst_d->begin != NULL) {
                throw runtime_error("Cannot assign to an already initialized dynd string");
            } else if (src_d->begin == NULL) {
                // Allow uninitialized -> uninitialized assignment as a special case, for
                // (future) missing data support
                return;
            }

            // If the blockrefs are different, require a copy operation
            if (dst_md->blockref != src_md->blockref) {
                char *dst_begin = NULL, *dst_end = NULL;
                const char *src_begin = src_d->begin;
                const char *src_end = src_d->end;
                memory_block_pod_allocator_api *allocator = get_memory_block_pod_allocator_api(dst_md->blockref);

                allocator->allocate(dst_md->blockref, src_end - src_begin,
                                e->dst_alignment, &dst_begin, &dst_end);
                memcpy(dst_begin, src_begin, src_end - src_begin);

                // Set the output
                dst_d->begin = dst_begin;
                dst_d->end = dst_end;
            } else if (e->dst_alignment <= e->src_alignment) {
                // Copy the pointers from the source bytes
                *dst_d = *src_d;
            } else {
                throw runtime_error("Attempted to reference source data when increasing bytes alignment");
            }
        }
    };
} // anonymous namespace

size_t dynd::make_blockref_bytes_assignment_kernel(
    ckernel_builder *out, size_t offset_out, size_t dst_alignment,
    const char *dst_arrmeta, size_t src_alignment, const char *src_arrmeta,
    kernel_request_t kernreq, const eval::eval_context *DYND_UNUSED(ectx))
{
    // Adapt the incoming request to a 'single' kernel
    offset_out =
        make_kernreq_to_single_kernel_adapter(out, offset_out, 1, kernreq);
    out->ensure_capacity_leaf(offset_out + sizeof(blockref_bytes_kernel_extra));
    blockref_bytes_kernel_extra *e = out->get_at<blockref_bytes_kernel_extra>(offset_out);
    e->base.set_function<expr_single_t>(&blockref_bytes_kernel_extra::single);
    e->dst_alignment = dst_alignment;
    e->src_alignment = src_alignment;
    e->dst_arrmeta = reinterpret_cast<const bytes_type_arrmeta *>(dst_arrmeta);
    e->src_arrmeta = reinterpret_cast<const bytes_type_arrmeta *>(src_arrmeta);
    return offset_out + sizeof(blockref_bytes_kernel_extra);
}

/////////////////////////////////////////
// fixedbytes to blockref bytes assignment

namespace {
    struct fixedbytes_to_blockref_bytes_kernel_extra {
        typedef fixedbytes_to_blockref_bytes_kernel_extra extra_type;

        ckernel_prefix base;
        size_t dst_alignment;
        intptr_t src_data_size;
        size_t src_alignment;
        const bytes_type_arrmeta *dst_arrmeta;

        /** Does a single fixed-bytes copy */
        static void single(char *dst, const char *const *src,
                           ckernel_prefix *extra)
        {
            extra_type *e = reinterpret_cast<extra_type *>(extra);
            const bytes_type_arrmeta *dst_md = e->dst_arrmeta;
            // TODO: With some additional mechanism to track the source memory block, could
            //       avoid copying the bytes data.
            char *dst_begin = NULL, *dst_end = NULL;
            const char *src_begin = src[0];
            const char *src_end = src_begin + e->src_data_size;
            bytes_type_data *dst_d = reinterpret_cast<bytes_type_data *>(dst);

            if (dst_d->begin != NULL) {
                throw runtime_error("Cannot assign to an already initialized dynd string");
            }

            memory_block_pod_allocator_api *allocator = get_memory_block_pod_allocator_api(dst_md->blockref);

            allocator->allocate(dst_md->blockref, src_end - src_begin,
                            e->dst_alignment, &dst_begin, &dst_end);
            memcpy(dst_begin, src_begin, src_end - src_begin);

            // Set the output
            dst_d->begin = dst_begin;
            dst_d->end = dst_end;
        }
    };
} // anonymous namespace

size_t dynd::make_fixedbytes_to_blockref_bytes_assignment_kernel(
    ckernel_builder *out, size_t offset_out, size_t dst_alignment,
    const char *dst_arrmeta, intptr_t src_data_size, size_t src_alignment,
    kernel_request_t kernreq, const eval::eval_context *DYND_UNUSED(ectx))
{
    // Adapt the incoming request to a 'single' kernel
    offset_out =
        make_kernreq_to_single_kernel_adapter(out, offset_out, 1, kernreq);
    out->ensure_capacity_leaf(offset_out + sizeof(fixedbytes_to_blockref_bytes_kernel_extra));
    fixedbytes_to_blockref_bytes_kernel_extra *e = out->get_at<fixedbytes_to_blockref_bytes_kernel_extra>(offset_out);
    e->base.set_function<expr_single_t>(&fixedbytes_to_blockref_bytes_kernel_extra::single);
    e->dst_alignment = dst_alignment;
    e->src_data_size = src_data_size;
    e->src_alignment = src_alignment;
    e->dst_arrmeta = reinterpret_cast<const bytes_type_arrmeta *>(dst_arrmeta);
    return offset_out + sizeof(fixedbytes_to_blockref_bytes_kernel_extra);
}
