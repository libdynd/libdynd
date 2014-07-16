//
// Copyright (C) 2011-14 Mark Wiebe, DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <stdexcept>
#include <sstream>

#include <dynd/type.hpp>
#include <dynd/diagnostics.hpp>
#include <dynd/kernels/assignment_kernels.hpp>
#include <dynd/kernels/var_dim_assignment_kernels.hpp>
#include <dynd/types/var_dim_type.hpp>
#include <dynd/types/strided_dim_type.hpp>
#include <dynd/types/cfixed_dim_type.hpp>

using namespace std;
using namespace dynd;

/////////////////////////////////////////
// broadcast to var array assignment

namespace {
    struct broadcast_to_var_assign_ck : public kernels::unary_ck<broadcast_to_var_assign_ck> {
        intptr_t m_dst_target_alignment;
        const var_dim_type_arrmeta *m_dst_md;

        inline void single(char *dst, const char *src)
        {
            var_dim_type_data *dst_d = reinterpret_cast<var_dim_type_data *>(dst);
            ckernel_prefix *child = get_child_ckernel();
            expr_strided_t child_fn = child->get_function<expr_strided_t>();
            if (dst_d->begin == NULL) {
                if (m_dst_md->offset != 0) {
                    throw runtime_error("Cannot assign to an uninitialized dynd var_dim which has a non-zero offset");
                }
                // If we're writing to an empty array, have to allocate the output
                memory_block_data *memblock = m_dst_md->blockref;
                if (memblock->m_type == objectarray_memory_block_type) {
                    memory_block_objectarray_allocator_api *allocator =
                                    get_memory_block_objectarray_allocator_api(memblock);

                    // Allocate the output array data
                    dst_d->begin = allocator->allocate(memblock, 1);
                } else {
                    memory_block_pod_allocator_api *allocator =
                                    get_memory_block_pod_allocator_api(memblock);

                    // Allocate the output array data
                    char *dst_end = NULL;
                    allocator->allocate(memblock, m_dst_md->stride,
                                m_dst_target_alignment, &dst_d->begin, &dst_end);
                }
                dst_d->size = 1;
                // Copy a single input to the newly allocated element
                intptr_t zero_stride = 0;
                child_fn(dst_d->begin, 0, &src, &zero_stride, 1, child);
            } else {
                // We're broadcasting elements to an already allocated array segment
                dst = dst_d->begin + m_dst_md->offset;
                intptr_t zero_stride = 0;
                child_fn(dst, m_dst_md->stride, &src, &zero_stride, dst_d->size, child);
            }
        }

        void destruct_children()
        {
            get_child_ckernel()->destroy();
        }
    };
} // anonymous namespace

size_t dynd::make_broadcast_to_var_dim_assignment_kernel(
    ckernel_builder *ckb, intptr_t ckb_offset, const ndt::type &dst_var_dim_tp,
    const char *dst_arrmeta, const ndt::type &src_tp, const char *src_arrmeta,
    kernel_request_t kernreq, const eval::eval_context *ectx)
{
    typedef broadcast_to_var_assign_ck self_type;
    if (dst_var_dim_tp.get_type_id() != var_dim_type_id) {
        stringstream ss;
        ss << "make_broadcast_to_blockref_array_assignment_kernel: provided "
              "destination type " << dst_var_dim_tp << " is not a var_dim";
        throw runtime_error(ss.str());
    }
    const var_dim_type *dst_vad = dst_var_dim_tp.tcast<var_dim_type>();
    const var_dim_type_arrmeta *dst_md =
                    reinterpret_cast<const var_dim_type_arrmeta *>(dst_arrmeta);

    self_type *self = self_type::create(ckb, kernreq, ckb_offset);
    self->m_dst_target_alignment = dst_vad->get_target_alignment();
    self->m_dst_md = dst_md;
    return ::make_assignment_kernel(
        ckb, ckb_offset, dst_vad->get_element_type(),
        dst_arrmeta + sizeof(var_dim_type_arrmeta), src_tp, src_arrmeta,
        kernel_request_strided, ectx);
}

/////////////////////////////////////////
// var array to var array assignment

namespace {
    struct var_assign_ck : public kernels::unary_ck<var_assign_ck> {
        intptr_t m_dst_target_alignment;
        const var_dim_type_arrmeta *m_dst_md, *m_src_md;

        inline void single(char *dst, const char *src)
        {
            var_dim_type_data *dst_d = reinterpret_cast<var_dim_type_data *>(dst);
            const var_dim_type_data *src_d = reinterpret_cast<const var_dim_type_data *>(src);
            ckernel_prefix *child = get_child_ckernel();
            expr_strided_t child_fn = child->get_function<expr_strided_t>();
            if (dst_d->begin == NULL) {
                if (m_dst_md->offset != 0) {
                    throw runtime_error("Cannot assign to an uninitialized dynd var_dim which has a non-zero offset");
                }
                // As a special case, allow uninitialized -> uninitialized assignment as a no-op
                if (src_d->begin != NULL) {
                    intptr_t dim_size = src_d->size;
                    intptr_t dst_stride = m_dst_md->stride, src_stride = m_src_md->stride;
                    // If we're writing to an empty array, have to allocate the output
                    memory_block_data *memblock = m_dst_md->blockref;
                    if (memblock->m_type == objectarray_memory_block_type) {
                        memory_block_objectarray_allocator_api *allocator =
                                        get_memory_block_objectarray_allocator_api(memblock);

                        // Allocate the output array data
                        dst_d->begin = allocator->allocate(memblock, dim_size);
                    } else {
                        memory_block_pod_allocator_api *allocator =
                                        get_memory_block_pod_allocator_api(memblock);

                        // Allocate the output array data
                        char *dst_end = NULL;
                        allocator->allocate(memblock, dim_size * dst_stride,
                                    m_dst_target_alignment, &dst_d->begin, &dst_end);
                    }
                    dst_d->size = dim_size;
                    // Copy to the newly allocated element
                    dst = dst_d->begin;
                    src = src_d->begin + m_src_md->offset;
                    child_fn(dst, dst_stride, &src, &src_stride, dim_size, child);
                }
            } else {
                if (src_d->begin == NULL) {
                    throw runtime_error("Cannot assign an uninitialized dynd "
                                        "var_dim to an initialized one");
                }
                intptr_t dst_dim_size = dst_d->size, src_dim_size = src_d->size;
                intptr_t dst_stride = m_dst_md->stride,
                                src_stride = src_dim_size != 1 ? m_src_md->stride : 0;
                // Check for a broadcasting error
                if (src_dim_size != 1 && dst_dim_size != src_dim_size) {
                    stringstream ss;
                    ss << "error broadcasting input var_dim sized ";
                    ss << src_dim_size << " to output var_dim sized " << dst_dim_size;
                    throw broadcast_error(ss.str());
                }
                // We're copying/broadcasting elements to an already allocated array segment
                dst = dst_d->begin + m_dst_md->offset;
                src = src_d->begin + m_src_md->offset;
                child_fn(dst, dst_stride, &src, &src_stride, dst_dim_size, child);
            }
        }

        inline void destruct_children()
        {
            get_child_ckernel()->destroy();
        }
    };
} // anonymous namespace

size_t dynd::make_var_dim_assignment_kernel(
    ckernel_builder *ckb, intptr_t ckb_offset, const ndt::type &dst_var_dim_tp,
    const char *dst_arrmeta, const ndt::type &src_var_dim_tp,
    const char *src_arrmeta, kernel_request_t kernreq,
    const eval::eval_context *ectx)
{
    typedef var_assign_ck self_type;
    if (dst_var_dim_tp.get_type_id() != var_dim_type_id) {
        stringstream ss;
        ss << "make_broadcast_to_blockref_array_assignment_kernel: provided "
              "destination type " << dst_var_dim_tp << " is not a var_dim";
        throw runtime_error(ss.str());
    }
    if (src_var_dim_tp.get_type_id() != var_dim_type_id) {
        stringstream ss;
        ss << "make_broadcast_to_blockref_array_assignment_kernel: provided "
              "source type " << src_var_dim_tp << " is not a var_dim";
        throw runtime_error(ss.str());
    }
    const var_dim_type *dst_vad = dst_var_dim_tp.tcast<var_dim_type>();
    const var_dim_type *src_vad = src_var_dim_tp.tcast<var_dim_type>();
    const var_dim_type_arrmeta *dst_md =
        reinterpret_cast<const var_dim_type_arrmeta *>(dst_arrmeta);
    const var_dim_type_arrmeta *src_md =
        reinterpret_cast<const var_dim_type_arrmeta *>(src_arrmeta);

    self_type *self = self_type::create(ckb, kernreq, ckb_offset);
    self->m_dst_target_alignment = dst_vad->get_target_alignment();
    self->m_dst_md = dst_md;
    self->m_src_md = src_md;
    return ::make_assignment_kernel(
        ckb, ckb_offset, dst_vad->get_element_type(),
        dst_arrmeta + sizeof(var_dim_type_arrmeta), src_vad->get_element_type(),
        src_arrmeta + sizeof(var_dim_type_arrmeta), kernel_request_strided,
        ectx);
}

/////////////////////////////////////////
// strided array to var array assignment

namespace {
    struct strided_to_var_assign_ck : public kernels::unary_ck<strided_to_var_assign_ck> {
        intptr_t m_dst_target_alignment;
        const var_dim_type_arrmeta *m_dst_md;
        intptr_t m_src_stride, m_src_dim_size;

        inline void single(char *dst, const char *src)
        {
            var_dim_type_data *dst_d = reinterpret_cast<var_dim_type_data *>(dst);
            ckernel_prefix *child = get_child_ckernel();
            expr_strided_t child_fn = child->get_function<expr_strided_t>();
            if (dst_d->begin == NULL) {
                if (m_dst_md->offset != 0) {
                    throw runtime_error("Cannot assign to an uninitialized dynd var_dim which has a non-zero offset");
                }
                intptr_t dim_size = m_src_dim_size;
                intptr_t dst_stride = m_dst_md->stride, src_stride = m_src_stride;
                // If we're writing to an empty array, have to allocate the output
                memory_block_data *memblock = m_dst_md->blockref;
                if (memblock->m_type == objectarray_memory_block_type) {
                    memory_block_objectarray_allocator_api *allocator =
                                    get_memory_block_objectarray_allocator_api(memblock);

                    // Allocate the output array data
                    dst_d->begin = allocator->allocate(memblock, dim_size);
                } else {
                    memory_block_pod_allocator_api *allocator =
                                    get_memory_block_pod_allocator_api(memblock);

                    // Allocate the output array data
                    char *dst_end = NULL;
                    allocator->allocate(memblock, dim_size * dst_stride,
                                m_dst_target_alignment, &dst_d->begin, &dst_end);
                }
                dst_d->size = dim_size;
                // Copy to the newly allocated element
                dst = dst_d->begin;
                child_fn(dst, dst_stride, &src, &src_stride, dim_size, child);
            } else {
                intptr_t dst_dim_size = dst_d->size, src_dim_size = m_src_dim_size;
                intptr_t dst_stride = m_dst_md->stride, src_stride = m_src_stride;
                // Check for a broadcasting error
                if (src_dim_size != 1 && dst_dim_size != src_dim_size) {
                    stringstream ss;
                    ss << "error broadcasting input strided array sized " << src_dim_size;
                    ss << " to output var_dim sized " << dst_dim_size;
                    throw broadcast_error(ss.str());
                }
                // We're copying/broadcasting elements to an already allocated array segment
                dst = dst_d->begin + m_dst_md->offset;
                child_fn(dst, dst_stride, &src, &src_stride, dst_dim_size, child);
            }
        }

        inline void destruct_children()
        {
            base.destroy_child_ckernel(sizeof(self_type));
        }
    };
} // anonymous namespace

size_t dynd::make_strided_to_var_dim_assignment_kernel(
    ckernel_builder *ckb, intptr_t ckb_offset, const ndt::type &dst_var_dim_tp,
    const char *dst_arrmeta, intptr_t src_dim_size, intptr_t src_stride,
    const ndt::type &src_el_tp, const char *src_el_arrmeta,
    kernel_request_t kernreq, const eval::eval_context *ectx)
{
    typedef strided_to_var_assign_ck self_type;
    if (dst_var_dim_tp.get_type_id() != var_dim_type_id) {
        stringstream ss;
        ss << "make_strided_to_var_dim_assignment_kernel: provided destination "
              "type " << dst_var_dim_tp << " is not a var_dim";
        throw runtime_error(ss.str());
    }
    const var_dim_type *dst_vad = dst_var_dim_tp.tcast<var_dim_type>();
    const var_dim_type_arrmeta *dst_md =
                    reinterpret_cast<const var_dim_type_arrmeta *>(dst_arrmeta);

    self_type *self = self_type::create(ckb, kernreq, ckb_offset);
    self->m_dst_target_alignment = dst_vad->get_target_alignment();
    self->m_dst_md = dst_md;
    self->m_src_stride = src_stride;
    self->m_src_dim_size = src_dim_size;

    return ::make_assignment_kernel(
        ckb, ckb_offset, dst_vad->get_element_type(),
        dst_arrmeta + sizeof(var_dim_type_arrmeta), src_el_tp, src_el_arrmeta,
        kernel_request_strided, ectx);
}

/////////////////////////////////////////
// var array to strided array assignment

namespace {
    struct var_to_strided_assign_ck : public kernels::unary_ck<var_to_strided_assign_ck> {
        intptr_t m_dst_stride, m_dst_dim_size;
        const var_dim_type_arrmeta *m_src_md;

        inline void single(char *dst, const char *src)
        {
            const var_dim_type_data *src_d = reinterpret_cast<const var_dim_type_data *>(src);
            ckernel_prefix *child = get_child_ckernel();
            expr_strided_t child_fn = child->get_function<expr_strided_t>();
            if (src_d->begin == NULL) {
                throw runtime_error("Cannot assign an uninitialized dynd var "
                                    "array to a strided one");
            }

            intptr_t dst_dim_size = m_dst_dim_size, src_dim_size = src_d->size;
            intptr_t dst_stride = m_dst_stride,
                     src_stride = src_dim_size != 1 ? m_src_md->stride : 0;
            // Check for a broadcasting error
            if (src_dim_size != 1 && dst_dim_size != src_dim_size) {
                stringstream ss;
                ss << "error broadcasting input var array sized " << src_dim_size;
                ss << " to output strided array sized " << dst_dim_size;
                throw broadcast_error(ss.str());
            }
            // Copying/broadcasting elements
            src = src_d->begin + m_src_md->offset;
            child_fn(dst, dst_stride, &src, &src_stride, dst_dim_size, child);
        }

        inline void destruct_children()
        {
            get_child_ckernel()->destroy();
        }
    };
} // anonymous namespace

size_t dynd::make_var_to_strided_dim_assignment_kernel(
    ckernel_builder *ckb, intptr_t ckb_offset,
    const ndt::type &dst_strided_dim_tp, const char *dst_arrmeta,
    const ndt::type &src_var_dim_tp, const char *src_arrmeta,
    kernel_request_t kernreq, const eval::eval_context *ectx)
{
    typedef var_to_strided_assign_ck self_type;
    if (src_var_dim_tp.get_type_id() != var_dim_type_id) {
        stringstream ss;
        ss << "make_var_to_strided_dim_assignment_kernel: provided source type " << src_var_dim_tp << " is not a var_dim";
        throw runtime_error(ss.str());
    }
    const var_dim_type *src_vad = src_var_dim_tp.tcast<var_dim_type>();
    const var_dim_type_arrmeta *src_md =
        reinterpret_cast<const var_dim_type_arrmeta *>(src_arrmeta);

    self_type *self = self_type::create(ckb, kernreq, ckb_offset);
    ndt::type dst_element_tp;
    const char *dst_element_arrmeta;
    if (!dst_strided_dim_tp.get_as_strided(dst_arrmeta, &self->m_dst_dim_size,
                                           &self->m_dst_stride, &dst_element_tp,
                                           &dst_element_arrmeta)) {
        stringstream ss;
        ss << "make_var_to_strided_dim_assignment_kernel: provided destination "
              "type " << dst_strided_dim_tp
           << " is not a strided_dim or fixed_array";
        throw runtime_error(ss.str());
    }

    self->m_src_md = src_md;
    return ::make_assignment_kernel(ckb, ckb_offset,
                                    dst_element_tp, dst_element_arrmeta,
                                    src_vad->get_element_type(),
                                    src_arrmeta + sizeof(var_dim_type_arrmeta),
                                    kernel_request_strided, ectx);
}
