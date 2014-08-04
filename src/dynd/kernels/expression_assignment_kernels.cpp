//
// Copyright (C) 2011-14 Mark Wiebe, DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <algorithm>

#include <dynd/type.hpp>
#include <dynd/types/base_expr_type.hpp>
#include <dynd/kernels/expression_assignment_kernels.hpp>

using namespace std;
using namespace dynd;


namespace {
    struct buffered_kernel_extra {
        typedef buffered_kernel_extra extra_type;

        ckernel_prefix base;
        // Offsets, from the start of &base, to the kernels
        // before and after the buffer
        size_t first_kernel_offset, second_kernel_offset;
        const base_type *buffer_tp;
        char *buffer_arrmeta;
        size_t buffer_data_offset, buffer_data_size;
        intptr_t buffer_stride;

        // Initializes the type and arrmeta for the buffer
        // NOTE: This does NOT initialize the buffer_data_offset,
        //       just the buffer_data_size.
        void init(const ndt::type& buffer_tp_, kernel_request_t kernreq) {
            size_t element_count = 1;
            switch (kernreq) {
                case kernel_request_single:
                    base.set_function<expr_single_t>(&single);
                    break;
                case kernel_request_strided:
                    base.set_function<expr_strided_t>(&strided);
                    element_count = DYND_BUFFER_CHUNK_SIZE;
                    break;
                default: {
                    stringstream ss;
                    ss << "buffered_kernel: unrecognized request " << (int)kernreq;
                    throw runtime_error(ss.str());
                }   
            }
            base.destructor = &destruct;
            // The kernel data owns a reference in buffer_tp
            buffer_tp = ndt::type(buffer_tp_).release();
            if (!buffer_tp_.is_builtin()) {
                size_t buffer_arrmeta_size = buffer_tp_.extended()->get_arrmeta_size();
                if (buffer_arrmeta_size > 0) {
                    buffer_arrmeta = reinterpret_cast<char *>(malloc(buffer_arrmeta_size));
                    if (buffer_arrmeta == NULL) {
                        throw bad_alloc();
                    }
                    buffer_tp->arrmeta_default_construct(buffer_arrmeta, 0, NULL);
                }
                // Make sure the buffer data size is pointer size-aligned
                buffer_stride = buffer_tp->get_default_data_size(0, NULL);
                buffer_data_size = inc_to_alignment(element_count * buffer_stride, sizeof(void *));
            } else {
                // Make sure the buffer data size is pointer size-aligned
                buffer_stride = buffer_tp_.get_data_size();
                buffer_data_size = inc_to_alignment(element_count * buffer_stride, sizeof(void *));
            }
            
        }

        static void single(char *dst, const char *const *src,
                           ckernel_prefix *extra)
        {
            char *eraw = reinterpret_cast<char *>(extra);
            extra_type *e = reinterpret_cast<extra_type *>(extra);
            ckernel_prefix *echild_first, *echild_second;
            expr_single_t opchild;
            const base_type *buffer_tp = e->buffer_tp;
            char *buffer_arrmeta = e->buffer_arrmeta;
            char *buffer_data_ptr = eraw + e->buffer_data_offset;
            echild_first = reinterpret_cast<ckernel_prefix *>(eraw + e->first_kernel_offset);
            echild_second = reinterpret_cast<ckernel_prefix *>(eraw + e->second_kernel_offset);

            // If the type needs it, initialize the buffer data to zero
            if (!is_builtin_type(buffer_tp) && (buffer_tp->get_flags()&type_flag_zeroinit) != 0) {
                memset(buffer_data_ptr, 0, e->buffer_data_size);
            }
            // First kernel (src -> buffer)
            opchild = echild_first->get_function<expr_single_t>();
            opchild(buffer_data_ptr, src, echild_first);
            // Second kernel (buffer -> dst)
            opchild = echild_second->get_function<expr_single_t>();
            opchild(dst, &buffer_data_ptr, echild_second);
            // Reset the buffer storage if used
            if (buffer_arrmeta != NULL) {
                buffer_tp->arrmeta_reset_buffers(buffer_arrmeta);
            }
        }
        static void strided(char *dst, intptr_t dst_stride,
                            const char *const *src, const intptr_t *src_stride,
                            size_t count, ckernel_prefix *extra)
        {
            char *eraw = reinterpret_cast<char *>(extra);
            extra_type *e = reinterpret_cast<extra_type *>(extra);
            ckernel_prefix *echild_first, *echild_second;
            expr_strided_t opchild_first, opchild_second;
            const base_type *buffer_tp = e->buffer_tp;
            char *buffer_arrmeta = e->buffer_arrmeta;
            char *buffer_data_ptr = eraw + e->buffer_data_offset;
            intptr_t buffer_stride = e->buffer_stride;
            echild_first = reinterpret_cast<ckernel_prefix *>(eraw + e->first_kernel_offset);
            echild_second = reinterpret_cast<ckernel_prefix *>(eraw + e->second_kernel_offset);

            opchild_first = echild_first->get_function<expr_strided_t>();
            opchild_second = echild_second->get_function<expr_strided_t>();
            const char *src0 = src[0];
            intptr_t src0_stride = src_stride[0];
            while (count > 0) {
                size_t chunk_size = min(count, (size_t)DYND_BUFFER_CHUNK_SIZE);
                // If the type needs it, initialize the buffer data to zero
                if (!is_builtin_type(buffer_tp) &&
                        (buffer_tp->get_flags() & type_flag_zeroinit) != 0) {
                    memset(buffer_data_ptr, 0, chunk_size * e->buffer_stride);
                }
                // First kernel (src -> buffer)
                opchild_first(buffer_data_ptr, buffer_stride, &src0,
                              &src0_stride, chunk_size, echild_first);
                // Second kernel (buffer -> dst)
                opchild_second(dst, dst_stride, &buffer_data_ptr,
                               &buffer_stride, chunk_size, echild_second);
                // Reset the buffer storage if used
                if (buffer_arrmeta != NULL) {
                    buffer_tp->arrmeta_reset_buffers(buffer_arrmeta);
                }
                dst += chunk_size * dst_stride;
                src0 += chunk_size * src0_stride;
                count -= chunk_size;
            }
        }
        static void destruct(ckernel_prefix *self)
        {
            extra_type *e = reinterpret_cast<extra_type *>(self);
            // Steal the buffer_tp reference count into a type
            ndt::type buffer_tp(e->buffer_tp, false);
            char *buffer_arrmeta = e->buffer_arrmeta;
            // Destruct and free the arrmeta for the buffer
            if (buffer_arrmeta != NULL) {
                buffer_tp.extended()->arrmeta_destruct(buffer_arrmeta);
                free(buffer_arrmeta);
            }
            // Destruct the child kernels
            self->destroy_child_ckernel(e->first_kernel_offset);
            self->destroy_child_ckernel(e->second_kernel_offset);
        }
    };
} // anonymous namespace

size_t dynd::make_expression_assignment_kernel(
    ckernel_builder *ckb, intptr_t ckb_offset, const ndt::type &dst_tp,
    const char *dst_arrmeta, const ndt::type &src_tp, const char *src_arrmeta,
    kernel_request_t kernreq, const eval::eval_context *ectx)
{
    intptr_t root_ckb_offset = ckb_offset;
    if (dst_tp.get_kind() == expr_kind) {
        const base_expr_type *dst_bed = dst_tp.tcast<base_expr_type>();
        if (src_tp == dst_bed->get_value_type()) {
            // In this case, it's just a chain of value -> operand on the dst side
            const ndt::type& opdt = dst_bed->get_operand_type();
            if (opdt.get_kind() != expr_kind) {
                // Leaf case, just a single value -> operand kernel
                return dst_bed->make_value_to_operand_assignment_kernel(ckb, ckb_offset,
                                dst_arrmeta, src_arrmeta, kernreq, ectx);
            } else {
                // Chain case, buffer one segment of the chain
                const ndt::type& buffer_tp = static_cast<const base_expr_type *>(
                                    opdt.extended())->get_value_type();
                buffered_kernel_extra *e = ckb->alloc_ck<buffered_kernel_extra>(ckb_offset);
                e->init(buffer_tp, kernreq);
                // Construct the first kernel (src -> buffer)
                e->first_kernel_offset = ckb_offset - root_ckb_offset;
                ckb_offset = dst_bed->make_value_to_operand_assignment_kernel(
                    ckb, ckb_offset, e->buffer_arrmeta, src_arrmeta, kernreq,
                    ectx);
                // Allocate the buffer data
                ckb_offset = inc_to_alignment(ckb_offset,
                                              buffer_tp.get_data_alignment());
                intptr_t buffer_data_offset = ckb_offset;
                kernels::inc_ckb_offset(ckb_offset, e->buffer_data_size);
                ckb->ensure_capacity(ckb_offset);
                // This may have invalidated the 'e' pointer, so get it again!
                e = ckb->get_at<buffered_kernel_extra>(root_ckb_offset);
                e->buffer_data_offset = buffer_data_offset - root_ckb_offset;
                // Construct the second kernel (buffer -> dst)
                e->second_kernel_offset = ckb_offset - root_ckb_offset;
                return ::make_assignment_kernel(ckb, ckb_offset,
                                opdt, dst_arrmeta,
                                buffer_tp, e->buffer_arrmeta,
                                kernreq, ectx);
            }
        } else {
            ndt::type buffer_tp;
            if (src_tp.get_kind() != expr_kind) {
                // In this case, need a data converting assignment to dst_tp.value_type(),
                // then the dst_tp expression chain
                buffer_tp = dst_bed->get_value_type();
            } else {
                // Both src and dst are expression types, use the src expression chain, and
                // the src value type to dst type as the two segments to buffer together
                buffer_tp = src_tp.value_type();
            }
            buffered_kernel_extra *e =
                ckb->alloc_ck<buffered_kernel_extra>(ckb_offset);
            e->init(buffer_tp, kernreq);
            // Construct the first kernel (src -> buffer)
            e->first_kernel_offset = ckb_offset - root_ckb_offset;
            ckb_offset = ::make_assignment_kernel(ckb, ckb_offset,
                            buffer_tp, e->buffer_arrmeta,
                            src_tp, src_arrmeta,
                            kernreq, ectx);
            ckb_offset = inc_to_alignment(ckb_offset, buffer_tp.get_data_alignment());
            // Allocate the buffer data
            intptr_t buffer_data_offset = ckb_offset;
            kernels::inc_ckb_offset(ckb_offset, e->buffer_data_size);
            ckb->ensure_capacity(ckb_offset);
            // This may have invalidated the 'e' pointer, so get it again!
            e = ckb->get_at<buffered_kernel_extra>(root_ckb_offset);
            e->buffer_data_offset = buffer_data_offset - root_ckb_offset;
            // Construct the second kernel (buffer -> dst)
            e->second_kernel_offset = ckb_offset - root_ckb_offset;
            return ::make_assignment_kernel(ckb, ckb_offset,
                            dst_tp, dst_arrmeta,
                            buffer_tp, e->buffer_arrmeta,
                            kernreq, ectx);
        }
    } else {
        const base_expr_type *src_bed = src_tp.tcast<base_expr_type>();
        if (dst_tp == src_bed->get_value_type()) {
            // In this case, it's just a chain of operand -> value on the src side
            const ndt::type& opdt = src_bed->get_operand_type();
            if (opdt.get_kind() != expr_kind) {
                // Leaf case, just a single value -> operand kernel
                return src_bed->make_operand_to_value_assignment_kernel(ckb, ckb_offset,
                                dst_arrmeta, src_arrmeta, kernreq, ectx);
            } else {
                // Chain case, buffer one segment of the chain
                const ndt::type& buffer_tp = static_cast<const base_expr_type *>(
                                opdt.extended())->get_value_type();
                buffered_kernel_extra *e = ckb->alloc_ck<buffered_kernel_extra>(ckb_offset);
                e->init(buffer_tp, kernreq);
                size_t buffer_data_size = e->buffer_data_size;
                // Construct the first kernel (src -> buffer)
                e->first_kernel_offset = ckb_offset - root_ckb_offset;
                ckb_offset = ::make_assignment_kernel(
                    ckb, ckb_offset, buffer_tp,
                    e->buffer_arrmeta, opdt, src_arrmeta, kernreq, ectx);
                // Allocate the buffer data
                ckb_offset = inc_to_alignment(ckb_offset, buffer_tp.get_data_alignment());
                size_t buffer_data_offset = ckb_offset;
                kernels::inc_ckb_offset(ckb_offset, buffer_data_size);
                ckb->ensure_capacity(ckb_offset);
                // This may have invalidated the 'e' pointer, so get it again!
                e = ckb->get_at<buffered_kernel_extra>(root_ckb_offset);
                e->buffer_data_offset = buffer_data_offset - root_ckb_offset;
                // Construct the second kernel (buffer -> dst)
                e->second_kernel_offset = ckb_offset - root_ckb_offset;
                return src_bed->make_operand_to_value_assignment_kernel(
                                ckb, ckb_offset,
                                dst_arrmeta, e->buffer_arrmeta, kernreq, ectx);
            }
        } else {
            // Put together the src expression chain and the src value type
            // to dst value type conversion
            const ndt::type& buffer_tp = src_tp.value_type();
            buffered_kernel_extra *e = ckb->alloc_ck<buffered_kernel_extra>(ckb_offset);
            e->init(buffer_tp, kernreq);
            size_t buffer_data_size = e->buffer_data_size;
            // Construct the first kernel (src -> buffer)
            e->first_kernel_offset = ckb_offset - root_ckb_offset;
            ckb_offset = ::make_assignment_kernel(ckb, ckb_offset, buffer_tp,
                                                  e->buffer_arrmeta, src_tp,
                                                  src_arrmeta, kernreq, ectx);
            // Allocate the buffer data
            ckb_offset =
                inc_to_alignment(ckb_offset, buffer_tp.get_data_alignment());
            size_t buffer_data_offset = ckb_offset;
            kernels::inc_ckb_offset(ckb_offset, buffer_data_size);
            ckb->ensure_capacity(ckb_offset);
            // This may have invalidated the 'e' pointer, so get it again!
            e = ckb->get_at<buffered_kernel_extra>(root_ckb_offset);
            e->buffer_data_offset = buffer_data_offset - root_ckb_offset;
            // Construct the second kernel (buffer -> dst)
            e->second_kernel_offset = ckb_offset - root_ckb_offset;
            return ::make_assignment_kernel(
                ckb, ckb_offset, dst_tp, dst_arrmeta,
                buffer_tp, e->buffer_arrmeta, kernreq, ectx);
        }
    }
}
