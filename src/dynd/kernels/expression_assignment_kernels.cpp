//
// Copyright (C) 2011-13 Mark Wiebe, DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <dynd/type.hpp>
#include <dynd/types/base_expression_type.hpp>
#include <dynd/kernels/expression_assignment_kernels.hpp>

using namespace std;
using namespace dynd;


namespace {
    struct buffered_kernel_extra {
        typedef buffered_kernel_extra extra_type;

        kernel_data_prefix base;
        // Offsets, from the start of &base, to the kernels
        // before and after the buffer
        size_t first_kernel_offset, second_kernel_offset;
        const base_type *buffer_dt;
        char *buffer_metadata;
        size_t buffer_data_offset, buffer_data_size;
        intptr_t buffer_stride;

        // Initializes the dtype and metadata for the buffer
        // NOTE: This does NOT initialize the buffer_data_offset,
        //       just the buffer_data_size.
        void init(const ndt::type& buffer_dt_, kernel_request_t kernreq) {
            size_t element_count = 1;
            switch (kernreq) {
                case kernel_request_single:
                    base.set_function<unary_single_operation_t>(&single);
                    break;
                case kernel_request_strided:
                    base.set_function<unary_strided_operation_t>(&strided);
                    element_count = DYND_BUFFER_CHUNK_SIZE;
                    break;
                default: {
                    stringstream ss;
                    ss << "buffered_kernel: unrecognized request " << (int)kernreq;
                    throw runtime_error(ss.str());
                }   
            }
            base.destructor = &destruct;
            // The kernel data owns a reference in buffer_dt
            buffer_dt = ndt::type(buffer_dt_).release();
            if (!buffer_dt_.is_builtin()) {
                size_t buffer_metadata_size = buffer_dt_.extended()->get_metadata_size();
                if (buffer_metadata_size > 0) {
                    buffer_metadata = reinterpret_cast<char *>(malloc(buffer_metadata_size));
                    if (buffer_metadata == NULL) {
                        throw bad_alloc();
                    }
                    buffer_dt->metadata_default_construct(buffer_metadata, 0, NULL);
                }
                // Make sure the buffer data size is pointer size-aligned
                buffer_stride = buffer_dt->get_default_data_size(0, NULL);
                buffer_data_size = inc_to_alignment(element_count * buffer_stride, sizeof(void *));
            } else {
                // Make sure the buffer data size is pointer size-aligned
                buffer_stride = buffer_dt_.get_data_size();
                buffer_data_size = inc_to_alignment(element_count * buffer_stride, sizeof(void *));
            }
            
        }

        static void single(char *dst, const char *src,
                            kernel_data_prefix *extra)
        {
            char *eraw = reinterpret_cast<char *>(extra);
            extra_type *e = reinterpret_cast<extra_type *>(extra);
            kernel_data_prefix *echild_first, *echild_second;
            unary_single_operation_t opchild;
            const base_type *buffer_dt = e->buffer_dt;
            char *buffer_metadata = e->buffer_metadata;
            char *buffer_data_ptr = eraw + e->buffer_data_offset;
            echild_first = reinterpret_cast<kernel_data_prefix *>(eraw + e->first_kernel_offset);
            echild_second = reinterpret_cast<kernel_data_prefix *>(eraw + e->second_kernel_offset);

            // If the type needs it, initialize the buffer data to zero
            if (!is_builtin_type(buffer_dt) && (buffer_dt->get_flags()&type_flag_zeroinit) != 0) {
                memset(buffer_data_ptr, 0, e->buffer_data_size);
            }
            // First kernel (src -> buffer)
            opchild = echild_first->get_function<unary_single_operation_t>();
            opchild(buffer_data_ptr, src, echild_first);
            // Second kernel (buffer -> dst)
            opchild = echild_second->get_function<unary_single_operation_t>();
            opchild(dst, buffer_data_ptr, echild_second);
            // Reset the buffer storage if used
            if (buffer_metadata != NULL) {
                buffer_dt->metadata_reset_buffers(buffer_metadata);
            }
        }
        static void strided(char *dst, intptr_t dst_stride,
                        const char *src, intptr_t src_stride,
                        size_t count, kernel_data_prefix *extra)
        {
            char *eraw = reinterpret_cast<char *>(extra);
            extra_type *e = reinterpret_cast<extra_type *>(extra);
            kernel_data_prefix *echild_first, *echild_second;
            unary_strided_operation_t opchild_first, opchild_second;
            const base_type *buffer_dt = e->buffer_dt;
            char *buffer_metadata = e->buffer_metadata;
            char *buffer_data_ptr = eraw + e->buffer_data_offset;
            size_t buffer_stride = e->buffer_stride;
            echild_first = reinterpret_cast<kernel_data_prefix *>(eraw + e->first_kernel_offset);
            echild_second = reinterpret_cast<kernel_data_prefix *>(eraw + e->second_kernel_offset);

            opchild_first = echild_first->get_function<unary_strided_operation_t>();
            opchild_second = echild_second->get_function<unary_strided_operation_t>();
            while (count > 0) {
                size_t chunk_size = min(DYND_BUFFER_CHUNK_SIZE, count);
                // If the type needs it, initialize the buffer data to zero
                if (!is_builtin_type(buffer_dt) && (buffer_dt->get_flags()&type_flag_zeroinit) != 0) {
                    memset(buffer_data_ptr, 0, chunk_size * e->buffer_stride);
                }
                // First kernel (src -> buffer)
                opchild_first(buffer_data_ptr, buffer_stride, src, src_stride, chunk_size, echild_first);
                // Second kernel (buffer -> dst)
                opchild_second(dst, dst_stride, buffer_data_ptr, buffer_stride, chunk_size, echild_second);
                // Reset the buffer storage if used
                if (buffer_metadata != NULL) {
                    buffer_dt->metadata_reset_buffers(buffer_metadata);
                }
                count -= chunk_size;
            }
        }
        static void destruct(kernel_data_prefix *extra)
        {
            char *eraw = reinterpret_cast<char *>(extra);
            extra_type *e = reinterpret_cast<extra_type *>(extra);
            // Steal the buffer_dt reference count into a dtype
            ndt::type buffer_dt(e->buffer_dt, false);
            char *buffer_metadata = e->buffer_metadata;
            // Destruct and free the metadata for the buffer
            if (buffer_metadata != NULL) {
                buffer_dt.extended()->metadata_destruct(buffer_metadata);
                free(buffer_metadata);
            }
            kernel_data_prefix *echild;
            // Destruct the first kernel
            if (e->first_kernel_offset != 0) {
                echild = reinterpret_cast<kernel_data_prefix *>(eraw + e->first_kernel_offset);
                if (echild->destructor) {
                    echild->destructor(echild);
                }
            }
            // Destruct the second kernel
            if (e->second_kernel_offset != 0) {
                echild = reinterpret_cast<kernel_data_prefix *>(eraw + e->second_kernel_offset);
                if (echild->destructor) {
                    echild->destructor(echild);
                }
            }
        }
    };
} // anonymous namespace

size_t dynd::make_expression_assignment_kernel(
                hierarchical_kernel *out, size_t offset_out,
                const ndt::type& dst_dt, const char *dst_metadata,
                const ndt::type& src_dt, const char *src_metadata,
                kernel_request_t kernreq, assign_error_mode errmode,
                const eval::eval_context *ectx)
{
    if (dst_dt.get_kind() == expression_kind) {
        const base_expression_type *dst_bed = static_cast<const base_expression_type *>(dst_dt.extended());
        if (src_dt == dst_bed->get_value_type()) {
            // In this case, it's just a chain of value -> operand on the dst side
            const ndt::type& opdt = dst_bed->get_operand_type();
            if (opdt.get_kind() != expression_kind) {
                // Leaf case, just a single value -> operand kernel
                return dst_bed->make_value_to_operand_assignment_kernel(out, offset_out,
                                dst_metadata, src_metadata, kernreq, ectx);
            } else {
                // Chain case, buffer one segment of the chain
                const ndt::type& buffer_dt = static_cast<const base_expression_type *>(
                                    opdt.extended())->get_value_type();
                out->ensure_capacity(offset_out + sizeof(buffered_kernel_extra));
                buffered_kernel_extra *e = out->get_at<buffered_kernel_extra>(offset_out);
                e->init(buffer_dt, kernreq);
                size_t buffer_data_size = e->buffer_data_size;
                // Construct the first kernel (src -> buffer)
                e->first_kernel_offset = sizeof(buffered_kernel_extra);
                size_t buffer_data_offset;
                buffer_data_offset = dst_bed->make_value_to_operand_assignment_kernel(
                                out, offset_out + e->first_kernel_offset,
                                e->buffer_metadata, src_metadata, kernreq, ectx);
                // Allocate the buffer data
                buffer_data_offset = inc_to_alignment(buffer_data_offset, buffer_dt.get_data_alignment());
                out->ensure_capacity(offset_out + buffer_data_offset + buffer_data_size);
                // This may have invalidated the 'e' pointer, so get it again!
                e = out->get_at<buffered_kernel_extra>(offset_out);
                e->buffer_data_offset = buffer_data_offset;
                // Construct the second kernel (buffer -> dst)
                e->second_kernel_offset = buffer_data_offset + e->buffer_data_size;
                return ::make_assignment_kernel(out, offset_out + e->second_kernel_offset,
                                opdt, dst_metadata,
                                buffer_dt, e->buffer_metadata,
                                kernreq, errmode, ectx);
            }
        } else {
            ndt::type buffer_dt;
            if (src_dt.get_kind() != expression_kind) {
                // In this case, need a data converting assignment to dst_dt.value_type(),
                // then the dst_dt expression chain
                buffer_dt = dst_bed->get_value_type();
            } else {
                // Both src and dst are expression types, use the src expression chain, and
                // the src value dtype to dst dtype as the two segments to buffer together
                buffer_dt = src_dt.value_type();
            }
            out->ensure_capacity(offset_out + sizeof(buffered_kernel_extra));
            buffered_kernel_extra *e = out->get_at<buffered_kernel_extra>(offset_out);
            e->init(buffer_dt, kernreq);
            size_t buffer_data_size = e->buffer_data_size;
            // Construct the first kernel (src -> buffer)
            e->first_kernel_offset = sizeof(buffered_kernel_extra);
            size_t buffer_data_offset;
            buffer_data_offset = ::make_assignment_kernel(out, offset_out + e->first_kernel_offset,
                            buffer_dt, e->buffer_metadata,
                            src_dt, src_metadata,
                            kernreq, errmode, ectx);
            // Allocate the buffer data
            buffer_data_offset = inc_to_alignment(buffer_data_offset, buffer_dt.get_data_alignment());
            out->ensure_capacity(offset_out + buffer_data_offset + buffer_data_size);
            // This may have invalidated the 'e' pointer, so get it again!
            e = out->get_at<buffered_kernel_extra>(offset_out);
            e->buffer_data_offset = buffer_data_offset;
            // Construct the second kernel (buffer -> dst)
            e->second_kernel_offset = buffer_data_offset + e->buffer_data_size;
            return ::make_assignment_kernel(out, offset_out + e->second_kernel_offset,
                            dst_dt, dst_metadata,
                            buffer_dt, e->buffer_metadata,
                            kernreq, errmode, ectx);
        }
    } else {
        const base_expression_type *src_bed = static_cast<const base_expression_type *>(src_dt.extended());
        if (dst_dt == src_bed->get_value_type()) {
            // In this case, it's just a chain of operand -> value on the src side
            const ndt::type& opdt = src_bed->get_operand_type();
            if (opdt.get_kind() != expression_kind) {
                // Leaf case, just a single value -> operand kernel
                return src_bed->make_operand_to_value_assignment_kernel(out, offset_out,
                                dst_metadata, src_metadata, kernreq, ectx);
            } else {
                // Chain case, buffer one segment of the chain
                const ndt::type& buffer_dt = static_cast<const base_expression_type *>(
                                opdt.extended())->get_value_type();
                out->ensure_capacity(offset_out + sizeof(buffered_kernel_extra));
                buffered_kernel_extra *e = out->get_at<buffered_kernel_extra>(offset_out);
                e->init(buffer_dt, kernreq);
                size_t buffer_data_size = e->buffer_data_size;
                // Construct the first kernel (src -> buffer)
                e->first_kernel_offset = sizeof(buffered_kernel_extra);
                size_t buffer_data_offset;
                buffer_data_offset = ::make_assignment_kernel(out, offset_out + e->first_kernel_offset,
                                buffer_dt, e->buffer_metadata,
                                opdt, src_metadata,
                                kernreq, errmode, ectx);
                // Allocate the buffer data
                buffer_data_offset = inc_to_alignment(buffer_data_offset, buffer_dt.get_data_alignment());
                out->ensure_capacity(offset_out + buffer_data_offset + buffer_data_size);
                // This may have invalidated the 'e' pointer, so get it again!
                e = out->get_at<buffered_kernel_extra>(offset_out);
                e->buffer_data_offset = buffer_data_offset;
                // Construct the second kernel (buffer -> dst)
                e->second_kernel_offset = buffer_data_offset + e->buffer_data_size;
                return src_bed->make_operand_to_value_assignment_kernel(
                                out, offset_out + e->second_kernel_offset,
                                dst_metadata, e->buffer_metadata, kernreq, ectx);
            }
        } else {
            // Put together the src expression chain and the src value dtype
            // to dst value dtype conversion
            const ndt::type& buffer_dt = src_dt.value_type();
            out->ensure_capacity(offset_out + sizeof(buffered_kernel_extra));
            buffered_kernel_extra *e = out->get_at<buffered_kernel_extra>(offset_out);
            e->init(buffer_dt, kernreq);
            size_t buffer_data_size = e->buffer_data_size;
            // Construct the first kernel (src -> buffer)
            e->first_kernel_offset = sizeof(buffered_kernel_extra);
            size_t buffer_data_offset;
            buffer_data_offset = ::make_assignment_kernel(out, offset_out + e->first_kernel_offset,
                            buffer_dt, e->buffer_metadata,
                            src_dt, src_metadata,
                            kernreq, errmode, ectx);
            // Allocate the buffer data
            buffer_data_offset = inc_to_alignment(buffer_data_offset, buffer_dt.get_data_alignment());
            out->ensure_capacity(offset_out + buffer_data_offset + buffer_data_size);
            // This may have invalidated the 'e' pointer, so get it again!
            e = out->get_at<buffered_kernel_extra>(offset_out);
            e->buffer_data_offset = buffer_data_offset;
            // Construct the second kernel (buffer -> dst)
            e->second_kernel_offset = buffer_data_offset + e->buffer_data_size;
            return ::make_assignment_kernel(out, offset_out + e->second_kernel_offset,
                            dst_dt, dst_metadata,
                            buffer_dt, e->buffer_metadata,
                            kernreq, errmode, ectx);
        }
    }
}
