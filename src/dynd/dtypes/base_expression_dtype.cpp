//
// Copyright (C) 2011-13, DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <dynd/dtype.hpp>
#include <dynd/kernels/assignment_kernels.hpp>

using namespace std;
using namespace dynd;

base_expression_dtype::~base_expression_dtype()
{
}

bool base_expression_dtype::is_expression() const
{
    return true;
}

dtype base_expression_dtype::get_canonical_dtype() const
{
    return get_value_dtype();
}

size_t base_expression_dtype::get_metadata_size() const
{
    const dtype& dt = get_operand_dtype();
    if (!dt.is_builtin()) {
        return dt.extended()->get_metadata_size();
    } else {
        return 0;
    }
}

void base_expression_dtype::metadata_default_construct(char *metadata, int ndim, const intptr_t* shape) const
{
    const dtype& dt = get_operand_dtype();
    if (!dt.is_builtin()) {
        dt.extended()->metadata_default_construct(metadata, ndim, shape);
    }
}

void base_expression_dtype::metadata_copy_construct(char *dst_metadata, const char *src_metadata, memory_block_data *embedded_reference) const
{
    const dtype& dt = get_operand_dtype();
    if (!dt.is_builtin()) {
        dt.extended()->metadata_copy_construct(dst_metadata, src_metadata, embedded_reference);
    }
}

void base_expression_dtype::metadata_destruct(char *metadata) const
{
    const dtype& dt = get_operand_dtype();
    if (!dt.is_builtin()) {
        dt.extended()->metadata_destruct(metadata);
    }
}

void base_expression_dtype::metadata_debug_print(const char *metadata, std::ostream& o, const std::string& indent) const
{
    const dtype& dt = get_operand_dtype();
    if (!dt.is_builtin()) {
        dt.extended()->metadata_debug_print(metadata, o, indent);
    }
}

size_t base_expression_dtype::get_iterdata_size(int DYND_UNUSED(ndim)) const
{
    return 0;
}

namespace {
    struct buffered_kernel_extra {
        typedef buffered_kernel_extra extra_type;

        hierarchical_kernel_common_base base;
        // Offsets, from the start of &base, to the kernels
        // before and after the buffer
        size_t first_kernel_offset, second_kernel_offset;
        const base_dtype *buffer_dt;
        char *buffer_metadata;
        size_t buffer_data_offset, buffer_data_size;

        // Initializes the dtype and metadata for the buffer
        // NOTE: This does NOT initialize the buffer_data_offset,
        //       just the buffer_data_size.
        void init(const dtype& buffer_dt_) {
            base.function = &single;
            base.destructor = &destruct;
            // The kernel data owns a reference in buffer_dt
            buffer_dt = dtype(buffer_dt_).release();
            if (buffer_dt_.is_builtin()) {
                size_t buffer_metadata_size = buffer_dt_.extended()->get_metadata_size();
                buffer_metadata = reinterpret_cast<char *>(malloc(buffer_metadata_size));
                if (buffer_metadata == NULL) {
                    throw bad_alloc();
                }
                buffer_dt->metadata_default_construct(buffer_metadata, 0, NULL);
                // Make sure the buffer data size is pointer size-aligned
                buffer_data_size = inc_to_alignment(buffer_dt->get_default_data_size(0, NULL), sizeof(void *));
            } else {
                // Make sure the buffer data size is pointer size-aligned
                buffer_data_size = inc_to_alignment(buffer_dt_.get_data_size(), sizeof(void *));
            }
            
        }

        static void single(char *dst, const char *src,
                            hierarchical_kernel_common_base *extra)
        {
            char *eraw = reinterpret_cast<char *>(extra);
            extra_type *e = reinterpret_cast<extra_type *>(extra);
            hierarchical_kernel_common_base *echild_first, *echild_second;
            unary_single_operation_t opchild;
            const base_dtype *buffer_dt = e->buffer_dt;
            char *buffer_metadata = e->buffer_metadata;
            char *buffer_data_ptr = eraw + e->buffer_data_offset;
            echild_first = reinterpret_cast<hierarchical_kernel_common_base *>(eraw + e->first_kernel_offset);
            echild_second = reinterpret_cast<hierarchical_kernel_common_base *>(eraw + e->second_kernel_offset);

            // If the type needs it, initialize the buffer data to zero
            if (buffer_dt->get_flags()&dtype_flag_zeroinit) {
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
        static void destruct(hierarchical_kernel_common_base *extra)
        {
            char *eraw = reinterpret_cast<char *>(extra);
            extra_type *e = reinterpret_cast<extra_type *>(extra);
            // Steal the buffer_dt reference count into a dtype
            dtype buffer_dt(e->buffer_dt, false);
            char *buffer_metadata = e->buffer_metadata;
            // Destruct and free the metadata for the buffer
            if (buffer_metadata != NULL) {
                buffer_dt.extended()->metadata_destruct(buffer_metadata);
                free(buffer_metadata);
            }
            hierarchical_kernel_common_base *echild;
            // Destruct the first kernel
            if (e->first_kernel_offset != 0) {
                echild = reinterpret_cast<hierarchical_kernel_common_base *>(eraw + e->first_kernel_offset);
                if (echild->destructor) {
                    echild->destructor(echild);
                }
            }
            // Destruct the second kernel
            if (e->second_kernel_offset != 0) {
                echild = reinterpret_cast<hierarchical_kernel_common_base *>(eraw + e->second_kernel_offset);
                if (echild->destructor) {
                    echild->destructor(echild);
                }
            }
        }
    };
} // anonymous namespace

size_t base_expression_dtype::make_operand_to_value_assignment_kernel(
                hierarchical_kernel<unary_single_operation_t> *out,
                size_t offset_out,
                const char *dst_metadata, const char *src_metadata,
                const eval::eval_context *ectx) const
{
    stringstream ss;
    ss << "dynd dtype " << dtype(this, true) << " does not support reading of its values";
    throw runtime_error(ss.str());
}

size_t base_expression_dtype::make_value_to_operand_assignment_kernel(
                hierarchical_kernel<unary_single_operation_t> *out,
                size_t offset_out,
                const char *dst_metadata, const char *src_metadata,
                const eval::eval_context *ectx) const
{
    stringstream ss;
    ss << "dynd dtype " << dtype(this, true) << " does not support writing to its values";
    throw runtime_error(ss.str());
}

size_t base_expression_dtype::make_assignment_kernel(
                hierarchical_kernel<unary_single_operation_t> *out,
                size_t offset_out,
                const dtype& dst_dt, const char *dst_metadata,
                const dtype& src_dt, const char *src_metadata,
                assign_error_mode errmode,
                const eval::eval_context *ectx) const
{
    if (dst_dt.extended() == this) {
        if (src_dt == get_value_dtype()) {
            // In this case, it's just a chain of value -> operand on the dst side
            const dtype& opdt = get_operand_dtype();
            if (opdt.get_kind() != expression_kind) {
                // Leaf case, just a single value -> operand kernel
                return make_value_to_operand_assignment_kernel(out, offset_out, dst_metadata, src_metadata, ectx);
            } else {
                // Chain case, buffer one segment of the chain
                const dtype& buffer_dt = static_cast<const base_expression_dtype *>(opdt.extended())->get_value_dtype();
                out->ensure_capacity(offset_out + sizeof(buffered_kernel_extra));
                buffered_kernel_extra *e = out->get_at<buffered_kernel_extra>(offset_out);
                e->init(buffer_dt);
                // Construct the first kernel (src -> buffer)
                e->first_kernel_offset = sizeof(buffered_kernel_extra);
                size_t buffer_data_offset;
                buffer_data_offset = make_value_to_operand_assignment_kernel(out, e->first_kernel_offset,
                                e->buffer_metadata, src_metadata, ectx);
                // Allocate the buffer data
                buffer_data_offset = inc_to_alignment(buffer_data_offset, buffer_dt.get_alignment());
                // This may have invalidated the 'e' pointer, so get it again!
                e = out->get_at<buffered_kernel_extra>(offset_out);
                e->buffer_data_offset = buffer_data_offset;
                // Construct the second kernel (buffer -> dst)
                e->second_kernel_offset = buffer_data_offset + e->buffer_data_size;
                return ::make_assignment_kernel(out, e->second_kernel_offset,
                                dst_dt, dst_metadata,
                                buffer_dt, e->buffer_metadata,
                                errmode, ectx);
            }
        } else {
            dtype buffer_dt;
            if (src_dt.get_kind() != expression_kind) {
                // In this case, need a data converting assignment to dst_dt.value_dtype(),
                // then the dst_dt expression chain
                buffer_dt = get_value_dtype();
            } else {
                // Both src and dst are expression dtypes, use the src expression chain, and
                // the src value dtype to dst dtype as the two segments to buffer together
                buffer_dt = src_dt.value_dtype();
            }
            out->ensure_capacity(offset_out + sizeof(buffered_kernel_extra));
            buffered_kernel_extra *e = out->get_at<buffered_kernel_extra>(offset_out);
            e->init(buffer_dt);
            // Construct the first kernel (src -> buffer)
            e->first_kernel_offset = sizeof(buffered_kernel_extra);
            size_t buffer_data_offset;
            buffer_data_offset = ::make_assignment_kernel(out, e->first_kernel_offset,
                            buffer_dt, e->buffer_metadata,
                            src_dt, src_metadata,
                            errmode, ectx);
            // Allocate the buffer data
            buffer_data_offset = inc_to_alignment(buffer_data_offset, buffer_dt.get_alignment());
            // This may have invalidated the 'e' pointer, so get it again!
            e = out->get_at<buffered_kernel_extra>(offset_out);
            e->buffer_data_offset = buffer_data_offset;
            // Construct the second kernel (buffer -> dst)
            e->second_kernel_offset = buffer_data_offset + e->buffer_data_size;
            return ::make_assignment_kernel(out, e->second_kernel_offset,
                            dst_dt, dst_metadata,
                            buffer_dt, e->buffer_metadata,
                            errmode, ectx);
        }
    } else {
        if (dst_dt == get_value_dtype()) {
            // In this case, it's just a chain of operand -> value on the src side
            const dtype& opdt = get_operand_dtype();
            if (opdt.get_kind() != expression_kind) {
                // Leaf case, just a single value -> operand kernel
                return make_operand_to_value_assignment_kernel(out, offset_out, dst_metadata, src_metadata, ectx);
            } else {
                // Chain case, buffer one segment of the chain
                const dtype& buffer_dt = static_cast<const base_expression_dtype *>(opdt.extended())->get_value_dtype();
                out->ensure_capacity(offset_out + sizeof(buffered_kernel_extra));
                buffered_kernel_extra *e = out->get_at<buffered_kernel_extra>(offset_out);
                e->init(buffer_dt);
                // Construct the first kernel (src -> buffer)
                e->first_kernel_offset = sizeof(buffered_kernel_extra);
                size_t buffer_data_offset;
                buffer_data_offset = ::make_assignment_kernel(out, e->first_kernel_offset,
                                buffer_dt, e->buffer_metadata,
                               src_dt, src_metadata,
                                errmode, ectx);
                // Allocate the buffer data
                buffer_data_offset = inc_to_alignment(buffer_data_offset, buffer_dt.get_alignment());
                // This may have invalidated the 'e' pointer, so get it again!
                e = out->get_at<buffered_kernel_extra>(offset_out);
                e->buffer_data_offset = buffer_data_offset;
                // Construct the second kernel (buffer -> dst)
                e->second_kernel_offset = buffer_data_offset + e->buffer_data_size;
                return make_operand_to_value_assignment_kernel(out, e->second_kernel_offset,
                                dst_metadata, e->buffer_metadata, ectx);
            }
        } else {
            // Put together the src expression chain and the src value dtype
            // to dst value dtype conversion
            const dtype& buffer_dt = src_dt.value_dtype();
            out->ensure_capacity(offset_out + sizeof(buffered_kernel_extra));
            buffered_kernel_extra *e = out->get_at<buffered_kernel_extra>(offset_out);
            e->init(buffer_dt);
            // Construct the first kernel (src -> buffer)
            e->first_kernel_offset = sizeof(buffered_kernel_extra);
            size_t buffer_data_offset;
            buffer_data_offset = ::make_assignment_kernel(out, e->first_kernel_offset,
                            buffer_dt, e->buffer_metadata,
                            src_dt, src_metadata,
                            errmode, ectx);
            // Allocate the buffer data
            buffer_data_offset = inc_to_alignment(buffer_data_offset, buffer_dt.get_alignment());
            // This may have invalidated the 'e' pointer, so get it again!
            e = out->get_at<buffered_kernel_extra>(offset_out);
            e->buffer_data_offset = buffer_data_offset;
            // Construct the second kernel (buffer -> dst)
            e->second_kernel_offset = buffer_data_offset + e->buffer_data_size;
            return ::make_assignment_kernel(out, e->second_kernel_offset,
                            dst_dt, dst_metadata,
                            buffer_dt, e->buffer_metadata,
                            errmode, ectx);
        }
    }
}