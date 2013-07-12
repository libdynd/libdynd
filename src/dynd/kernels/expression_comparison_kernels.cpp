//
// Copyright (C) 2011-13 Mark Wiebe, DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <dynd/type.hpp>
#include <dynd/types/base_expression_type.hpp>
#include <dynd/kernels/expression_comparison_kernels.hpp>
#include <dynd/kernels/assignment_kernels.hpp>

using namespace std;
using namespace dynd;

namespace {
    struct single_buffer {
        // Offset, from the start of &base, to the kernel for buffering
        size_t kernel_offset;
        const base_type *tp;
        char *metadata;
        size_t data_offset, data_size;
    };

    struct buffered_kernel_extra {
        typedef buffered_kernel_extra extra_type;

        kernel_data_prefix base;
        // Offset, from the start of 'base' to the comparison kernel
        size_t cmp_kernel_offset;
        single_buffer buf[2];

        // Initializes the type and metadata for one of the two buffers
        // NOTE: This does NOT initialize buf[i].data_offset or buf[i].kernel_offset
        void init_buffer(int i, const ndt::type& buffer_dt_) {
            single_buffer& b = buf[i];
            // The kernel data owns a reference in buffer_dt
            b.tp = ndt::type(buffer_dt_).release();
            if (!buffer_dt_.is_builtin()) {
                size_t buffer_metadata_size = buffer_dt_.extended()->get_metadata_size();
                if (buffer_metadata_size > 0) {
                    b.metadata = reinterpret_cast<char *>(malloc(buffer_metadata_size));
                    if (b.metadata == NULL) {
                        throw bad_alloc();
                    }
                    b.tp->metadata_default_construct(b.metadata, 0, NULL);
                }
                // Make sure the buffer data size is pointer size-aligned
                b.data_size = inc_to_alignment(b.tp->get_default_data_size(0, NULL),
                                sizeof(void *));
            } else {
                // Make sure the buffer data size is pointer size-aligned
                b.data_size = inc_to_alignment(buffer_dt_.get_data_size(), sizeof(void *));
            }
        }

        inline const char *buffer_operand(const single_buffer& b, const char *src)
        {
            char *eraw = reinterpret_cast<char *>(this);
            char *dst = eraw + b.data_offset;

            // If the type needs it, initialize the buffer data to zero
            if (!is_builtin_type(b.tp) && (b.tp->get_flags()&type_flag_zeroinit) != 0) {
                memset(dst, 0, b.data_size);
            }

            // Get and execute the assignment kernel
            kernel_data_prefix *echild;
            unary_single_operation_t opchild;
            echild = reinterpret_cast<kernel_data_prefix *>(eraw + b.kernel_offset);
            opchild = echild->get_function<unary_single_operation_t>();
            opchild(dst, src, echild);

            // Return the buffer
            return dst;
        }

        static int kernel(const char *src0, const char *src1, kernel_data_prefix *extra)
        {
            char *eraw = reinterpret_cast<char *>(extra);
            extra_type *e = reinterpret_cast<extra_type *>(extra);
            // Buffer the first operand if necessary
            if (e->buf[0].kernel_offset != 0) {
                src0 = e->buffer_operand(e->buf[0], src0);
            }
            // Buffer the second operand if necessary
            if (e->buf[1].kernel_offset != 0) {
                src1 = e->buffer_operand(e->buf[1], src1);
            }
            // Call the comparison kernel
            kernel_data_prefix *echild;
            binary_single_predicate_t opchild;
            echild = reinterpret_cast<kernel_data_prefix *>(eraw + e->cmp_kernel_offset);
            opchild = echild->get_function<binary_single_predicate_t>();
            int result = opchild(src0, src1, echild);

            // Clear the buffer data if necessary
            if (e->buf[0].metadata != NULL) {
                e->buf[0].tp->metadata_reset_buffers(e->buf[0].metadata);
            }
            if (e->buf[1].metadata != NULL) {
                e->buf[1].tp->metadata_reset_buffers(e->buf[1].metadata);
            }

            return result;
        }

        static void destruct(kernel_data_prefix *extra)
        {
            char *eraw = reinterpret_cast<char *>(extra);
            extra_type *e = reinterpret_cast<extra_type *>(extra);
            kernel_data_prefix *echild;

            for (int i = 0; i < 2; ++i) {
                single_buffer& b = e->buf[i];

                ndt::type dt(b.tp, false);
                // Steal the buffer0_tp reference count into an ndt::type
                char *metadata = b.metadata;
                // Destruct and free the metadata for the buffer
                if (metadata != NULL) {
                    dt.extended()->metadata_destruct(metadata);
                    free(metadata);
                }
                // Destruct the kernel for the buffer
                if (b.kernel_offset != 0) {
                    echild = reinterpret_cast<kernel_data_prefix *>(eraw + b.kernel_offset);
                    if (echild->destructor) {
                        echild->destructor(echild);
                    }
                }
            }

            // Destruct the comparison kernel
            if (e->cmp_kernel_offset != 0) {
                echild = reinterpret_cast<kernel_data_prefix *>(eraw + e->cmp_kernel_offset);
                if (echild->destructor) {
                    echild->destructor(echild);
                }
            }
        }
    };
} // anonymous namespace

size_t dynd::make_expression_comparison_kernel(
                hierarchical_kernel *out, size_t offset_out,
                const ndt::type& src0_dt, const char *src0_metadata,
                const ndt::type& src1_dt, const char *src1_metadata,
                comparison_type_t comptype,
                const eval::eval_context *ectx)
{
    size_t current_offset = offset_out + sizeof(buffered_kernel_extra);
    out->ensure_capacity(current_offset);
    buffered_kernel_extra *e = out->get_at<buffered_kernel_extra>(offset_out);
    e->base.set_function<binary_single_predicate_t>(&buffered_kernel_extra::kernel);
    e->base.destructor = &buffered_kernel_extra::destruct;
    // Initialize the information for buffering the operands
    if (src0_dt.get_kind() == expression_kind) {
        e->init_buffer(0, src0_dt.value_type());
        e->buf[0].kernel_offset = current_offset - offset_out;
        current_offset = make_assignment_kernel(out, current_offset,
                        src0_dt.value_type(), e->buf[0].metadata,
                        src0_dt, src0_metadata,
                        kernel_request_single, assign_error_none, ectx);
        // Have to re-retrieve 'e', because creating another kernel may invalidate it
        e = out->get_at<buffered_kernel_extra>(offset_out);
    }
    if (src1_dt.get_kind() == expression_kind) {
        e->init_buffer(1, src1_dt.value_type());
        e->buf[1].kernel_offset = current_offset - offset_out;
        current_offset = make_assignment_kernel(out, current_offset,
                        src1_dt.value_type(), e->buf[1].metadata,
                        src1_dt, src1_metadata,
                        kernel_request_single, assign_error_none, ectx);
        // Have to re-retrieve 'e', because creating another kernel may invalidate it
        e = out->get_at<buffered_kernel_extra>(offset_out);
    }
    // Allocate the data for the buffers
    if (e->buf[0].kernel_offset != 0) {
        current_offset = inc_to_alignment(current_offset, src0_dt.get_data_alignment());
        e->buf[0].data_offset = current_offset - offset_out;
        current_offset += e->buf[0].data_size;
    }
    if (e->buf[1].kernel_offset != 0) {
        current_offset = inc_to_alignment(current_offset, src1_dt.get_data_alignment());
        e->buf[1].data_offset = current_offset - offset_out;
        current_offset += e->buf[1].data_size;
    }
    out->ensure_capacity(current_offset);
    // Have to re-retrieve 'e', because allocating the buffer data may invalidate it
    e = out->get_at<buffered_kernel_extra>(offset_out);
    e->cmp_kernel_offset = current_offset - offset_out;
    return make_comparison_kernel(out, current_offset,
                    src0_dt.value_type(),
                    (e->buf[0].kernel_offset != 0) ? e->buf[0].metadata : src0_metadata,
                    src1_dt.value_type(),
                    (e->buf[1].kernel_offset != 0) ? e->buf[1].metadata : src1_metadata,
                    comptype, ectx);
}
