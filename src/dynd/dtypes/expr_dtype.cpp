//
// Copyright (C) 2011-13, DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <dynd/dtypes/expr_dtype.hpp>
#include <dynd/shortvector.hpp>
#include <dynd/dtypes/fixedstruct_dtype.hpp>
#include <dynd/shape_tools.hpp>

using namespace std;
using namespace dynd;

expr_dtype::expr_dtype(const dtype& value_dtype, const dtype& operand_dtype,
                const expr_kernel_generator *kgen)
    : base_expression_dtype(expr_type_id, expression_kind,
                        operand_dtype.get_data_size(), operand_dtype.get_alignment(),
                        dtype_flag_none,
                        operand_dtype.get_metadata_size(), value_dtype.get_undim()),
                    m_value_dtype(value_dtype), m_operand_dtype(operand_dtype),
                    m_kgen(kgen)
{
}

expr_dtype::~expr_dtype()
{
    expr_kernel_generator_decref(m_kgen);
}

void expr_dtype::print_data(std::ostream& DYND_UNUSED(o),
                const char *DYND_UNUSED(metadata), const char *DYND_UNUSED(data)) const
{
    throw runtime_error("internal error: expr_dtype::print_data isn't supposed to be called");
}

void expr_dtype::print_dtype(std::ostream& o) const
{
    o << "expr<";
    o << m_value_dtype << ", ";
    m_kgen->print_dtype(o);
    o << ">";
}

dtype expr_dtype::apply_linear_index(size_t nindices, const irange *indices,
            size_t current_i, const dtype& root_dt, bool DYND_UNUSED(leading_dimension)) const
{
    if (m_kgen->is_elwise()) {
        size_t undim = get_undim();
        const fixedstruct_dtype *fsd = static_cast<const fixedstruct_dtype *>(m_operand_dtype.extended());
        size_t field_count = fsd->get_field_count();
        const dtype *field_types = fsd->get_field_types();

        dtype result_value_dt = m_value_dtype.apply_linear_index(nindices, indices,
                        current_i, root_dt, true);
        vector<dtype> result_src_dt(field_count);
        // Apply the portion of the indexing to each of the src operand types
        for (size_t i = 0; i != field_count; ++i) {
            const dtype& dt = field_types[i];
            size_t field_undim = dt.get_undim();
            if (nindices + field_undim <= undim) {
                result_src_dt[i] = dt;
            } else {
                size_t index_offset = undim - field_undim;
                result_src_dt[i] = dt.apply_linear_index(
                                                nindices - index_offset, indices + index_offset,
                                                current_i, root_dt, false);
            }
        }
        dtype result_operand_dtype = make_fixedstruct_dtype(field_count, &result_src_dt[0],
                        fsd->get_field_names());
        expr_kernel_generator_incref(m_kgen);
        return make_expr_dtype(result_value_dt, result_operand_dtype, m_kgen);
    } else {
        throw runtime_error("expr_dtype::apply_linear_index is only implemented for elwise kernel generators");
    }
}

intptr_t expr_dtype::apply_linear_index(size_t nindices, const irange *indices, const char *metadata,
                const dtype& result_dtype, char *out_metadata,
                memory_block_data *embedded_reference,
                size_t current_i, const dtype& root_dt,
                bool leading_dimension, char **inout_data,
                memory_block_data **inout_dataref) const
{
    if (m_kgen->is_elwise()) {
        size_t undim = get_undim();
        const expr_dtype *out_ed = static_cast<const expr_dtype *>(result_dtype.extended());
        const fixedstruct_dtype *fsd = static_cast<const fixedstruct_dtype *>(m_operand_dtype.extended());
        const fixedstruct_dtype *out_fsd = static_cast<const fixedstruct_dtype *>(out_ed->m_operand_dtype.extended());
        const size_t *metadata_offsets = fsd->get_metadata_offsets();
        const size_t *out_metadata_offsets = out_fsd->get_metadata_offsets();
        size_t field_count = fsd->get_field_count();
        const dtype *field_types = fsd->get_field_types();
        const dtype *out_field_types = out_fsd->get_field_types();
        // Apply the portion of the indexing to each of the src operand types
        for (size_t i = 0; i != field_count; ++i) {
            const pointer_dtype *pd = static_cast<const pointer_dtype *>(field_types[i].extended());
            const pointer_dtype *out_pd = static_cast<const pointer_dtype *>(out_field_types[i].extended());
            size_t field_undim = pd->get_undim();
            if (nindices + field_undim <= undim) {
                pd->metadata_copy_construct(out_metadata + out_metadata_offsets[i],
                                metadata + metadata_offsets[i],
                                embedded_reference);
            } else {
                size_t index_offset = undim - field_undim;
                intptr_t offset = pd->apply_linear_index(nindices - index_offset, indices + index_offset,
                                metadata + metadata_offsets[i],
                                out_field_types[i], out_metadata + out_metadata_offsets[i],
                                embedded_reference, current_i, root_dt, false, NULL, NULL);
                if (offset != 0) {
                    throw runtime_error("internal error: expr_dtype::apply_linear_index"
                                    " expected 0 offset from pointer_dtype::apply_linear_index");
                }
            }
        }
        return 0;
    } else {
        throw runtime_error("expr_dtype::apply_linear_index is only implemented for elwise kernel generators");
    }
}

void expr_dtype::get_shape(size_t i, intptr_t *out_shape) const
{
    size_t undim = get_undim();
    // Initialize the shape to all ones
    out_shape += i;
    for (size_t j = 0; j != undim; ++j) {
        out_shape[j] = 1;
    }

    // Get each operand shape, and broadcast them together
    dimvector shape(undim);
    const fixedstruct_dtype *fsd = static_cast<const fixedstruct_dtype *>(m_operand_dtype.extended());
    size_t field_count = fsd->get_field_count();
    for (size_t fi = 0; fi != field_count; ++fi) {
        const dtype& dt = fsd->get_field_types()[fi];
        size_t field_undim = dt.get_undim();
        if (field_undim > 0) {
            dt.extended()->get_shape(0, shape.get());
            incremental_broadcast(undim, out_shape, field_undim, shape.get());
        }
    }
}

void expr_dtype::get_shape(size_t i,
                intptr_t *out_shape,
                const char *metadata) const
{
    size_t undim = get_undim();
    // Initialize the shape to all ones
    out_shape += i;
    for (size_t j = 0; j != undim; ++j) {
        out_shape[j] = 1;
    }

    // Get each operand shape, and broadcast them together
    dimvector shape(undim);
    const fixedstruct_dtype *fsd = static_cast<const fixedstruct_dtype *>(m_operand_dtype.extended());
    const size_t *metadata_offsets = fsd->get_metadata_offsets();
    size_t field_count = fsd->get_field_count();
    for (size_t fi = 0; fi != field_count; ++fi) {
        const dtype& dt = fsd->get_field_types()[fi];
        size_t field_undim = dt.get_undim();
        if (field_undim > 0) {
            dt.extended()->get_shape(0, shape.get(), metadata + metadata_offsets[fi]);
            incremental_broadcast(undim, out_shape, field_undim, shape.get());
        }
    }
}

bool expr_dtype::is_lossless_assignment(
                const dtype& DYND_UNUSED(dst_dt),
                const dtype& DYND_UNUSED(src_dt)) const
{
    return false;
}

bool expr_dtype::operator==(const base_dtype& rhs) const
{
    if (this == &rhs) {
        return true;
    } else if (rhs.get_type_id() != expr_type_id) {
        return false;
    } else {
        const expr_dtype *dt = static_cast<const expr_dtype*>(&rhs);
        return m_value_dtype == dt->m_value_dtype && m_operand_dtype == dt->m_operand_dtype;
    }
}

namespace {
    template<int N>
    struct expr_dtype_offset_applier_extra {
        typedef expr_dtype_offset_applier_extra<N> extra_type;

        kernel_data_prefix base;
        size_t offsets[N];

        // Only the single kernel is needed for this one
        static void single(char *dst, const char * const *src,
                        kernel_data_prefix *extra)
        {
            extra_type *e = reinterpret_cast<extra_type *>(extra);
            const size_t *offsets = e->offsets;
            const char *src_modified[N];
            for (int i = 0; i < N; ++i) {
                src_modified[i] = src[i] + offsets[i];
            }
            kernel_data_prefix *echild = &(e + 1)->base;
            expr_single_operation_t opchild = echild->get_function<expr_single_operation_t>();
            opchild(dst, src_modified, echild);
        }

        static void destruct(kernel_data_prefix *extra)
        {
            extra_type *e = reinterpret_cast<extra_type *>(extra);
            kernel_data_prefix *echild = &(e + 1)->base;
            if (echild->destructor) {
                echild->destructor(echild);
            }
        }
    };

    struct expr_dtype_offset_applier_general_extra {
        typedef expr_dtype_offset_applier_general_extra extra_type;

        kernel_data_prefix base;
        size_t src_count;
        // After this are src_count size_t offsets

       // Only the single kernel is needed for this one
        static void single(char *dst, const char * const *src,
                        kernel_data_prefix *extra)
        {
            extra_type *e = reinterpret_cast<extra_type *>(extra);
            size_t src_count = e->src_count;
            const size_t *offsets = reinterpret_cast<const size_t *>(e + 1);
            shortvector<const char *> src_modified(src_count);
            for (int i = 0; i < src_count; ++i) {
                src_modified[i] = src[i] + offsets[i];
            }
            kernel_data_prefix *echild = reinterpret_cast<kernel_data_prefix *>(
                            reinterpret_cast<char *>(extra) + sizeof(extra_type) +
                            src_count * sizeof(size_t));
            expr_single_operation_t opchild = echild->get_function<expr_single_operation_t>();
            opchild(dst, src_modified.get(), echild);
        }

        static void destruct(kernel_data_prefix *extra)
        {
            extra_type *e = reinterpret_cast<extra_type *>(extra);
            size_t src_count = e->src_count;
            kernel_data_prefix *echild = reinterpret_cast<kernel_data_prefix *>(
                            reinterpret_cast<char *>(extra) + sizeof(extra_type) +
                            src_count * sizeof(size_t));
            if (echild->destructor) {
                echild->destructor(echild);
            }
        }
    };
} // anonymous namespace

static size_t make_expr_dtype_offset_applier(
                assignment_kernel *out, size_t offset_out,
                size_t src_count, const intptr_t *src_data_offsets)
{
    // A few specializations with fixed size, and a general case version
    switch (src_count) {
        case 1: {
            out->ensure_capacity(offset_out + sizeof(expr_dtype_offset_applier_extra<1>));
            expr_dtype_offset_applier_extra<1> *e = out->get_at<expr_dtype_offset_applier_extra<1> >(offset_out);
            memcpy(e->offsets, src_data_offsets, sizeof(e->offsets));
            e->base.template set_function<expr_single_operation_t>(&expr_dtype_offset_applier_extra<1>::single);
            e->base.destructor = &expr_dtype_offset_applier_extra<1>::destruct;
            return offset_out + sizeof(expr_dtype_offset_applier_extra<1>);
        }
        case 2: {
            out->ensure_capacity(offset_out + sizeof(expr_dtype_offset_applier_extra<2>));
            expr_dtype_offset_applier_extra<2> *e = out->get_at<expr_dtype_offset_applier_extra<2> >(offset_out);
            memcpy(e->offsets, src_data_offsets, sizeof(e->offsets));
            e->base.template set_function<expr_single_operation_t>(&expr_dtype_offset_applier_extra<2>::single);
            e->base.destructor = &expr_dtype_offset_applier_extra<2>::destruct;
            return offset_out + sizeof(expr_dtype_offset_applier_extra<2>);
        }
        case 3: {
            out->ensure_capacity(offset_out + sizeof(expr_dtype_offset_applier_extra<3>));
            expr_dtype_offset_applier_extra<3> *e = out->get_at<expr_dtype_offset_applier_extra<3> >(offset_out);
            memcpy(e->offsets, src_data_offsets, sizeof(e->offsets));
            e->base.template set_function<expr_single_operation_t>(&expr_dtype_offset_applier_extra<3>::single);
            e->base.destructor = &expr_dtype_offset_applier_extra<3>::destruct;
            return offset_out + sizeof(expr_dtype_offset_applier_extra<3>);
        }
        case 4: {
            out->ensure_capacity(offset_out + sizeof(expr_dtype_offset_applier_extra<4>));
            expr_dtype_offset_applier_extra<4> *e = out->get_at<expr_dtype_offset_applier_extra<4> >(offset_out);
            memcpy(e->offsets, src_data_offsets, sizeof(e->offsets));
            e->base.template set_function<expr_single_operation_t>(&expr_dtype_offset_applier_extra<4>::single);
            e->base.destructor = &expr_dtype_offset_applier_extra<4>::destruct;
            return offset_out + sizeof(expr_dtype_offset_applier_extra<4>);
        }
        default: {
            size_t extra_size = sizeof(expr_dtype_offset_applier_general_extra) +
                            src_count * sizeof(size_t);
            out->ensure_capacity(offset_out + extra_size);
            expr_dtype_offset_applier_general_extra *e = out->get_at<expr_dtype_offset_applier_general_extra>(offset_out);
            e->src_count = src_count;
            size_t *out_offsets = reinterpret_cast<size_t *>(e + 1);
            memcpy(out_offsets, src_data_offsets, src_count * sizeof(size_t));
            e->base.set_function<expr_single_operation_t>(&expr_dtype_offset_applier_general_extra::single);
            e->base.destructor = &expr_dtype_offset_applier_general_extra::destruct;
            return offset_out + extra_size;
        }
    }
    
}

size_t expr_dtype::make_operand_to_value_assignment_kernel(
                assignment_kernel *out, size_t offset_out,
                const char *dst_metadata, const char *src_metadata,
                kernel_request_t kernreq, const eval::eval_context *ectx) const
{
    const fixedstruct_dtype *fsd = static_cast<const fixedstruct_dtype *>(m_operand_dtype.extended());

    offset_out = make_kernreq_to_single_kernel_adapter(out, offset_out, kernreq);
    size_t input_count = fsd->get_field_count();
    const size_t *metadata_offsets = fsd->get_metadata_offsets();
    shortvector<const char *> src_metadata_array(input_count);
    dimvector src_data_offsets(input_count);
    bool nonzero_offsets = false;

    const dtype *src_ptr_dt = fsd->get_field_types();
    vector<dtype> src_dt(input_count);
    for (size_t i = 0; i != input_count; ++i) {
        const pointer_dtype *pd = static_cast<const pointer_dtype *>(src_ptr_dt[i].extended());
        src_dt[i] = pd->get_target_dtype();
    }
    for (size_t i = 0; i != input_count; ++i) {
        const char *ptr_metadata = src_metadata + metadata_offsets[i];
        intptr_t offset = reinterpret_cast<const pointer_dtype_metadata *>(ptr_metadata)->offset;
        if (offset != 0) {
            nonzero_offsets = true;
        }
        src_data_offsets[i] = offset;
        src_metadata_array[i] = ptr_metadata + sizeof(pointer_dtype_metadata);
    }
    // If there were any non-zero pointer offsets, we need to add a kernel
    // adapter which applies those offsets.
    if (nonzero_offsets) {
        offset_out = make_expr_dtype_offset_applier(out, offset_out,
                        input_count, src_data_offsets.get());
    }
    return m_kgen->make_expr_kernel(out, offset_out,
                    m_value_dtype, dst_metadata,
                    input_count, &src_dt[0],
                    src_metadata_array.get(),
                    kernel_request_single, ectx);
}

size_t expr_dtype::make_value_to_operand_assignment_kernel(
                assignment_kernel *DYND_UNUSED(out),
                size_t DYND_UNUSED(offset_out),
                const char *DYND_UNUSED(dst_metadata), const char *DYND_UNUSED(src_metadata),
                kernel_request_t DYND_UNUSED(kernreq), const eval::eval_context *DYND_UNUSED(ectx)) const
{
    throw runtime_error("Cannot assign to a dynd expr object value");
}

dtype expr_dtype::with_replaced_storage_dtype(const dtype& DYND_UNUSED(replacement_dtype)) const
{
    throw runtime_error("TODO: implement expr_dtype::with_replaced_storage_dtype");
}

