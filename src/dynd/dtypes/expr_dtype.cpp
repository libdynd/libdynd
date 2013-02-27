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

size_t expr_dtype::make_operand_to_value_assignment_kernel(
                assignment_kernel *out,
                size_t offset_out,
                const char *dst_metadata, const char *src_metadata,
                kernel_request_t kernreq, const eval::eval_context *ectx) const
{
    const fixedstruct_dtype *fsd = static_cast<const fixedstruct_dtype *>(m_operand_dtype.extended());

    offset_out = make_kernreq_to_single_kernel_adapter(out, offset_out, kernreq);
    size_t input_count = fsd->get_field_count();
    const size_t *metadata_offsets = fsd->get_metadata_offsets();
    shortvector<const char *> src_metadata_array(input_count);
    const dtype *src_ptr_dt = fsd->get_field_types();
    vector<dtype> src_dt(input_count);
    for (size_t i = 0; i != input_count; ++i) {
        const pointer_dtype *pd = static_cast<const pointer_dtype *>(src_ptr_dt[i].extended());
        src_dt[i] = pd->get_target_dtype();
    }
    for (size_t i = 0; i != input_count; ++i) {
        src_metadata_array[i] = src_metadata + metadata_offsets[i] + sizeof(pointer_dtype_metadata);
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

