//
// Copyright (C) 2011-13 Mark Wiebe, DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <dynd/dtypes/unary_expr_dtype.hpp>
#include <dynd/shortvector.hpp>
#include <dynd/dtypes/builtin_dtype_properties.hpp>
#include <dynd/shape_tools.hpp>

using namespace std;
using namespace dynd;

unary_expr_dtype::unary_expr_dtype(const dtype& value_dtype, const dtype& operand_dtype,
                const expr_kernel_generator *kgen)
    : base_expression_dtype(unary_expr_type_id, expression_kind,
                        operand_dtype.get_data_size(), operand_dtype.get_data_alignment(),
                        inherited_flags(value_dtype.get_flags(), operand_dtype.get_flags()),
                        operand_dtype.get_metadata_size(), value_dtype.get_undim()),
                    m_value_dtype(value_dtype), m_operand_dtype(operand_dtype),
                    m_kgen(kgen)
{
}

unary_expr_dtype::~unary_expr_dtype()
{
    expr_kernel_generator_decref(m_kgen);
}

void unary_expr_dtype::print_data(std::ostream& DYND_UNUSED(o),
                const char *DYND_UNUSED(metadata), const char *DYND_UNUSED(data)) const
{
    throw runtime_error("internal error: unary_expr_dtype::print_data isn't supposed to be called");
}

void unary_expr_dtype::print_dtype(std::ostream& o) const
{
    o << "expr<";
    o << m_value_dtype;
    o << ", op0=" << m_operand_dtype;
    o << ", expr=";
    m_kgen->print_dtype(o);
    o << ">";
}

dtype unary_expr_dtype::apply_linear_index(size_t nindices, const irange *DYND_UNUSED(indices),
            size_t current_i, const dtype& DYND_UNUSED(root_dt), bool DYND_UNUSED(leading_dimension)) const
{
    if (m_kgen->is_elwise()) {
        // Scalar behavior
        if (nindices == 0) {
            return dtype(this, true);
        } else {
            throw too_many_indices(dtype(this, true), current_i + nindices, current_i);
        }
    } else {
        throw runtime_error("unary_expr_dtype::apply_linear_index is only implemented for elwise kernel generators");
    }
}

intptr_t unary_expr_dtype::apply_linear_index(size_t nindices, const irange *DYND_UNUSED(indices), const char *metadata,
                const dtype& DYND_UNUSED(result_dtype), char *out_metadata,
                memory_block_data *embedded_reference,
                size_t current_i, const dtype& DYND_UNUSED(root_dt),
                bool DYND_UNUSED(leading_dimension), char **DYND_UNUSED(inout_data),
                memory_block_data **DYND_UNUSED(inout_dataref)) const
{
    if (m_kgen->is_elwise()) {
        // Scalar behavior
        if (nindices == 0) {
            // Copy any metadata verbatim
            if (get_metadata_size() > 0) {
                m_operand_dtype.extended()->metadata_copy_construct(out_metadata, metadata, embedded_reference);
            }
            return 0;
        } else {
            throw too_many_indices(dtype(this, true), current_i + nindices, current_i);
        }
    } else {
        throw runtime_error("unary_expr_dtype::apply_linear_index is only implemented for elwise kernel generators");
    }
}

bool unary_expr_dtype::is_lossless_assignment(
                const dtype& DYND_UNUSED(dst_dt),
                const dtype& DYND_UNUSED(src_dt)) const
{
    return false;
}

bool unary_expr_dtype::operator==(const base_dtype& rhs) const
{
    if (this == &rhs) {
        return true;
    } else if (rhs.get_type_id() != unary_expr_type_id) {
        return false;
    } else {
        const unary_expr_dtype *dt = static_cast<const unary_expr_dtype*>(&rhs);
        return m_value_dtype == dt->m_value_dtype &&
                        m_operand_dtype == dt->m_operand_dtype &&
                        m_kgen == dt->m_kgen;
    }
}

size_t unary_expr_dtype::make_operand_to_value_assignment_kernel(
                hierarchical_kernel *out, size_t offset_out,
                const char *dst_metadata, const char *src_metadata,
                kernel_request_t kernreq, const eval::eval_context *ectx) const
{
    // As a special case, when src_count == 1, the kernel generated
    // is a unary_single_operation_t/unary_strided_operation_t instead of
    // expr_single_operation_t/expr_strided_operation_t
    return m_kgen->make_expr_kernel(out, offset_out,
                    m_value_dtype, dst_metadata,
                    1, &m_operand_dtype.value_dtype(),
                    &src_metadata,
                    kernreq, ectx);
}

size_t unary_expr_dtype::make_value_to_operand_assignment_kernel(
                hierarchical_kernel *DYND_UNUSED(out), size_t DYND_UNUSED(offset_out),
                const char *DYND_UNUSED(dst_metadata), const char *DYND_UNUSED(src_metadata),
                kernel_request_t DYND_UNUSED(kernreq), const eval::eval_context *DYND_UNUSED(ectx)) const
{
    throw runtime_error("Cannot assign to a dynd unary_expr object value");
}

dtype unary_expr_dtype::with_replaced_storage_dtype(const dtype& DYND_UNUSED(replacement_dtype)) const
{
    throw runtime_error("TODO: implement unary_expr_dtype::with_replaced_storage_dtype");
}

void unary_expr_dtype::get_dynamic_array_properties(const std::pair<std::string, gfunc::callable> **out_properties,
                size_t *out_count) const
{
    const dtype& udt = m_value_dtype.get_udtype();
    if (!udt.is_builtin()) {
        udt.extended()->get_dynamic_array_properties(out_properties, out_count);
    } else {
        get_builtin_dtype_dynamic_array_properties(udt.get_type_id(), out_properties, out_count);
    }
}

void unary_expr_dtype::get_dynamic_array_functions(const std::pair<std::string, gfunc::callable> **out_functions,
                size_t *out_count) const
{
    const dtype& udt = m_value_dtype.get_udtype();
    if (!udt.is_builtin()) {
        udt.extended()->get_dynamic_array_functions(out_functions, out_count);
    } else {
        //get_builtin_dtype_dynamic_ndobject_functions(udt.get_type_id(), out_functions, out_count);
    }
}
