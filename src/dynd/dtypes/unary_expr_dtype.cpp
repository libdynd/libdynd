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

unary_expr_dtype::unary_expr_dtype(const ndt::type& value_type, const ndt::type& operand_type,
                const expr_kernel_generator *kgen)
    : base_expression_type(unary_expr_type_id, expression_kind,
                        operand_type.get_data_size(), operand_type.get_data_alignment(),
                        inherited_flags(value_type.get_flags(), operand_type.get_flags()),
                        operand_type.get_metadata_size(), value_type.get_undim()),
                    m_value_type(value_type), m_operand_type(operand_type),
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
    o << m_value_type;
    o << ", op0=" << m_operand_type;
    o << ", expr=";
    m_kgen->print_dtype(o);
    o << ">";
}

ndt::type unary_expr_dtype::apply_linear_index(size_t nindices, const irange *DYND_UNUSED(indices),
            size_t current_i, const ndt::type& DYND_UNUSED(root_dt), bool DYND_UNUSED(leading_dimension)) const
{
    if (m_kgen->is_elwise()) {
        // Scalar behavior
        if (nindices == 0) {
            return ndt::type(this, true);
        } else {
            throw too_many_indices(ndt::type(this, true), current_i + nindices, current_i);
        }
    } else {
        throw runtime_error("unary_expr_dtype::apply_linear_index is only implemented for elwise kernel generators");
    }
}

intptr_t unary_expr_dtype::apply_linear_index(size_t nindices, const irange *DYND_UNUSED(indices), const char *metadata,
                const ndt::type& DYND_UNUSED(result_dtype), char *out_metadata,
                memory_block_data *embedded_reference,
                size_t current_i, const ndt::type& DYND_UNUSED(root_dt),
                bool DYND_UNUSED(leading_dimension), char **DYND_UNUSED(inout_data),
                memory_block_data **DYND_UNUSED(inout_dataref)) const
{
    if (m_kgen->is_elwise()) {
        // Scalar behavior
        if (nindices == 0) {
            // Copy any metadata verbatim
            if (get_metadata_size() > 0) {
                m_operand_type.extended()->metadata_copy_construct(out_metadata, metadata, embedded_reference);
            }
            return 0;
        } else {
            throw too_many_indices(ndt::type(this, true), current_i + nindices, current_i);
        }
    } else {
        throw runtime_error("unary_expr_dtype::apply_linear_index is only implemented for elwise kernel generators");
    }
}

bool unary_expr_dtype::is_lossless_assignment(
                const ndt::type& DYND_UNUSED(dst_dt),
                const ndt::type& DYND_UNUSED(src_dt)) const
{
    return false;
}

bool unary_expr_dtype::operator==(const base_type& rhs) const
{
    if (this == &rhs) {
        return true;
    } else if (rhs.get_type_id() != unary_expr_type_id) {
        return false;
    } else {
        const unary_expr_dtype *dt = static_cast<const unary_expr_dtype*>(&rhs);
        return m_value_type == dt->m_value_type &&
                        m_operand_type == dt->m_operand_type &&
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
                    m_value_type, dst_metadata,
                    1, &m_operand_type.value_type(),
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

ndt::type unary_expr_dtype::with_replaced_storage_type(const ndt::type& DYND_UNUSED(replacement_type)) const
{
    throw runtime_error("TODO: implement unary_expr_dtype::with_replaced_storage_type");
}

void unary_expr_dtype::get_dynamic_array_properties(const std::pair<std::string, gfunc::callable> **out_properties,
                size_t *out_count) const
{
    const ndt::type& udt = m_value_type.get_udtype();
    if (!udt.is_builtin()) {
        udt.extended()->get_dynamic_array_properties(out_properties, out_count);
    } else {
        get_builtin_dtype_dynamic_array_properties(udt.get_type_id(), out_properties, out_count);
    }
}

void unary_expr_dtype::get_dynamic_array_functions(const std::pair<std::string, gfunc::callable> **out_functions,
                size_t *out_count) const
{
    const ndt::type& udt = m_value_type.get_udtype();
    if (!udt.is_builtin()) {
        udt.extended()->get_dynamic_array_functions(out_functions, out_count);
    } else {
        //get_builtin_dtype_dynamic_ndobject_functions(udt.get_type_id(), out_functions, out_count);
    }
}
