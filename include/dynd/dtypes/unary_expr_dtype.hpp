//
// Copyright (C) 2011-13, DyND Developers
// BSD 2-Clause License, see LICENSE.txt


#ifndef _DYND__UNARY_EXPR_DTYPE_HPP_
#define _DYND__UNARY_EXPR_DTYPE_HPP_

#include <dynd/dtype.hpp>
#include <dynd/kernels/expr_kernel_generator.hpp>

namespace dynd {

/**
 * The unary_expr dtype is like the expr dtype, but
 * special-cased for unary operations. These specializations
 * are:
 *
 *  - The operand type is just the single operand, instead
 *    of being a pointer to the operand as is done in the
 *    expr dtype.
 *  - Elementwise unary expr dtypes are applied at the
 *    element level, so it can efficiently interoperate
 *    with other elementwise expression dtypes such as
 *    type conversion, byte swapping, etc.
 */
class unary_expr_dtype : public base_expression_dtype {
    dtype m_value_dtype, m_operand_dtype;
    const expr_kernel_generator *m_kgen;

public:
    unary_expr_dtype(const dtype& value_dtype, const dtype& operand_dtype,
                    const expr_kernel_generator *kgen);

    virtual ~unary_expr_dtype();

    const dtype& get_value_dtype() const {
        return m_value_dtype;
    }
    const dtype& get_operand_dtype() const {
        return m_operand_dtype;
    }

    void print_data(std::ostream& o, const char *metadata, const char *data) const;

    void print_dtype(std::ostream& o) const;

    dtype apply_linear_index(size_t nindices, const irange *indices,
                size_t current_i, const dtype& root_dt, bool leading_dimension) const;
    intptr_t apply_linear_index(size_t nindices, const irange *indices, const char *metadata,
                    const dtype& result_dtype, char *out_metadata,
                    memory_block_data *embedded_reference,
                    size_t current_i, const dtype& root_dt,
                    bool leading_dimension, char **inout_data,
                    memory_block_data **inout_dataref) const;

    bool is_lossless_assignment(const dtype& dst_dt, const dtype& src_dt) const;

    bool operator==(const base_dtype& rhs) const;

    dtype with_replaced_storage_dtype(const dtype& replacement_dtype) const;

    size_t make_operand_to_value_assignment_kernel(
                    hierarchical_kernel *out, size_t offset_out,
                    const char *dst_metadata, const char *src_metadata,
                    kernel_request_t kernreq, const eval::eval_context *ectx) const;
    size_t make_value_to_operand_assignment_kernel(
                    hierarchical_kernel *out, size_t offset_out,
                    const char *dst_metadata, const char *src_metadata,
                    kernel_request_t kernreq, const eval::eval_context *ectx) const;

    void get_dynamic_ndobject_properties(const std::pair<std::string, gfunc::callable> **out_properties,
                    size_t *out_count) const;
    void get_dynamic_ndobject_functions(const std::pair<std::string, gfunc::callable> **out_functions,
                    size_t *out_count) const;
};

/**
 * Makes a unary expr dtype.
 */
inline dtype make_unary_expr_dtype(const dtype& value_dtype,
                const dtype& operand_dtype,
                const expr_kernel_generator *kgen)
{
    return dtype(new unary_expr_dtype(value_dtype, operand_dtype, kgen), false);
}

} // namespace dynd

#endif // _DYND__UNARY_EXPR_DTYPE_HPP_
