//
// Copyright (C) 2011-14 Mark Wiebe, DyND Developers
// BSD 2-Clause License, see LICENSE.txt


#ifndef _DYND__UNARY_EXPR_TYPE_HPP_
#define _DYND__UNARY_EXPR_TYPE_HPP_

#include <dynd/type.hpp>
#include <dynd/kernels/expr_kernel_generator.hpp>

namespace dynd {

/**
 * The unary_expr type is like the expr type, but
 * special-cased for unary operations. These specializations
 * are:
 *
 *  - The operand type is just the single operand, instead
 *    of being a pointer to the operand as is done in the
 *    expr type.
 *  - Elementwise unary expr types are applied at the
 *    element level, so it can efficiently interoperate
 *    with other elementwise expression types such as
 *    type conversion, byte swapping, etc.
 */
class unary_expr_type : public base_expr_type {
    ndt::type m_value_type, m_operand_type;
    const expr_kernel_generator *m_kgen;

public:
    unary_expr_type(const ndt::type& value_type, const ndt::type& operand_type,
                    const expr_kernel_generator *kgen);

    virtual ~unary_expr_type();

    const ndt::type& get_value_type() const {
        return m_value_type;
    }
    const ndt::type& get_operand_type() const {
        return m_operand_type;
    }

    void print_data(std::ostream& o, const char *arrmeta, const char *data) const;

    void print_type(std::ostream& o) const;

    ndt::type apply_linear_index(intptr_t nindices, const irange *indices,
                size_t current_i, const ndt::type& root_tp, bool leading_dimension) const;
    intptr_t apply_linear_index(intptr_t nindices, const irange *indices, const char *arrmeta,
                    const ndt::type& result_tp, char *out_arrmeta,
                    memory_block_data *embedded_reference,
                    size_t current_i, const ndt::type& root_tp,
                    bool leading_dimension, char **inout_data,
                    memory_block_data **inout_dataref) const;

    bool is_lossless_assignment(const ndt::type& dst_tp, const ndt::type& src_tp) const;

    bool operator==(const base_type& rhs) const;

    ndt::type with_replaced_storage_type(const ndt::type& replacement_type) const;

    size_t make_operand_to_value_assignment_kernel(
                    ckernel_builder *ckb, intptr_t ckb_offset,
                    const char *dst_arrmeta, const char *src_arrmeta,
                    kernel_request_t kernreq, const eval::eval_context *ectx) const;
    size_t make_value_to_operand_assignment_kernel(
                    ckernel_builder *ckb, intptr_t ckb_offset,
                    const char *dst_arrmeta, const char *src_arrmeta,
                    kernel_request_t kernreq, const eval::eval_context *ectx) const;

    void get_dynamic_array_properties(const std::pair<std::string, gfunc::callable> **out_properties,
                    size_t *out_count) const;
    void get_dynamic_array_functions(const std::pair<std::string, gfunc::callable> **out_functions,
                    size_t *out_count) const;
};

namespace ndt {
    /**
     * Makes a unary expr type.
     */
    inline ndt::type make_unary_expr(const ndt::type& value_tp,
                    const ndt::type& operand_tp,
                    const expr_kernel_generator *kgen)
    {
        return ndt::type(new unary_expr_type(value_tp, operand_tp, kgen), false);
    }
}

} // namespace dynd

#endif // _DYND__UNARY_EXPR_TYPE_HPP_
