//
// Copyright (C) 2011-14 Mark Wiebe, DyND Developers
// BSD 2-Clause License, see LICENSE.txt


#ifndef _DYND__EXPR_TYPE_HPP_
#define _DYND__EXPR_TYPE_HPP_

#include <dynd/type.hpp>
#include <dynd/types/pointer_type.hpp>
#include <dynd/kernels/expr_kernel_generator.hpp>

namespace dynd {

/**
 * The expr type represents an expression on
 * an arbitrary number of operands. It stores
 * its operands as a structure of pointer types.
 *
 * The computation is held by an expr_kernel_generator
 * instance, which is able to create the kernels
 * on demand, as a subkernel of an unary_ckernel_builder
 * object.
 *
 * The unary kernel is always treated specially, it
 * doesn't go into an expr_type unless it's a component
 * of a larger kernel. Instead, it goes into the
 * unary_expr_type, which skips the extra pointer indirection
 * needed for multiple operands.
 *
 * TODO: It would be nice to put the expr_kernel_generator
 *       into the arrmeta, so that static expr_type instances
 *       could be shared between different operations like
 *       +, -, *, /. The operand type defines the arrmeta, though,
 *       so a special type just for this purpose may be required.
 */
class expr_type : public base_expr_type {
    ndt::type m_value_type, m_operand_type;
    const expr_kernel_generator *m_kgen;

public:
    expr_type(const ndt::type& value_type, const ndt::type& operand_type,
                    const expr_kernel_generator *kgen);

    virtual ~expr_type();

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

        void get_shape(intptr_t ndim, intptr_t i, intptr_t *out_shape, const char *arrmeta, const char *data) const;

    bool is_lossless_assignment(const ndt::type& dst_tp, const ndt::type& src_tp) const;

    bool operator==(const base_type& rhs) const;

    ndt::type with_replaced_storage_type(const ndt::type& replacement_type) const;

    inline const expr_kernel_generator& get_kgen() const {
        return *m_kgen;
    }

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
     * Makes an expr type.
     */
    inline ndt::type make_expr(const ndt::type& value_type,
                    const ndt::type& operand_type,
                    const expr_kernel_generator *kgen)
    {
        return ndt::type(new expr_type(value_type, operand_type, kgen), false);
    }
} // namespace ndt

} // namespace dynd

#endif // _DYND__EXPR_TYPE_HPP_
