//
// Copyright (C) 2011-15 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#pragma once

#include <dynd/type.hpp>
#include <dynd/kernels/expr_kernel_generator.hpp>

namespace dynd {
namespace ndt {

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
  class DYND_API unary_expr_type : public base_expr_type {
    type m_value_type, m_operand_type;
    const expr_kernel_generator *m_kgen;

  public:
    unary_expr_type(const type &value_type, const type &operand_type,
                    const expr_kernel_generator *kgen);

    virtual ~unary_expr_type();

    const type &get_value_type() const { return m_value_type; }
    const type &get_operand_type() const { return m_operand_type; }

    void print_data(std::ostream &o, const char *arrmeta,
                    const char *data) const;

    void print_type(std::ostream &o) const;

    type apply_linear_index(intptr_t nindices, const irange *indices,
                            size_t current_i, const type &root_tp,
                            bool leading_dimension) const;
    intptr_t apply_linear_index(intptr_t nindices, const irange *indices,
                                const char *arrmeta, const type &result_tp,
                                char *out_arrmeta,
                                const intrusive_ptr<memory_block_data> &embedded_reference,
                                size_t current_i, const type &root_tp,
                                bool leading_dimension, char **inout_data,
                                intrusive_ptr<memory_block_data> &inout_dataref) const;

    bool is_lossless_assignment(const type &dst_tp, const type &src_tp) const;

    bool operator==(const base_type &rhs) const;

    type with_replaced_storage_type(const type &replacement_type) const;

    size_t make_operand_to_value_assignment_kernel(
        void *ckb, intptr_t ckb_offset, const char *dst_arrmeta,
        const char *src_arrmeta, kernel_request_t kernreq,
        const eval::eval_context *ectx) const;
    size_t make_value_to_operand_assignment_kernel(
        void *ckb, intptr_t ckb_offset, const char *dst_arrmeta,
        const char *src_arrmeta, kernel_request_t kernreq,
        const eval::eval_context *ectx) const;

    void get_dynamic_array_properties(
        const std::pair<std::string, gfunc::callable> **out_properties,
        size_t *out_count) const;
    void get_dynamic_array_functions(
        const std::pair<std::string, gfunc::callable> **out_functions,
        size_t *out_count) const;

    /**
     * Makes a unary expr type.
     */
    static type make(const type &value_tp, const type &operand_tp,
                     const expr_kernel_generator *kgen)
    {
      return type(new unary_expr_type(value_tp, operand_tp, kgen), false);
    }
  };

} // namespace dynd::ndt
} // namespace dynd
