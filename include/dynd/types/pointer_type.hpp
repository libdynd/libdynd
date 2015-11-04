//
// Copyright (C) 2011-15 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

/**
 * The pointer type contains C/C++ raw pointers
 * pointing at data in other memory_blocks, using
 * blockrefs to manage the memory.
 *
 * This type operates in a "gather/scatter" fashion,
 * exposing itself as an expression type whose expression
 * copies the data to/from the pointer targets.
 */

#pragma once

#include <dynd/type.hpp>
#include <dynd/types/void_pointer_type.hpp>

namespace dynd {

struct DYND_API pointer_type_arrmeta {
  /**
   * A reference to the memory block which contains the data.
   */
  intrusive_ptr<memory_block_data> blockref;
  /* Each pointed-to destination is offset by this amount */
  intptr_t offset;
};

namespace ndt {

  class DYND_API pointer_type : public base_expr_type {
    type m_target_tp;

  public:
    pointer_type(const type &target_tp);

    virtual ~pointer_type();

    const type &get_value_type() const { return m_target_tp.value_type(); }
    const type &get_operand_type() const;

    const type &get_target_type() const { return m_target_tp; }

    void print_data(std::ostream &o, const char *arrmeta,
                    const char *data) const;

    void print_type(std::ostream &o) const;

    inline bool is_type_subarray(const type &subarray_tp) const
    {
      // Uniform dimensions can share one implementation
      return (!subarray_tp.is_builtin() &&
              (*this) == (*subarray_tp.extended())) ||
             m_target_tp.is_type_subarray(subarray_tp);
    }

    bool is_expression() const;
    bool is_unique_data_owner(const char *arrmeta) const;
    void transform_child_types(type_transform_fn_t transform_fn,
                               intptr_t arrmeta_offset, void *extra,
                               type &out_transformed_tp,
                               bool &out_was_transformed) const;
    type get_canonical_type() const;

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
    type at_single(intptr_t i0, const char **inout_arrmeta,
                   const char **inout_data) const;

    type get_type_at_dimension(char **inout_arrmeta, intptr_t i,
                               intptr_t total_ndim = 0) const;

    void get_shape(intptr_t ndim, intptr_t i, intptr_t *out_shape,
                   const char *arrmeta, const char *data) const;

    axis_order_classification_t classify_axis_order(const char *arrmeta) const;

    bool is_lossless_assignment(const type &dst_tp, const type &src_tp) const;

    bool operator==(const base_type &rhs) const;

    type with_replaced_storage_type(const type &replacement_type) const;

    void arrmeta_default_construct(char *arrmeta, bool blockref_alloc) const;
    void arrmeta_copy_construct(char *dst_arrmeta, const char *src_arrmeta,
                                const intrusive_ptr<memory_block_data> &embedded_reference) const;
    void arrmeta_reset_buffers(char *arrmeta) const;
    void arrmeta_finalize_buffers(char *arrmeta) const;
    void arrmeta_destruct(char *arrmeta) const;
    void arrmeta_debug_print(const char *arrmeta, std::ostream &o,
                             const std::string &indent) const;

    intptr_t make_assignment_kernel(void *ckb, intptr_t ckb_offset,
                                    const type &dst_tp, const char *dst_arrmeta,
                                    const type &src_tp, const char *src_arrmeta,
                                    kernel_request_t kernreq,
                                    const eval::eval_context *ectx) const;

    size_t make_operand_to_value_assignment_kernel(
        void *ckb, intptr_t ckb_offset, const char *dst_arrmeta,
        const char *src_arrmeta, kernel_request_t kernreq,
        const eval::eval_context *ectx) const;

    nd::array get_option_nafunc() const;

    bool match(const char *arrmeta, const type &candidate_tp,
               const char *candidate_arrmeta,
               std::map<std::string, type> &tp_vars) const;

    void get_dynamic_type_properties(
        const std::pair<std::string, nd::callable> **out_properties,
        size_t *out_count) const;
    void get_dynamic_array_functions(
        const std::pair<std::string, gfunc::callable> **out_functions,
        size_t *out_count) const;

    static type make(const type &target_tp);

    template <typename T>
    static type make()
    {
      return make(type::make<T>());
    }
  };

  template <typename T>
  struct type::equivalent<T *> {
    static type make() { return pointer_type::make(type::make<T>()); }
  };

  inline type make_pointer_type(const type &target_tp)
  {
    return pointer_type::make(target_tp);
  }

} // namespace dynd::ndt
} // namespace dynd
