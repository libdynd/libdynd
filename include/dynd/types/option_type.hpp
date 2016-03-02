//
// Copyright (C) 2011-15 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#pragma once

#include <dynd/callable.hpp>

namespace dynd {

template <typename ValueType>
struct option {
  ValueType value;

  option() : value(ndt::traits<ValueType>::na()) {}

  option(ValueType value) : value(value) {}
};

template <typename ValueType>
option<ValueType> opt()
{
  return option<ValueType>();
}

template <typename ValueType>
option<ValueType> opt(ValueType value)
{
  return option<ValueType>(value);
}

namespace ndt {

  /**
   * The option type represents data which may or may not be there.
   */
  class DYND_API option_type : public base_type {
    type m_value_tp;

  public:
    option_type(const type &value_tp);

    size_t get_default_data_size() const { return m_value_tp.get_default_data_size(); }

    void get_vars(std::unordered_set<std::string> &vars) const;

    const type &get_value_type() const { return m_value_tp.value_type(); }

    /** Assigns NA to one value */
    void assign_na(const char *arrmeta, char *data, const eval::eval_context *ectx) const;

    /** Returns true if the value is available */
    bool is_avail(const char *arrmeta, const char *data, const eval::eval_context *ectx) const;

    void print_type(std::ostream &o) const;
    void print_data(std::ostream &o, const char *arrmeta, const char *data) const;

    bool is_expression() const;
    bool is_unique_data_owner(const char *arrmeta) const;
    void transform_child_types(type_transform_fn_t transform_fn, intptr_t arrmeta_offset, void *extra,
                               type &out_transformed_tp, bool &out_was_transformed) const;
    type get_canonical_type() const;

    void set_from_utf8_string(const char *arrmeta, char *data, const char *utf8_begin, const char *utf8_end,
                              const eval::eval_context *ectx) const;

    type get_type_at_dimension(char **inout_arrmeta, intptr_t i, intptr_t total_ndim = 0) const;

    bool is_lossless_assignment(const type &dst_tp, const type &src_tp) const;

    bool operator==(const base_type &rhs) const;

    void arrmeta_default_construct(char *arrmeta, bool blockref_alloc) const;
    void arrmeta_copy_construct(char *dst_arrmeta, const char *src_arrmeta,
                                const intrusive_ptr<memory_block_data> &embedded_reference) const;
    void arrmeta_reset_buffers(char *arrmeta) const;
    void arrmeta_finalize_buffers(char *arrmeta) const;
    void arrmeta_destruct(char *arrmeta) const;
    void arrmeta_debug_print(const char *arrmeta, std::ostream &o, const std::string &indent) const;

    void data_destruct(const char *arrmeta, char *data) const;
    void data_destruct_strided(const char *arrmeta, char *data, intptr_t stride, size_t count) const;

    bool match(const char *arrmeta, const type &candidate_tp, const char *candidate_arrmeta,
               std::map<std::string, type> &tp_vars) const;

    std::map<std::string, std::pair<ndt::type, void *>> get_dynamic_type_properties() const;
  };

  template <typename ValueType>
  struct traits<option<ValueType>> {
    static const bool is_same_layout = true;

    static type equivalent() { return make_type<option_type>(make_type<ValueType>()); }
  };

} // namespace dynd::ndt
} // namespace dynd
