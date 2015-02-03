//
// Copyright (C) 2011-15 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#pragma once

#include <vector>
#include <string>

#include <dynd/array.hpp>
#include <dynd/string.hpp>

namespace dynd {

class typevar_constructed_type : public base_type {
  nd::string m_name;
  ndt::type m_arg;

public:
  typevar_constructed_type(const nd::string &name, const ndt::type &arg);

  virtual ~typevar_constructed_type() {}

  const nd::string &get_name() const { return m_name; }

  std::string get_name_str() const { return m_name.str(); }

  ndt::type get_arg() const { return m_arg; }

  void print_data(std::ostream &o, const char *arrmeta, const char *data) const;

  void print_type(std::ostream &o) const;

  intptr_t get_dim_size(const char *arrmeta, const char *data) const;

  ndt::type apply_linear_index(intptr_t nindices, const irange *indices,
                               size_t current_i, const ndt::type &root_tp,
                               bool leading_dimension) const;
  intptr_t apply_linear_index(intptr_t nindices, const irange *indices,
                              const char *arrmeta, const ndt::type &result_tp,
                              char *out_arrmeta,
                              memory_block_data *embedded_reference,
                              size_t current_i, const ndt::type &root_tp,
                              bool leading_dimension, char **inout_data,
                              memory_block_data **inout_dataref) const;

  bool is_lossless_assignment(const ndt::type &dst_tp,
                              const ndt::type &src_tp) const;

  bool operator==(const base_type &rhs) const;

  void arrmeta_default_construct(char *arrmeta, bool blockref_alloc) const;
  void arrmeta_copy_construct(char *dst_arrmeta, const char *src_arrmeta,
                              memory_block_data *embedded_reference) const;
  void arrmeta_destruct(char *arrmeta) const;

  bool matches(const ndt::type &self_tp, const char *self_arrmeta,
               const ndt::type &other_tp, const char *other_arrmeta,
               std::map<nd::string, ndt::type> &tp_vars) const;

  void get_dynamic_type_properties(
      const std::pair<std::string, gfunc::callable> **out_properties,
      size_t *out_count) const;
}; // class typevar_type

namespace ndt {
  /** Makes a typevar_constructed type with the specified types */
  inline ndt::type make_typevar_constructed(const nd::string &name,
                                            const ndt::type &arg)
  {
    return ndt::type(new typevar_constructed_type(name, arg), false);
  }
} // namespace ndt

} // namespace dynd