//
// Copyright (C) 2011-15 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#pragma once

#include <dynd/types/base_type.hpp>
#include <dynd/type.hpp>

namespace dynd {

class c_contiguous_type : public base_type {
  ndt::type m_child_tp;

public:
  c_contiguous_type(const ndt::type &child_tp);

  const ndt::type &get_child_type() const {
    return m_child_tp;
  }

  void print_data(std::ostream &o, const char *arrmeta, const char *data) const;

  void print_type(std::ostream &o) const;

  void get_shape(intptr_t ndim, intptr_t i, intptr_t *out_shape,
                 const char *arrmeta, const char *data) const;

  ndt::type apply_linear_index(intptr_t nindices, const irange *indices,
                               size_t current_i, const ndt::type &root_tp,
                               bool leading_dimension) const;

  intptr_t apply_linear_index(intptr_t nindices, const irange *indices,
                              const char *arrmeta, const ndt::type &result_type,
                              char *out_arrmeta,
                              memory_block_data *embedded_reference,
                              size_t current_i, const ndt::type &root_tp,
                              bool leading_dimension, char **inout_data,
                              memory_block_data **inout_dataref) const;

  ndt::type at_single(intptr_t i0, const char **inout_arrmeta,
                      const char **inout_data) const;

  ndt::type get_type_at_dimension(char **inout_arrmeta, intptr_t i,
                                  intptr_t total_ndim = 0) const;

  bool is_c_contiguous(const char *arrmeta) const;

  bool operator==(const base_type &rhs) const;

  virtual void arrmeta_default_construct(char *arrmeta,
                                         bool blockref_alloc) const;
  virtual void
  arrmeta_copy_construct(char *dst_arrmeta, const char *src_arrmeta,
                         memory_block_data *embedded_reference) const;
  virtual void arrmeta_destruct(char *arrmeta) const;

  virtual bool match(const char *arrmeta, const ndt::type &candidate_tp,
                     const char *candidate_arrmeta,
                     std::map<nd::string, ndt::type> &tp_vars) const;
};

namespace ndt {
  inline type make_c_contiguous(const type &child_tp)
  {
    return type(new c_contiguous_type(child_tp), false);
  }
} // namespace ndt

} // namespace dynd
