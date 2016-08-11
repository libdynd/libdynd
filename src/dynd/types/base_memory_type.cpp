//
// Copyright (C) 2011-16 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <dynd/types/base_memory_type.hpp>

using namespace std;
using namespace dynd;

size_t ndt::base_memory_type::get_default_data_size() const {
  if (m_element_tp.is_builtin()) {
    return m_element_tp.get_data_size();
  } else {
    return m_element_tp.extended()->get_default_data_size();
  }
}

void ndt::base_memory_type::print_data(std::ostream &o, const char *arrmeta, const char *data) const {
  m_element_tp.print_data(o, arrmeta + m_storage_arrmeta_offset, data);
}

void ndt::base_memory_type::get_shape(intptr_t ndim, intptr_t i, intptr_t *out_shape, const char *arrmeta,
                                      const char *data) const {
  m_element_tp.extended()->get_shape(ndim, i, out_shape, arrmeta, data);
}

void ndt::base_memory_type::get_strides(size_t i, intptr_t *out_strides, const char *arrmeta) const {
  m_element_tp.extended()->get_strides(i, out_strides, arrmeta);
}

void ndt::base_memory_type::get_vars(std::unordered_set<std::string> &vars) const { m_element_tp.get_vars(vars); }

ndt::type ndt::base_memory_type::apply_linear_index(intptr_t nindices, const irange *indices, size_t current_i,
                                                    const type &root_tp, bool leading_dimension) const {
  return with_replaced_storage_type(
      m_element_tp.extended()->apply_linear_index(nindices, indices, current_i, root_tp, leading_dimension));
}

intptr_t ndt::base_memory_type::apply_linear_index(intptr_t nindices, const irange *indices, const char *arrmeta,
                                                   const type &result_type, char *out_arrmeta,
                                                   const nd::memory_block &embedded_reference, size_t current_i,
                                                   const type &root_tp, bool leading_dimension, char **inout_data,
                                                   nd::memory_block &inout_dataref) const {
  if (m_element_tp.is_builtin()) {
    return 0;
  }

  return m_element_tp.extended()->apply_linear_index(
      nindices, indices, arrmeta, result_type.extended<base_memory_type>()->get_element_type(), out_arrmeta,
      embedded_reference, current_i, root_tp, leading_dimension, inout_data, inout_dataref);
}

ndt::type ndt::base_memory_type::at_single(intptr_t i0, const char **inout_arrmeta, const char **inout_data) const {
  return with_replaced_storage_type(m_element_tp.extended()->at_single(i0, inout_arrmeta, inout_data));
}

ndt::type ndt::base_memory_type::get_type_at_dimension(char **inout_arrmeta, intptr_t i, intptr_t total_ndim) const {
  if (i == 0) {
    return type(this, true);
  }

  return with_replaced_storage_type(m_element_tp.extended()->get_type_at_dimension(inout_arrmeta, i, total_ndim));
}

bool ndt::base_memory_type::is_lossless_assignment(const type &dst_tp, const type &src_tp) const {
  // Default to calling with the storage types
  if (dst_tp.extended() == this) {
    return ::is_lossless_assignment(m_element_tp, src_tp);
  } else {
    return ::is_lossless_assignment(dst_tp, m_element_tp);
  }
}

void ndt::base_memory_type::transform_child_types(type_transform_fn_t transform_fn, intptr_t arrmeta_offset,
                                                  void *extra, type &out_transformed_tp,
                                                  bool &out_was_transformed) const {
  type tmp_tp;
  bool was_transformed = false;
  transform_fn(m_element_tp, arrmeta_offset + m_storage_arrmeta_offset, extra, tmp_tp, was_transformed);
  if (was_transformed) {
    out_transformed_tp = with_replaced_storage_type(tmp_tp);
    out_was_transformed = true;
  } else {
    out_transformed_tp = type(this, true);
  }
}

ndt::type ndt::base_memory_type::get_canonical_type() const { return m_element_tp.get_canonical_type(); }

void ndt::base_memory_type::arrmeta_default_construct(char *arrmeta, bool blockref_alloc) const {
  if (!m_element_tp.is_builtin()) {
    m_element_tp.extended()->arrmeta_default_construct(arrmeta + m_storage_arrmeta_offset, blockref_alloc);
  }
}

void ndt::base_memory_type::arrmeta_copy_construct(char *dst_arrmeta, const char *src_arrmeta,
                                                   const nd::memory_block &embedded_reference) const {
  if (!m_element_tp.is_builtin()) {
    m_element_tp.extended()->arrmeta_copy_construct(dst_arrmeta + m_storage_arrmeta_offset,
                                                    src_arrmeta + m_storage_arrmeta_offset, embedded_reference);
  }
}

void ndt::base_memory_type::arrmeta_destruct(char *arrmeta) const {
  if (!m_element_tp.is_builtin()) {
    m_element_tp.extended()->arrmeta_destruct(arrmeta + m_storage_arrmeta_offset);
  }
}

bool ndt::base_memory_type::match(const type &candidate_tp, std::map<std::string, type> &tp_vars) const {
  if (candidate_tp.get_base_id() != memory_id) {
    return false;
  }

  return m_element_tp.match(candidate_tp.extended<base_memory_type>()->m_element_tp, tp_vars);
}

std::map<std::string, std::pair<ndt::type, const char *>> ndt::base_memory_type::get_dynamic_type_properties() const {
  std::map<std::string, std::pair<ndt::type, const char *>> properties;
  properties["storage_type"] = {ndt::make_type<type_type>(), reinterpret_cast<const char *>(&m_element_tp)};

  return properties;
}
