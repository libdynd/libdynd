//
// Copyright (C) 2011-15 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <dynd/types/c_contiguous_type.hpp>

using namespace std;
using namespace dynd;

c_contiguous_type::c_contiguous_type(const ndt::type &child_tp)
    : base_type(c_contiguous_type_id, child_tp.get_kind(),
                child_tp.get_data_size(), child_tp.get_data_alignment(),
                child_tp.get_flags(), child_tp.get_arrmeta_size(),
                child_tp.get_ndim(), 0),
      m_child_tp(child_tp)
{
  // Restrict c_contiguous_type to fixed_dim_type (and, eventually, tuple_type
  // and struct_type)
  if (m_child_tp.get_type_id() != fixed_dim_type_id) {
    throw std::invalid_argument(
        "c_contiguous_type must have a child that is a fixed_dim_type");
  }

  // Propagate the inherited flags from the element
  //  m_members.flags |=
  //    (child_tp.get_flags() &
  //   ((type_flags_operand_inherited | type_flags_value_inherited) &
  //  ~type_flag_scalar));
}

void c_contiguous_type::print_data(std::ostream &o, const char *arrmeta,
                                   const char *data) const
{
  m_child_tp.print_data(o, arrmeta, data);
}

void c_contiguous_type::print_type(std::ostream &o) const
{
  o << "c_contiguous[" << m_child_tp << "]";
}

void c_contiguous_type::get_shape(intptr_t ndim, intptr_t i,
                                  intptr_t *out_shape, const char *arrmeta,
                                  const char *data) const
{
  m_child_tp.extended()->get_shape(ndim, i, out_shape, arrmeta, data);
}

ndt::type c_contiguous_type::apply_linear_index(intptr_t nindices,
                                                const irange *indices,
                                                size_t current_i,
                                                const ndt::type &root_tp,
                                                bool leading_dimension) const
{
  if (nindices == 0) {
    return ndt::type(this, true);
  }

  ndt::type child_tp = m_child_tp.extended()->apply_linear_index(
      nindices, indices, current_i, root_tp, leading_dimension);

  // TODO: We can preserve c_contiguous in some cases
  return child_tp;
}

intptr_t c_contiguous_type::apply_linear_index(
    intptr_t nindices, const irange *indices, const char *arrmeta,
    const ndt::type &result_type, char *out_arrmeta,
    memory_block_data *embedded_reference, size_t current_i,
    const ndt::type &root_tp, bool leading_dimension, char **inout_data,
    memory_block_data **inout_dataref) const
{
  if (m_child_tp.is_builtin()) {
    return 0;
  }

  return m_child_tp.extended()->apply_linear_index(
      nindices, indices, arrmeta,
      result_type.extended<c_contiguous_type>()->m_child_tp, out_arrmeta,
      embedded_reference, current_i, root_tp, leading_dimension, inout_data,
      inout_dataref);
}

ndt::type c_contiguous_type::at_single(intptr_t i0, const char **inout_arrmeta,
                                       const char **inout_data) const
{
  return ndt::make_c_contiguous(
      m_child_tp.extended()->at_single(i0, inout_arrmeta, inout_data));
}

ndt::type c_contiguous_type::get_type_at_dimension(char **inout_arrmeta,
                                                   intptr_t i,
                                                   intptr_t total_ndim) const
{
  if (i == 0) {
    return ndt::type(this, true);
  }

  ndt::type child_tp = m_child_tp.extended()->get_type_at_dimension(
      inout_arrmeta, i, total_ndim);
  if (child_tp.is_builtin()) {
    return child_tp;
  }

  return ndt::make_c_contiguous(child_tp);
}

bool c_contiguous_type::is_c_contiguous(const char *DYND_UNUSED(arrmeta)) const
{
  return true;
}

bool c_contiguous_type::operator==(const base_type &rhs) const
{
  if (this == &rhs) {
    return true;
  } else if (rhs.get_type_id() != c_contiguous_type_id) {
    return false;
  } else {
    const c_contiguous_type *tp = static_cast<const c_contiguous_type *>(&rhs);
    return m_child_tp == tp->m_child_tp;
  }
}

void c_contiguous_type::arrmeta_default_construct(char *arrmeta,
                                                  bool blockref_alloc) const
{
  if (!m_child_tp.is_builtin()) {
    m_child_tp.extended()->arrmeta_default_construct(arrmeta, blockref_alloc);
  }

  if (!m_child_tp.is_c_contiguous(arrmeta)) {
    throw std::runtime_error(
        "c_contiguous_type must construct arrmeta that is c_contiguous");
  }
}

void c_contiguous_type::arrmeta_copy_construct(
    char *dst_arrmeta, const char *src_arrmeta,
    memory_block_data *embedded_reference) const
{
  if (!m_child_tp.is_builtin()) {
    m_child_tp.extended()->arrmeta_copy_construct(dst_arrmeta, src_arrmeta,
                                                  embedded_reference);
  }
}

void c_contiguous_type::arrmeta_destruct(char *arrmeta) const
{
  if (!m_child_tp.is_builtin()) {
    m_child_tp.extended()->arrmeta_destruct(arrmeta);
  }
}

bool c_contiguous_type::match(const char *arrmeta,
                              const ndt::type &candidate_tp,
                              const char *candidate_arrmeta,
                              std::map<nd::string, ndt::type> &tp_vars) const
{
  if (candidate_tp.get_type_id() == c_contiguous_type_id) {
    return m_child_tp.match(
        arrmeta, candidate_tp.extended<c_contiguous_type>()->m_child_tp,
        candidate_arrmeta, tp_vars);
  }

  return candidate_tp.is_c_contiguous(candidate_arrmeta) &&
         m_child_tp.match(arrmeta, candidate_tp, candidate_arrmeta, tp_vars);
}