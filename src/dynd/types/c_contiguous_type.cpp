//
// Copyright (C) 2011-15 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <dynd/types/c_contiguous_type.hpp>

using namespace std;
using namespace dynd;

ndt::c_contiguous_type::c_contiguous_type(const type &child_tp)
    : base_type(c_contiguous_id, 0, 1, type_flag_symbolic | (child_tp.get_flags() & (type_flags_value_inherited |
                                                                                     type_flags_operand_inherited)),
                0, child_tp.get_ndim(), 0),
      m_child_tp(child_tp)
{
  // Restrict c_contiguous_type to fixed_dim_type (and, eventually, tuple_type
  // and struct_type)
  if (m_child_tp.get_id() != fixed_dim_id) {
    throw std::invalid_argument("c_contiguous_type must have a child that is a fixed_dim_type");
  }
}

void ndt::c_contiguous_type::print_data(std::ostream &o, const char *arrmeta, const char *data) const
{
  m_child_tp.print_data(o, arrmeta, data);
}

void ndt::c_contiguous_type::print_type(std::ostream &o) const { o << "C[" << m_child_tp << "]"; }

void ndt::c_contiguous_type::get_shape(intptr_t ndim, intptr_t i, intptr_t *out_shape, const char *arrmeta,
                                       const char *data) const
{
  m_child_tp.extended()->get_shape(ndim, i, out_shape, arrmeta, data);
}

ndt::type ndt::c_contiguous_type::apply_linear_index(intptr_t nindices, const irange *indices, size_t current_i,
                                                     const type &root_tp, bool leading_dimension) const
{
  if (nindices == 0) {
    return type(this, true);
  }

  type child_tp = m_child_tp.extended()->apply_linear_index(nindices, indices, current_i, root_tp, leading_dimension);

  // TODO: We can preserve c_contiguous in some cases
  return child_tp;
}

intptr_t ndt::c_contiguous_type::apply_linear_index(intptr_t nindices, const irange *indices, const char *arrmeta,
                                                    const type &result_type, char *out_arrmeta,
                                                    const intrusive_ptr<memory_block_data> &embedded_reference,
                                                    size_t current_i, const type &root_tp, bool leading_dimension,
                                                    char **inout_data,
                                                    intrusive_ptr<memory_block_data> &inout_dataref) const
{
  if (m_child_tp.is_builtin()) {
    return 0;
  }

  return m_child_tp.extended()->apply_linear_index(
      nindices, indices, arrmeta, result_type.extended<c_contiguous_type>()->m_child_tp, out_arrmeta,
      embedded_reference, current_i, root_tp, leading_dimension, inout_data, inout_dataref);
}

ndt::type ndt::c_contiguous_type::at_single(intptr_t i0, const char **inout_arrmeta, const char **inout_data) const
{
  return make(m_child_tp.extended()->at_single(i0, inout_arrmeta, inout_data));
}

ndt::type ndt::c_contiguous_type::get_type_at_dimension(char **inout_arrmeta, intptr_t i, intptr_t total_ndim) const
{
  if (i == 0) {
    return type(this, true);
  }

  type child_tp = m_child_tp.extended()->get_type_at_dimension(inout_arrmeta, i, total_ndim);
  if (child_tp.is_builtin()) {
    return child_tp;
  }

  return make(child_tp);
}

bool ndt::c_contiguous_type::is_c_contiguous(const char *DYND_UNUSED(arrmeta)) const { return true; }

bool ndt::c_contiguous_type::operator==(const base_type &rhs) const
{
  if (this == &rhs) {
    return true;
  }
  else if (rhs.get_id() != c_contiguous_id) {
    return false;
  }
  else {
    const c_contiguous_type *tp = static_cast<const c_contiguous_type *>(&rhs);
    return m_child_tp == tp->m_child_tp;
  }
}

void ndt::c_contiguous_type::arrmeta_default_construct(char *DYND_UNUSED(arrmeta),
                                                       bool DYND_UNUSED(blockref_alloc)) const
{
  stringstream ss;
  ss << "Cannot default construct arrmeta for symbolic type " << type(this, true);
  throw runtime_error(ss.str());
}

void ndt::c_contiguous_type::arrmeta_copy_construct(
    char *DYND_UNUSED(dst_arrmeta), const char *DYND_UNUSED(src_arrmeta),
    const intrusive_ptr<memory_block_data> &DYND_UNUSED(embedded_reference)) const
{
  stringstream ss;
  ss << "Cannot copy construct arrmeta for symbolic type " << type(this, true);
  throw runtime_error(ss.str());
}

void ndt::c_contiguous_type::arrmeta_destruct(char *DYND_UNUSED(arrmeta)) const {}

bool ndt::c_contiguous_type::match(const char *arrmeta, const type &candidate_tp, const char *candidate_arrmeta,
                                   std::map<std::string, type> &tp_vars) const
{
  if (candidate_tp.get_id() == c_contiguous_id) {
    return m_child_tp.match(arrmeta, candidate_tp.extended<c_contiguous_type>()->m_child_tp, candidate_arrmeta,
                            tp_vars);
  }

  return candidate_tp.is_c_contiguous(candidate_arrmeta) &&
         m_child_tp.match(arrmeta, candidate_tp, candidate_arrmeta, tp_vars);
}
