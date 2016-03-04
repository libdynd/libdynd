//
// Copyright (C) 2011-15 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <dynd/callable.hpp>
#include <dynd/types/callable_type.hpp>
#include <dynd/types/fixed_dim_type.hpp>
#include <dynd/types/str_util.hpp>
#include <dynd/ensure_immutable_contig.hpp>
#include <dynd/types/typevar_type.hpp>

using namespace std;
using namespace dynd;

ndt::callable_type::callable_type(const type &ret_type, const type &pos_types, const type &kwd_types)
    : base_type(callable_id, sizeof(data_type), alignof(data_type), type_flag_zeroinit | type_flag_destructor, 0, 0, 0),
      m_return_type(ret_type), m_pos_tuple(pos_types), m_kwd_struct(kwd_types)
{
  if (m_pos_tuple.get_id() != tuple_id) {
    stringstream ss;
    ss << "dynd callable positional arg types require a tuple type, got a "
          "type \"" << m_pos_tuple << "\"";
    throw invalid_argument(ss.str());
  }
  if (m_kwd_struct.get_id() != struct_id) {
    stringstream ss;
    ss << "dynd callable keyword arg types require a struct type, got a "
          "type \"" << m_kwd_struct << "\"";
    throw invalid_argument(ss.str());
  }

  for (intptr_t i = 0, i_end = get_nkwd(); i < i_end; ++i) {
    if (m_kwd_struct.extended<tuple_type>()->get_field_type(i).get_id() == option_id) {
      m_opt_kwd_indices.push_back(i);
    }
  }

  // TODO: Should check that all the kwd names are simple identifier names
  //       because struct_type does not check that.

  // Note that we don't base the flags of this type on that of its arguments
  // and return types, because it is something the can be instantiated, even
  // for arguments that are symbolic.
}

static void print_callable(std::ostream &o, const ndt::callable_type *DYND_UNUSED(af_tp),
                           const ndt::callable_type::data_type *af)
{
  o << "<callable at " << (void *)af << ">";
}

void ndt::callable_type::print_data(std::ostream &o, const char *DYND_UNUSED(arrmeta), const char *data) const
{
  const data_type *af = reinterpret_cast<const data_type *>(data);
  print_callable(o, this, af);
}

void ndt::callable_type::print_type(std::ostream &o) const
{
  intptr_t npos = get_npos();
  intptr_t nkwd = get_nkwd();

  o << "(";

  for (intptr_t i = 0; i < npos; ++i) {
    if (i > 0) {
      o << ", ";
    }

    o << get_pos_type(i);
  }
  if (m_pos_tuple.extended<tuple_type>()->is_variadic()) {
    if (npos > 0) {
      o << ", ...";
    }
    else {
      o << "...";
    }
  }
  for (intptr_t i = 0; i < nkwd; ++i) {
    if (i > 0 || npos > 0) {
      o << ", ";
    }

    // TODO: names should be validated on input, not just
    //       printed specially like in struct_type.
    const std::string &name = get_kwd_name(i);
    if (is_simple_identifier_name(name)) {
      o << name;
    }
    else {
      print_escaped_utf8_string(o, name, true);
    }
    o << ": " << get_kwd_type(i);
  }
  if (nkwd > 0 && m_kwd_struct.extended<struct_type>()->is_variadic()) {
    o << "...";
  }

  o << ") -> " << m_return_type;
}

void ndt::callable_type::transform_child_types(type_transform_fn_t transform_fn, intptr_t arrmeta_offset, void *extra,
                                               type &out_transformed_tp, bool &out_was_transformed) const
{
  type tmp_return_type, tmp_pos_types, tmp_kwd_types;

  bool was_transformed = false;
  transform_fn(m_return_type, arrmeta_offset, extra, tmp_return_type, was_transformed);
  transform_fn(m_pos_tuple, arrmeta_offset, extra, tmp_pos_types, was_transformed);
  transform_fn(m_kwd_struct, arrmeta_offset, extra, tmp_kwd_types, was_transformed);
  if (was_transformed) {
    out_transformed_tp = make(tmp_return_type, tmp_pos_types, tmp_kwd_types);
    out_was_transformed = true;
  }
  else {
    out_transformed_tp = type(this, true);
  }
}

ndt::type ndt::callable_type::get_canonical_type() const
{
  type tmp_return_type, tmp_pos_types, tmp_kwd_types;

  tmp_return_type = m_return_type.get_canonical_type();
  tmp_pos_types = m_pos_tuple.get_canonical_type();
  tmp_kwd_types = m_kwd_struct.get_canonical_type();
  return make(tmp_return_type, tmp_pos_types, tmp_kwd_types);
}

void ndt::callable_type::get_vars(std::unordered_set<std::string> &vars) const
{
  m_return_type.get_vars(vars);
  m_pos_tuple.get_vars(vars);
  m_kwd_struct.get_vars(vars);
}

ndt::type ndt::callable_type::apply_linear_index(intptr_t DYND_UNUSED(nindices), const irange *DYND_UNUSED(indices),
                                                 size_t DYND_UNUSED(current_i), const type &DYND_UNUSED(root_tp),
                                                 bool DYND_UNUSED(leading_dimension)) const
{
  throw type_error("Cannot store data of funcproto type");
}

intptr_t ndt::callable_type::apply_linear_index(intptr_t DYND_UNUSED(nindices), const irange *DYND_UNUSED(indices),
                                                const char *DYND_UNUSED(arrmeta), const type &DYND_UNUSED(result_tp),
                                                char *DYND_UNUSED(out_arrmeta),
                                                const intrusive_ptr<memory_block_data> &DYND_UNUSED(embedded_reference),
                                                size_t DYND_UNUSED(current_i), const type &DYND_UNUSED(root_tp),
                                                bool DYND_UNUSED(leading_dimension), char **DYND_UNUSED(inout_data),
                                                intrusive_ptr<memory_block_data> &DYND_UNUSED(inout_dataref)) const
{
  throw type_error("Cannot store data of funcproto type");
}

bool ndt::callable_type::is_lossless_assignment(const type &dst_tp, const type &src_tp) const
{
  if (dst_tp.extended() == this) {
    if (src_tp.extended() == this) {
      return true;
    }
    else if (src_tp.get_id() == callable_id) {
      return *dst_tp.extended() == *src_tp.extended();
    }
  }

  return false;
}

bool ndt::callable_type::operator==(const base_type &rhs) const
{
  if (this == &rhs) {
    return true;
  }
  else if (rhs.get_id() != callable_id) {
    return false;
  }
  else {
    const callable_type *fpt = static_cast<const callable_type *>(&rhs);
    return m_return_type == fpt->m_return_type && m_pos_tuple == fpt->m_pos_tuple && m_kwd_struct == fpt->m_kwd_struct;
  }
}

void ndt::callable_type::arrmeta_default_construct(char *DYND_UNUSED(arrmeta), bool DYND_UNUSED(blockref_alloc)) const
{
}

void ndt::callable_type::arrmeta_copy_construct(
    char *DYND_UNUSED(dst_arrmeta), const char *DYND_UNUSED(src_arrmeta),
    const intrusive_ptr<memory_block_data> &DYND_UNUSED(embedded_reference)) const
{
}

void ndt::callable_type::arrmeta_reset_buffers(char *DYND_UNUSED(arrmeta)) const {}

void ndt::callable_type::arrmeta_finalize_buffers(char *DYND_UNUSED(arrmeta)) const {}

void ndt::callable_type::arrmeta_destruct(char *DYND_UNUSED(arrmeta)) const {}

void ndt::callable_type::data_destruct(const char *DYND_UNUSED(arrmeta), char *data) const
{
  const data_type *d = reinterpret_cast<data_type *>(data);
  d->~data_type();
}

void ndt::callable_type::data_destruct_strided(const char *DYND_UNUSED(arrmeta), char *data, intptr_t stride,
                                               size_t count) const
{
  for (size_t i = 0; i != count; ++i, data += stride) {
    const data_type *d = reinterpret_cast<data_type *>(data);
    d->~data_type();
  }
}

bool ndt::callable_type::match(const type &candidate_tp, std::map<std::string, type> &tp_vars) const
{
  if (candidate_tp.get_id() != callable_id) {
    return false;
  }

  // First match the return type
  if (!m_return_type.match(candidate_tp.extended<callable_type>()->m_return_type, tp_vars)) {
    return false;
  }

  // Next match all the positional parameters
  if (!m_pos_tuple.match(candidate_tp.extended<callable_type>()->m_pos_tuple, tp_vars)) {
    return false;
  }

  // Finally match all the keyword parameters
  if (!m_kwd_struct.match(candidate_tp.extended<callable_type>()->get_kwd_struct(), tp_vars)) {
    return false;
  }

  return true;
}

std::map<std::string, std::pair<ndt::type, const char *>> ndt::callable_type::get_dynamic_type_properties() const
{
  std::map<std::string, std::pair<ndt::type, const char *>> properties;
  const std::vector<ndt::type> &pos_types = m_pos_tuple.extended<ndt::tuple_type>()->get_field_types();
  const std::vector<ndt::type> &kwd_types = m_kwd_struct.extended<ndt::struct_type>()->get_field_types();
  const std::vector<std::string> &kwd_names = m_kwd_struct.extended<ndt::struct_type>()->get_field_names();

  properties["pos_types"] = {ndt::type_for(pos_types), reinterpret_cast<const char *>(&pos_types)};
  properties["kwd_types"] = {ndt::type_for(kwd_types), reinterpret_cast<const char *>(&kwd_types)};
  properties["kwd_names"] = {ndt::type_for(kwd_names), reinterpret_cast<const char *>(&kwd_names)};
  properties["return_type"] = {ndt::type("type"), reinterpret_cast<const char *>(&m_return_type)};

  return properties;
}

ndt::type ndt::make_generic_funcproto(intptr_t nargs)
{
  return callable_type::make(typevar_type::make("R"), make_typevar_range("T", nargs));
}
