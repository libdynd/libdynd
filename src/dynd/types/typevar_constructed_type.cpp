//
// Copyright (C) 2011-16 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <dynd/types/base_memory_type.hpp>
#include <dynd/types/typevar_constructed_type.hpp>
#include <dynd/types/typevar_type.hpp>

using namespace std;
using namespace dynd;

ndt::typevar_constructed_type::typevar_constructed_type(const std::string &name, const type &arg)
    : base_type(typevar_constructed_id, 0, 1, type_flag_symbolic, 0, arg.get_ndim(), arg.get_strided_ndim()),
      m_name(name), m_arg(arg)
{
  //  static ndt::type args_pattern("((...), {...})");
  if (m_name.empty()) {
    throw type_error("dynd typevar name cannot be null");
  }
  else if (!is_valid_typevar_name(m_name.c_str(), m_name.c_str() + m_name.size())) {
    stringstream ss;
    ss << "dynd typevar name ";
    print_escaped_utf8_string(ss, m_name);
    ss << " is not valid, it must be alphanumeric and begin with a capital";
    throw type_error(ss.str());
  }
  // else if (!args.get_type().match(args_pattern)) {
  //  stringstream ss;
  // ss << "dynd constructed typevar must have args matching " << args_pattern
  // << ", which " << args.get_type() << " does not";
  // throw type_error(ss.str());
  //}
}

void ndt::typevar_constructed_type::get_vars(std::unordered_set<std::string> &vars) const
{
  vars.insert(m_name);
  m_arg.get_vars(vars);
}

void ndt::typevar_constructed_type::print_data(std::ostream &DYND_UNUSED(o), const char *DYND_UNUSED(arrmeta),
                                               const char *DYND_UNUSED(data)) const
{
  throw type_error("Cannot store data of typevar_constructed type");
}

void ndt::typevar_constructed_type::print_type(std::ostream &o) const
{
  // Type variables are barewords starting with a capital letter
  o << m_name << "[" << m_arg << "]";
}

intptr_t ndt::typevar_constructed_type::get_dim_size(const char *DYND_UNUSED(arrmeta),
                                                     const char *DYND_UNUSED(data)) const
{
  return -1;
}

ndt::type ndt::typevar_constructed_type::apply_linear_index(intptr_t DYND_UNUSED(nindices),
                                                            const irange *DYND_UNUSED(indices),
                                                            size_t DYND_UNUSED(current_i),
                                                            const type &DYND_UNUSED(root_tp),
                                                            bool DYND_UNUSED(leading_dimension)) const
{
  throw type_error("Cannot store data of typevar_constructed type");
}

intptr_t ndt::typevar_constructed_type::apply_linear_index(
    intptr_t DYND_UNUSED(nindices), const irange *DYND_UNUSED(indices), const char *DYND_UNUSED(arrmeta),
    const type &DYND_UNUSED(result_tp), char *DYND_UNUSED(out_arrmeta),
    const intrusive_ptr<memory_block_data> &DYND_UNUSED(embedded_reference), size_t DYND_UNUSED(current_i),
    const type &DYND_UNUSED(root_tp), bool DYND_UNUSED(leading_dimension), char **DYND_UNUSED(inout_data),
    intrusive_ptr<memory_block_data> &DYND_UNUSED(inout_dataref)) const
{
  throw type_error("Cannot store data of typevar_constructed type");
}

bool ndt::typevar_constructed_type::is_lossless_assignment(const type &dst_tp, const type &src_tp) const
{
  if (dst_tp.extended() == this) {
    if (src_tp.extended() == this) {
      return true;
    }
    else if (src_tp.get_id() == typevar_constructed_id) {
      return *dst_tp.extended() == *src_tp.extended();
    }
  }

  return false;
}

bool ndt::typevar_constructed_type::operator==(const base_type &rhs) const
{
  if (this == &rhs) {
    return true;
  }
  else if (rhs.get_id() != typevar_constructed_id) {
    return false;
  }
  else {
    const typevar_constructed_type *tvt = static_cast<const typevar_constructed_type *>(&rhs);
    return m_name == tvt->m_name && m_arg == tvt->m_arg;
  }
}

void ndt::typevar_constructed_type::arrmeta_default_construct(char *DYND_UNUSED(arrmeta),
                                                              bool DYND_UNUSED(blockref_alloc)) const
{
  throw type_error("Cannot store data of typevar_constructed type");
}

void ndt::typevar_constructed_type::arrmeta_copy_construct(
    char *DYND_UNUSED(dst_arrmeta), const char *DYND_UNUSED(src_arrmeta),
    const intrusive_ptr<memory_block_data> &DYND_UNUSED(embedded_reference)) const
{
  throw type_error("Cannot store data of typevar_constructed type");
}

void ndt::typevar_constructed_type::arrmeta_destruct(char *DYND_UNUSED(arrmeta)) const
{
  throw type_error("Cannot store data of typevar_constructed type");
}

bool ndt::typevar_constructed_type::match(const type &candidate_tp, std::map<std::string, type> &tp_vars) const
{
  if (candidate_tp.get_id() == typevar_constructed_id) {
    return m_arg.match(candidate_tp.extended<typevar_constructed_type>()->m_arg, tp_vars);
  }

  if (candidate_tp.get_base_id() != memory_id) {
    if (m_arg.match(candidate_tp, tp_vars)) {
      type &tv_type = tp_vars[m_name];
      if (tv_type.is_null()) {
        tv_type = make_type<void>();
      }
      return true;
    }
    return false;
  }

  type &tv_type = tp_vars[m_name];
  if (tv_type.is_null()) {
    // This typevar hasn't been seen yet
    tv_type = candidate_tp.extended<base_memory_type>()->with_replaced_storage_type(make_type<void>());
  }

  return m_arg.match(candidate_tp.extended<base_memory_type>()->get_element_type(), tp_vars);
}

std::map<std::string, std::pair<ndt::type, const char *>>
ndt::typevar_constructed_type::get_dynamic_type_properties() const
{
  std::map<std::string, std::pair<ndt::type, const char *>> properties;
  properties["name"] = {ndt::type("string"), reinterpret_cast<const char *>(&m_name)};

  return properties;
}
