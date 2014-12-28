//
// Copyright (C) 2011-14 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <dynd/types/pow_dimsym_type.hpp>
#include <dynd/types/typevar_type.hpp>
#include <dynd/types/fixed_dim_type.hpp>
#include <dynd/types/typevar_dim_type.hpp>
#include <dynd/func/make_callable.hpp>

using namespace std;
using namespace dynd;

pow_dimsym_type::pow_dimsym_type(const ndt::type &base_tp, const nd::string &exponent,
                                   const ndt::type &element_type)
    : base_dim_type(pow_dimsym_type_id, element_type, 0, 1, 0,
                            type_flag_symbolic, false),
      m_base_tp(base_tp), m_exponent(exponent)
{
  if (base_tp.get_kind() != dim_kind ||
      base_tp.extended<base_dim_type>()->get_element_type().get_type_id() !=
          void_type_id) {
    stringstream ss;
    ss << "dynd base type for dimensional power symbolic type is not valid: "
       << base_tp;
    throw type_error(ss.str());
  }
  if (m_exponent.is_null()) {
    throw type_error("dynd typevar name cannot be null");
  }
  else if (!is_valid_typevar_name(m_exponent.begin(), m_exponent.end())) {
    stringstream ss;
    ss << "dynd typevar name ";
    print_escaped_utf8_string(ss, m_exponent.begin(), m_exponent.end());
    ss << " is not valid, it must be alphanumeric and begin with a capital";
    throw type_error(ss.str());
  }
}

void pow_dimsym_type::print_data(std::ostream &DYND_UNUSED(o),
                                 const char *DYND_UNUSED(arrmeta),
                                 const char *DYND_UNUSED(data)) const
{
  throw type_error("Cannot store data of typevar type");
}

void pow_dimsym_type::print_type(std::ostream& o) const
{
  switch (m_base_tp.get_type_id()) {
  case fixed_dim_type_id:
    o << m_base_tp.extended<fixed_dim_type>()->get_fixed_dim_size();
    break;
  case cfixed_dim_type_id:
    o << m_base_tp.extended<cfixed_dim_type>()->get_fixed_dim_size();
    break;
  case fixed_dimsym_type_id:
    o << "Fixed";
    break;
  case var_dim_type_id:
    o << "var";
    break;
  case typevar_dim_type_id:
    o << m_base_tp.extended<typevar_dim_type>()->get_name_str();
    break;
  default:
    break;
  }

  o << "**" << m_exponent.str() << " * " << get_element_type();
}

ndt::type pow_dimsym_type::apply_linear_index(
    intptr_t DYND_UNUSED(nindices), const irange *DYND_UNUSED(indices),
    size_t DYND_UNUSED(current_i), const ndt::type &DYND_UNUSED(root_tp),
    bool DYND_UNUSED(leading_dimension)) const
{
    throw type_error("Cannot store data of typevar type");
}

intptr_t pow_dimsym_type::apply_linear_index(
    intptr_t DYND_UNUSED(nindices), const irange *DYND_UNUSED(indices),
    const char *DYND_UNUSED(arrmeta), const ndt::type &DYND_UNUSED(result_tp),
    char *DYND_UNUSED(out_arrmeta),
    memory_block_data *DYND_UNUSED(embedded_reference),
    size_t DYND_UNUSED(current_i), const ndt::type &DYND_UNUSED(root_tp),
    bool DYND_UNUSED(leading_dimension), char **DYND_UNUSED(inout_data),
    memory_block_data **DYND_UNUSED(inout_dataref)) const
{
  throw type_error("Cannot store data of typevar type");
}

intptr_t pow_dimsym_type::get_dim_size(const char *DYND_UNUSED(arrmeta),
                                       const char *DYND_UNUSED(data)) const
{
  return -1;
}

bool pow_dimsym_type::is_lossless_assignment(
    const ndt::type &DYND_UNUSED(dst_tp),
    const ndt::type &DYND_UNUSED(src_tp)) const
{
  return false;
}

bool pow_dimsym_type::operator==(const base_type& rhs) const
{
  if (this == &rhs) {
    return true;
  }
  else if (rhs.get_type_id() != pow_dimsym_type_id) {
    return false;
  }
  else {
    const pow_dimsym_type *tvt = static_cast<const pow_dimsym_type *>(&rhs);
    return m_exponent == tvt->m_exponent && m_base_tp == tvt->m_base_tp &&
           m_element_tp == tvt->m_element_tp;
  }
}

void pow_dimsym_type::arrmeta_default_construct(
    char *DYND_UNUSED(arrmeta), bool DYND_UNUSED(blockref_alloc)) const
{
  throw type_error("Cannot store data of typevar type");
}

void pow_dimsym_type::arrmeta_copy_construct(
    char *DYND_UNUSED(dst_arrmeta), const char *DYND_UNUSED(src_arrmeta),
    memory_block_data *DYND_UNUSED(embedded_reference)) const
{
    throw type_error("Cannot store data of typevar type");
}

size_t pow_dimsym_type::arrmeta_copy_construct_onedim(
    char *DYND_UNUSED(dst_arrmeta), const char *DYND_UNUSED(src_arrmeta),
    memory_block_data *DYND_UNUSED(embedded_reference)) const
{
    throw type_error("Cannot store data of typevar type");
}

void pow_dimsym_type::arrmeta_destruct(char *DYND_UNUSED(arrmeta)) const
{
    throw type_error("Cannot store data of typevar type");
}

/*
static nd::array property_get_name(const ndt::type& tp) {
    return tp.extended<typevar_dim_type>()->get_name();
}

static ndt::type property_get_element_type(const ndt::type& dt) {
    return dt.extended<typevar_dim_type>()->get_element_type();
}

void typevar_dim_type::get_dynamic_type_properties(
    const std::pair<std::string, gfunc::callable> **out_properties,
    size_t *out_count) const
{
    static pair<string, gfunc::callable> type_properties[] = {
        pair<string, gfunc::callable>(
            "name", gfunc::make_callable(&property_get_name, "self")),
        pair<string, gfunc::callable>(
            "element_type",
            gfunc::make_callable(&property_get_element_type, "self")), };

    *out_properties = type_properties;
    *out_count = sizeof(type_properties) / sizeof(type_properties[0]);
}
*/
