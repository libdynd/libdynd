//
// Copyright (C) 2011-15 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <dynd/types/base_memory_type.hpp>
#include <dynd/types/typevar_constructed_type.hpp>
#include <dynd/types/typevar_type.hpp>
#include <dynd/func/make_callable.hpp>

using namespace std;
using namespace dynd;

typevar_constructed_type::typevar_constructed_type(const nd::string &name,
                                                   const ndt::type &arg)
    : base_type(typevar_constructed_type_id, pattern_kind, 0, 1,
                type_flag_symbolic, 0, arg.get_ndim(), arg.get_strided_ndim()),
      m_name(name), m_arg(arg)
{
//  static ndt::type args_pattern("((...), {...})");
  if (m_name.is_null()) {
    throw type_error("dynd typevar name cannot be null");
  } else if (!is_valid_typevar_name(m_name.begin(), m_name.end())) {
    stringstream ss;
    ss << "dynd typevar name ";
    print_escaped_utf8_string(ss, m_name.begin(), m_name.end());
    ss << " is not valid, it must be alphanumeric and begin with a capital";
    throw type_error(ss.str());
  }
//else if (!args.get_type().match(args_pattern)) {
  //  stringstream ss;
    //ss << "dynd constructed typevar must have args matching " << args_pattern
      // << ", which " << args.get_type() << " does not";
    //throw type_error(ss.str());
  //}
}

void typevar_constructed_type::get_vars(std::unordered_set<std::string> &vars) const
{
  vars.insert(m_name.str());
  m_arg.get_vars(vars);
}

void typevar_constructed_type::print_data(std::ostream &DYND_UNUSED(o),
                                          const char *DYND_UNUSED(arrmeta),
                                          const char *DYND_UNUSED(data)) const
{
  throw type_error("Cannot store data of typevar_constructed type");
}

void typevar_constructed_type::print_type(std::ostream &o) const
{
  // Type variables are barewords starting with a capital letter
  o << m_name.str() << "[" << m_arg << "]";
}

intptr_t typevar_constructed_type::get_dim_size(const char *DYND_UNUSED(arrmeta),
                                                const char *DYND_UNUSED(data)) const
{
    return -1;
}

ndt::type typevar_constructed_type::apply_linear_index(
    intptr_t DYND_UNUSED(nindices), const irange *DYND_UNUSED(indices),
    size_t DYND_UNUSED(current_i), const ndt::type &DYND_UNUSED(root_tp),
    bool DYND_UNUSED(leading_dimension)) const
{
  throw type_error("Cannot store data of typevar_constructed type");
}

intptr_t typevar_constructed_type::apply_linear_index(
    intptr_t DYND_UNUSED(nindices), const irange *DYND_UNUSED(indices),
    const char *DYND_UNUSED(arrmeta), const ndt::type &DYND_UNUSED(result_tp),
    char *DYND_UNUSED(out_arrmeta),
    memory_block_data *DYND_UNUSED(embedded_reference),
    size_t DYND_UNUSED(current_i), const ndt::type &DYND_UNUSED(root_tp),
    bool DYND_UNUSED(leading_dimension), char **DYND_UNUSED(inout_data),
    memory_block_data **DYND_UNUSED(inout_dataref)) const
{
  throw type_error("Cannot store data of typevar_constructed type");
}

bool
typevar_constructed_type::is_lossless_assignment(const ndt::type &dst_tp,
                                                 const ndt::type &src_tp) const
{
  if (dst_tp.extended() == this) {
    if (src_tp.extended() == this) {
      return true;
    } else if (src_tp.get_type_id() == typevar_constructed_type_id) {
      return *dst_tp.extended() == *src_tp.extended();
    }
  }

  return false;
}

bool typevar_constructed_type::operator==(const base_type &rhs) const
{
  if (this == &rhs) {
    return true;
  } else if (rhs.get_type_id() != typevar_constructed_type_id) {
    return false;
  } else {
    const typevar_constructed_type *tvt =
        static_cast<const typevar_constructed_type *>(&rhs);
    return m_name == tvt->m_name && m_arg == tvt->m_arg;
  }
}

void typevar_constructed_type::arrmeta_default_construct(
    char *DYND_UNUSED(arrmeta), bool DYND_UNUSED(blockref_alloc)) const
{
  throw type_error("Cannot store data of typevar_constructed type");
}

void typevar_constructed_type::arrmeta_copy_construct(
    char *DYND_UNUSED(dst_arrmeta), const char *DYND_UNUSED(src_arrmeta),
    memory_block_data *DYND_UNUSED(embedded_reference)) const
{
  throw type_error("Cannot store data of typevar_constructed type");
}

void
typevar_constructed_type::arrmeta_destruct(char *DYND_UNUSED(arrmeta)) const
{
  throw type_error("Cannot store data of typevar_constructed type");
}

bool typevar_constructed_type::match(
    const char *arrmeta, const ndt::type &candidate_tp,
    const char *candidate_arrmeta,
    std::map<nd::string, ndt::type> &tp_vars) const
{
  if (candidate_tp.get_type_id() == typevar_constructed_type_id) {
    return m_arg.match(arrmeta,
                         candidate_tp.extended<typevar_constructed_type>()->m_arg,
                         candidate_arrmeta, tp_vars);
  }

  if (candidate_tp.get_kind() != memory_kind) {
    if (m_arg.match(arrmeta, candidate_tp, candidate_arrmeta, tp_vars)) {
      ndt::type &tv_type = tp_vars[m_name];
      if (tv_type.is_null()) {
        tv_type = ndt::make_type<void>();
      }
      return true;
    }
    return false;
  }

  ndt::type &tv_type = tp_vars[m_name];
  if (tv_type.is_null()) {
    // This typevar hasn't been seen yet
    tv_type = candidate_tp.extended<base_memory_type>()->with_replaced_storage_type(
        ndt::make_type<void>());
  }

  return m_arg.match(
      arrmeta, candidate_tp.extended<base_memory_type>()->get_element_type(),
      candidate_arrmeta, tp_vars);
}

static nd::array property_get_name(const ndt::type &tp)
{
  return tp.extended<typevar_constructed_type>()->get_name();
}

void typevar_constructed_type::get_dynamic_type_properties(
    const std::pair<std::string, gfunc::callable> **out_properties,
    size_t *out_count) const
{
  static pair<string, gfunc::callable> type_properties[] = {
      pair<string, gfunc::callable>(
          "name", gfunc::make_callable(&property_get_name, "self")),
  };

  *out_properties = type_properties;
  *out_count = sizeof(type_properties) / sizeof(type_properties[0]);
}
