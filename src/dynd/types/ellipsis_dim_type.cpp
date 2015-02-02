//
// Copyright (C) 2011-15 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <dynd/types/dim_fragment_type.hpp>
#include <dynd/types/ellipsis_dim_type.hpp>
#include <dynd/types/typevar_type.hpp>
#include <dynd/func/make_callable.hpp>

using namespace std;
using namespace dynd;

ellipsis_dim_type::ellipsis_dim_type(const nd::string &name,
                                     const ndt::type &element_type)
    : base_dim_type(ellipsis_dim_type_id, element_type, 0, 1, 0,
                    type_flag_symbolic | type_flag_dim_variadic, false),
      m_name(name)
{
  if (!m_name.is_null()) {
    // Make sure name begins with a capital letter, and is an identifier
    const char *begin = m_name.begin(), *end = m_name.end();
    if (end - begin == 0) {
      // Convert empty string into NULL
      m_name = nd::string();
    } else if (!is_valid_typevar_name(begin, end)) {
      stringstream ss;
      ss << "dynd ellipsis name \"";
      print_escaped_utf8_string(ss, begin, end);
      ss << "\" is not valid, it must be alphanumeric and begin with a capital";
      throw type_error(ss.str());
    }
  }
}

void ellipsis_dim_type::print_data(std::ostream &DYND_UNUSED(o),
                                   const char *DYND_UNUSED(arrmeta),
                                   const char *DYND_UNUSED(data)) const
{
  throw type_error("Cannot store data of ellipsis type");
}

void ellipsis_dim_type::print_type(std::ostream &o) const
{
  // Type variables are barewords starting with a capital letter
  if (!m_name.is_null()) {
    o << m_name.str();
  }
  o << "... * " << get_element_type();
}

intptr_t ellipsis_dim_type::get_dim_size(const char *DYND_UNUSED(arrmeta),
                                         const char *DYND_UNUSED(data)) const
{
  return -1;
}

bool ellipsis_dim_type::is_lossless_assignment(const ndt::type &dst_tp,
                                               const ndt::type &src_tp) const
{
  if (dst_tp.extended() == this) {
    if (src_tp.extended() == this) {
      return true;
    } else if (src_tp.get_type_id() == ellipsis_dim_type_id) {
      return *dst_tp.extended() == *src_tp.extended();
    }
  }

  return false;
}

bool ellipsis_dim_type::operator==(const base_type &rhs) const
{
  if (this == &rhs) {
    return true;
  } else if (rhs.get_type_id() != ellipsis_dim_type_id) {
    return false;
  } else {
    const ellipsis_dim_type *tvt = static_cast<const ellipsis_dim_type *>(&rhs);
    return m_name == tvt->m_name && m_element_tp == tvt->m_element_tp;
  }
}

ndt::type
ellipsis_dim_type::get_type_at_dimension(char **DYND_UNUSED(inout_arrmeta),
                                         intptr_t i, intptr_t total_ndim) const
{
  if (i == 0) {
    return ndt::type(this, true);
  } else if (i <= m_element_tp.get_ndim()) {
    return m_element_tp.get_type_at_dimension(NULL, i - 1, total_ndim + 1);
  } else {
    return m_element_tp.get_type_at_dimension(NULL, m_element_tp.get_ndim(),
                                              total_ndim + 1 +
                                                  m_element_tp.get_ndim());
  }
}

void ellipsis_dim_type::arrmeta_default_construct(
    char *DYND_UNUSED(arrmeta), bool DYND_UNUSED(blockref_alloc)) const
{
  throw type_error("Cannot store data of ellipsis type");
}

void ellipsis_dim_type::arrmeta_copy_construct(
    char *DYND_UNUSED(dst_arrmeta), const char *DYND_UNUSED(src_arrmeta),
    memory_block_data *DYND_UNUSED(embedded_reference)) const
{
  throw type_error("Cannot store data of ellipsis type");
}

size_t ellipsis_dim_type::arrmeta_copy_construct_onedim(
    char *DYND_UNUSED(dst_arrmeta), const char *DYND_UNUSED(src_arrmeta),
    memory_block_data *DYND_UNUSED(embedded_reference)) const
{
  throw type_error("Cannot store data of ellipsis type");
}

void ellipsis_dim_type::arrmeta_destruct(char *DYND_UNUSED(arrmeta)) const
{
  throw type_error("Cannot store data of ellipsis type");
}

bool ellipsis_dim_type::matches(const char *arrmeta, const ndt::type &other,
                                std::map<nd::string, ndt::type> &tp_vars) const
{
  if (other.get_type_id() == any_sym_type_id) {
    return true;
  }

  if (other.get_ndim() == 0) {
    const nd::string &tv_name = get_name();
    if (!tv_name.is_null()) {
      ndt::type &tv_type = tp_vars[tv_name];
      if (tv_type.is_null()) {
        // This typevar hasn't been seen yet, make it
        // be an empty dim fragment
        tv_type = ndt::make_dim_fragment();
      } else {
        // Make sure the type matches previous
        // instances of the type var, which is
        // always ok from the zero dims case
        // because "Dims..." combine
        // with broadcasting rules.
        if (tv_type.get_type_id() != dim_fragment_type_id) {
          // Inconsistent type var usage, previously
          // wasn't a dim fragment
          return false;
        }
      }
    }
    return other.matches(arrmeta, get_element_type(), tp_vars);
  } else if (other.get_type_id() == ellipsis_dim_type_id) {
    return m_element_tp.matches(
        arrmeta, other.extended<ellipsis_dim_type>()->m_element_tp, tp_vars);
  } else if (other.get_ndim() >= get_ndim() - 1) {
    intptr_t matched_ndim = other.get_ndim() - get_ndim() + 1;
    const nd::string &tv_name = get_name();
    if (!tv_name.is_null()) {
      ndt::type &tv_type = tp_vars[tv_name];
      if (tv_type.is_null()) {
        // This typevar hasn't been seen yet, so it's
        // a dim fragment of the given size.
        tv_type = ndt::make_dim_fragment(matched_ndim, other);
      } else {
        // Make sure the type matches previous  instances of the type var,
        // which in this case means they should broadcast together.
        if (tv_type.get_type_id() == dim_fragment_type_id) {
          ndt::type result =
              tv_type.extended<dim_fragment_type>()->broadcast_with_type(
                  matched_ndim, other);
          if (result.is_null()) {
            return false;
          } else {
            tv_type.swap(result);
          }
        } else {
          // Inconsistent type var usage, previously
          // wasn't a dim fragment
          return false;
        }
      }
    }
    return other.get_type_at_dimension(NULL, matched_ndim)
        .matches(arrmeta, get_element_type(), tp_vars);
  }

  return false;
}

static nd::array property_get_name(const ndt::type &tp)
{
  return tp.extended<ellipsis_dim_type>()->get_name();
}

static ndt::type property_get_element_type(const ndt::type &dt)
{
  return dt.extended<ellipsis_dim_type>()->get_element_type();
}

void ellipsis_dim_type::get_dynamic_type_properties(
    const std::pair<std::string, gfunc::callable> **out_properties,
    size_t *out_count) const
{
  static pair<string, gfunc::callable> type_properties[] = {
      pair<string, gfunc::callable>(
          "name", gfunc::make_callable(&property_get_name, "self")),
      pair<string, gfunc::callable>(
          "element_type",
          gfunc::make_callable(&property_get_element_type, "self")),
  };

  *out_properties = type_properties;
  *out_count = sizeof(type_properties) / sizeof(type_properties[0]);
}
