//
// Copyright (C) 2011-15 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <dynd/types/dim_fragment_type.hpp>
#include <dynd/types/ellipsis_dim_type.hpp>
#include <dynd/types/typevar_type.hpp>
#include <dynd/func/apply.hpp>
#include <dynd/gfunc/make_gcallable.hpp>
#include <dynd/kernels/base_property_kernel.hpp>

using namespace std;
using namespace dynd;

ndt::ellipsis_dim_type::ellipsis_dim_type(const std::string &name, const type &element_type)
    : base_dim_type(ellipsis_dim_type_id, pattern_kind, element_type, 0, 1, 0, type_flag_symbolic | type_flag_variadic,
                    false),
      m_name(name)
{
  if (!m_name.empty()) {
    // Make sure name begins with a capital letter, and is an identifier
    const char *begin = m_name.c_str(), *end = m_name.c_str() + m_name.size();
    if (end - begin == 0) {
      // Convert empty string into NULL
      m_name = "";
    } else if (!is_valid_typevar_name(begin, end)) {
      stringstream ss;
      ss << "dynd ellipsis name \"";
      print_escaped_utf8_string(ss, m_name);
      ss << "\" is not valid, it must be alphanumeric and begin with a capital";
      throw type_error(ss.str());
    }
  }
}

void ndt::ellipsis_dim_type::get_vars(std::unordered_set<std::string> &vars) const
{
  vars.insert(m_name);
  m_element_tp.get_vars(vars);
}

void ndt::ellipsis_dim_type::print_data(std::ostream &DYND_UNUSED(o), const char *DYND_UNUSED(arrmeta),
                                        const char *DYND_UNUSED(data)) const
{
  throw type_error("Cannot store data of ellipsis type");
}

void ndt::ellipsis_dim_type::print_type(std::ostream &o) const
{
  // Type variables are barewords starting with a capital letter
  if (!m_name.empty()) {
    o << m_name;
  }
  o << "... * " << get_element_type();
}

intptr_t ndt::ellipsis_dim_type::get_dim_size(const char *DYND_UNUSED(arrmeta), const char *DYND_UNUSED(data)) const
{
  return -1;
}

bool ndt::ellipsis_dim_type::is_lossless_assignment(const type &dst_tp, const type &src_tp) const
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

bool ndt::ellipsis_dim_type::operator==(const base_type &rhs) const
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

ndt::type ndt::ellipsis_dim_type::get_type_at_dimension(char **DYND_UNUSED(inout_arrmeta), intptr_t i,
                                                        intptr_t total_ndim) const
{
  if (i == 0) {
    return type(this, true);
  } else if (i <= m_element_tp.get_ndim()) {
    return m_element_tp.get_type_at_dimension(NULL, i - 1, total_ndim + 1);
  } else {
    return m_element_tp.get_type_at_dimension(NULL, m_element_tp.get_ndim(), total_ndim + 1 + m_element_tp.get_ndim());
  }
}

void ndt::ellipsis_dim_type::arrmeta_default_construct(char *DYND_UNUSED(arrmeta),
                                                       bool DYND_UNUSED(blockref_alloc)) const
{
  throw type_error("Cannot store data of ellipsis type");
}

void ndt::ellipsis_dim_type::arrmeta_copy_construct(char *DYND_UNUSED(dst_arrmeta),
                                                    const char *DYND_UNUSED(src_arrmeta),
                                                    memory_block_data *DYND_UNUSED(embedded_reference)) const
{
  throw type_error("Cannot store data of ellipsis type");
}

size_t ndt::ellipsis_dim_type::arrmeta_copy_construct_onedim(char *DYND_UNUSED(dst_arrmeta),
                                                             const char *DYND_UNUSED(src_arrmeta),
                                                             memory_block_data *DYND_UNUSED(embedded_reference)) const
{
  throw type_error("Cannot store data of ellipsis type");
}

void ndt::ellipsis_dim_type::arrmeta_destruct(char *DYND_UNUSED(arrmeta)) const
{
  throw type_error("Cannot store data of ellipsis type");
}

bool ndt::ellipsis_dim_type::match(const char *arrmeta, const type &candidate_tp, const char *candidate_arrmeta,
                                   std::map<std::string, type> &tp_vars) const
{
  // TODO XXX This is wrong, "Any" could represent a type that doesn't match
  // against this one...
  if (candidate_tp.get_type_id() == any_kind_type_id) {
    return true;
  }

  if (candidate_tp.get_ndim() == 0) {
    const std::string &tv_name = get_name();
    if (!tv_name.empty()) {
      type &tv_type = tp_vars[tv_name];
      if (tv_type.is_null()) {
        // This typevar hasn't been seen yet, make it
        // be an empty dim fragment
        tv_type = make_dim_fragment();
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
    return m_element_tp.match(arrmeta, candidate_tp, candidate_arrmeta, tp_vars);
  } else if (candidate_tp.get_type_id() == ellipsis_dim_type_id) {
    return m_element_tp.match(arrmeta, candidate_tp.extended<ellipsis_dim_type>()->m_element_tp, candidate_arrmeta,
                              tp_vars);
  } else if (candidate_tp.get_ndim() >= get_ndim() - 1) {
    intptr_t matched_ndim = candidate_tp.get_ndim() - get_ndim() + 1;
    const std::string &tv_name = get_name();
    if (!tv_name.empty()) {
      type &tv_type = tp_vars[tv_name];
      if (tv_type.is_null()) {
        // This typevar hasn't been seen yet, so it's
        // a dim fragment of the given size.
        tv_type = make_dim_fragment(matched_ndim, candidate_tp);
      } else {
        // Make sure the type matches previous  instances of the type var,
        // which in this case means they should broadcast together.
        if (tv_type.get_type_id() == dim_fragment_type_id) {
          type result = tv_type.extended<dim_fragment_type>()->broadcast_with_type(matched_ndim, candidate_tp);
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
    return m_element_tp.match(arrmeta, candidate_tp.get_type_at_dimension(NULL, matched_ndim), NULL, tp_vars);
  }

  return false;
}

/*
static nd::array property_get_name(const ndt::type &tp)
{
  return tp.extended<ndt::ellipsis_dim_type>()->get_name();
}
*/

static ndt::type property_get_element_type(ndt::type dt)
{
  return dt.extended<ndt::ellipsis_dim_type>()->get_element_type();
}

void ndt::ellipsis_dim_type::get_dynamic_type_properties(const std::pair<std::string, nd::callable> **out_properties,
                                                         size_t *out_count) const
{
  struct name_kernel : nd::base_property_kernel<name_kernel> {
    name_kernel(const ndt::type &tp, const ndt::type &dst_tp, const char *dst_arrmeta)
        : base_property_kernel<name_kernel>(tp, dst_tp, dst_arrmeta)
    {
    }

    void single(char *dst, char *const *DYND_UNUSED(src))
    {
      nd::array a = static_cast<nd::array>(tp.extended<ellipsis_dim_type>()->get_name());
      typed_data_copy(dst_tp, dst_arrmeta, dst,
                      static_cast<nd::array>(tp.extended<ellipsis_dim_type>()->get_name()).get_arrmeta(), a.get_data());
    }

    static void resolve_dst_type(char *DYND_UNUSED(static_data), size_t DYND_UNUSED(data_size), char *data,
                                 ndt::type &dst_tp, intptr_t DYND_UNUSED(nsrc), const ndt::type *DYND_UNUSED(src_tp),
                                 const dynd::nd::array &DYND_UNUSED(kwds),
                                 const std::map<std::string, ndt::type> &DYND_UNUSED(tp_vars))
    {
      const type &tp = *reinterpret_cast<const ndt::type *>(data);
      dst_tp = static_cast<nd::array>(tp.extended<ellipsis_dim_type>()->get_name()).get_type();
    }
  };

  static pair<std::string, nd::callable> type_properties[] = {
      //      pair<string, nd::callable>(
      //        "name", nd::callable::make<name_kernel>(type("(self: type) ->
      //        Any"))),
      pair<std::string, nd::callable>("element_type", nd::functional::apply(&property_get_element_type, "self"))};

  *out_properties = type_properties;
  *out_count = sizeof(type_properties) / sizeof(type_properties[0]);
}

ndt::type ndt::ellipsis_dim_type::with_element_type(const type &element_tp) const
{
  return make_ellipsis_dim(m_name, element_tp);
}
