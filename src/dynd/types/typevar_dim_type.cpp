//
// Copyright (C) 2011-16 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <dynd/types/fixed_dim_type.hpp>
#include <dynd/types/typevar_dim_type.hpp>
#include <dynd/types/typevar_type.hpp>

using namespace std;
using namespace dynd;

void ndt::typevar_dim_type::get_vars(std::unordered_set<std::string> &vars) const {
  vars.insert(m_name);
  m_element_tp.get_vars(vars);
}

void ndt::typevar_dim_type::print_data(std::ostream &DYND_UNUSED(o), const char *DYND_UNUSED(arrmeta),
                                       const char *DYND_UNUSED(data)) const {
  throw type_error("Cannot store data of typevar type");
}

void ndt::typevar_dim_type::print_type(std::ostream &o) const {
  // Type variables are barewords starting with a capital letter
  o << m_name << " * " << get_element_type();
}

intptr_t ndt::typevar_dim_type::get_dim_size(const char *DYND_UNUSED(arrmeta), const char *DYND_UNUSED(data)) const {
  return -1;
}

bool ndt::typevar_dim_type::is_lossless_assignment(const type &dst_tp, const type &src_tp) const {
  if (dst_tp.extended() == this) {
    if (src_tp.extended() == this) {
      return true;
    } else if (src_tp.get_id() == typevar_dim_id) {
      return *dst_tp.extended() == *src_tp.extended();
    }
  }

  return false;
}

bool ndt::typevar_dim_type::operator==(const base_type &rhs) const {
  if (this == &rhs) {
    return true;
  } else if (rhs.get_id() != typevar_dim_id) {
    return false;
  } else {
    const typevar_dim_type *tvt = static_cast<const typevar_dim_type *>(&rhs);
    return m_name == tvt->m_name && m_element_tp == tvt->m_element_tp;
  }
}

ndt::type ndt::typevar_dim_type::get_type_at_dimension(char **DYND_UNUSED(inout_arrmeta), intptr_t i,
                                                       intptr_t total_ndim) const {
  if (i == 0) {
    return type(this, true);
  } else {
    return m_element_tp.get_type_at_dimension(NULL, i - 1, total_ndim + 1);
  }
}

void ndt::typevar_dim_type::arrmeta_default_construct(char *DYND_UNUSED(arrmeta),
                                                      bool DYND_UNUSED(blockref_alloc)) const {
  throw type_error("Cannot store data of typevar type");
}

void ndt::typevar_dim_type::arrmeta_copy_construct(char *DYND_UNUSED(dst_arrmeta), const char *DYND_UNUSED(src_arrmeta),
                                                   const nd::memory_block &DYND_UNUSED(embedded_reference)) const {
  throw type_error("Cannot store data of typevar type");
}

size_t
ndt::typevar_dim_type::arrmeta_copy_construct_onedim(char *DYND_UNUSED(dst_arrmeta),
                                                     const char *DYND_UNUSED(src_arrmeta),
                                                     const nd::memory_block &DYND_UNUSED(embedded_reference)) const {
  throw type_error("Cannot store data of typevar type");
}

void ndt::typevar_dim_type::arrmeta_destruct(char *DYND_UNUSED(arrmeta)) const {
  throw type_error("Cannot store data of typevar type");
}

bool ndt::typevar_dim_type::match(const type &candidate_tp, std::map<std::string, type> &tp_vars) const {
  if (candidate_tp.is_scalar()) {
    return false;
  }

  type &tv_type = tp_vars[get_name()];
  if (tv_type.is_null()) {
    // This typevar hasn't been seen yet
    tv_type = candidate_tp;
    return m_element_tp.match(candidate_tp.get_type_at_dimension(NULL, 1), tp_vars);
  } else {
    // Make sure the type matches previous
    // instances of the type var
    if (candidate_tp.get_id() != tv_type.get_id()) {
      return false;
    }
    switch (candidate_tp.get_id()) {
    case fixed_dim_id:
      if (candidate_tp.extended<fixed_dim_type>()->get_fixed_dim_size() !=
          tv_type.extended<fixed_dim_type>()->get_fixed_dim_size()) {
        return false;
      }
      break;
    default:
      break;
    }
    return m_element_tp.match(candidate_tp.get_type_at_dimension(NULL, 1), tp_vars);
  }
}

std::map<std::string, std::pair<ndt::type, const char *>> ndt::typevar_dim_type::get_dynamic_type_properties() const {
  std::map<std::string, std::pair<ndt::type, const char *>> properties;
  properties["name"] = {ndt::type("string"), reinterpret_cast<const char *>(&m_name)};
  properties["element_type"] = {ndt::make_type<type_type>(), reinterpret_cast<const char *>(&m_element_tp)};

  return properties;
}

ndt::type ndt::typevar_dim_type::with_element_type(const type &element_tp) const {
  return make_type<typevar_dim_type>(m_name, element_tp);
}
