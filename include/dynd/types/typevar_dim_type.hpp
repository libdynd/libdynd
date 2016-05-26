//
// Copyright (C) 2011-16 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#pragma once

#include <string>
#include <vector>

#include <dynd/types/base_dim_type.hpp>
#include <dynd/types/typevar_type.hpp>

namespace dynd {
namespace ndt {

  class DYNDT_API typevar_dim_type : public base_dim_type {
    std::string m_name;

  public:
    typevar_dim_type(type_id_t id, const std::string &name, const type &element_type)
        : base_dim_type(id, element_type, 0, 1, 0, type_flag_symbolic, false), m_name(name) {
      if (m_name.empty()) {
        throw type_error("dynd typevar name cannot be null");
      } else if (!is_valid_typevar_name(m_name.c_str(), m_name.c_str() + m_name.size())) {
        std::stringstream ss;
        ss << "dynd typevar name ";
        print_escaped_utf8_string(ss, m_name);
        ss << " is not valid, it must be alphanumeric and begin with a capital";
        throw type_error(ss.str());
      }
    }

    const std::string &get_name() const { return m_name; }

    void print_data(std::ostream &o, const char *arrmeta, const char *data) const;

    void print_type(std::ostream &o) const;

    intptr_t get_dim_size(const char *arrmeta, const char *data) const;

    void get_vars(std::unordered_set<std::string> &vars) const;

    bool is_lossless_assignment(const type &dst_tp, const type &src_tp) const;

    bool operator==(const base_type &rhs) const;

    type get_type_at_dimension(char **inout_arrmeta, intptr_t i, intptr_t total_ndim = 0) const;

    void arrmeta_default_construct(char *arrmeta, bool blockref_alloc) const;
    void arrmeta_copy_construct(char *dst_arrmeta, const char *src_arrmeta,
                                const nd::memory_block &embedded_reference) const;
    size_t arrmeta_copy_construct_onedim(char *dst_arrmeta, const char *src_arrmeta,
                                         const nd::memory_block &embedded_reference) const;
    void arrmeta_destruct(char *arrmeta) const;

    bool match(const type &candidate_tp, std::map<std::string, type> &tp_vars) const;

    std::map<std::string, std::pair<ndt::type, const char *>> get_dynamic_type_properties() const;

    virtual type with_element_type(const type &element_tp) const;
  };

  template <>
  struct id_of<typevar_dim_type> : std::integral_constant<type_id_t, typevar_dim_id> {};

} // namespace dynd::ndt
} // namespace dynd
