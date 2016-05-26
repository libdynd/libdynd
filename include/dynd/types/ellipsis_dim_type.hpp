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

  type make_ellipsis_dim(const std::string &name, const type &element_type);

  class DYNDT_API ellipsis_dim_type : public base_dim_type {
    std::string m_name;

  public:
    ellipsis_dim_type(type_id_t id, const std::string &name, const type &element_type)
        : base_dim_type(id, element_type, 0, 1, 0, type_flag_symbolic | type_flag_variadic, false), m_name(name) {
      if (!m_name.empty()) {
        // Make sure name begins with a capital letter, and is an identifier
        const char *begin = m_name.c_str(), *end = m_name.c_str() + m_name.size();
        if (end - begin == 0) {
          // Convert empty string into NULL
          m_name = "";
        } else if (!is_valid_typevar_name(begin, end)) {
          std::stringstream ss;
          ss << "dynd ellipsis name \"";
          print_escaped_utf8_string(ss, m_name);
          ss << "\" is not valid, it must be alphanumeric and begin with a capital";
          throw type_error(ss.str());
        }
      }
    }

    const std::string &get_name() const { return m_name; }

    void get_vars(std::unordered_set<std::string> &vars) const;

    void print_data(std::ostream &o, const char *arrmeta, const char *data) const;

    void print_type(std::ostream &o) const;

    intptr_t get_dim_size(const char *arrmeta, const char *data) const;

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

    static type make_if_not_variadic(const type &element_tp) {
      if (element_tp.is_variadic()) {
        return element_tp;
      }

      return make_ellipsis_dim("Dims", element_tp);
    }
  };

  template <>
  struct id_of<ellipsis_dim_type> : std::integral_constant<type_id_t, ellipsis_dim_id> {};

  /** Makes an ellipsis type with the specified name and element type */
  inline type make_ellipsis_dim(const std::string &name, const type &element_type) {
    return make_type<ellipsis_dim_type>(name, element_type);
  }

  /** Make an unnamed ellipsis type */
  inline type make_ellipsis_dim(const type &element_type) { return make_type<ellipsis_dim_type>("", element_type); }

} // namespace dynd::ndt
} // namespace dynd
