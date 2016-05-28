//
// Copyright (C) 2011-16 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#pragma once

#include <string>
#include <vector>

#include <dynd/types/base_type.hpp>

namespace dynd {
namespace ndt {

  /**
   * Checks if the provided string range is a valid typevar name.
   */
  bool is_valid_typevar_name(const char *begin, const char *end);

  class DYNDT_API typevar_type : public base_type {
    std::string m_name;

  public:
    typevar_type(type_id_t id, const std::string &name)
        : base_type(id, scalar_kind_id, 0, 1, type_flag_symbolic, 0, 0, 0), m_name(name) {
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

    void get_vars(std::unordered_set<std::string> &vars) const;

    void print_data(std::ostream &o, const char *arrmeta, const char *data) const;

    void print_type(std::ostream &o) const;

    type apply_linear_index(intptr_t nindices, const irange *indices, size_t current_i, const type &root_tp,
                            bool leading_dimension) const;
    intptr_t apply_linear_index(intptr_t nindices, const irange *indices, const char *arrmeta, const type &result_tp,
                                char *out_arrmeta, const nd::memory_block &embedded_reference, size_t current_i,
                                const type &root_tp, bool leading_dimension, char **inout_data,
                                nd::memory_block &inout_dataref) const;

    bool is_lossless_assignment(const type &dst_tp, const type &src_tp) const;

    bool operator==(const base_type &rhs) const;

    void arrmeta_default_construct(char *arrmeta, bool blockref_alloc) const;
    void arrmeta_copy_construct(char *dst_arrmeta, const char *src_arrmeta,
                                const nd::memory_block &embedded_reference) const;
    void arrmeta_destruct(char *arrmeta) const;

    bool match(const type &candidate_tp, std::map<std::string, type> &tp_vars) const;

    std::map<std::string, std::pair<ndt::type, const char *>> get_dynamic_type_properties() const;
  };

  template <>
  struct id_of<typevar_type> : std::integral_constant<type_id_t, typevar_id> {};

} // namespace dynd::ndt
} // namespace dynd
