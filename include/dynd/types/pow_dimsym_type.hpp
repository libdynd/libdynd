//
// Copyright (C) 2011-16 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#pragma once

#include <string>
#include <vector>

#include <dynd/types/base_dim_type.hpp>
#include <dynd/types/dim_kind_type.hpp>
#include <dynd/types/typevar_type.hpp>

namespace dynd {
namespace ndt {

  class DYNDT_API pow_dimsym_type : public base_dim_type {
    type m_base_tp;
    std::string m_exponent;

  public:
    pow_dimsym_type(type_id_t id, const type &base_tp, const std::string &exponent, const type &element_type)
        : base_dim_type(id, element_type, 0, 1, 0, type_flag_symbolic, false), m_base_tp(base_tp),
          m_exponent(exponent) {
      if (base_tp.is_scalar() || base_tp.extended<base_dim_type>()->get_element_type().get_id() != void_id) {
        std::stringstream ss;
        ss << "dynd base type for dimensional power symbolic type is not valid: " << base_tp;
        throw type_error(ss.str());
      }
      if (m_exponent.empty()) {
        throw type_error("dynd typevar name cannot be null");
      } else if (!is_valid_typevar_name(m_exponent.c_str(), m_exponent.c_str() + m_exponent.size())) {
        std::stringstream ss;
        ss << "dynd typevar name ";
        print_escaped_utf8_string(ss, m_exponent);
        ss << " is not valid, it must be alphanumeric and begin with a capital";
        throw type_error(ss.str());
      }
    }

    const type &get_base_type() const { return m_base_tp; }

    const std::string &get_exponent() const { return m_exponent; }

    void print_data(std::ostream &o, const char *arrmeta, const char *data) const;

    void print_type(std::ostream &o) const;

    type apply_linear_index(intptr_t nindices, const irange *indices, size_t current_i, const type &root_tp,
                            bool leading_dimension) const;
    intptr_t apply_linear_index(intptr_t nindices, const irange *indices, const char *arrmeta, const type &result_tp,
                                char *out_arrmeta, const nd::memory_block &embedded_reference, size_t current_i,
                                const type &root_tp, bool leading_dimension, char **inout_data,
                                nd::memory_block &inout_dataref) const;

    intptr_t get_dim_size(const char *arrmeta, const char *data) const;

    bool is_lossless_assignment(const type &dst_tp, const type &src_tp) const;

    bool operator==(const base_type &rhs) const;

    void arrmeta_default_construct(char *arrmeta, bool blockref_alloc) const;
    void arrmeta_copy_construct(char *dst_arrmeta, const char *src_arrmeta,
                                const nd::memory_block &embedded_reference) const;
    size_t arrmeta_copy_construct_onedim(char *dst_arrmeta, const char *src_arrmeta,
                                         const nd::memory_block &embedded_reference) const;
    void arrmeta_destruct(char *arrmeta) const;

    std::map<std::string, std::pair<ndt::type, const char *>> get_dynamic_type_properties() const;

    bool match(const type &candidate_tp, std::map<std::string, type> &tp_vars) const;

    virtual type with_element_type(const type &element_tp) const;
  };

  template <>
  struct id_of<pow_dimsym_type> : std::integral_constant<type_id_t, pow_dimsym_id> {};

} // namespace dynd::ndt
} // namespace dynd
