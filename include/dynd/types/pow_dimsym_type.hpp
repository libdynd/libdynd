//
// Copyright (C) 2011-16 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#pragma once

#include <string>
#include <vector>

#include <dynd/types/base_dim_type.hpp>

namespace dynd {
namespace ndt {

  class DYNDT_API pow_dimsym_type : public base_dim_type {
    type m_base_tp;
    std::string m_exponent;

  public:
    pow_dimsym_type(const type &base_tp, const std::string &exponent, const type &element_type);

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

} // namespace dynd::ndt
} // namespace dynd
