//
// Copyright (C) 2011-15 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#pragma once

#include <string>
#include <vector>

#include <dynd/types/base_type.hpp>

namespace dynd {
namespace ndt {

  class DYNDT_API typevar_type : public base_type {
    std::string m_name;

  public:
    typevar_type(const std::string &name);

    const std::string &get_name() const { return m_name; }

    void get_vars(std::unordered_set<std::string> &vars) const;

    void print_data(std::ostream &o, const char *arrmeta, const char *data) const;

    void print_type(std::ostream &o) const;

    type apply_linear_index(intptr_t nindices, const irange *indices, size_t current_i, const type &root_tp,
                            bool leading_dimension) const;
    intptr_t apply_linear_index(intptr_t nindices, const irange *indices, const char *arrmeta, const type &result_tp,
                                char *out_arrmeta, const intrusive_ptr<memory_block_data> &embedded_reference,
                                size_t current_i, const type &root_tp, bool leading_dimension, char **inout_data,
                                intrusive_ptr<memory_block_data> &inout_dataref) const;

    bool is_lossless_assignment(const type &dst_tp, const type &src_tp) const;

    bool operator==(const base_type &rhs) const;

    void arrmeta_default_construct(char *arrmeta, bool blockref_alloc) const;
    void arrmeta_copy_construct(char *dst_arrmeta, const char *src_arrmeta,
                                const intrusive_ptr<memory_block_data> &embedded_reference) const;
    void arrmeta_destruct(char *arrmeta) const;

    bool match(const type &candidate_tp, std::map<std::string, type> &tp_vars) const;

    std::map<std::string, std::pair<ndt::type, const char *>> get_dynamic_type_properties() const;

    /** Makes a typevar type with the specified types */
    static type make(const std::string &name) { return type(new typevar_type(name), false); }
  }; // class typevar_type

  std::vector<ndt::type> make_typevar_range(const char *name, intptr_t count);

  /**
   * Checks if the provided string range is a valid typevar name.
   */
  bool is_valid_typevar_name(const char *begin, const char *end);

} // namespace dynd::ndt
} // namespace dynd
