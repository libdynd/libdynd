//
// Copyright (C) 2011-15 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#pragma once

#include <vector>
#include <string>

#include <dynd/array.hpp>
#include <dynd/string.hpp>

namespace dynd {
namespace ndt {

  class DYND_API typevar_constructed_type : public base_type {
    std::string m_name;
    type m_arg;

  public:
    typevar_constructed_type(const std::string &name, const type &arg);

    virtual ~typevar_constructed_type() {}

    std::string get_name() const { return m_name; }

    type get_arg() const { return m_arg; }

    void get_vars(std::unordered_set<std::string> &vars) const;

    void print_data(std::ostream &o, const char *arrmeta,
                    const char *data) const;

    void print_type(std::ostream &o) const;

    intptr_t get_dim_size(const char *arrmeta, const char *data) const;

    type apply_linear_index(intptr_t nindices, const irange *indices,
                            size_t current_i, const type &root_tp,
                            bool leading_dimension) const;
    intptr_t apply_linear_index(intptr_t nindices, const irange *indices,
                                const char *arrmeta, const type &result_tp,
                                char *out_arrmeta,
                                const intrusive_ptr<memory_block_data> &embedded_reference,
                                size_t current_i, const type &root_tp,
                                bool leading_dimension, char **inout_data,
                                intrusive_ptr<memory_block_data> &inout_dataref) const;

    bool is_lossless_assignment(const type &dst_tp, const type &src_tp) const;

    bool operator==(const base_type &rhs) const;

    void arrmeta_default_construct(char *arrmeta, bool blockref_alloc) const;
    void arrmeta_copy_construct(char *dst_arrmeta, const char *src_arrmeta,
                                const intrusive_ptr<memory_block_data> &embedded_reference) const;
    void arrmeta_destruct(char *arrmeta) const;

    bool match(const char *arrmeta, const type &candidate_tp,
               const char *candidate_arrmeta,
               std::map<std::string, type> &tp_vars) const;

    /*
        void get_dynamic_type_properties(
            const std::pair<std::string, nd::arrfunc> **out_properties,
            size_t *out_count) const;
    */

    /** Makes a typevar_constructed type with the specified types */
    static type make(const std::string &name, const type &arg)
    {
      return type(new typevar_constructed_type(name, arg), false);
    }
  };

} // namespace dynd::ndt
} // namespace dynd
