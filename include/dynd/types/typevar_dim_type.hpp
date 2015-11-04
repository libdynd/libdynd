//
// Copyright (C) 2011-15 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#pragma once

#include <vector>
#include <string>

#include <dynd/array.hpp>
#include <dynd/string.hpp>
#include <dynd/types/base_dim_type.hpp>

namespace dynd {
namespace ndt {

  class DYND_API typevar_dim_type : public base_dim_type {
    std::string m_name;

  public:
    typevar_dim_type(const std::string &name, const type &element_type);

    virtual ~typevar_dim_type() {}

    const std::string &get_name() const { return m_name; }

    void print_data(std::ostream &o, const char *arrmeta,
                    const char *data) const;

    void print_type(std::ostream &o) const;

    intptr_t get_dim_size(const char *arrmeta, const char *data) const;

    void get_vars(std::unordered_set<std::string> &vars) const;

    bool is_lossless_assignment(const type &dst_tp, const type &src_tp) const;

    bool operator==(const base_type &rhs) const;

    type get_type_at_dimension(char **inout_arrmeta, intptr_t i,
                               intptr_t total_ndim = 0) const;

    void arrmeta_default_construct(char *arrmeta, bool blockref_alloc) const;
    void arrmeta_copy_construct(char *dst_arrmeta, const char *src_arrmeta,
                                const intrusive_ptr<memory_block_data> &embedded_reference) const;
    size_t
    arrmeta_copy_construct_onedim(char *dst_arrmeta, const char *src_arrmeta,
                                  const intrusive_ptr<memory_block_data> &embedded_reference) const;
    void arrmeta_destruct(char *arrmeta) const;

    bool match(const char *arrmeta, const type &candidate_tp,
               const char *candidate_arrmeta,
               std::map<std::string, type> &tp_vars) const;

    void get_dynamic_type_properties(
        const std::pair<std::string, nd::callable> **out_properties,
        size_t *out_count) const;

    virtual type with_element_type(const type &element_tp) const;

    /** Makes a typevar type with the specified name and element type */
    static type make(const std::string &name, const type &element_type)
    {
      return type(new typevar_dim_type(name, element_type), false);
    }

    static type make(const std::string &name, const type &element_tp,
                     intptr_t ndim)
    {
      type result = element_tp;
      for (intptr_t i = 0; i < ndim; ++i) {
        result = make(name, result);
      }

      return result;
    }
  };

} // namespace dynd::ndt
} // namespace dynd
