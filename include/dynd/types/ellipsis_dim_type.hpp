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

  type make_ellipsis_dim(const std::string &name, const type &element_type);

  class DYND_API ellipsis_dim_type : public base_dim_type {
    // m_name is either NULL or an immutable array of type "string"
    std::string m_name;

  public:
    ellipsis_dim_type(const std::string &name, const type &element_type);

    virtual ~ellipsis_dim_type()
    {
    }

    const std::string &get_name() const
    {
      return m_name;
    }

    void get_vars(std::unordered_set<std::string> &vars) const;

    void print_data(std::ostream &o, const char *arrmeta,
                    const char *data) const;

    void print_type(std::ostream &o) const;

    intptr_t get_dim_size(const char *arrmeta, const char *data) const;

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

    static type make_if_not_variadic(const type &element_tp)
    {
      if (element_tp.is_variadic()) {
        return element_tp;
      }

      return make_ellipsis_dim("Dims", element_tp);
    }
  };

  /** Makes an ellipsis type with the specified name and element type */
  inline type make_ellipsis_dim(const std::string &name,
                                const type &element_type)
  {
    return type(new ellipsis_dim_type(name, element_type), false);
  }

  /** Make an unnamed ellipsis type */
  inline type make_ellipsis_dim(const type &element_type)
  {
    return type(new ellipsis_dim_type("", element_type), false);
  }

} // namespace dynd::ndt
} // namespace dynd
