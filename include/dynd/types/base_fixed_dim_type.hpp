//
// Copyright (C) 2011-15 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#pragma once

#include <dynd/types/base_dim_type.hpp>

namespace dynd {
namespace ndt {

  class DYNDT_API base_fixed_dim_type : public base_dim_type {
  public:
    using base_dim_type::base_dim_type;

    base_fixed_dim_type(const type &element_tp);

    size_t get_default_data_size() const;

    void print_data(std::ostream &o, const char *arrmeta, const char *data) const;

    void print_type(std::ostream &o) const;

    bool is_expression() const;
    bool is_unique_data_owner(const char *arrmeta) const;
    void transform_child_types(type_transform_fn_t transform_fn, intptr_t arrmeta_offset, void *extra,
                               type &out_transformed_tp, bool &out_was_transformed) const;
    type get_canonical_type() const;

    type at_single(intptr_t i0, const char **inout_arrmeta, const char **inout_data) const;

    type get_type_at_dimension(char **inout_arrmeta, intptr_t i, intptr_t total_ndim = 0) const;

    intptr_t get_dim_size(const char *arrmeta, const char *data) const;
    void get_shape(intptr_t ndim, intptr_t i, intptr_t *out_shape, const char *arrmeta, const char *data) const;

    bool is_lossless_assignment(const type &dst_tp, const type &src_tp) const;

    virtual bool is_sized() const { return false; }

    bool operator==(const base_type &rhs) const;

    void arrmeta_default_construct(char *arrmeta, bool blockref_alloc) const;
    void arrmeta_copy_construct(char *dst_arrmeta, const char *src_arrmeta,
                                const intrusive_ptr<memory_block_data> &embedded_reference) const;
    void arrmeta_reset_buffers(char *arrmeta) const;
    void arrmeta_finalize_buffers(char *arrmeta) const;
    void arrmeta_destruct(char *arrmeta) const;
    void arrmeta_debug_print(const char *arrmeta, std::ostream &o, const std::string &indent) const;
    size_t arrmeta_copy_construct_onedim(char *dst_arrmeta, const char *src_arrmeta,
                                         const intrusive_ptr<memory_block_data> &embedded_reference) const;

    void data_destruct(const char *arrmeta, char *data) const;
    void data_destruct_strided(const char *arrmeta, char *data, intptr_t stride, size_t count) const;

    bool match(const type &candidate_tp, std::map<std::string, type> &tp_vars) const;

    std::map<std::string, std::pair<ndt::type, const char *>> get_dynamic_type_properties() const;

    virtual type with_element_type(const type &element_tp) const;

    static type make(const type &element_tp) { return type(new base_fixed_dim_type(element_tp), false); }

    static type make(const type &element_tp, intptr_t ndim)
    {
      if (ndim > 0) {
        type result = make(element_tp);
        for (intptr_t i = 1; i < ndim; ++i) {
          result = make(result);
        }
        return result;
      }
      else {
        return element_tp;
      }
    }
  };

  DYNDT_API type make_fixed_dim_kind(const type &element_tp);

  inline type make_fixed_dim_kind(const type &uniform_tp, intptr_t ndim)
  {
    return base_fixed_dim_type::make(uniform_tp, ndim);
  }

  template <typename T>
  struct traits<T[]> {
    static type equivalent() { return base_fixed_dim_type::make(make_type<T>()); }
  };

  // Need to handle const properly
  template <typename T>
  struct traits<const T[]> {
    static type equivalent() { return make_type<T[]>(); }
  };

} // namespace dynd::ndt
} // namespace dynd
