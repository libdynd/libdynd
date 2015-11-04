//
// Copyright (C) 2011-15 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#pragma once

#include <dynd/type.hpp>
#include <dynd/types/base_dim_type.hpp>
#include <dynd/typed_data_assign.hpp>
#include <dynd/types/view_type.hpp>
#include <dynd/types/string_type.hpp>
#include <dynd/types/type_type.hpp>
#include <dynd/array.hpp>

namespace dynd {
namespace nd {

  template <typename T, int N>
  class strided_vals;

} // namespace dynd::nd

namespace ndt {

  class DYND_API fixed_dim_kind_type : public base_dim_type {
  public:
    fixed_dim_kind_type(const type &element_tp);

    virtual ~fixed_dim_kind_type();

    size_t get_default_data_size() const;

    void print_data(std::ostream &o, const char *arrmeta,
                    const char *data) const;

    void print_type(std::ostream &o) const;

    bool is_expression() const;
    bool is_unique_data_owner(const char *arrmeta) const;
    void transform_child_types(type_transform_fn_t transform_fn,
                               intptr_t arrmeta_offset, void *extra,
                               type &out_transformed_tp,
                               bool &out_was_transformed) const;
    type get_canonical_type() const;

    type at_single(intptr_t i0, const char **inout_arrmeta,
                   const char **inout_data) const;

    type get_type_at_dimension(char **inout_arrmeta, intptr_t i,
                               intptr_t total_ndim = 0) const;

    intptr_t get_dim_size(const char *arrmeta, const char *data) const;
    void get_shape(intptr_t ndim, intptr_t i, intptr_t *out_shape,
                   const char *arrmeta, const char *data) const;

    bool is_lossless_assignment(const type &dst_tp, const type &src_tp) const;

    bool operator==(const base_type &rhs) const;

    void arrmeta_default_construct(char *arrmeta, bool blockref_alloc) const;
    void arrmeta_copy_construct(char *dst_arrmeta, const char *src_arrmeta,
                                const intrusive_ptr<memory_block_data> &embedded_reference) const;
    void arrmeta_reset_buffers(char *arrmeta) const;
    void arrmeta_finalize_buffers(char *arrmeta) const;
    void arrmeta_destruct(char *arrmeta) const;
    void arrmeta_debug_print(const char *arrmeta, std::ostream &o,
                             const std::string &indent) const;
    size_t
    arrmeta_copy_construct_onedim(char *dst_arrmeta, const char *src_arrmeta,
                                  const intrusive_ptr<memory_block_data> &embedded_reference) const;

    void data_destruct(const char *arrmeta, char *data) const;
    void data_destruct_strided(const char *arrmeta, char *data, intptr_t stride,
                               size_t count) const;

    bool match(const char *arrmeta, const type &candidate_tp,
               const char *candidate_arrmeta,
               std::map<std::string, type> &tp_vars) const;

    void get_dynamic_type_properties(
        const std::pair<std::string, nd::callable> **out_properties,
        size_t *out_count) const;
    void get_dynamic_array_properties(
        const std::pair<std::string, gfunc::callable> **out_properties,
        size_t *out_count) const;
    void get_dynamic_array_functions(
        const std::pair<std::string, gfunc::callable> **out_functions,
        size_t *out_count) const;

    virtual type with_element_type(const type &element_tp) const;

    static type make(const type &element_tp)
    {
      return type(new fixed_dim_kind_type(element_tp), false);
    }

    static type make(const type &element_tp, intptr_t ndim)
    {
      if (ndim > 0) {
        type result = make(element_tp);
        for (intptr_t i = 1; i < ndim; ++i) {
          result = make(result);
        }
        return result;
      } else {
        return element_tp;
      }
    }
  };

  DYND_API type make_fixed_dim_kind(const type &element_tp);

  inline type make_fixed_dim_kind(const type &uniform_tp, intptr_t ndim)
  {
    return fixed_dim_kind_type::make(uniform_tp, ndim);
  }

  template <typename T>
  struct type::equivalent<T[]> {
    static type make() { return fixed_dim_kind_type::make(type::make<T>()); }
  };

  // Need to handle const properly
  template <typename T>
  struct type::equivalent<const T[]> {
    static type make() { return type::make<T[]>(); }
  };

  // Produces type "Fixed ** <N> * <T>"
  template <typename T, int N>
  struct type::equivalent<nd::strided_vals<T, N>> {
    static type make() { return fixed_dim_kind_type::make(type::make<T>(), N); }
  };

} // namespace dynd::ndt
} // namespace dynd
