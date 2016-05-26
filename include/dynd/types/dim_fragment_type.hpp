//
// Copyright (C) 2011-16 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#pragma once

#include <string>
#include <vector>

#include <dynd/shortvector.hpp>
#include <dynd/types/base_dim_type.hpp>

namespace dynd {

/**
 * Names for the values in the tagged dims
 * of the dim_fragment_type
 */
enum {
  dim_fragment_var = -1,
  dim_fragment_fixed_sym = -2
  // values >= 0 mean fixed[N]
};

namespace ndt {

  class DYNDT_API dim_fragment_type : public base_dim_type {
    dimvector m_tagged_dims;

  public:
    dim_fragment_type(type_id_t id, intptr_t ndim, const intptr_t *tagged_dims)
        : base_dim_type(id, make_type<void>(), 0, 1, 0, type_flag_symbolic, false), m_tagged_dims(ndim, tagged_dims) {
      this->m_ndim = static_cast<uint8_t>(ndim);
    }

    dim_fragment_type(type_id_t new_id, intptr_t ndim, const type &tp);

    dim_fragment_type(type_id_t new_id) : dim_fragment_type(new_id, 0, nullptr) {}

    /**
     * The tagged_dims should be interpreted as an array of
     * size get_ndim() containing:
     *   -1 : var
     *   -2 : strided
     *   N >= 0 : fixed[N]
     */
    inline const intptr_t *get_tagged_dims() const { return m_tagged_dims.get(); }

    /**
     * Broadcasts this dim_fragment with some dimensions of
     * another type, producing another dim_fragment, or
     * a null type if there is a broadcasting error.
     */
    type broadcast_with_type(intptr_t ndim, const type &tp) const;

    /**
     * Applies the dim_fragment to the provided dtype.
     */
    type apply_to_dtype(const type &dtp) const;

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

    virtual type with_element_type(const type &element_tp) const;
  }; // class dim_fragment_type

  template <>
  struct id_of<dim_fragment_type> : std::integral_constant<type_id_t, dim_fragment_id> {};

  /** Makes a dim fragment out of the tagged dims provided */
  inline type make_dim_fragment(intptr_t ndim, const intptr_t *tagged_dims) {
    if (ndim > 0) {
      return make_type<dim_fragment_type>(ndim, tagged_dims);
    } else {
      return make_type<dim_fragment_type>();
    }
  }

  /** Make a dim fragment from the provided type */
  inline type make_dim_fragment(intptr_t ndim, const type &tp) {
    if (ndim > 0) {
      return make_type<dim_fragment_type>(ndim, tp);
    } else {
      return make_type<dim_fragment_type>();
    }
  }

} // namespace dynd::ndt
} // namespace dynd
