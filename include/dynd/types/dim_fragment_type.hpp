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

  class DYND_API dim_fragment_type : public base_dim_type {
    dimvector m_tagged_dims;

  public:
    dim_fragment_type(intptr_t ndim, const intptr_t *tagged_dims);
    dim_fragment_type(intptr_t ndim, const type &tp);

    virtual ~dim_fragment_type() {}

    /**
     * The tagged_dims should be interpreted as an array of
     * size get_ndim() containing:
     *   -1 : var
     *   -2 : strided
     *   N >= 0 : fixed[N]
     */
    inline const intptr_t *get_tagged_dims() const
    {
      return m_tagged_dims.get();
    }

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

    void print_data(std::ostream &o, const char *arrmeta,
                    const char *data) const;

    void print_type(std::ostream &o) const;

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

    intptr_t get_dim_size(const char *arrmeta, const char *data) const;

    bool is_lossless_assignment(const type &dst_tp, const type &src_tp) const;

    bool operator==(const base_type &rhs) const;

    void arrmeta_default_construct(char *arrmeta, bool blockref_alloc) const;
    void arrmeta_copy_construct(char *dst_arrmeta, const char *src_arrmeta,
                                const intrusive_ptr<memory_block_data> &embedded_reference) const;
    size_t
    arrmeta_copy_construct_onedim(char *dst_arrmeta, const char *src_arrmeta,
                                  const intrusive_ptr<memory_block_data> &embedded_reference) const;
    void arrmeta_destruct(char *arrmeta) const;

    virtual type with_element_type(const type &element_tp) const;
  }; // class dim_fragment_type

  /** Makes an empty dim fragment */
  DYND_API const type &make_dim_fragment();

  /** Makes a dim fragment out of the tagged dims provided */
  inline type make_dim_fragment(intptr_t ndim, const intptr_t *tagged_dims)
  {
    if (ndim > 0) {
      return type(new dim_fragment_type(ndim, tagged_dims), false);
    } else {
      return make_dim_fragment();
    }
  }

  /** Make a dim fragment from the provided type */
  inline type make_dim_fragment(intptr_t ndim, const type &tp)
  {
    if (ndim > 0) {
      return type(new dim_fragment_type(ndim, tp), false);
    } else {
      return make_dim_fragment();
    }
  }

} // namespace dynd::ndt
} // namespace dynd
