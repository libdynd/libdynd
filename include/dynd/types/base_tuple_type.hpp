//
// Copyright (C) 2011-15 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#pragma once

#include <dynd/types/base_type.hpp>
#include <dynd/types/fixed_dim_type.hpp>
#include <dynd/func/callable.hpp>
#include <dynd/array.hpp>

namespace dynd {

/**
 * Base class for all tuple and struct types. If a type has kind tuple_kind or
 * struct_kind, it must be a subclass of base_tuple_type.
 */
class base_tuple_type : public base_type {
protected:
  /**
   * The number of values in m_field_types and m_arrmeta_offsets.
   */
  intptr_t m_field_count;
  /**
   * Immutable contiguous array of field types. Always has type "N * type".
   */
  nd::array m_field_types;
  /**
   * Immutable contiguous array of arrmeta offsets. Always has type "N * intptr".
   */
  nd::array m_arrmeta_offsets;
  /**
   * If true, the tuple is variadic, which means it is symbolic, and matches
   * against the beginning of a concrete tuple.
   */
  bool m_variadic;

  virtual uintptr_t *get_arrmeta_data_offsets(char *DYND_UNUSED(arrmeta)) const
  {
    return NULL;
  }

public:
  base_tuple_type(type_id_t type_id, const nd::array &field_types,
                  flags_type flags, bool layout_in_arrmeta, bool variadic);

  virtual ~base_tuple_type();

  /** The number of fields in the struct. This is the size of the other arrays.
   */
  intptr_t get_field_count() const { return m_field_count; }
  /** The array of the field types */
  const nd::array &get_field_types() const { return m_field_types; }
  const ndt::type *get_field_types_raw() const
  {
    return reinterpret_cast<const ndt::type *>(
        m_field_types.get_readonly_originptr());
  }
  /** The array of the field data offsets */
  virtual const uintptr_t *get_data_offsets(const char *arrmeta) const = 0;
  /** The array of the field arrmeta offsets */
  const nd::array &get_arrmeta_offsets() const { return m_arrmeta_offsets; }
  const uintptr_t *get_arrmeta_offsets_raw() const
  {
    return reinterpret_cast<const uintptr_t *>(
        m_arrmeta_offsets.get_readonly_originptr());
  }

  const ndt::type &get_field_type(intptr_t i) const
  {
    return unchecked_fixed_dim_get<ndt::type>(m_field_types, i);
  }
  const uintptr_t &get_arrmeta_offset(intptr_t i) const
  {
    return unchecked_fixed_dim_get<uintptr_t>(m_arrmeta_offsets, i);
  }

  bool is_variadic() const { return m_variadic; }

  void print_data(std::ostream &o, const char *arrmeta, const char *data) const;
  bool is_expression() const;
  bool is_unique_data_owner(const char *arrmeta) const;

  size_t get_default_data_size() const;

  void get_shape(intptr_t ndim, intptr_t i, intptr_t *out_shape,
                 const char *arrmeta, const char *data) const;

  ndt::type apply_linear_index(intptr_t nindices, const irange *indices,
                               size_t current_i, const ndt::type &root_tp,
                               bool leading_dimension) const;
  intptr_t apply_linear_index(intptr_t nindices, const irange *indices,
                              const char *arrmeta, const ndt::type &result_tp,
                              char *out_arrmeta,
                              memory_block_data *embedded_reference,
                              size_t current_i, const ndt::type &root_tp,
                              bool leading_dimension, char **inout_data,
                              memory_block_data **inout_dataref) const;

  void arrmeta_default_construct(char *arrmeta, bool blockref_alloc) const;
  void arrmeta_copy_construct(char *dst_arrmeta, const char *src_arrmeta,
                              memory_block_data *embedded_reference) const;
  void arrmeta_reset_buffers(char *arrmeta) const;
  void arrmeta_finalize_buffers(char *arrmeta) const;
  void arrmeta_destruct(char *arrmeta) const;

  void data_destruct(const char *arrmeta, char *data) const;
  void data_destruct_strided(const char *arrmeta, char *data, intptr_t stride,
                             size_t count) const;

  void foreach_leading(const char *arrmeta, char *data, foreach_fn_t callback,
                       void *callback_data) const;

  virtual bool matches(const char *arrmeta, const ndt::type &other,
                       std::map<nd::string, ndt::type> &tp_vars) const;

  /**
   * Fills in the array of default data offsets based on the data sizes
   * and alignments of the types.
   */
  static void fill_default_data_offsets(intptr_t n, const ndt::type *tps,
                                        uintptr_t *out_data_offsets)
  {
    size_t offs = 0;
    if (n > 0) {
      out_data_offsets[0] = 0;
      for (intptr_t i = 1; i < n; ++i) {
        const ndt::type &tp = tps[i - 1];
        if (!tp.is_builtin()) {
          offs += tp.extended()->get_default_data_size();
        }
        else {
          offs += tp.get_data_size();
        }
        offs = inc_to_alignment(offs, tps[i].get_data_alignment());
        out_data_offsets[i] = offs;
      }
    }
  }
};

} // namespace dynd
