//
// Copyright (C) 2011-15 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#pragma once

#include <string>
#include <vector>

#include <dynd/type.hpp>
#include <dynd/types/fixed_dim_type.hpp>

namespace dynd {
namespace ndt {

  class DYNDT_API tuple_type : public base_type {
  protected:
    /**
     * The number of values in m_field_types and m_arrmeta_offsets.
     */
    intptr_t m_field_count;

    /**
     * Immutable vector of field types. Always has type "N * type".
     */
    const std::vector<type> m_field_types;

    /**
     * Immutable vector of arrmeta offsets. Always has type "N * uintptr".
     */
    std::vector<uintptr_t> m_arrmeta_offsets;

    /**
     * If true, the tuple is variadic, which means it is symbolic, and matches
     * against the beginning of a concrete tuple.
     */
    bool m_variadic;

    tuple_type(type_id_t type_id, const std::vector<type> &field_types, uint32_t flags, bool layout_in_arrmeta,
               bool variadic);

    uintptr_t *get_arrmeta_data_offsets(char *arrmeta) const { return reinterpret_cast<uintptr_t *>(arrmeta); }

  public:
    tuple_type(const std::vector<type> &field_types, bool variadic);

    /** The array of the field data offsets */
    inline const uintptr_t *get_data_offsets(const char *arrmeta) const
    {
      return reinterpret_cast<const uintptr_t *>(arrmeta);
    }

    /** The number of fields in the struct. This is the size of the other
     * arrays.
     */
    intptr_t get_field_count() const { return m_field_count; }
    /** The type of the tuple */
    const ndt::type get_type() const { return ndt::type_for(m_field_types); }

    /** The field types */
    const std::vector<type> &get_field_types() const { return m_field_types; }

    const type *get_field_types_raw() const { return m_field_types.data(); }
    /** The field arrmeta offsets */
    const std::vector<uintptr_t> &get_arrmeta_offsets() const { return m_arrmeta_offsets; }
    const uintptr_t *get_arrmeta_offsets_raw() const { return m_arrmeta_offsets.data(); }

    const type &get_field_type(intptr_t i) const { return m_field_types[i]; }
    uintptr_t get_arrmeta_offset(intptr_t i) const { return m_arrmeta_offsets[i]; }

    bool is_variadic() const { return m_variadic; }

    virtual void get_vars(std::unordered_set<std::string> &vars) const;

    void print_data(std::ostream &o, const char *arrmeta, const char *data) const;
    bool is_expression() const;
    bool is_unique_data_owner(const char *arrmeta) const;

    size_t get_default_data_size() const;

    void get_shape(intptr_t ndim, intptr_t i, intptr_t *out_shape, const char *arrmeta, const char *data) const;

    type apply_linear_index(intptr_t nindices, const irange *indices, size_t current_i, const type &root_tp,
                            bool leading_dimension) const;
    intptr_t apply_linear_index(intptr_t nindices, const irange *indices, const char *arrmeta, const type &result_tp,
                                char *out_arrmeta, const intrusive_ptr<memory_block_data> &embedded_reference,
                                size_t current_i, const type &root_tp, bool leading_dimension, char **inout_data,
                                intrusive_ptr<memory_block_data> &inout_dataref) const;

    void print_type(std::ostream &o) const;

    void transform_child_types(type_transform_fn_t transform_fn, intptr_t arrmeta_offset, void *extra,
                               type &out_transformed_tp, bool &out_was_transformed) const;
    type get_canonical_type() const;

    bool is_lossless_assignment(const type &dst_tp, const type &src_tp) const;

    bool operator==(const base_type &rhs) const;

    void arrmeta_default_construct(char *arrmeta, bool blockref_alloc) const;
    void arrmeta_copy_construct(char *dst_arrmeta, const char *src_arrmeta,
                                const intrusive_ptr<memory_block_data> &embedded_reference) const;
    void arrmeta_reset_buffers(char *arrmeta) const;
    void arrmeta_finalize_buffers(char *arrmeta) const;
    void arrmeta_destruct(char *arrmeta) const;

    void data_destruct(const char *arrmeta, char *data) const;
    void data_destruct_strided(const char *arrmeta, char *data, intptr_t stride, size_t count) const;

    void foreach_leading(const char *arrmeta, char *data, foreach_fn_t callback, void *callback_data) const;

    void arrmeta_debug_print(const char *arrmeta, std::ostream &o, const std::string &indent) const;

    virtual bool match(const type &candidate_tp, std::map<std::string, type> &tp_vars) const;

    std::map<std::string, std::pair<ndt::type, const char *>> get_dynamic_type_properties() const;

    /**
     * Fills in the array of default data offsets based on the data sizes
     * and alignments of the types.
     */
    static void fill_default_data_offsets(intptr_t nfields, const type *field_tps, uintptr_t *out_data_offsets)
    {
      if (nfields > 0) {
        out_data_offsets[0] = 0;
        size_t offs = 0;
        for (intptr_t i = 1; i < nfields; ++i) {
          offs += field_tps[i - 1].get_default_data_size();
          offs = inc_to_alignment(offs, field_tps[i].get_data_alignment());
          out_data_offsets[i] = offs;
        }
      }
    }

    /** Makes a tuple type with the specified types */
    static type make(const std::vector<type> &field_types, bool variadic = false)
    {
      return type(new tuple_type(field_types, variadic), false);
    }

    /** Makes an empty tuple */
    static type make(bool variadic = false) { return make(std::vector<type>(), variadic); }
  };

} // namespace dynd::ndt
} // namespace dynd
