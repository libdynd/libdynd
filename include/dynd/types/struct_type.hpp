//
// Copyright (C) 2011-16 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#pragma once

#include <string>
#include <vector>

#include <dynd/type.hpp>
#include <dynd/types/pointer_type.hpp>
#include <dynd/types/string_type.hpp>
#include <dynd/types/tuple_type.hpp>
#include <dynd/types/type_type.hpp>

namespace dynd {
namespace ndt {
  namespace detail {

    inline std::vector<std::string> names_from_fields(const std::vector<std::pair<ndt::type, std::string>> &fields) {
      std::vector<std::string> field_names;
      for (size_t i = 0; i < fields.size(); ++i) {
        field_names.push_back(fields[i].second);
      }

      return field_names;
    }

    inline std::vector<ndt::type> types_from_fields(const std::vector<std::pair<ndt::type, std::string>> &fields) {
      std::vector<ndt::type> field_types;
      for (size_t i = 0; i < fields.size(); ++i) {
        field_types.push_back(fields[i].first);
      }

      return field_types;
    }
  } // namespace dynd::ndt::detail

  class DYNDT_API struct_type : public base_type {
  protected:
    intptr_t m_field_count;

    const std::vector<std::string> m_field_names;
    std::vector<std::pair<type, std::string>> m_field_tp;
    std::vector<type> m_field_types;
    std::vector<uintptr_t> m_arrmeta_offsets;

    bool m_variadic;

  public:
    struct_type(type_id_t id, const std::vector<std::string> &field_names, const std::vector<type> &field_types,
                bool variadic = false)
        : base_type(id, make_type<scalar_kind_type>(), 0, 1, type_flag_indexable | (variadic ? type_flag_symbolic : 0),
                    0, 0, 0),
          m_field_count(field_types.size()), m_field_names(field_names), m_field_types(field_names.size()),
          m_arrmeta_offsets(field_names.size()), m_variadic(variadic) {
      size_t arrmeta_offset = get_field_count() * sizeof(size_t);

      this->m_data_alignment = 1;

      for (intptr_t i = 0; i < m_field_count; ++i) {
        m_field_types[i] = field_types[i];
      }

      for (intptr_t i = 0; i != m_field_count; ++i) {
        const type &ft = get_field_type(i);
        size_t field_alignment = ft.get_data_alignment();
        // Accumulate the biggest field alignment as the type alignment
        if (field_alignment > this->m_data_alignment) {
          this->m_data_alignment = (uint8_t)field_alignment;
        }
        // Inherit any operand flags from the fields
        this->flags |= (ft.get_flags() & type_flags_operand_inherited);
        // Calculate the arrmeta offsets
        m_arrmeta_offsets[i] = arrmeta_offset;
        arrmeta_offset += ft.get_arrmeta_size();
      }

      this->m_metadata_size = arrmeta_offset;

      // Make sure that the number of names matches
      uintptr_t name_count = field_names.size();
      if (name_count != (uintptr_t)m_field_count) {
        std::stringstream ss;
        ss << "dynd struct type requires that the number of names, " << name_count << " matches the number of types, "
           << m_field_count;
        throw std::invalid_argument(ss.str());
      }

      for (intptr_t i = 0; i < m_field_count; ++i) {
        m_field_tp.emplace_back(field_types[i], field_names[i]);
      }
    }

    struct_type(type_id_t id, const std::vector<std::pair<type, std::string>> &fields, bool variadic = false)
        : struct_type(id, detail::names_from_fields(fields), detail::types_from_fields(fields), variadic) {}

    struct_type(type_id_t id, bool variadic = false) : struct_type(id, {}, variadic) {}

    /** The array of the field names */
    const std::vector<std::string> &get_field_names() const { return m_field_names; }
    const std::string &get_field_name(intptr_t i) const { return m_field_names[i]; }

    intptr_t get_field_count() const { return m_field_count; }
    const ndt::type get_type() const { return ndt::type_for(m_field_types); }
    const std::vector<type> &get_field_types() const { return m_field_types; }
    const type *get_field_types_raw() const { return m_field_types.data(); }
    const std::vector<uintptr_t> &get_arrmeta_offsets() const { return m_arrmeta_offsets; }
    const uintptr_t *get_arrmeta_offsets_raw() const { return m_arrmeta_offsets.data(); }

    uintptr_t get_arrmeta_offset(intptr_t i) const { return m_arrmeta_offsets[i]; }

    size_t get_default_data_size() const;

    /**
     * Gets the field index for the given name. Returns -1 if
     * the struct doesn't have a field of the given name.
     *
     * \param field_name  The name of the field.
     *
     * \returns  The field index, or -1 if there is no field
     *           of the given name.
     */
    intptr_t get_field_index(const std::string &field_name) const;

    /**
     * Gets the field type for the given name. Raises std::invalid_argument if
     * the struct doesn't have a field of the given name.
     *
     * \param field_name  The name of the field.
     *
     * \returns  The field type.
     */
    const type &get_field_type(intptr_t i) const;

    bool is_expression() const;
    bool is_unique_data_owner(const char *arrmeta) const;

    bool is_variadic() const { return m_variadic; }

    const std::vector<std::pair<type, std::string>> &get_named_field_types() const { return m_field_tp; }

    void print_type(std::ostream &o) const;

    void transform_child_types(type_transform_fn_t transform_fn, intptr_t arrmeta_offset, void *extra,
                               type &out_transformed_tp, bool &out_was_transformed) const;
    type get_canonical_type() const;

    type at_single(intptr_t i0, const char **inout_arrmeta, const char **inout_data) const;

    bool is_lossless_assignment(const type &dst_tp, const type &src_tp) const;

    bool operator==(const base_type &rhs) const;

    void arrmeta_default_construct(char *arrmeta, bool blockref_alloc) const;
    void arrmeta_copy_construct(char *dst_arrmeta, const char *src_arrmeta,
                                const nd::memory_block &embedded_reference) const;
    void arrmeta_reset_buffers(char *arrmeta) const;
    void arrmeta_finalize_buffers(char *arrmeta) const;
    void arrmeta_destruct(char *arrmeta) const;

    void data_destruct(const char *arrmeta, char *data) const;
    void data_destruct_strided(const char *arrmeta, char *data, intptr_t stride, size_t count) const;

    void arrmeta_debug_print(const char *arrmeta, std::ostream &o, const std::string &indent) const;

    type apply_linear_index(intptr_t nindices, const irange *indices, size_t current_i, const type &root_tp,
                            bool leading_dimension) const;
    intptr_t apply_linear_index(intptr_t nindices, const irange *indices, const char *arrmeta, const type &result_tp,
                                char *out_arrmeta, const nd::memory_block &embedded_reference, size_t current_i,
                                const type &root_tp, bool leading_dimension, char **inout_data,
                                nd::memory_block &inout_dataref) const;

    std::map<std::string, std::pair<ndt::type, const char *>> get_dynamic_type_properties() const;

    virtual bool match(const type &candidate_tp, std::map<std::string, type> &tp_vars) const;

    /**
     * Fills in the array of default data offsets based on the data sizes
     * and alignments of the types.
     */
    static void fill_default_data_offsets(intptr_t nfields, const type *field_tps, uintptr_t *out_data_offsets) {
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
  };

  template <>
  struct id_of<struct_type> : std::integral_constant<type_id_t, struct_id> {};

} // namespace dynd::ndt
} // namespace dynd
