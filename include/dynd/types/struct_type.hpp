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

  class DYNDT_API struct_type : public tuple_type {
  protected:
    const std::vector<std::string> m_field_names;
    std::vector<std::pair<type, std::string>> m_field_tp;

  public:
    struct_type(type_id_t id, const std::vector<std::string> &field_names, const std::vector<type> &field_types,
                bool variadic = false)
        : tuple_type(id, field_types.size(), field_types.data(), variadic, type_flag_none), m_field_names(field_names) {
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

    const std::vector<std::pair<type, std::string>> &get_named_field_types() const { return m_field_tp; }

    void print_type(std::ostream &o) const;

    void transform_child_types(type_transform_fn_t transform_fn, intptr_t arrmeta_offset, void *extra,
                               type &out_transformed_tp, bool &out_was_transformed) const;
    type get_canonical_type() const;

    type at_single(intptr_t i0, const char **inout_arrmeta, const char **inout_data) const;

    bool is_lossless_assignment(const type &dst_tp, const type &src_tp) const;

    bool operator==(const base_type &rhs) const;

    void arrmeta_debug_print(const char *arrmeta, std::ostream &o, const std::string &indent) const;

    type apply_linear_index(intptr_t nindices, const irange *indices, size_t current_i, const type &root_tp,
                            bool leading_dimension) const;
    intptr_t apply_linear_index(intptr_t nindices, const irange *indices, const char *arrmeta, const type &result_tp,
                                char *out_arrmeta, const nd::memory_block &embedded_reference, size_t current_i,
                                const type &root_tp, bool leading_dimension, char **inout_data,
                                nd::memory_block &inout_dataref) const;

    std::map<std::string, std::pair<ndt::type, const char *>> get_dynamic_type_properties() const;

    virtual bool match(const type &candidate_tp, std::map<std::string, type> &tp_vars) const;
  };

  template <>
  struct id_of<struct_type> : std::integral_constant<type_id_t, struct_id> {};

} // namespace dynd::ndt
} // namespace dynd
