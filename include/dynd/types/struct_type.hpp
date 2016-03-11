//
// Copyright (C) 2011-15 DyND Developers
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

  class DYNDT_API struct_type : public tuple_type {
    const std::vector<std::string> m_field_names;

  protected:
    uintptr_t *get_arrmeta_data_offsets(char *arrmeta) const { return reinterpret_cast<uintptr_t *>(arrmeta); }

  public:
    struct_type(const std::vector<std::string> &field_names, const std::vector<type> &field_types,
                bool variadic = false);

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
    const type &get_field_type(const std::string &field_name) const;
    const type &get_field_type(intptr_t i) const;

    inline const uintptr_t *get_data_offsets(const char *arrmeta) const
    {
      return reinterpret_cast<const uintptr_t *>(arrmeta);
    }

    uintptr_t get_data_offset(const char *arrmeta, const std::string &field_name) const;

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
                                char *out_arrmeta, const intrusive_ptr<memory_block_data> &embedded_reference,
                                size_t current_i, const type &root_tp, bool leading_dimension, char **inout_data,
                                intrusive_ptr<memory_block_data> &inout_dataref) const;

    std::map<std::string, std::pair<ndt::type, const char *>> get_dynamic_type_properties() const;

    virtual bool match(const type &candidate_tp, std::map<std::string, type> &tp_vars) const;

    /** Makes a struct type with the specified fields */
    static type make(const std::vector<std::string> &field_names, const std::vector<type> &field_types,
                     bool variadic = false)
    {
      return type(new struct_type(field_names, field_types, variadic), false);
    }

    /** Makes an empty struct type */
    static type make(bool variadic = false) { return make(std::vector<std::string>(), std::vector<type>(), variadic); }
  };

} // namespace dynd::ndt
} // namespace dynd
