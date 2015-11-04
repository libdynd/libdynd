//
// Copyright (C) 2011-15 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#pragma once

#include <dynd/types/base_type.hpp>
#include <dynd/types/base_tuple_type.hpp>
#include <dynd/types/string_type.hpp>

namespace dynd {
namespace ndt {

  /**
   * Base class for all struct types. If a type
   * has kind struct_kind, it must be a subclass of
   * base_struct_type.
   *
   * This class uses the base_tuple_type for the definition
   * of the field types, and adds field names to that.
   */
  class DYND_API base_struct_type : public base_tuple_type {
  protected:
    nd::array m_field_names;

  public:
    base_struct_type(type_id_t type_id, const nd::array &field_names, const nd::array &field_types, flags_type flags,
                     bool layout_in_arrmeta, bool variadic);

    virtual ~base_struct_type();

    /** The array of the field names */
    const nd::array &get_field_names() const
    {
      return m_field_names;
    }
    const string &get_field_name_raw(intptr_t i) const
    {
      return unchecked_fixed_dim_get<string>(m_field_names, i);
    }
    const std::string get_field_name(intptr_t i) const
    {
      const string &std(get_field_name_raw(i));
      return std::string(std.begin(), std.end());
    }

    /**
     * Gets the field index for the given name. Returns -1 if
     * the struct doesn't have a field of the given name.
     *
     * \param field_name  The name of the field.
     *
     * \returns  The field index, or -1 if there is not field
     *           of the given name.
     */
    inline intptr_t get_field_index(const std::string &field_name) const
    {
      return get_field_index(field_name.data(), field_name.data() + field_name.size());
    }
    intptr_t get_field_index(const char *field_name_begin, const char *field_name_end) const;

    type apply_linear_index(intptr_t nindices, const irange *indices, size_t current_i, const type &root_tp,
                            bool leading_dimension) const;
    intptr_t apply_linear_index(intptr_t nindices, const irange *indices, const char *arrmeta, const type &result_tp,
                                char *out_arrmeta, const intrusive_ptr<memory_block_data> &embedded_reference, size_t current_i,
                                const type &root_tp, bool leading_dimension, char **inout_data,
                                intrusive_ptr<memory_block_data> &inout_dataref) const;

    virtual bool match(const char *arrmeta, const type &candidate_tp, const char *candidate_arrmeta,
                       std::map<std::string, type> &tp_vars) const;

    size_t get_elwise_property_index(const std::string &property_name) const;
    type get_elwise_property_type(size_t elwise_property_index, bool &out_readable, bool &out_writable) const;
    size_t make_elwise_property_getter_kernel(void *ckb, intptr_t ckb_offset, const char *dst_arrmeta,
                                              const char *src_arrmeta, size_t src_elwise_property_index,
                                              kernel_request_t kernreq, const eval::eval_context *ectx) const;
    size_t make_elwise_property_setter_kernel(void *ckb, intptr_t ckb_offset, const char *dst_arrmeta,
                                              size_t dst_elwise_property_index, const char *src_arrmeta,
                                              kernel_request_t kernreq, const eval::eval_context *ectx) const;
  };

} // namespace dynd::ndt
} // namespace dynd
