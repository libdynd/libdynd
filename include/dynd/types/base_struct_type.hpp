//
// Copyright (C) 2011-14 Mark Wiebe, DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#ifndef _DYND__BASE_STRUCT_TYPE_HPP_
#define _DYND__BASE_STRUCT_TYPE_HPP_

#include <dynd/types/base_type.hpp>
#include <dynd/types/base_tuple_type.hpp>

namespace dynd {


/**
 * Base class for all struct types. If a type
 * has kind struct_kind, it must be a subclass of
 * base_struct_type.
 *
 * This class uses the base_tuple_type for the definition
 * of the field types, and adds field names to that.
 */
class base_struct_type : public base_tuple_type {
public:
    inline base_struct_type(type_id_t type_id, size_t data_size, size_t alignment,
                            size_t field_count, flags_type flags,
                            size_t metadata_size)
      : base_tuple_type(type_id, struct_kind, data_size, alignment, field_count,
                        flags, metadata_size)
    {}

    virtual ~base_struct_type();

    /** The array of the field names */
    virtual const std::string *get_field_names() const = 0;
    /**
     * Gets the field index for the given name. Returns -1 if
     * the struct doesn't have a field of the given name.
     *
     * \param field_name  The name of the field.
     *
     * \returns  The field index, or -1 if there is not field
     *           of the given name.
     */
    virtual intptr_t get_field_index(const std::string& field_name) const = 0;
    intptr_t get_field_index(const char *field_index_begin,
                             const char *field_index_end) const;

    size_t get_elwise_property_index(const std::string& property_name) const;
    ndt::type get_elwise_property_type(size_t elwise_property_index,
                    bool& out_readable, bool& out_writable) const;
    size_t make_elwise_property_getter_kernel(
                    ckernel_builder *out, size_t offset_out,
                    const char *dst_metadata,
                    const char *src_metadata, size_t src_elwise_property_index,
                    kernel_request_t kernreq, const eval::eval_context *ectx) const;
    size_t make_elwise_property_setter_kernel(
                    ckernel_builder *out, size_t offset_out,
                    const char *dst_metadata, size_t dst_elwise_property_index,
                    const char *src_metadata,
                    kernel_request_t kernreq, const eval::eval_context *ectx) const;
};


} // namespace dynd

#endif // _DYND__BASE_STRUCT_TYPE_HPP_
