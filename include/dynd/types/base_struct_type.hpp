//
// Copyright (C) 2011-13 Mark Wiebe, DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#ifndef _DYND__BASE_STRUCT_TYPE_HPP_
#define _DYND__BASE_STRUCT_TYPE_HPP_

#include <dynd/types/base_type.hpp>

namespace dynd {


/**
 * Base class for all struct types. If a type
 * has kind struct_kind, it must be a subclass of
 * base_struct_type.
 */
class base_struct_type : public base_type {
protected:
    size_t m_field_count;
public:
    inline base_struct_type(type_id_t type_id, size_t data_size,
                    size_t alignment, size_t field_count, flags_type flags, size_t metadata_size)
        : base_type(type_id, struct_kind, data_size, alignment, flags, metadata_size, 0),
            m_field_count(field_count)
    {}

    virtual ~base_struct_type();

    /** The number of fields in the struct. This is the size of the other arrays. */
    inline size_t get_field_count() const {
        return m_field_count;
    }
    /** The array of the field types */
    virtual const ndt::type *get_field_types() const = 0;
    /** The array of the field names */
    virtual const std::string *get_field_names() const = 0;
    /** The array of the field data offsets */
    virtual const size_t *get_data_offsets(const char *metadata) const = 0;
    /** The array of the field metadata offsets */
    virtual const size_t *get_metadata_offsets() const = 0;
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

    void get_shape(size_t ndim, size_t i, intptr_t *out_shape,
                    const char *metadata, const char *data) const;

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

    void data_destruct(const char *metadata, char *data) const;
    void data_destruct_strided(const char *metadata, char *data,
                    intptr_t stride, size_t count) const;
};


} // namespace dynd

#endif // _DYND__BASE_STRUCT_TYPE_HPP_
