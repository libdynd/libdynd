//
// Copyright (C) 2011-13, DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#ifndef _DYND__FIXEDSTRUCT_DTYPE_HPP_
#define _DYND__FIXEDSTRUCT_DTYPE_HPP_

#include <vector>
#include <string>

#include <dynd/dtype.hpp>
#include <dynd/dtypes/base_struct_dtype.hpp>
#include <dynd/memblock/memory_block.hpp>

namespace dynd {

class fixedstruct_dtype : public base_struct_dtype {
    std::vector<dtype> m_field_types;
    std::vector<std::string> m_field_names;
    std::vector<size_t> m_data_offsets;
    std::vector<size_t> m_metadata_offsets;
    std::vector<std::pair<std::string, gfunc::callable> > m_ndobject_properties;
    dtype_memory_management_t m_memory_management;

    void create_ndobject_properties();

    // Special constructor to break the property parameter cycle in
    // create_ndobject_properties
    fixedstruct_dtype(int, int);
public:
    fixedstruct_dtype(size_t field_count, const dtype *field_types,
                    const std::string *field_names);

    virtual ~fixedstruct_dtype();

    size_t get_default_data_size(size_t DYND_UNUSED(ndim), const intptr_t *DYND_UNUSED(shape)) const {
        return get_data_size();
    }

    const dtype *get_field_types() const {
        return &m_field_types[0];
    }

    const std::string *get_field_names() const {
        return &m_field_names[0];
    }

    const std::vector<std::string>& get_field_names_vector() const {
        return m_field_names;
    }

    intptr_t get_field_index(const std::string& field_name) const;

    const size_t *get_data_offsets(const char *DYND_UNUSED(metadata)) const {
        return &m_data_offsets[0];
    }

    const std::vector<size_t>& get_data_offsets_vector() const {
        return m_data_offsets;
    }

    const size_t *get_metadata_offsets() const {
        return &m_metadata_offsets[0];
    }

    const std::vector<size_t>& get_metadata_offsets_vector() const {
        return m_metadata_offsets;
    }

    void print_data(std::ostream& o, const char *metadata, const char *data) const;

    void print_dtype(std::ostream& o) const;

    // This is about the native storage, buffering code needs to check whether
    // the value_dtype is an object type separately.
    dtype_memory_management_t get_memory_management() const {
        return m_memory_management;
    }

    bool is_expression() const;
    bool is_unique_data_owner(const char *metadata) const;
    void transform_child_dtypes(dtype_transform_fn_t transform_fn, const void *extra,
                    dtype& out_transformed_dtype, bool& out_was_transformed) const;
    dtype get_canonical_dtype() const;

    dtype apply_linear_index(size_t nindices, const irange *indices,
                size_t current_i, const dtype& root_dt, bool leading_dimension) const;
    intptr_t apply_linear_index(size_t nindices, const irange *indices, const char *metadata,
                    const dtype& result_dtype, char *out_metadata,
                    memory_block_data *embedded_reference,
                    size_t current_i, const dtype& root_dt,
                    bool leading_dimension, char **inout_data,
                    memory_block_data **inout_dataref) const;
    dtype at_single(intptr_t i0, const char **inout_metadata, const char **inout_data) const;

    void get_shape(size_t i, intptr_t *out_shape) const;
    intptr_t get_representative_stride(const char *metadata) const;

    bool is_lossless_assignment(const dtype& dst_dt, const dtype& src_dt) const;

    bool operator==(const base_dtype& rhs) const;

    void metadata_default_construct(char *metadata, size_t ndim, const intptr_t* shape) const;
    void metadata_copy_construct(char *dst_metadata, const char *src_metadata, memory_block_data *embedded_reference) const;
    void metadata_reset_buffers(char *metadata) const;
    void metadata_finalize_buffers(char *metadata) const;
    void metadata_destruct(char *metadata) const;
    void metadata_debug_print(const char *metadata, std::ostream& o, const std::string& indent) const;

    size_t make_assignment_kernel(
                    hierarchical_kernel *out, size_t offset_out,
                    const dtype& dst_dt, const char *dst_metadata,
                    const dtype& src_dt, const char *src_metadata,
                    kernel_request_t kernreq, assign_error_mode errmode,
                    const eval::eval_context *ectx) const;

    size_t make_comparison_kernel(
                    hierarchical_kernel *out, size_t offset_out,
                    const dtype& src0_dt, const char *src0_metadata,
                    const dtype& src1_dt, const char *src1_metadata,
                    comparison_type_t comptype,
                    const eval::eval_context *ectx) const;

    void foreach_leading(char *data, const char *metadata, foreach_fn_t callback, void *callback_data) const;

    void get_dynamic_dtype_properties(const std::pair<std::string, gfunc::callable> **out_properties, size_t *out_count) const;
    void get_dynamic_ndobject_properties(const std::pair<std::string, gfunc::callable> **out_properties, size_t *out_count) const;
}; // class fixedstruct_dtype

/** Makes a struct dtype with the specified fields */
inline dtype make_fixedstruct_dtype(size_t field_count, const dtype *field_types,
                const std::string *field_names) {
    return dtype(new fixedstruct_dtype(field_count, field_types, field_names), false);
}

/** Makes a struct dtype with the specified fields */
inline dtype make_fixedstruct_dtype(const dtype& dt0, const std::string& name0)
{
    return make_fixedstruct_dtype(1, &dt0, &name0);
}

/** Makes a struct dtype with the specified fields */
inline dtype make_fixedstruct_dtype(const dtype& dt0, const std::string& name0, const dtype& dt1, const std::string& name1)
{
    dtype field_types[2];
    std::string field_names[2];
    field_types[0] = dt0;
    field_types[1] = dt1;
    field_names[0] = name0;
    field_names[1] = name1;
    return make_fixedstruct_dtype(2, field_types, field_names);
}

/** Makes a struct dtype with the specified fields */
inline dtype make_fixedstruct_dtype(const dtype& dt0, const std::string& name0, const dtype& dt1, const std::string& name1, const dtype& dt2, const std::string& name2)
{
    dtype field_types[3];
    std::string field_names[3];
    field_types[0] = dt0;
    field_types[1] = dt1;
    field_types[2] = dt2;
    field_names[0] = name0;
    field_names[1] = name1;
    field_names[2] = name2;
    return make_fixedstruct_dtype(3, field_types, field_names);
}

/** Makes a struct dtype with the specified fields */
inline dtype make_fixedstruct_dtype(const dtype& dt0, const std::string& name0,
                const dtype& dt1, const std::string& name1, const dtype& dt2, const std::string& name2,
                const dtype& dt3, const std::string& name3)
{
    dtype field_types[4];
    std::string field_names[4];
    field_types[0] = dt0;
    field_types[1] = dt1;
    field_types[2] = dt2;
    field_types[3] = dt3;
    field_names[0] = name0;
    field_names[1] = name1;
    field_names[2] = name2;
    field_names[3] = name3;
    return make_fixedstruct_dtype(4, field_types, field_names);
}

/** Makes a struct dtype with the specified fields */
inline dtype make_fixedstruct_dtype(const dtype& dt0, const std::string& name0,
                const dtype& dt1, const std::string& name1, const dtype& dt2, const std::string& name2,
                const dtype& dt3, const std::string& name3, const dtype& dt4, const std::string& name4)
{
    dtype field_types[5];
    std::string field_names[5];
    field_types[0] = dt0;
    field_types[1] = dt1;
    field_types[2] = dt2;
    field_types[3] = dt3;
    field_types[4] = dt4;
    field_names[0] = name0;
    field_names[1] = name1;
    field_names[2] = name2;
    field_names[3] = name3;
    field_names[4] = name4;
    return make_fixedstruct_dtype(5, field_types, field_names);
}

/**
 * \brief Checks whether a set of offsets can be used for fixedstruct.
 *
 * Because fixedstruct does not support customizable offset (use struct for
 * that), this function can be used to check that offsets are compatible with
 * fixedstruct.
 *
 * \param field_count  The number of array entries in `field_types` and `field_offsets`
 * \param field_types  An array of the field dtypes.
 * \param field_offsets  The offsets corresponding to the types.
 * \param total_size  The total size of the struct in bytes.
 *
 * \returns  True if constructing a fixedstruct with the same dtypes and field offsets will
 *           produce the provided offsets.
 */
inline bool is_fixedstruct_compatible_offsets(size_t field_count,
                const dtype *field_types, const size_t *field_offsets, size_t total_size)
{
    size_t offset = 0, max_alignment = 1;
    for (size_t i = 0; i != field_count; ++i) {
        size_t field_alignment = field_types[i].get_alignment();
        size_t field_data_size = field_types[i].get_data_size();
        offset = inc_to_alignment(offset, field_alignment);
        if (field_offsets[i] != offset || field_data_size == 0) {
            return false;
        }
        max_alignment = (field_alignment > max_alignment) ? field_alignment : max_alignment;
        offset += field_data_size;
    }
    offset = inc_to_alignment(offset, max_alignment);
    return total_size == offset;
}

} // namespace dynd

#endif // _DYND__FIXEDSTRUCT_DTYPE_HPP_
