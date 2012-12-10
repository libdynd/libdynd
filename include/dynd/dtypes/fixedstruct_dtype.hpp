//
// Copyright (C) 2011-12, Dynamic NDArray Developers
// BSD 2-Clause License, see LICENSE.txt
//

#ifndef _DYND__FIXEDSTRUCT_DTYPE_HPP_
#define _DYND__FIXEDSTRUCT_DTYPE_HPP_

#include <vector>
#include <string>

#include <dynd/dtype.hpp>
#include <dynd/memblock/memory_block.hpp>

namespace dynd {

class fixedstruct_dtype : public extended_dtype {
    std::vector<dtype> m_field_types;
    std::vector<std::string> m_field_names;
    std::vector<size_t> m_data_offsets;
    std::vector<size_t> m_metadata_offsets;
    std::vector<std::pair<std::string, gfunc::callable> > m_ndobject_properties;
    size_t m_element_size;
    size_t m_metadata_size;
    dtype_memory_management_t m_memory_management;
    unsigned char m_alignment;

    void create_ndobject_properties();

    // Used as the parameters dtype for the ndobject properties callables
    static dtype ndobject_parameters_dtype;
public:
    fixedstruct_dtype(const std::vector<dtype>& field_types, const std::vector<std::string>& field_names);

    type_id_t get_type_id() const {
        return fixedstruct_type_id;
    }
    dtype_kind_t get_kind() const {
        return struct_kind;
    }
    // Expose the storage traits here
    size_t get_alignment() const {
        return m_alignment;
    }
    size_t get_element_size() const {
        return m_element_size;
    }
    size_t get_default_element_size(int DYND_UNUSED(ndim), const intptr_t *DYND_UNUSED(shape)) const {
        return m_element_size;
    }

    size_t get_field_count() const {
        return m_field_types.size();
    }

    const std::vector<dtype>& get_field_types() const {
        return m_field_types;
    }

    const std::vector<std::string>& get_field_names() const {
        return m_field_names;
    }

    const std::vector<size_t>& get_data_offsets() const {
        return m_data_offsets;
    }

    const std::vector<size_t>& get_metadata_offsets() const {
        return m_metadata_offsets;
    }

    void print_element(std::ostream& o, const char *metadata, const char *data) const;

    void print_dtype(std::ostream& o) const;

    // This is about the native storage, buffering code needs to check whether
    // the value_dtype is an object type separately.
    dtype_memory_management_t get_memory_management() const {
        return m_memory_management;
    }

    bool is_scalar() const;
    bool is_expression() const;
    dtype with_transformed_scalar_types(dtype_transform_fn_t transform_fn, const void *extra) const;
    dtype get_canonical_dtype() const;

    dtype apply_linear_index(int nindices, const irange *indices, int current_i, const dtype& root_dt) const;
    intptr_t apply_linear_index(int nindices, const irange *indices, char *data, const char *metadata,
                    const dtype& result_dtype, char *out_metadata,
                    memory_block_data *embedded_reference,
                    int current_i, const dtype& root_dt) const;
    dtype at(intptr_t i0, const char **inout_metadata, const char **inout_data) const;

    intptr_t get_dim_size(const char *data, const char *metadata) const;
    void get_shape(int i, intptr_t *out_shape) const;
    intptr_t get_representative_stride(const char *metadata) const;

    bool is_lossless_assignment(const dtype& dst_dt, const dtype& src_dt) const;

    void get_single_compare_kernel(single_compare_kernel_instance& out_kernel) const;

    void get_dtype_assignment_kernel(const dtype& dst_dt, const dtype& src_dt,
                    assign_error_mode errmode,
                    kernel_instance<unary_operation_pair_t>& out_kernel) const;

    bool operator==(const extended_dtype& rhs) const;

    size_t get_metadata_size() const;
    void metadata_default_construct(char *metadata, int ndim, const intptr_t* shape) const;
    void metadata_copy_construct(char *dst_metadata, const char *src_metadata, memory_block_data *embedded_reference) const;
    void metadata_destruct(char *metadata) const;
    void metadata_debug_print(const char *metadata, std::ostream& o, const std::string& indent) const;

    void foreach_leading(char *data, const char *metadata, foreach_fn_t callback, void *callback_data) const;

    void get_dynamic_dtype_properties(const std::pair<std::string, gfunc::callable> **out_properties, int *out_count) const;
    void get_dynamic_ndobject_properties(const std::pair<std::string, gfunc::callable> **out_properties, int *out_count) const;
}; // class fixedstruct_dtype

/** Makes a struct dtype with the specified fields */
inline dtype make_fixedstruct_dtype(const std::vector<dtype>& field_types, const std::vector<std::string>& field_names) {
    return dtype(new fixedstruct_dtype(field_types, field_names));
}

/** Makes a struct dtype with the specified fields */
inline dtype make_fixedstruct_dtype(const dtype& dt0, const std::string& name0)
{
    std::vector<dtype> fields;
    std::vector<std::string> field_names;
    fields.push_back(dt0);
    field_names.push_back(name0);
    return make_fixedstruct_dtype(fields, field_names);
}

/** Makes a struct dtype with the specified fields */
inline dtype make_fixedstruct_dtype(const dtype& dt0, const std::string& name0, const dtype& dt1, const std::string& name1)
{
    std::vector<dtype> fields;
    std::vector<std::string> field_names;
    fields.push_back(dt0);
    fields.push_back(dt1);
    field_names.push_back(name0);
    field_names.push_back(name1);
    return make_fixedstruct_dtype(fields, field_names);
}

/** Makes a struct dtype with the specified fields */
inline dtype make_fixedstruct_dtype(const dtype& dt0, const std::string& name0, const dtype& dt1, const std::string& name1, const dtype& dt2, const std::string& name2)
{
    std::vector<dtype> fields;
    std::vector<std::string> field_names;
    fields.push_back(dt0);
    fields.push_back(dt1);
    fields.push_back(dt2);
    field_names.push_back(name0);
    field_names.push_back(name1);
    field_names.push_back(name2);
    return make_fixedstruct_dtype(fields, field_names);
}

/** Makes a struct dtype with the specified fields */
inline dtype make_fixedstruct_dtype(const dtype& dt0, const std::string& name0,
                const dtype& dt1, const std::string& name1, const dtype& dt2, const std::string& name2,
                const dtype& dt3, const std::string& name3)
{
    std::vector<dtype> fields;
    std::vector<std::string> field_names;
    fields.push_back(dt0);
    fields.push_back(dt1);
    fields.push_back(dt2);
    fields.push_back(dt3);
    field_names.push_back(name0);
    field_names.push_back(name1);
    field_names.push_back(name2);
    field_names.push_back(name3);
    return make_fixedstruct_dtype(fields, field_names);
}

/** Makes a struct dtype with the specified fields */
inline dtype make_fixedstruct_dtype(const dtype& dt0, const std::string& name0,
                const dtype& dt1, const std::string& name1, const dtype& dt2, const std::string& name2,
                const dtype& dt3, const std::string& name3, const dtype& dt4, const std::string& name4)
{
    std::vector<dtype> fields;
    std::vector<std::string> field_names;
    fields.push_back(dt0);
    fields.push_back(dt1);
    fields.push_back(dt2);
    fields.push_back(dt3);
    fields.push_back(dt4);
    field_names.push_back(name0);
    field_names.push_back(name1);
    field_names.push_back(name2);
    field_names.push_back(name3);
    field_names.push_back(name4);
    return make_fixedstruct_dtype(fields, field_names);
}

/**
 * \brief Checks whether a set of offsets can be used for fixedstruct.
 *
 * Because fixedstruct does not support customizable offset (use struct for
 * that), this function can be used to check that offsets are compatible with
 * fixedstruct.
 *
 * \param nfields  The number of array entries in `field_types` and `field_offsets`
 * \param field_types  An array of the field dtypes.
 * \param field_offsets  The offsets corresponding to the types.
 * \param total_size  The total size of the struct in bytes.
 *
 * \returns  True if constructing a fixedstruct with the same dtypes and field offsets will
 *           produce the provided offsets.
 */
inline bool is_fixedstruct_compatible_offsets(int nfields, const dtype *field_types, const size_t *field_offsets, size_t total_size)
{
    size_t offset = 0, max_alignment = 1;
    for (int i = 0; i < nfields; ++i) {
        size_t field_alignment = field_types[i].get_alignment();
        size_t field_data_size = field_types[i].get_element_size();
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
