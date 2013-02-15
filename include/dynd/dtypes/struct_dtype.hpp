//
// Copyright (C) 2011-13, DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#ifndef _DYND__STRUCT_DTYPE_HPP_
#define _DYND__STRUCT_DTYPE_HPP_

#include <vector>
#include <string>

#include <dynd/dtype.hpp>
#include <dynd/dtypes/base_struct_dtype.hpp>
#include <dynd/memblock/memory_block.hpp>

namespace dynd {

class struct_dtype : public base_struct_dtype {
    std::vector<dtype> m_field_types;
    std::vector<std::string> m_field_names;
    std::vector<size_t> m_metadata_offsets;
    std::vector<std::pair<std::string, gfunc::callable> > m_ndobject_properties;
    dtype_memory_management_t m_memory_management;

    void create_ndobject_properties();

    // Used as the parameters dtype for the ndobject properties callables
    static dtype ndobject_parameters_dtype;
public:
    struct_dtype(const std::vector<dtype>& fields, const std::vector<std::string>& field_names);

    virtual ~struct_dtype();

    size_t get_default_data_size(size_t ndim, const intptr_t *shape) const;

    size_t get_field_count() const {
        return m_field_types.size();
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

    const size_t *get_metadata_offsets() const {
        return &m_metadata_offsets[0];
    }

    const std::vector<size_t>& get_metadata_offsets_vector() const {
        return m_metadata_offsets;
    }

    const size_t *get_data_offsets(const char *metadata) const {
        return reinterpret_cast<const size_t *>(metadata);
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

    intptr_t get_dim_size(const char *data, const char *metadata) const;
    void get_shape(size_t i, intptr_t *out_shape) const;
    intptr_t get_representative_stride(const char *metadata) const;

    bool is_lossless_assignment(const dtype& dst_dt, const dtype& src_dt) const;

    void get_single_compare_kernel(kernel_instance<compare_operations_t>& out_kernel) const;

    bool operator==(const base_dtype& rhs) const;

    void metadata_default_construct(char *metadata, size_t ndim, const intptr_t* shape) const;
    void metadata_copy_construct(char *dst_metadata, const char *src_metadata, memory_block_data *embedded_reference) const;
    void metadata_reset_buffers(char *metadata) const;
    void metadata_finalize_buffers(char *metadata) const;
    void metadata_destruct(char *metadata) const;
    void metadata_debug_print(const char *metadata, std::ostream& o, const std::string& indent) const;


    size_t make_assignment_kernel(
                    assignment_kernel *out,
                    size_t offset_out,
                    const dtype& dst_dt, const char *dst_metadata,
                    const dtype& src_dt, const char *src_metadata,
                    assign_error_mode errmode,
                    const eval::eval_context *ectx) const;

    void foreach_leading(char *data, const char *metadata, foreach_fn_t callback, void *callback_data) const;

    void get_dynamic_dtype_properties(const std::pair<std::string, gfunc::callable> **out_properties, size_t *out_count) const;
    void get_dynamic_ndobject_properties(const std::pair<std::string, gfunc::callable> **out_properties, size_t *out_count) const;
}; // class struct_dtype

/** Makes a tuple dtype with the specified fields, using the standard layout */
inline dtype make_struct_dtype(const std::vector<dtype>& fields, const std::vector<std::string>& field_names) {
    return dtype(new struct_dtype(fields, field_names), false);
}

/** Makes a tuple dtype with the specified fields, using the standard layout */
inline dtype make_struct_dtype(const dtype& dt0, const std::string& name0)
{
    std::vector<dtype> fields;
    std::vector<std::string> field_names;
    fields.push_back(dt0);
    field_names.push_back(name0);
    return make_struct_dtype(fields, field_names);
}

/** Makes a tuple dtype with the specified fields, using the standard layout */
inline dtype make_struct_dtype(const dtype& dt0, const std::string& name0, const dtype& dt1, const std::string& name1)
{
    std::vector<dtype> fields;
    std::vector<std::string> field_names;
    fields.push_back(dt0);
    fields.push_back(dt1);
    field_names.push_back(name0);
    field_names.push_back(name1);
    return make_struct_dtype(fields, field_names);
}

/** Makes a tuple dtype with the specified fields, using the standard layout */
inline dtype make_struct_dtype(const dtype& dt0, const std::string& name0, const dtype& dt1, const std::string& name1, const dtype& dt2, const std::string& name2)
{
    std::vector<dtype> fields;
    std::vector<std::string> field_names;
    fields.push_back(dt0);
    fields.push_back(dt1);
    fields.push_back(dt2);
    field_names.push_back(name0);
    field_names.push_back(name1);
    field_names.push_back(name2);
    return make_struct_dtype(fields, field_names);
}

} // namespace dynd

#endif // _DYND__STRUCT_DTYPE_HPP_
