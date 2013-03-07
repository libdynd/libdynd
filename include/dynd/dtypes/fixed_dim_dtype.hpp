//
// Copyright (C) 2011-13, DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#ifndef _DYND__FIXEDARRAY_DTYPE_HPP_
#define _DYND__FIXEDARRAY_DTYPE_HPP_

#include <dynd/dtype.hpp>
#include <dynd/dtype_assign.hpp>
#include <dynd/dtypes/view_dtype.hpp>
#include <dynd/dtypes/base_uniform_dim_dtype.hpp>

namespace dynd {

struct fixed_dim_dtype_iterdata {
    iterdata_common common;
    char *data;
    intptr_t stride;
};

class fixed_dim_dtype : public base_uniform_dim_dtype {
    intptr_t m_stride;
    size_t m_dim_size;
    std::vector<std::pair<std::string, gfunc::callable> > m_ndobject_properties, m_ndobject_functions;

    void create_ndobject_properties();
public:
    fixed_dim_dtype(size_t dimension_size, const dtype& element_dtype);
    fixed_dim_dtype(size_t dimension_size, const dtype& element_dtype, intptr_t stride);

    virtual ~fixed_dim_dtype();

    size_t get_default_data_size(size_t DYND_UNUSED(ndim), const intptr_t *DYND_UNUSED(shape)) const {
        return get_data_size();
    }

    intptr_t get_fixed_stride() const {
        return m_stride;
    }

    size_t get_fixed_dim_size() const {
        return m_dim_size;
    }

    void print_data(std::ostream& o, const char *metadata, const char *data) const;

    void print_dtype(std::ostream& o) const;

    bool is_expression() const;
    bool is_unique_data_owner(const char *metadata) const;
    void transform_child_dtypes(dtype_transform_fn_t transform_fn, const void *extra,
                    dtype& out_transformed_dtype, bool& out_was_transformed) const;
    dtype get_canonical_dtype() const;
    bool is_strided() const;
    void process_strided(const char *metadata, const char *data,
                    dtype& out_dt, const char *&out_origin,
                    intptr_t& out_stride, intptr_t& out_dim_size) const;

    dtype apply_linear_index(size_t nindices, const irange *indices,
                size_t current_i, const dtype& root_dt, bool leading_dimension) const;
    intptr_t apply_linear_index(size_t nindices, const irange *indices, const char *metadata,
                    const dtype& result_dtype, char *out_metadata,
                    memory_block_data *embedded_reference,
                    size_t current_i, const dtype& root_dt,
                    bool leading_dimension, char **inout_data,
                    memory_block_data **inout_dataref) const;
    dtype at_single(intptr_t i0, const char **inout_metadata, const char **inout_data) const;

    dtype get_dtype_at_dimension(char **inout_metadata, size_t i, size_t total_ndim = 0) const;

    intptr_t get_dim_size(const char *metadata, const char *data) const;
    void get_shape(size_t i, intptr_t *out_shape) const;
    void get_shape(size_t i, intptr_t *out_shape, const char *metadata) const;
    void get_strides(size_t i, intptr_t *out_strides, const char *metadata) const;
    intptr_t get_representative_stride(const char *metadata) const;

    bool is_lossless_assignment(const dtype& dst_dt, const dtype& src_dt) const;

    bool operator==(const base_dtype& rhs) const;

    void metadata_default_construct(char *metadata, size_t ndim, const intptr_t* shape) const;
    void metadata_copy_construct(char *dst_metadata, const char *src_metadata, memory_block_data *embedded_reference) const;
    void metadata_reset_buffers(char *metadata) const;
    void metadata_finalize_buffers(char *metadata) const;
    void metadata_destruct(char *metadata) const;
    void metadata_debug_print(const char *metadata, std::ostream& o, const std::string& indent) const;

    size_t get_iterdata_size(size_t ndim) const;
    size_t iterdata_construct(iterdata_common *iterdata, const char **inout_metadata, size_t ndim, const intptr_t* shape, dtype& out_uniform_dtype) const;
    size_t iterdata_destruct(iterdata_common *iterdata, size_t ndim) const;

    void data_destruct(const char *metadata, char *data) const;
    void data_destruct_strided(const char *metadata, char *data,
                    intptr_t stride, size_t count) const;

    size_t make_assignment_kernel(
                    hierarchical_kernel *out, size_t offset_out,
                    const dtype& dst_dt, const char *dst_metadata,
                    const dtype& src_dt, const char *src_metadata,
                    kernel_request_t kernreq, assign_error_mode errmode,
                    const eval::eval_context *ectx) const;

    void foreach_leading(char *data, const char *metadata, foreach_fn_t callback, void *callback_data) const;
    
    void reorder_default_constructed_strides(char *dst_metadata, const dtype& src_dtype, const char *src_metadata) const;

    void get_dynamic_dtype_properties(
                    const std::pair<std::string, gfunc::callable> **out_properties,
                    size_t *out_count) const;
    void get_dynamic_ndobject_properties(
                    const std::pair<std::string, gfunc::callable> **out_properties,
                    size_t *out_count) const;
    void get_dynamic_ndobject_functions(
                    const std::pair<std::string, gfunc::callable> **out_functions,
                    size_t *out_count) const;
};

inline dtype make_fixed_dim_dtype(size_t size, const dtype& element_dtype) {
    return dtype(new fixed_dim_dtype(size, element_dtype), false);
}

inline dtype make_fixed_dim_dtype(size_t size, const dtype& element_dtype, intptr_t stride) {
    return dtype(new fixed_dim_dtype(size, element_dtype, stride), false);
}

dtype make_fixed_dim_dtype(size_t ndim, const intptr_t *shape, const dtype& uniform_dtype, const int *axis_perm);

} // namespace dynd

#endif // _DYND__FIXEDARRAY_DTYPE_HPP_
