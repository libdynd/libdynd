//
// Copyright (C) 2011-12, Dynamic NDArray Developers
// BSD 2-Clause License, see LICENSE.txt
//

#ifndef _DYND__FIXEDARRAY_DTYPE_HPP_
#define _DYND__FIXEDARRAY_DTYPE_HPP_

#include <dynd/dtype.hpp>
#include <dynd/dtype_assign.hpp>
#include <dynd/dtypes/view_dtype.hpp>

namespace dynd {

struct fixedarray_dtype_iterdata {
    iterdata_common common;
    char *data;
    intptr_t stride;
};

class fixedarray_dtype : public extended_dtype {
    dtype m_element_dtype;
    intptr_t m_stride;
    size_t m_dimension_size, m_element_size;
    std::vector<std::pair<std::string, gfunc::callable> > m_ndobject_properties, m_ndobject_functions;

    void create_ndobject_properties();
public:
    fixedarray_dtype(const dtype& element_dtype, size_t dimension_size);
    fixedarray_dtype(const dtype& element_dtype, size_t dimension_size, intptr_t stride);

    type_id_t get_type_id() const {
        return fixedarray_type_id;
    }
    dtype_kind_t get_kind() const {
        return uniform_array_kind;
    }
    // Expose the storage traits here
    size_t get_alignment() const {
        return m_element_dtype.get_alignment();
    }
    size_t get_element_size() const {
        return m_element_size;
    }
    size_t get_default_element_size(int DYND_UNUSED(ndim), const intptr_t *DYND_UNUSED(shape)) const {
        return m_element_size;
    }

    const dtype& get_element_dtype() const {
        return m_element_dtype;
    }

    intptr_t get_fixed_stride() const {
        return m_stride;
    }

    size_t get_fixed_dim_size() const {
        return m_dimension_size;
    }

    void print_element(std::ostream& o, const char *metadata, const char *data) const;

    void print_dtype(std::ostream& o) const;

    // This is about the native storage, buffering code needs to check whether
    // the value_dtype is an object type separately.
    dtype_memory_management_t get_memory_management() const {
        return pod_memory_management;
    }

    bool is_scalar() const;
    bool is_uniform_dim() const;
    bool is_expression() const;
    void transform_child_dtypes(dtype_transform_fn_t transform_fn, const void *extra,
                    dtype& out_transformed_dtype, bool& out_was_transformed) const;
    dtype get_canonical_dtype() const;

    dtype apply_linear_index(int nindices, const irange *indices, int current_i, const dtype& root_dt) const;
    intptr_t apply_linear_index(int nindices, const irange *indices, char *data, const char *metadata,
                    const dtype& result_dtype, char *out_metadata,
                    memory_block_data *embedded_reference,
                    int current_i, const dtype& root_dt) const;
    dtype at(intptr_t i0, const char **inout_metadata, const char **inout_data) const;

    int get_undim() const;
    dtype get_dtype_at_dimension(char **inout_metadata, int i, int total_ndim = 0) const;

    intptr_t get_dim_size(const char *data, const char *metadata) const;
    void get_shape(int i, intptr_t *out_shape) const;
    void get_shape(int i, intptr_t *out_shape, const char *metadata) const;
    void get_strides(int i, intptr_t *out_strides, const char *metadata) const;
    intptr_t get_representative_stride(const char *metadata) const;

    bool is_lossless_assignment(const dtype& dst_dt, const dtype& src_dt) const;

    void get_single_compare_kernel(single_compare_kernel_instance& out_kernel) const;

    void get_dtype_assignment_kernel(const dtype& dst_dt, const dtype& src_dt,
                    assign_error_mode errmode,
                    kernel_instance<unary_operation_pair_t>& out_kernel) const;

    bool operator==(const extended_dtype& rhs) const;

    size_t get_metadata_size() const {
        return 0;
    }
    void metadata_default_construct(char *metadata, int ndim, const intptr_t* shape) const;
    void metadata_copy_construct(char *dst_metadata, const char *src_metadata, memory_block_data *embedded_reference) const;
    void metadata_reset_buffers(char *metadata) const;
    void metadata_finalize_buffers(char *metadata) const;
    void metadata_destruct(char *metadata) const;
    void metadata_debug_print(const char *metadata, std::ostream& o, const std::string& indent) const;

    size_t get_iterdata_size(int ndim) const;
    size_t iterdata_construct(iterdata_common *iterdata, const char **inout_metadata, int ndim, const intptr_t* shape, dtype& out_uniform_dtype) const;
    size_t iterdata_destruct(iterdata_common *iterdata, int ndim) const;

    void foreach_leading(char *data, const char *metadata, foreach_fn_t callback, void *callback_data) const;
    
    void reorder_default_constructed_strides(char *dst_metadata, const dtype& src_dtype, const char *src_metadata) const;

    void get_dynamic_ndobject_properties(const std::pair<std::string, gfunc::callable> **out_properties, int *out_count) const;
    void get_dynamic_ndobject_functions(const std::pair<std::string, gfunc::callable> **out_functions, int *out_count) const;
};

inline dtype make_fixedarray_dtype(const dtype& element_dtype, size_t size) {
    return dtype(new fixedarray_dtype(element_dtype, size));
}

inline dtype make_fixedarray_dtype(const dtype& element_dtype, size_t size, intptr_t stride) {
    return dtype(new fixedarray_dtype(element_dtype, size, stride));
}

dtype make_fixedarray_dtype(const dtype& uniform_dtype, int ndim, const intptr_t *shape, const int *axis_perm);

} // namespace dynd

#endif // _DYND__FIXEDARRAY_DTYPE_HPP_
