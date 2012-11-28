//
// Copyright (C) 2011-12, Dynamic NDArray Developers
// BSD 2-Clause License, see LICENSE.txt
//

#ifndef _DYND__STRIDED_ARRAY_DTYPE_HPP_
#define _DYND__STRIDED_ARRAY_DTYPE_HPP_

#include <dynd/dtype.hpp>
#include <dynd/dtype_assign.hpp>
#include <dynd/dtypes/view_dtype.hpp>

namespace dynd {

struct strided_array_dtype_metadata {
    intptr_t size;
    intptr_t stride;
};

struct strided_array_dtype_iterdata {
    iterdata_common common;
    char *data;
    intptr_t stride;
};

class strided_array_dtype : public extended_dtype {
    dtype m_element_dtype;
public:
    strided_array_dtype(const dtype& element_dtype);

    type_id_t type_id() const {
        return strided_array_type_id;
    }
    dtype_kind_t kind() const {
        return composite_kind;
    }
    // Expose the storage traits here
    size_t alignment() const {
        return m_element_dtype.alignment();
    }
    size_t get_element_size() const {
        return 0;
    }
    size_t get_default_element_size(int ndim, const intptr_t *shape) const;

    const dtype& get_element_dtype() const {
        return m_element_dtype;
    }

    void print_element(std::ostream& o, const char *data, const char *metadata) const;

    void print_dtype(std::ostream& o) const;

    // This is about the native storage, buffering code needs to check whether
    // the value_dtype is an object type separately.
    dtype_memory_management_t get_memory_management() const {
        return pod_memory_management;
    }

    bool is_uniform_dim() const;
    bool is_scalar() const;
    dtype with_transformed_scalar_types(dtype_transform_fn_t transform_fn, const void *extra) const;
    dtype get_canonical_dtype() const;

    dtype apply_linear_index(int nindices, const irange *indices, int current_i, const dtype& root_dt) const;
    intptr_t apply_linear_index(int nindices, const irange *indices, char *data, const char *metadata,
                    const dtype& result_dtype, char *out_metadata, int current_i, const dtype& root_dt) const;

    int get_uniform_ndim() const;
    dtype get_dtype_at_dimension(char **inout_metadata, int i, int total_ndim = 0) const;

    intptr_t get_dim_size(const char *data, const char *metadata) const;
    void get_shape(int i, intptr_t *out_shape) const;
    void get_shape(int i, intptr_t *out_shape, const char *data, const char *metadata) const;
    void get_strides(int i, intptr_t *out_strides, const char *data, const char *metadata) const;

    bool is_lossless_assignment(const dtype& dst_dt, const dtype& src_dt) const;

    void get_single_compare_kernel(single_compare_kernel_instance& out_kernel) const;

    void get_dtype_assignment_kernel(const dtype& dst_dt, const dtype& src_dt,
                    assign_error_mode errmode,
                    unary_specialization_kernel_instance& out_kernel) const;

    bool operator==(const extended_dtype& rhs) const;

    size_t get_metadata_size() const;
    void metadata_default_construct(char *metadata, int ndim, const intptr_t* shape) const;
    void metadata_copy_construct(char *dst_metadata, const char *src_metadata, memory_block_data *embedded_reference) const;
    void metadata_destruct(char *metadata) const;
    void metadata_debug_print(const char *metadata, std::ostream& o, const std::string& indent) const;

    size_t get_iterdata_size(int ndim) const;
    size_t iterdata_construct(iterdata_common *iterdata, const char **inout_metadata, int ndim, const intptr_t* shape, dtype& out_uniform_dtype) const;
    size_t iterdata_destruct(iterdata_common *iterdata, int ndim) const;

    void foreach_leading(char *data, const char *metadata, foreach_fn_t callback, void *callback_data) const;
};

inline dtype make_strided_array_dtype(const dtype& element_dtype) {
    return dtype(new strided_array_dtype(element_dtype));
}

inline dtype make_strided_array_dtype(const dtype& uniform_dtype, int ndim) {
    dtype result = uniform_dtype;
    for (int i = 0; i < ndim; ++i) {
        result = make_strided_array_dtype(result);
    }
    return result;
}

} // namespace dynd

#endif // _DYND__STRIDED_ARRAY_DTYPE_HPP_
