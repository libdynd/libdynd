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

    bool is_scalar(const char *data, const char *metadata) const;

    dtype apply_linear_index(int nindices, const irange *indices, int current_i, const dtype& root_dt) const;

    void get_shape(int i, std::vector<intptr_t>& out_shape) const;

    void get_shape(int i, std::vector<intptr_t>& out_shape, const char *data, const char *metadata) const;

    void get_strides(int i, std::vector<intptr_t>& out_strides, const char *data, const char *metadata) const;

    bool is_lossless_assignment(const dtype& dst_dt, const dtype& src_dt) const;

    void get_single_compare_kernel(single_compare_kernel_instance& out_kernel) const;

    void get_dtype_assignment_kernel(const dtype& dst_dt, const dtype& src_dt,
                    assign_error_mode errmode,
                    unary_specialization_kernel_instance& out_kernel) const;

    bool operator==(const extended_dtype& rhs) const;

    size_t get_metadata_size() const;
    void metadata_default_construct(char *metadata, int ndim, const intptr_t* shape) const;
    void metadata_destruct(char *metadata) const;
    void metadata_debug_dump(const char *metadata, std::ostream& o, const std::string& indent) const;

    size_t get_iterdata_size() const;
    size_t iterdata_construct(iterdata_common *iterdata, const char *metadata, int ndim, const intptr_t* shape, dtype& out_uniform_dtype) const;
    size_t iterdata_destruct(iterdata_common *iterdata, int ndim) const;

    void foreach(int ndim, char *data, const char *metadata, foreach_fn_t callback, const void *callback_data) const;
};

inline dtype make_strided_array_dtype(const dtype& element_dtype) {
    return dtype(new strided_array_dtype(element_dtype));
}

} // namespace dynd

#endif // _DYND__STRIDED_ARRAY_DTYPE_HPP_
