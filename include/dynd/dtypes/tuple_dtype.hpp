//
// Copyright (C) 2011-12, Dynamic NDArray Developers
// BSD 2-Clause License, see LICENSE.txt
//

#ifndef _DYND__TUPLE_DTYPE_HPP_
#define _DYND__TUPLE_DTYPE_HPP_

#include <vector>

#include <dynd/dtype.hpp>

namespace dynd {

class tuple_dtype : public extended_dtype {
    std::vector<dtype> m_fields;
    std::vector<size_t> m_offsets;
    std::vector<size_t> m_metadata_offsets;
    size_t m_metadata_size;
    uintptr_t m_element_size;
    dtype_memory_management_t m_memory_management;
    unsigned char m_alignment;
    bool m_is_standard_layout;

    bool compute_is_standard_layout() const;
public:
    tuple_dtype(const std::vector<dtype>& fields);
    tuple_dtype(const std::vector<dtype>& fields, const std::vector<size_t> offsets,
                        size_t element_size, size_t alignment);

    type_id_t get_type_id() const {
        return tuple_type_id;
    }
    dtype_kind_t get_kind() const {
        return composite_kind;
    }
    // Expose the storage traits here
    size_t get_alignment() const {
        return m_alignment;
    }
    size_t get_element_size() const {
        return m_element_size;
    }

    const std::vector<dtype>& get_fields() const {
        return m_fields;
    }

    const std::vector<size_t>& get_offsets() const {
        return m_offsets;
    }

    /**
     * Returns true if the layout is standard, i.e. constructable without
     * specifying the offsets/alignment/element_size.
     */
    bool is_standard_layout() const {
        return m_is_standard_layout;
    }

    void print_element(std::ostream& o, const char *metadata, const char *data) const;

    void print_dtype(std::ostream& o) const;

    // This is about the native storage, buffering code needs to check whether
    // the value_dtype is an object type separately.
    dtype_memory_management_t get_memory_management() const {
        return m_memory_management;
    }

    dtype apply_linear_index(int nindices, const irange *indices, int current_i, const dtype& root_dt) const;

    void get_shape(int i, intptr_t *out_shape) const;

    bool is_lossless_assignment(const dtype& dst_dt, const dtype& src_dt) const;

    void get_single_compare_kernel(single_compare_kernel_instance& out_kernel) const;

    void get_dtype_assignment_kernel(const dtype& dst_dt, const dtype& src_dt,
                    assign_error_mode errmode,
                    unary_specialization_kernel_instance& out_kernel) const;

    bool operator==(const extended_dtype& rhs) const;
}; // class tuple_dtype

/** Makes a tuple dtype with the specified fields, using the standard layout */
inline dtype make_tuple_dtype(const std::vector<dtype>& fields) {
    return dtype(new tuple_dtype(fields));
}

/** Makes a tuple dtype with the specified fields and layout */
inline dtype make_tuple_dtype(const std::vector<dtype>& fields, const std::vector<size_t> offsets,
                size_t element_size, size_t alignment)
{
    return dtype(new tuple_dtype(fields, offsets, element_size, alignment));
}

/** Makes a tuple dtype with the specified fields, using the standard layout */
inline dtype make_tuple_dtype(const dtype& dt0)
{
    std::vector<dtype> fields;
    fields.push_back(dt0);
    return make_tuple_dtype(fields);
}

/** Makes a tuple dtype with the specified fields, using the standard layout */
inline dtype make_tuple_dtype(const dtype& dt0, const dtype& dt1)
{
    std::vector<dtype> fields;
    fields.push_back(dt0);
    fields.push_back(dt1);
    return make_tuple_dtype(fields);
}

/** Makes a tuple dtype with the specified fields, using the standard layout */
inline dtype make_tuple_dtype(const dtype& dt0, const dtype& dt1, const dtype& dt2)
{
    std::vector<dtype> fields;
    fields.push_back(dt0);
    fields.push_back(dt1);
    fields.push_back(dt2);
    return make_tuple_dtype(fields);
}

/** Makes a tuple dtype with the specified fields, using the standard layout */
inline dtype make_tuple_dtype(const dtype& dt0, const dtype& dt1, const dtype& dt2, const dtype& dt3)
{
    std::vector<dtype> fields;
    fields.push_back(dt0);
    fields.push_back(dt1);
    fields.push_back(dt2);
    fields.push_back(dt3);
    return make_tuple_dtype(fields);
}

} // namespace dynd

#endif // _DYND__TUPLE_DTYPE_HPP_
