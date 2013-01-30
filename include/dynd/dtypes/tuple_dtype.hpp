//
// Copyright (C) 2011-13, DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#ifndef _DYND__TUPLE_DTYPE_HPP_
#define _DYND__TUPLE_DTYPE_HPP_

#include <vector>

#include <dynd/dtype.hpp>

namespace dynd {

class tuple_dtype : public base_dtype {
    std::vector<dtype> m_fields;
    std::vector<size_t> m_offsets;
    std::vector<size_t> m_metadata_offsets;
    size_t m_metadata_size;
    dtype_memory_management_t m_memory_management;
    bool m_is_standard_layout;

    bool compute_is_standard_layout() const;
public:
    tuple_dtype(const std::vector<dtype>& fields);
    tuple_dtype(const std::vector<dtype>& fields, const std::vector<size_t> offsets,
                        size_t data_size, size_t alignment);

    virtual ~tuple_dtype();

    const std::vector<dtype>& get_fields() const {
        return m_fields;
    }

    const std::vector<size_t>& get_offsets() const {
        return m_offsets;
    }

    /**
     * Returns true if the layout is standard, i.e. constructable without
     * specifying the offsets/alignment/data_size.
     */
    bool is_standard_layout() const {
        return m_is_standard_layout;
    }

    void print_data(std::ostream& o, const char *metadata, const char *data) const;

    void print_dtype(std::ostream& o) const;

    // This is about the native storage, buffering code needs to check whether
    // the value_dtype is an object type separately.
    dtype_memory_management_t get_memory_management() const {
        return m_memory_management;
    }

    void get_shape(size_t i, intptr_t *out_shape) const;

    bool is_lossless_assignment(const dtype& dst_dt, const dtype& src_dt) const;

    void get_single_compare_kernel(kernel_instance<compare_operations_t>& out_kernel) const;

    void get_dtype_assignment_kernel(const dtype& dst_dt, const dtype& src_dt,
                    assign_error_mode errmode,
                    kernel_instance<unary_operation_pair_t>& out_kernel) const;

    bool operator==(const base_dtype& rhs) const;
}; // class tuple_dtype

/** Makes a tuple dtype with the specified fields, using the standard layout */
inline dtype make_tuple_dtype(const std::vector<dtype>& fields) {
    return dtype(new tuple_dtype(fields), false);
}

/** Makes a tuple dtype with the specified fields and layout */
inline dtype make_tuple_dtype(const std::vector<dtype>& fields, const std::vector<size_t> offsets,
                size_t data_size, size_t alignment)
{
    return dtype(new tuple_dtype(fields, offsets, data_size, alignment), false);
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
