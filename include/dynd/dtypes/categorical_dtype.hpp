//
// Copyright (C) 2011-12, Dynamic NDArray Developers
// BSD 2-Clause License, see LICENSE.txt
//
// The categorical dtype always represents categorical data
//
#ifndef _DYND__CATEGORICAL_DTYPE_HPP_
#define _DYND__CATEGORICAL_DTYPE_HPP_

#include <map>
#include <vector>

#include <dynd/dtype.hpp>
#include <dynd/ndobject.hpp>
#include <dynd/dtypes/strided_array_dtype.hpp>


namespace {

struct assign_to_same_category_type;

} // anonymous namespace

namespace dynd {

class categorical_dtype : public base_dtype {
    // The data type of the category
    dtype m_category_dtype;
    // list of categories, in sorted order
    ndobject m_categories;
    // mapping from category indices to values
    std::vector<intptr_t> m_category_index_to_value;
    // mapping from values to category indices
    std::vector<intptr_t> m_value_to_category_index;

public:
    categorical_dtype(const ndobject& categories, bool presorted=false);

    virtual ~categorical_dtype() {
    }

    void print_data(std::ostream& o, const char *metadata, const char *data) const;

    void print_dtype(std::ostream& o) const;

    dtype_memory_management_t get_memory_management() const {
        return pod_memory_management;
    }

    dtype apply_linear_index(int nindices, const irange *indices, int current_i, const dtype& root_dt) const;

    void get_shape(size_t i, intptr_t *out_shape) const;

    intptr_t get_category_count() const {
        return reinterpret_cast<const strided_array_dtype_metadata *>(m_categories.get_ndo_meta())->size;
    }

    const dtype& get_category_dtype() const {
        return m_category_dtype;
    }

    uint32_t get_value_from_category(const char *category_metadata, const char *category_data) const;
    uint32_t get_value_from_category(const ndobject& category) const;

    const char *get_category_data_from_value(uint32_t value) const {
        if (value >= get_category_count()) {
            throw std::runtime_error("category value is out of bounds");
        }
        return m_categories.get_readonly_originptr() +
                    m_value_to_category_index[value] * reinterpret_cast<const strided_array_dtype_metadata *>(m_categories.get_ndo_meta())->stride;
    }

    bool is_lossless_assignment(const dtype& dst_dt, const dtype& src_dt) const;

    void get_dtype_assignment_kernel(const dtype& dst_dt, const dtype& src_dt,
                    assign_error_mode errmode,
                    kernel_instance<unary_operation_pair_t>& out_kernel) const;

    bool operator==(const base_dtype& rhs) const;

    size_t get_metadata_size() const;
    void metadata_default_construct(char *metadata, int ndim, const intptr_t* shape) const;
    void metadata_copy_construct(char *dst_metadata, const char *src_metadata, memory_block_data *embedded_reference) const;
    void metadata_destruct(char *metadata) const;
    void metadata_debug_print(const char *metadata, std::ostream& o, const std::string& indent) const;

    friend struct assign_to_same_category_type;
    friend struct assign_from_same_category_type;
    friend struct assign_from_commensurate_category_type;
};

inline dtype make_categorical_dtype(const ndobject& values) {
    return dtype(new categorical_dtype(values));
}


dtype factor_categorical_dtype(const ndobject& values);


} // namespace dynd

#endif // _DYND__CATEGORICAL_DTYPE_HPP_
