//
// Copyright (C) 2011-12, Dynamic NDArray Developers
// BSD 2-Clause License, see LICENSE.txt
//
// The categorical dtype always represents categorical data
//
#ifndef _DND__CATEGORICAL_DTYPE_HPP_
#define _DND__CATEGORICAL_DTYPE_HPP_

#include <map>
#include <vector>

#include <dnd/dtype.hpp>
#include <dnd/ndarray.hpp>


namespace {

struct assign_to_same_category_type;

} // anonymous namespace

namespace dnd {

class categorical_dtype : public extended_dtype {
    // The data type of the category
    dtype m_category_dtype;
    // list of categories, sorted lexicographically
    std::vector<const char *> m_categories;
    // mapping from category indices to values
    std::vector<intptr_t> m_category_index_to_value;
    // mapping from values to category indices
    std::vector<intptr_t> m_value_to_category_index;

public:
    categorical_dtype(const ndarray& categories);

    ~categorical_dtype();

    type_id_t type_id() const {
        return categorical_type_id;
    }
    dtype_kind_t kind() const {
        return custom_kind;
    }
    unsigned char alignment() const {
        return 4; // TODO
    }
    uintptr_t element_size() const {
        return 4; // TODO
    }

    void print_element(std::ostream& o, const char *data) const;

    void print_dtype(std::ostream& o) const;

    dtype_memory_management_t get_memory_management() const {
        return pod_memory_management;
    }

    const dtype& value_dtype(const dtype& self) const {
        return self;
    }

    intptr_t get_category_count() const {
        return m_categories.size();
    }

    uint32_t get_value_from_category(const char *category) const;

    bool is_lossless_assignment(const dtype& dst_dt, const dtype& src_dt) const;

    void get_dtype_assignment_kernel(const dtype& dst_dt, const dtype& src_dt,
                    assign_error_mode errmode,
                    unary_specialization_kernel_instance& out_kernel) const;

    bool operator==(const extended_dtype& rhs) const;

    friend struct assign_to_same_category_type;
    friend struct assign_from_same_category_type;
    friend struct assign_from_commensurate_category_type;

};

inline dtype make_categorical_dtype(const ndarray& values) {
    return dtype(make_shared<categorical_dtype>(values));
}


} // namespace dnd

#endif // _DND__CATEGORICAL_DTYPE_HPP_
