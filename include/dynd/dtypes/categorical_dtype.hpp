//
// Copyright (C) 2011-13 Mark Wiebe, DyND Developers
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
#include <dynd/dtypes/strided_dim_dtype.hpp>


namespace {

struct assign_to_same_category_type;

} // anonymous namespace

namespace dynd {

class categorical_dtype : public base_dtype {
    // The data type of the category
    dtype m_category_dtype;
    // The integer type used for storage
    dtype m_storage_dtype;
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

    void get_shape(size_t ndim, size_t i, intptr_t *out_shape, const char *metadata) const;

    size_t get_category_count() const {
        return (size_t)reinterpret_cast<const strided_dim_dtype_metadata *>(m_categories.get_ndo_meta())->size;
    }

    /**
     * Returns the dtype of the category values.
     */
    const dtype& get_category_dtype() const {
        return m_category_dtype;
    }

    /**
     * Return the dtype of the underlying integer used
     * to index the category list.
     */
    const dtype& get_storage_dtype() const {
        return m_storage_dtype;
    }

    uint32_t get_value_from_category(const char *category_metadata, const char *category_data) const;
    uint32_t get_value_from_category(const ndobject& category) const;

    const char *get_category_data_from_value(size_t value) const {
        if (value >= get_category_count()) {
            throw std::runtime_error("category value is out of bounds");
        }
        return m_categories.get_readonly_originptr() +
                    m_value_to_category_index[value] * reinterpret_cast<const strided_dim_dtype_metadata *>(m_categories.get_ndo_meta())->stride;
    }
    /** Returns the metadata corresponding to data from get_category_data_from_value */
    const char *get_category_metadata() const;

    /** Returns the categories as an immutable ndobject */
    ndobject get_categories() const;

    bool is_lossless_assignment(const dtype& dst_dt, const dtype& src_dt) const;

    bool operator==(const base_dtype& rhs) const;

    void metadata_default_construct(char *metadata, size_t ndim, const intptr_t* shape) const;
    void metadata_copy_construct(char *dst_metadata, const char *src_metadata, memory_block_data *embedded_reference) const;
    void metadata_destruct(char *metadata) const;
    void metadata_debug_print(const char *metadata, std::ostream& o, const std::string& indent) const;

    size_t make_assignment_kernel(
                    hierarchical_kernel *out, size_t offset_out,
                    const dtype& dst_dt, const char *dst_metadata,
                    const dtype& src_dt, const char *src_metadata,
                    kernel_request_t kernreq, assign_error_mode errmode,
                    const eval::eval_context *ectx) const;

    void get_dynamic_ndobject_properties(
                    const std::pair<std::string, gfunc::callable> **out_properties,
                    size_t *out_count) const;
    void get_dynamic_dtype_properties(
                    const std::pair<std::string, gfunc::callable> **out_properties,
                    size_t *out_count) const;

    friend struct assign_to_same_category_type;
    friend struct assign_from_same_category_type;
    friend struct assign_from_commensurate_category_type;
};

inline dtype make_categorical_dtype(const ndobject& values) {
    return dtype(new categorical_dtype(values), false);
}


dtype factor_categorical_dtype(const ndobject& values);


} // namespace dynd

#endif // _DYND__CATEGORICAL_DTYPE_HPP_
