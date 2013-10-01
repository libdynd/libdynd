//
// Copyright (C) 2011-13 Mark Wiebe, DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//
// The categorical type always represents categorical data
//
#ifndef _DYND__CATEGORICAL_TYPE_HPP_
#define _DYND__CATEGORICAL_TYPE_HPP_

#include <map>
#include <vector>

#include <dynd/type.hpp>
#include <dynd/array.hpp>
#include <dynd/types/strided_dim_type.hpp>


namespace {

struct assign_to_same_category_type;

} // anonymous namespace

namespace dynd {

class categorical_type : public base_type {
    // The data type of the category
    ndt::type m_category_tp;
    // The integer type used for storage
    ndt::type m_storage_type;
    // list of categories, in sorted order
    nd::array m_categories;
    // mapping from category indices to values
    std::vector<intptr_t> m_category_index_to_value;
    // mapping from values to category indices
    std::vector<intptr_t> m_value_to_category_index;

public:
    categorical_type(const nd::array& categories, bool presorted=false);

    virtual ~categorical_type() {
    }

    void print_data(std::ostream& o, const char *metadata, const char *data) const;

    void print_type(std::ostream& o) const;

        void get_shape(intptr_t ndim, intptr_t i, intptr_t *out_shape, const char *metadata, const char *data) const;

    size_t get_category_count() const {
        return (size_t)reinterpret_cast<const strided_dim_type_metadata *>(m_categories.get_ndo_meta())->size;
    }

    /**
     * Returns the type of the category values.
     */
    const ndt::type& get_category_type() const {
        return m_category_tp;
    }

    /**
     * Return the type of the underlying integer used
     * to index the category list.
     */
    const ndt::type& get_storage_type() const {
        return m_storage_type;
    }

    uint32_t get_value_from_category(const char *category_metadata, const char *category_data) const;
    uint32_t get_value_from_category(const nd::array& category) const;

    const char *get_category_data_from_value(size_t value) const {
        if (value >= get_category_count()) {
            throw std::runtime_error("category value is out of bounds");
        }
        return m_categories.get_readonly_originptr() +
                    m_value_to_category_index[value] * reinterpret_cast<const strided_dim_type_metadata *>(m_categories.get_ndo_meta())->stride;
    }
    /** Returns the metadata corresponding to data from get_category_data_from_value */
    const char *get_category_metadata() const;

    /** Returns the categories as an immutable nd::array */
    nd::array get_categories() const;

    bool is_lossless_assignment(const ndt::type& dst_tp, const ndt::type& src_tp) const;

    bool operator==(const base_type& rhs) const;

    void metadata_default_construct(char *metadata, intptr_t ndim, const intptr_t* shape) const;
    void metadata_copy_construct(char *dst_metadata, const char *src_metadata, memory_block_data *embedded_reference) const;
    void metadata_destruct(char *metadata) const;
    void metadata_debug_print(const char *metadata, std::ostream& o, const std::string& indent) const;

    size_t make_assignment_kernel(
                    ckernel_builder *out, size_t offset_out,
                    const ndt::type& dst_tp, const char *dst_metadata,
                    const ndt::type& src_tp, const char *src_metadata,
                    kernel_request_t kernreq, assign_error_mode errmode,
                    const eval::eval_context *ectx) const;

    void get_dynamic_array_properties(
                    const std::pair<std::string, gfunc::callable> **out_properties,
                    size_t *out_count) const;
    void get_dynamic_type_properties(
                    const std::pair<std::string, gfunc::callable> **out_properties,
                    size_t *out_count) const;

    friend struct assign_to_same_category_type;
    friend struct assign_from_same_category_type;
    friend struct assign_from_commensurate_category_type;
};

namespace ndt {
    inline ndt::type make_categorical(const nd::array& values) {
        return ndt::type(new categorical_type(values), false);
    }

    ndt::type factor_categorical(const nd::array& values);
} // namespace ndt

} // namespace dynd

#endif // _DYND__CATEGORICAL_TYPE_HPP_
