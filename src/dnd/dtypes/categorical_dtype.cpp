//
// Copyright (C) 2011-12, Dynamic NDArray Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <cstring>

#include <dnd/dtypes/categorical_dtype.hpp>
#include <dnd/kernels/single_compare_kernel_instance.hpp>
#include <dnd/raw_iteration.hpp>

using namespace dnd;
using namespace std;

namespace {

class sorter {
    const vector<char *>& m_categories;
    const single_compare_operation_t m_less;
    const auxiliary_data& m_auxdata;
public:
    sorter(vector<char *>& values, const single_compare_operation_t less, const auxiliary_data& auxdata) :
        m_categories(values), m_less(less), m_auxdata(auxdata) {}
    bool operator()(int a, int b) {
        return m_less(m_categories[a], m_categories[b], m_auxdata);
    }
};

}


categorical_dtype::categorical_dtype(const ndarray& categories)
    : extended_dtype(), m_category_dtype(categories.get_dtype())
{
    m_categories.resize(categories.get_num_elements());
    m_value_to_category_index.resize(categories.get_num_elements());
    m_category_index_to_value.resize(categories.get_num_elements());

    // create the mapping from indices of (to be lexicographically sorted) categories to values
    vector<char *> categories_user_order(categories.get_num_elements());
    for (uint32_t i = 0; i < m_category_index_to_value.size(); ++i) {
        m_category_index_to_value[i] = i;
        categories_user_order[i] = new char[m_category_dtype.element_size()];
        memcpy(categories_user_order[i], categories(i).get_readonly_originptr(), m_category_dtype.element_size());
    }
    single_compare_kernel_instance k;
    m_category_dtype.get_single_compare_kernel(k);
    std::sort(m_category_index_to_value.begin(), m_category_index_to_value.end(), sorter(categories_user_order, k.comparisons[less_id], k.auxdata));

    // reorder categories lexicographically, and create mapping from values to indices of (lexicographically sorted) categories
    for (uint32_t i = 0; i < m_category_index_to_value.size(); ++i) {
        m_categories[i] = categories_user_order[m_category_index_to_value[i]];
        m_value_to_category_index[i] = m_category_index_to_value[m_category_index_to_value[i]];
    }
}

categorical_dtype::~categorical_dtype() {
    for (vector<char*>::iterator it = m_categories.begin(); it != m_categories.end(); ++it) {
        delete[] *it;
    }

}

void categorical_dtype::print_element(std::ostream& o, const char *data) const
{
    uint32_t value;
    memcpy(&value, data, sizeof(index));
    m_category_dtype.print_element(o, m_categories[m_value_to_category_index[value]]);
}


void categorical_dtype::print_dtype(std::ostream& o) const
{
    o << "categorical(";
    m_category_dtype.print_element(o, m_categories[m_value_to_category_index[0]]);
    for (uint32_t i = 1; i < m_categories.size(); ++i) {
        o << ", ";
        m_category_dtype.print_element(o, m_categories[m_value_to_category_index[i]]);
    }
    o << ")>";
}


bool categorical_dtype::is_lossless_assignment(const dtype& dst_dt, const dtype& src_dt) const
{
    if (src_dt.extended() == this) {
        if (dst_dt.extended() == this) {
            // Casting from identical types
            return true;
        } else if (dst_dt.type_id() != categorical_type_id) {
            return false;
        } else {
            return false; // TODO
        }

    } else {
        return false; // TODO
    }
}

bool categorical_dtype::operator==(const extended_dtype& rhs) const
{
    if (this == &rhs) return true;

    if (rhs.type_id() != categorical_type_id) return false;

    if (static_cast<const categorical_dtype&>(rhs).m_category_index_to_value != m_category_index_to_value) return false;

    if (static_cast<const categorical_dtype&>(rhs).m_value_to_category_index != m_value_to_category_index) return false;

    if (static_cast<const categorical_dtype&>(rhs).m_categories.size() != m_categories.size()) return false;

    if (static_cast<const categorical_dtype&>(rhs).m_category_dtype!= m_category_dtype) return false;

    single_compare_kernel_instance k;
    m_category_dtype.get_single_compare_kernel(k);
    single_compare_operation_t cmp = k.comparisons[equal_id];
    for (uint32_t i = 0; i < m_categories.size(); ++i) {
        if (!cmp(static_cast<const categorical_dtype&>(rhs).m_categories[i], m_categories[i], k.auxdata)) return false;
    }
    return true;

}
