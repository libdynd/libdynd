//
// Copyright (C) 2011-12, Dynamic NDArray Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <cstring>

#include <dnd/auxiliary_data.hpp>
#include <dnd/dtypes/categorical_dtype.hpp>
#include <dnd/kernels/assignment_kernels.hpp>
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
        bool operator()(intptr_t i, intptr_t j) {
            return m_less(m_categories[i], m_categories[j], m_auxdata);
        }
    };

    class cmp{
        const single_compare_operation_t m_less;
        const auxiliary_data& m_auxdata;
    public:
        cmp(const single_compare_operation_t less, const auxiliary_data& auxdata) :
            m_less(less), m_auxdata(auxdata) {}
        bool operator()(const char *a, const char *b) {
            return m_less(a, b, m_auxdata);
        }
    };

    struct assign_to_same_category_type {
        static void general_kernel(char *dst, intptr_t dst_stride, const char *src, intptr_t src_stride,
                            intptr_t count, const AuxDataBase *auxdata)
        {
            categorical_dtype *cat = reinterpret_cast<categorical_dtype *>(
                get_raw_auxiliary_data(auxdata)&~1
            );

            // TODO handle non POD or throw
            size_t N = cat->m_category_dtype.element_size();
            for (intptr_t i = 0; i < count; ++i) {
                uint32_t value = *reinterpret_cast<const uint32_t *>(src);
                const char *src_val = cat->m_categories[cat->m_value_to_category_index[value]];
                memcpy(dst, src_val, N);

                dst += dst_stride;
                src += src_stride;
            }
        }

        static void scalar_kernel(char *dst, intptr_t DND_UNUSED(dst_stride), const char *src, intptr_t DND_UNUSED(src_stride),
                            intptr_t DND_UNUSED(count), const AuxDataBase *auxdata)
        {
            categorical_dtype *cat = reinterpret_cast<categorical_dtype *>(
                get_raw_auxiliary_data(auxdata)&~1
            );

            // TODO handle non POD or throw
            size_t N = cat->m_category_dtype.element_size();
            uint32_t value = *reinterpret_cast<const uint32_t *>(src);
            const char *src_val = cat->m_categories[cat->m_value_to_category_index[value]];
            memcpy(dst, src_val, N);
        }

        static void scalar_to_contiguous_kernel(char *dst, intptr_t DND_UNUSED(dst_stride), const char *src, intptr_t DND_UNUSED(src_stride),
                            intptr_t count, const AuxDataBase *auxdata)
        {
            categorical_dtype *cat = reinterpret_cast<categorical_dtype *>(
                get_raw_auxiliary_data(auxdata)&~1
            );

            // TODO handle non POD or throw
            size_t N = cat->m_category_dtype.element_size();
            uint32_t value = *reinterpret_cast<const uint32_t *>(src);
            const char *src_val = cat->m_categories[cat->m_value_to_category_index[value]];
            for (intptr_t i = 0; i < count; ++i) {
                memcpy(dst, src_val, N);

                dst += N;
            }
        }
    };

    static specialized_unary_operation_table_t assign_to_same_category_type_specializations = {
        assign_to_same_category_type::general_kernel,
        assign_to_same_category_type::scalar_kernel,
        assign_to_same_category_type::general_kernel,
        assign_to_same_category_type::scalar_to_contiguous_kernel
    };

    struct assign_from_same_category_type {
        static void general_kernel(char *dst, intptr_t dst_stride, const char *src, intptr_t src_stride,
                            intptr_t count, const AuxDataBase *auxdata)
        {
            categorical_dtype *cat = reinterpret_cast<categorical_dtype *>(
                get_raw_auxiliary_data(auxdata)&~1
            );

            for (intptr_t i = 0; i < count; ++i) {
                uint32_t src_val = cat->get_value_from_category(src);
                *reinterpret_cast<uint32_t *>(dst) = src_val;

                dst += dst_stride;
                src += src_stride;
            }
        }

        static void scalar_kernel(char *dst, intptr_t DND_UNUSED(dst_stride), const char *src, intptr_t DND_UNUSED(src_stride),
                            intptr_t DND_UNUSED(count), const AuxDataBase *auxdata)
        {
            categorical_dtype *cat = reinterpret_cast<categorical_dtype *>(
                get_raw_auxiliary_data(auxdata)&~1
            );
            size_t N = cat->element_size();
            uint32_t src_val = cat->get_value_from_category(src);
            memcpy(dst, reinterpret_cast<const char *>(&src_val), N);
        }

        static void scalar_to_contiguous_kernel(char *dst, intptr_t DND_UNUSED(dst_stride), const char *src, intptr_t DND_UNUSED(src_stride),
                            intptr_t count, const AuxDataBase *auxdata)
        {
            categorical_dtype *cat = reinterpret_cast<categorical_dtype *>(
                get_raw_auxiliary_data(auxdata)&~1
            );

            size_t N = cat->element_size();
            uint32_t src_val = cat->get_value_from_category(src);
            for (intptr_t i = 0; i < count; ++i) {
                memcpy(dst, reinterpret_cast<const char *>(&src_val), N);

                dst += N;
            }
        }
    };

    static specialized_unary_operation_table_t assign_from_same_category_type_specializations = {
        assign_from_same_category_type::general_kernel,
        assign_from_same_category_type::scalar_kernel,
        assign_from_same_category_type::general_kernel,
        assign_from_same_category_type::scalar_to_contiguous_kernel
    };

    // struct assign_from_commensurate_category {
    //     static void general_kernel(char *dst, intptr_t dst_stride, const char *src, intptr_t src_stride,
    //                         intptr_t count, const AuxDataBase *auxdata)
    //     {
    //         categorical_dtype *cat = reinterpret_cast<categorical_dtype *>(
    //             get_raw_auxiliary_data(auxdata)&~1
    //         );
    //     }

    //     static void scalar_kernel(char *dst, intptr_t DND_UNUSED(dst_stride), const char *src, intptr_t DND_UNUSED(src_stride),
    //                         intptr_t, const AuxDataBase *auxdata)
    //     {
    //         categorical_dtype *cat = reinterpret_cast<categorical_dtype *>(
    //             get_raw_auxiliary_data(auxdata)&~1
    //         );
    //     }

    //     static void contiguous_kernel(char *dst, intptr_t DND_UNUSED(dst_stride), const char *src, intptr_t DND_UNUSED(src_stride),
    //                         intptr_t count, const AuxDataBase *auxdata)
    //     {
    //         categorical_dtype *cat = reinterpret_cast<categorical_dtype *>(
    //             get_raw_auxiliary_data(auxdata)&~1
    //         );
    //     }

    //     static void scalar_to_contiguous_kernel(char *dst, intptr_t DND_UNUSED(dst_stride), const char *src, intptr_t DND_UNUSED(src_stride),
    //                         intptr_t count, const AuxDataBase *auxdata)
    //     {
    //         categorical_dtype *cat = reinterpret_cast<categorical_dtype *>(
    //             get_raw_auxiliary_data(auxdata)&~1
    //         );
    //     }
    // };

    // static specialized_unary_operation_table_t assign_from_commensurate_category_specializations = {
    //     assign_from_commensurate_category::general_kernel,
    //     assign_from_commensurate_category::scalar_kernel,
    //     assign_from_commensurate_category::contiguous_kernel,
    //     assign_from_commensurate_category::scalar_to_contiguous_kernel
    // };

} // anoymous namespace


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
    for (vector<const char *>::iterator it = m_categories.begin(); it != m_categories.end(); ++it) {
        delete[] *it;
    }

}

void categorical_dtype::print_element(std::ostream& o, const char *data) const
{
    uint32_t value = *reinterpret_cast<const uint32_t*>(data);
    if (value < m_value_to_category_index.size()) {
        m_category_dtype.print_element(o, m_categories[m_value_to_category_index[value]]);
    }
    else {
        o << "UNK"; // TODO better outpout?
    }
}


void categorical_dtype::print_dtype(std::ostream& o) const
{
    o << "categorical(";
    m_category_dtype.print_element(o, m_categories[m_value_to_category_index[0]]);
    for (uint32_t i = 1; i < m_categories.size(); ++i) {
        o << ", ";
        m_category_dtype.print_element(o, m_categories[m_value_to_category_index[i]]);
    }
    o << ")";
}

uint32_t categorical_dtype::get_value_from_category(const char *category) const
{
    single_compare_kernel_instance k;
    m_category_dtype.get_single_compare_kernel(k);
    pair<vector<const char *>::const_iterator,vector<const char *>::const_iterator> bounds;
    bounds = equal_range(
        m_categories.begin(), m_categories.end(), category, cmp(k.comparisons[less_id], k.auxdata)
    );
    if (bounds.first == m_categories.end() || bounds.first == bounds.second) {
        stringstream ss;
        m_category_dtype.print_element(ss, category);
        throw std::runtime_error("Unknown category: '" + ss.str() + "'");
    }
    else {
        return m_category_index_to_value[bounds.first-m_categories.begin()];
    }

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

void categorical_dtype::get_dtype_assignment_kernel(const dtype& dst_dt, const dtype& src_dt,
                    assign_error_mode DND_UNUSED(errmode),
                    unary_specialization_kernel_instance& out_kernel) const {

    // POD copy already handled if category mappings are identical

    if (this == dst_dt.extended()) {
        // try to assign from another categorical dtype if it can be mapped
        if (src_dt.type_id() == categorical_type_id) {
            //out_kernel.specializations = assign_from_commensurate_category_specializations;
            // TODO auxdata
        }
        // assign from the same category value dtype
        else if (src_dt.value_dtype() == m_category_dtype) {
            out_kernel.specializations = assign_from_same_category_type_specializations;
            make_raw_auxiliary_data(out_kernel.auxdata, reinterpret_cast<uintptr_t>(this));
        }
        else {
            stringstream ss;
            ss << "Cannot assign categorical type '";
            this->print_dtype(ss);
            //ss << "' from type '";
            //src_dt.extended()->print_dtype(ss);
            //ss << "'";
            throw std::runtime_error(ss.str()); // TODO better message
        }
    }
    else {
        // assign to the same category value dtype
        if (dst_dt.value_dtype() == m_category_dtype) {
            out_kernel.specializations = assign_to_same_category_type_specializations;
            make_raw_auxiliary_data(out_kernel.auxdata, reinterpret_cast<uintptr_t>(this));
        }
        else {
            stringstream ss;
            ss << "Cannot assign categorical type '";
            this->print_dtype(ss);
            //ss << "' to type '";
            //dst_dt.extended()->print_dtype(ss);
            //ss << "'";
            throw std::runtime_error(ss.str()); // TODO better message
        }

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
    single_compare_operation_t cmp_equal = k.comparisons[equal_id];
    for (uint32_t i = 0; i < m_categories.size(); ++i) {
        if (!cmp_equal(static_cast<const categorical_dtype&>(rhs).m_categories[i], m_categories[i], k.auxdata)) return false;
    }
    return true;

}


