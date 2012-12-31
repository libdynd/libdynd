//
// Copyright (C) 2011-12, Dynamic NDArray Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <cstring>
#include <set>

#include <dynd/auxiliary_data.hpp>
#include <dynd/ndobject_iter.hpp>
#include <dynd/dtypes/categorical_dtype.hpp>
#include <dynd/kernels/assignment_kernels.hpp>
#include <dynd/kernels/single_compare_kernel_instance.hpp>
#include <dynd/dtypes/strided_array_dtype.hpp>

using namespace dynd;
using namespace std;

namespace {

    class sorter {
        const vector<char *>& m_categories;
        const single_compare_operation_t m_less;
        const auxiliary_data& m_auxdata;
    public:
        sorter(vector<char *>& values, const single_compare_operation_t less, const auxiliary_data& auxdata) :
            m_categories(values), m_less(less), m_auxdata(auxdata) {}
        bool operator()(intptr_t i, intptr_t j) const {
            return m_less(m_categories[i], m_categories[j], m_auxdata);
        }
    };

    class cmp {
        const single_compare_operation_t m_less;
        const auxiliary_data& m_auxdata;
    public:
        cmp(const single_compare_operation_t less, const auxiliary_data& auxdata) :
            m_less(less), m_auxdata(auxdata) {}
        bool operator()(const char *a, const char *b) const {
            return m_less(a, b, m_auxdata);
        }
    };

    struct categorical_to_other_assign {
        // Assign from a categorical dtype to some other dtype
        struct auxdata_storage {
            kernel_instance<unary_operation_pair_t> kernel;
            dtype cat_dt;
            size_t dst_size;
        };

        static void single_kernel(char *dst, const char *src, unary_kernel_static_data *extra)
        {
            auxdata_storage& ad = get_auxiliary_data<auxdata_storage>(extra->auxdata);
            const categorical_dtype *cat = static_cast<const categorical_dtype *>(ad.cat_dt.extended());
            unary_kernel_static_data kernel_extra(ad.kernel.auxdata, extra->dst_metadata, extra->src_metadata);

            uint32_t value = *reinterpret_cast<const uint32_t *>(src);
            const char *src_val = cat->get_category_from_value(value);
            ad.kernel.kernel.single(dst, src_val, &kernel_extra);
        }

        static void contig_kernel(char *dst, const char *src, size_t count, unary_kernel_static_data *extra)
        {
            auxdata_storage& ad = get_auxiliary_data<auxdata_storage>(extra->auxdata);
            const categorical_dtype *cat = static_cast<const categorical_dtype *>(ad.cat_dt.extended());
            size_t dst_size = ad.dst_size;
            unary_kernel_static_data kernel_extra;
            kernel_extra.auxdata = ad.kernel.auxdata;
            kernel_extra.dst_metadata = extra->dst_metadata;
            kernel_extra.src_metadata = extra->src_metadata;

            const uint32_t *src_vals = reinterpret_cast<const uint32_t *>(src);
            for (size_t i = 0; i != count; ++i) {
                uint32_t value = *src_vals;
                const char *src_val = cat->get_category_from_value(value);
                ad.kernel.kernel.single(dst, src_val, &kernel_extra);

                dst += dst_size;
                ++src_vals;
            }
        }
    };

    struct category_to_categorical_assign {
        // Assign from an input matching the category dtype to a categorical type
        static void single_kernel(char *dst, const char *src, unary_kernel_static_data *extra)
        {
            const categorical_dtype *cat = reinterpret_cast<const categorical_dtype *>(
                get_raw_auxiliary_data(extra->auxdata)&~1
            );

            uint32_t src_val = cat->get_value_from_category(src);
            *reinterpret_cast<uint32_t *>(dst) = src_val;
        }

        static void contig_kernel(char *dst, const char *src, size_t count, unary_kernel_static_data *extra)
        {
            const categorical_dtype *cat = reinterpret_cast<const categorical_dtype *>(
                get_raw_auxiliary_data(extra->auxdata)&~1
            );
            size_t src_size = cat->get_category_dtype().get_data_size();

            uint32_t *dst_vals = reinterpret_cast<uint32_t *>(dst);
            for (size_t i = 0; i != count; ++i) {
                uint32_t src_val = cat->get_value_from_category(src);
                *dst_vals = src_val;

                ++dst;
                src += src_size;
            }
        }
    };

    // struct assign_from_commensurate_category {
    //     static void general_kernel(char *dst, intptr_t dst_stride, const char *src, intptr_t src_stride,
    //                         intptr_t count, const AuxDataBase *auxdata)
    //     {
    //         categorical_dtype *cat = reinterpret_cast<categorical_dtype *>(
    //             get_raw_auxiliary_data(auxdata)&~1
    //         );
    //     }

    //     static void scalar_kernel(char *dst, intptr_t DYND_UNUSED(dst_stride), const char *src, intptr_t DYND_UNUSED(src_stride),
    //                         intptr_t, const AuxDataBase *auxdata)
    //     {
    //         categorical_dtype *cat = reinterpret_cast<categorical_dtype *>(
    //             get_raw_auxiliary_data(auxdata)&~1
    //         );
    //     }

    //     static void contiguous_kernel(char *dst, intptr_t DYND_UNUSED(dst_stride), const char *src, intptr_t DYND_UNUSED(src_stride),
    //                         intptr_t count, const AuxDataBase *auxdata)
    //     {
    //         categorical_dtype *cat = reinterpret_cast<categorical_dtype *>(
    //             get_raw_auxiliary_data(auxdata)&~1
    //         );
    //     }

    //     static void scalar_to_contiguous_kernel(char *dst, intptr_t DYND_UNUSED(dst_stride), const char *src, intptr_t DYND_UNUSED(src_stride),
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

categorical_dtype::categorical_dtype(const ndobject& categories)
    : base_dtype(categorical_type_id, custom_kind, 4, 4)
{
    const dtype& cdt = categories.get_dtype();
    if (cdt.get_type_id() != strided_array_type_id) {
        throw runtime_error("categorical_dtype only supports construction from a strided_array of categories");
    }
    m_category_dtype = categories.get_dtype().at(0);
    if (!m_category_dtype.is_scalar()) {
        throw runtime_error("categorical_dtype only supports construction from a 1-dimensional strided_array of categories");
    }
    if (m_category_dtype.get_memory_management() != pod_memory_management) {
        stringstream ss;
        ss << "categorical_dtype does not yet support " << m_category_dtype << " as the category because it is not POD";
        throw runtime_error(ss.str());
    }

    intptr_t num_categories;
    categories.get_shape(&num_categories);

    single_compare_kernel_instance k;
    m_category_dtype.get_single_compare_kernel(k);

    cmp less(k.comparisons[less_id], k.auxdata);
    set<char *, cmp> uniques(less);

    m_categories.resize(num_categories);
    m_value_to_category_index.resize(num_categories);
    m_category_index_to_value.resize(num_categories);

    // create the mapping from indices of (to be lexicographically sorted) categories to values
    vector<char *> categories_user_order(num_categories);
    for (uint32_t i = 0; i < m_category_index_to_value.size(); ++i) {
        m_category_index_to_value[i] = i;
        categories_user_order[i] = new char[m_category_dtype.get_data_size()];
        memcpy(categories_user_order[i], categories.at(i).get_readonly_originptr(), m_category_dtype.get_data_size());

        if (uniques.find(categories_user_order[i]) == uniques.end()) {
            uniques.insert(categories_user_order[i]);
        } else {
            stringstream ss;
            ss << "categories must be unique: category value ";
            m_category_dtype.print_data(ss, NULL, categories_user_order[i]);
            ss << " appears more than once";
            throw std::runtime_error(ss.str());
        }
    }
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

void categorical_dtype::print_data(std::ostream& o, const char *metadata, const char *data) const
{
    uint32_t value = *reinterpret_cast<const uint32_t*>(data);
    if (value < m_value_to_category_index.size()) {
        m_category_dtype.print_data(o, metadata, m_categories[m_value_to_category_index[value]]);
    }
    else {
        o << "UNK"; // TODO better outpout?
    }
}


void categorical_dtype::print_dtype(std::ostream& o) const
{
    o << "categorical<" << m_category_dtype;
    o << ", [";
    m_category_dtype.print_data(o, NULL, m_categories[m_value_to_category_index[0]]); // TODO: ndobject metadata
    for (uint32_t i = 1; i < m_categories.size(); ++i) {
        o << ", ";
        m_category_dtype.print_data(o, NULL, m_categories[m_value_to_category_index[i]]); // TODO: ndobject metadata
    }
    o << "]>";
}

dtype dynd::categorical_dtype::apply_linear_index(int nindices, const irange *indices, int current_i, const dtype& root_dt) const
{
    if (nindices == 0) {
        return dtype(this, true);
    } else {
        return m_category_dtype.apply_linear_index(nindices, indices, current_i, root_dt);
    }
}

void dynd::categorical_dtype::get_shape(size_t i, intptr_t *out_shape) const
{
    if (!m_category_dtype.is_builtin()) {
        m_category_dtype.extended()->get_shape(i, out_shape);
    }
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
        m_category_dtype.print_data(ss, NULL, category); // TODO: ndobject metadata
        throw std::runtime_error("Unknown category: '" + ss.str() + "'");
    }
    else {
        return m_category_index_to_value[bounds.first-m_categories.begin()];
    }

}

bool categorical_dtype::is_lossless_assignment(const dtype& dst_dt, const dtype& src_dt) const
{
    if (dst_dt.extended() == this) {
        if (src_dt.extended() == this) {
            // Casting from identical types
            return true;
        } else  {
            return false; // TODO
        }

    } else {
        return ::is_lossless_assignment(dst_dt, m_category_dtype); // TODO
    }
}

void categorical_dtype::get_dtype_assignment_kernel(const dtype& dst_dt, const dtype& src_dt,
                    assign_error_mode errmode,
                    kernel_instance<unary_operation_pair_t>& out_kernel) const
{
    // POD copy already handled if category mappings are identical

    if (this == dst_dt.extended()) {
        // try to assign from another categorical dtype if it can be mapped
        if (src_dt.get_type_id() == categorical_type_id) {
            //out_kernel.specializations = assign_from_commensurate_category_specializations;
            // TODO auxdata
            throw std::runtime_error("assignment between different categorical dtypes isn't supported yet");
        }
        // assign from the same category value dtype
        else if (src_dt.value_dtype() == m_category_dtype) {
            out_kernel.kernel = unary_operation_pair_t(category_to_categorical_assign::single_kernel,
                            category_to_categorical_assign::contig_kernel);
            make_raw_auxiliary_data(out_kernel.auxdata, reinterpret_cast<uintptr_t>(this));
        }
        else {
            stringstream ss;
            ss << "assignment from " << src_dt << " to " << dst_dt << " is not implemented yet";
            throw runtime_error(ss.str());
        }
    }
    else {
        if (dst_dt.value_dtype().get_type_id() != categorical_type_id) {
            out_kernel.kernel = unary_operation_pair_t(categorical_to_other_assign::single_kernel,
                            categorical_to_other_assign::contig_kernel);
            make_auxiliary_data<categorical_to_other_assign::auxdata_storage>(out_kernel.auxdata);
            categorical_to_other_assign::auxdata_storage& ad =
                        out_kernel.auxdata.get<categorical_to_other_assign::auxdata_storage>();
            ad.cat_dt = src_dt;
            ad.dst_size = dst_dt.get_data_size();
            ::get_dtype_assignment_kernel(dst_dt, m_category_dtype, errmode, NULL, ad.kernel);
        }
        else {
            stringstream ss;
            ss << "assignment from " << src_dt << " to " << dst_dt << " is not implemented yet";
            throw runtime_error(ss.str());
        }

    }

}

bool categorical_dtype::operator==(const base_dtype& rhs) const
{
    if (this == &rhs) return true;

    if (rhs.get_type_id() != categorical_type_id) return false;

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

size_t categorical_dtype::get_metadata_size() const
{
    if (!m_category_dtype.is_builtin()) {
        return m_category_dtype.extended()->get_metadata_size();
    } else {
        return 0;
    }
}

void categorical_dtype::metadata_default_construct(char *metadata, int ndim, const intptr_t* shape) const
{
    if (!m_category_dtype.is_builtin()) {
        m_category_dtype.extended()->metadata_default_construct(metadata, ndim, shape);
    }
}

void categorical_dtype::metadata_copy_construct(char *dst_metadata, const char *src_metadata, memory_block_data *embedded_reference) const
{
    if (!m_category_dtype.is_builtin()) {
        m_category_dtype.extended()->metadata_copy_construct(dst_metadata, src_metadata, embedded_reference);
    }
}

void categorical_dtype::metadata_destruct(char *metadata) const
{
    if (!m_category_dtype.is_builtin()) {
        m_category_dtype.extended()->metadata_destruct(metadata);
    }
}

void categorical_dtype::metadata_debug_print(const char *metadata, std::ostream& o, const std::string& indent) const
{
    if (!m_category_dtype.is_builtin()) {
        m_category_dtype.extended()->metadata_debug_print(metadata, o, indent);
    }
}

dtype dynd::factor_categorical_dtype(const ndobject& values)
{
    single_compare_kernel_instance k;

    ndobject_iter<1, 0> iter(values);

    iter.get_uniform_dtype().get_single_compare_kernel(k);
    cmp less(k.comparisons[less_id], k.auxdata);
    set<const char *, cmp> uniques(less);

    if (!iter.empty()) {
        do {
            if (uniques.find(iter.data()) == uniques.end()) {
                uniques.insert(iter.data());
            }
        } while (iter.next());
    }

    // TODO: This voodoo needs to be simplified so it's easy to understand what's going on

    // Copy the values (now sorted and unique) into a new ndobject
    ndobject categories = make_strided_ndobject(uniques.size(), iter.get_uniform_dtype());
    kernel_instance<unary_operation_pair_t> kernel;
    get_dtype_assignment_kernel(iter.get_uniform_dtype(), kernel);

    intptr_t stride = reinterpret_cast<const strided_array_dtype_metadata *>(categories.get_ndo_meta())->stride;
    char *dst_ptr = categories.get_readwrite_originptr();
    uint32_t i = 0;
    unary_kernel_static_data extra(kernel.auxdata, categories.get_ndo_meta() + sizeof(strided_array_dtype_metadata),
                    iter.metadata());
    for (set<const char *, cmp>::const_iterator it = uniques.begin(); it != uniques.end(); ++it) {
        kernel.kernel.single(dst_ptr, *it, &extra);
        ++i;
        dst_ptr += stride;
    }

    return make_categorical_dtype(categories);
}

