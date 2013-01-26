//
// Copyright (C) 2011-13, DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <cstring>
#include <set>

#include <dynd/auxiliary_data.hpp>
#include <dynd/ndobject_iter.hpp>
#include <dynd/dtypes/categorical_dtype.hpp>
#include <dynd/kernels/assignment_kernels.hpp>
#include <dynd/dtypes/strided_array_dtype.hpp>

using namespace dynd;
using namespace std;

namespace {

    class sorter {
        const char *m_originptr;
        intptr_t m_stride;
        const single_compare_operation_t m_less;
        single_compare_static_data *m_extra;
    public:
        sorter(const char *originptr, intptr_t stride, const single_compare_operation_t less, single_compare_static_data *extra) :
            m_originptr(originptr), m_stride(stride), m_less(less), m_extra(extra) {}
        bool operator()(intptr_t i, intptr_t j) const {
            return m_less(m_originptr + i * m_stride, m_originptr + j * m_stride, m_extra);
        }
    };

    class cmp {
        const single_compare_operation_t m_less;
        single_compare_static_data *m_extra;
    public:
        cmp(const single_compare_operation_t less, single_compare_static_data *extra) :
            m_less(less), m_extra(extra) {}
        bool operator()(const char *a, const char *b) const {
            bool result = m_less(a, b, m_extra);
            return result;
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
            ad.kernel.extra.dst_metadata = extra->dst_metadata;
            ad.kernel.extra.src_metadata = extra->src_metadata;

            uint32_t value = *reinterpret_cast<const uint32_t *>(src);
            const char *src_val = cat->get_category_data_from_value(value);
            ad.kernel.kernel.single(dst, src_val, &ad.kernel.extra);
        }

        static void strided_kernel(char *dst, intptr_t dst_stride, const char *src, intptr_t src_stride,
                        size_t count, unary_kernel_static_data *extra)
        {
            auxdata_storage& ad = get_auxiliary_data<auxdata_storage>(extra->auxdata);
            const categorical_dtype *cat = static_cast<const categorical_dtype *>(ad.cat_dt.extended());
            ad.kernel.extra.dst_metadata = extra->dst_metadata;
            ad.kernel.extra.src_metadata = extra->src_metadata;

            for (size_t i = 0; i != count; ++i) {
                uint32_t value = *reinterpret_cast<const uint32_t *>(src);
                const char *src_val = cat->get_category_data_from_value(value);
                ad.kernel.kernel.single(dst, src_val, &ad.kernel.extra);

                dst += dst_stride;
                src += src_stride;
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

            uint32_t src_val = cat->get_value_from_category(extra->src_metadata, src);
            *reinterpret_cast<uint32_t *>(dst) = src_val;
        }

        static void strided_kernel(char *dst, intptr_t dst_stride, const char *src, intptr_t src_stride, size_t count, unary_kernel_static_data *extra)
        {
            const categorical_dtype *cat = reinterpret_cast<const categorical_dtype *>(
                get_raw_auxiliary_data(extra->auxdata)&~1
            );

            for (size_t i = 0; i != count; ++i) {
                uint32_t src_val = cat->get_value_from_category(extra->src_metadata, src);
                *reinterpret_cast<uint32_t *>(dst) = src_val;

                dst += dst_stride;
                src += src_stride;
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

/** This function converts the set of char* pointers into a strided immutable ndobject of the categories */
static ndobject make_sorted_categories(const set<const char *, cmp>& uniques, const dtype& udtype, const char *metadata)
{
    ndobject categories = make_strided_ndobject(uniques.size(), udtype);
    kernel_instance<unary_operation_pair_t> assign;
    get_dtype_assignment_kernel(udtype, assign);

    intptr_t stride = reinterpret_cast<const strided_array_dtype_metadata *>(categories.get_ndo_meta())->stride;
    char *dst_ptr = categories.get_readwrite_originptr();
    assign.extra.dst_metadata = categories.get_ndo_meta() + sizeof(strided_array_dtype_metadata);
    assign.extra.src_metadata = metadata;
    for (set<const char *, cmp>::const_iterator it = uniques.begin(); it != uniques.end(); ++it) {
        assign.kernel.single(dst_ptr, *it, &assign.extra);
        dst_ptr += stride;
    }
    categories.get_dtype().extended()->metadata_finalize_buffers(categories.get_ndo_meta());
    categories.flag_as_immutable();

    return categories;
}

categorical_dtype::categorical_dtype(const ndobject& categories, bool presorted)
    : base_dtype(categorical_type_id, custom_kind, 4, 4)
{
    if (presorted) {
        // This is construction shortcut, for the case when the categories are already
        // sorted. No validation of this is done, the caller should have ensured it
        // was correct already, typically by construction.
        m_categories = categories.eval_immutable();
        m_category_dtype = m_categories.get_dtype().at(0);

        intptr_t num_categories = categories.get_dim_size();
        m_value_to_category_index.resize(num_categories);
        m_category_index_to_value.resize(num_categories);
        for (size_t i = 0; i != (size_t)num_categories; ++i) {
            m_value_to_category_index[i] = i;
            m_category_index_to_value[i] = i;
        }
        return;
    }

    const dtype& cdt = categories.get_dtype();
    if (cdt.get_type_id() != strided_array_type_id) {
        throw runtime_error("categorical_dtype only supports construction from a strided_array of categories");
    }
    m_category_dtype = categories.get_dtype().at(0);
    if (!m_category_dtype.is_scalar()) {
        throw runtime_error("categorical_dtype only supports construction from a 1-dimensional strided_array of categories");
    }

    intptr_t num_categories = categories.get_dim_size();
    intptr_t categories_stride = reinterpret_cast<const strided_array_dtype_metadata *>(categories.get_ndo_meta())->stride;

    kernel_instance<compare_operations_t> k;
    m_category_dtype.get_single_compare_kernel(k);
    k.extra.src0_metadata = k.extra.src1_metadata = categories.get_ndo_meta() + sizeof(strided_array_dtype_metadata);

    cmp less(k.kernel.ops[compare_operations_t::less_id], &k.extra);
    set<const char *, cmp> uniques(less);

    m_value_to_category_index.resize(num_categories);
    m_category_index_to_value.resize(num_categories);

    // create the mapping from indices of (to be lexicographically sorted) categories to values
    for (size_t i = 0; i != (size_t)num_categories; ++i) {
        m_category_index_to_value[i] = i;
        const char *category_value = categories.get_readonly_originptr() +
                        i * categories_stride;

        if (uniques.find(category_value) == uniques.end()) {
            uniques.insert(category_value);
        } else {
            stringstream ss;
            ss << "categories must be unique: category value ";
            m_category_dtype.print_data(ss, k.extra.src0_metadata, category_value);
            ss << " appears more than once";
            throw std::runtime_error(ss.str());
        }
    }
    // TODO: Putting everything in a set already caused a sort operation to occur,
    //       there's no reason we should need a second sort.
    std::sort(m_category_index_to_value.begin(), m_category_index_to_value.end(),
                    sorter(categories.get_readonly_originptr(), categories_stride, k.kernel.ops[compare_operations_t::less_id], &k.extra));

    // invert the m_category_index_to_value permutation
    for (uint32_t i = 0; i < m_category_index_to_value.size(); ++i) {
        m_value_to_category_index[m_category_index_to_value[i]] = i;
    }

    m_categories = make_sorted_categories(uniques, m_category_dtype, k.extra.src0_metadata);
}

void categorical_dtype::print_data(std::ostream& o, const char *metadata, const char *data) const
{
    uint32_t value = *reinterpret_cast<const uint32_t*>(data);
    if (value < m_value_to_category_index.size()) {
        m_category_dtype.print_data(o, metadata, get_category_data_from_value(value));
    }
    else {
        o << "UNK"; // TODO better outpout?
    }
}


void categorical_dtype::print_dtype(std::ostream& o) const
{
    size_t category_count = get_category_count();
    const char *metadata = m_categories.get_ndo_meta() + sizeof(strided_array_dtype_metadata);

    o << "categorical<" << m_category_dtype;
    o << ", [";
    m_category_dtype.print_data(o, metadata, get_category_data_from_value(0));
    for (size_t i = 1; i != category_count; ++i) {
        o << ", ";
        m_category_dtype.print_data(o, metadata, get_category_data_from_value(i));
    }
    o << "]>";
}

void dynd::categorical_dtype::get_shape(size_t i, intptr_t *out_shape) const
{
    if (!m_category_dtype.is_builtin()) {
        m_category_dtype.extended()->get_shape(i, out_shape);
    }
}

uint32_t categorical_dtype::get_value_from_category(const char *category_metadata, const char *category_data) const
{
    intptr_t i = dynd::binary_search(m_categories, category_metadata, category_data);
    if (i < 0) {
        stringstream ss;
        m_category_dtype.print_data(ss, category_metadata, category_data);
        throw std::runtime_error("Unknown category: '" + ss.str() + "'");
    } else {
        return m_category_index_to_value[i];
    }
}

uint32_t categorical_dtype::get_value_from_category(const ndobject& category) const
{
    if (category.get_dtype() == m_category_dtype) {
        // If the dtype is right, get the category value directly
        return get_value_from_category(category.get_ndo_meta(), category.get_readonly_originptr());
    } else {
        // Otherwise convert to the correct dtype, then get the category value
        ndobject c(m_category_dtype);
        c.val_assign(category);
        return get_value_from_category(c.get_ndo_meta(), c.get_readonly_originptr());
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
                            category_to_categorical_assign::strided_kernel);
            make_raw_auxiliary_data(out_kernel.extra.auxdata, reinterpret_cast<uintptr_t>(this));
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
                            categorical_to_other_assign::strided_kernel);
            make_auxiliary_data<categorical_to_other_assign::auxdata_storage>(out_kernel.extra.auxdata);
            categorical_to_other_assign::auxdata_storage& ad =
                        out_kernel.extra.auxdata.get<categorical_to_other_assign::auxdata_storage>();
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
    if (this == &rhs)
        return true;
    if (rhs.get_type_id() != categorical_type_id)
        return false;
    if (!m_categories.equals_exact(static_cast<const categorical_dtype&>(rhs).m_categories))
        return false;
    if (static_cast<const categorical_dtype&>(rhs).m_category_index_to_value != m_category_index_to_value)
        return false;
    if (static_cast<const categorical_dtype&>(rhs).m_value_to_category_index != m_value_to_category_index)
        return false;

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

void categorical_dtype::metadata_default_construct(char *DYND_UNUSED(metadata),
                int DYND_UNUSED(ndim), const intptr_t* DYND_UNUSED(shape)) const
{
    // Data is stored as int32, no metadata to process
}

void categorical_dtype::metadata_copy_construct(char *DYND_UNUSED(dst_metadata),
                const char *DYND_UNUSED(src_metadata),
                memory_block_data *DYND_UNUSED(embedded_reference)) const
{
    // Data is stored as int32, no metadata to process
}

void categorical_dtype::metadata_destruct(char *DYND_UNUSED(metadata)) const
{
    // Data is stored as int32, no metadata to process
}

void categorical_dtype::metadata_debug_print(const char *DYND_UNUSED(metadata),
                std::ostream& DYND_UNUSED(o), const std::string& DYND_UNUSED(indent)) const
{
    // Data is stored as int32, no metadata to process
}

dtype dynd::factor_categorical_dtype(const ndobject& values)
{
    kernel_instance<compare_operations_t> k;

    ndobject_iter<1, 0> iter(values);

    iter.get_uniform_dtype().get_single_compare_kernel(k);
    k.extra.src0_metadata = iter.metadata();
    k.extra.src1_metadata = iter.metadata();
    cmp less(k.kernel.ops[compare_operations_t::less_id], &k.extra);
    set<const char *, cmp> uniques(less);

    if (!iter.empty()) {
        do {
            if (uniques.find(iter.data()) == uniques.end()) {
                uniques.insert(iter.data());
            }
        } while (iter.next());
    }

    // Copy the values (now sorted and unique) into a new ndobject
    ndobject categories = make_sorted_categories(uniques, iter.get_uniform_dtype(), iter.metadata());

    return dtype(new categorical_dtype(categories, true), false);
}

