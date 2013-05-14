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
#include <dynd/kernels/comparison_kernels.hpp>
#include <dynd/dtypes/strided_dim_dtype.hpp>
#include <dynd/dtypes/convert_dtype.hpp>
#include <dynd/gfunc/make_callable.hpp>

using namespace dynd;
using namespace std;

namespace {

    class sorter {
        const char *m_originptr;
        intptr_t m_stride;
        const binary_single_predicate_t m_less;
        kernel_data_prefix *m_extra;
    public:
        sorter(const char *originptr, intptr_t stride,
                        const binary_single_predicate_t less, kernel_data_prefix *extra) :
            m_originptr(originptr), m_stride(stride), m_less(less), m_extra(extra) {}
        bool operator()(intptr_t i, intptr_t j) const {
            return m_less(m_originptr + i * m_stride, m_originptr + j * m_stride, m_extra) != 0;
        }
    };

    class cmp {
        const binary_single_predicate_t m_less;
        kernel_data_prefix *m_extra;
    public:
        cmp(const binary_single_predicate_t less, kernel_data_prefix *extra) :
            m_less(less), m_extra(extra) {}
        bool operator()(const char *a, const char *b) const {
            bool result = m_less(a, b, m_extra) != 0;
            return result;
        }
    };

    // Assign from a categorical dtype to some other dtype
    struct categorical_to_other_kernel_extra {
        typedef categorical_to_other_kernel_extra extra_type;

        kernel_data_prefix base;
        const categorical_dtype *src_cat_dt;

        template<typename UIntType>
        inline static void single(char *dst, const char *src, kernel_data_prefix *extra)
        {
            extra_type *e = reinterpret_cast<extra_type *>(extra);
            kernel_data_prefix *echild = &(e + 1)->base;
            unary_single_operation_t opchild = echild->get_function<unary_single_operation_t>();

            uint32_t value = *reinterpret_cast<const UIntType *>(src);
            const char *src_val = e->src_cat_dt->get_category_data_from_value(value);
            opchild(dst, src_val, echild);
        }

        // Some compilers are finicky about getting single<T> as a function pointer, so this...
        static void single_uint8(char *dst, const char *src, kernel_data_prefix *extra) {
            single<uint8_t>(dst, src, extra);
        }
        static void single_uint16(char *dst, const char *src, kernel_data_prefix *extra) {
            single<uint16_t>(dst, src, extra);
        }
        static void single_uint32(char *dst, const char *src, kernel_data_prefix *extra) {
            single<uint32_t>(dst, src, extra);
        }

        static void destruct(kernel_data_prefix *extra)
        {
            extra_type *e = reinterpret_cast<extra_type *>(extra);
            if (e->src_cat_dt != NULL) {
                base_dtype_decref(e->src_cat_dt);
            }
            kernel_data_prefix *echild = &(e + 1)->base;
            if (echild->destructor) {
                echild->destructor(echild);
            }
        }
    };

    struct category_to_categorical_kernel_extra {
        typedef category_to_categorical_kernel_extra extra_type;

        kernel_data_prefix base;
        const categorical_dtype *dst_cat_dt;
        const char *src_metadata;

        // Assign from an input matching the category dtype to a categorical type
        template<typename UIntType>
        inline static void single(char *dst, const char *src, kernel_data_prefix *extra)
        {
            extra_type *e = reinterpret_cast<extra_type *>(extra);
            uint32_t src_val = e->dst_cat_dt->get_value_from_category(e->src_metadata, src);
            *reinterpret_cast<UIntType *>(dst) = src_val;
        }

        // Some compilers are finicky about getting single<T> as a function pointer, so this...
        static void single_uint8(char *dst, const char *src, kernel_data_prefix *extra) {
            single<uint8_t>(dst, src, extra);
        }
        static void single_uint16(char *dst, const char *src, kernel_data_prefix *extra) {
            single<uint16_t>(dst, src, extra);
        }
        static void single_uint32(char *dst, const char *src, kernel_data_prefix *extra) {
            single<uint32_t>(dst, src, extra);
        }

        static void destruct(kernel_data_prefix *extra)
        {
            extra_type *e = reinterpret_cast<extra_type *>(extra);
            if (e->dst_cat_dt != NULL) {
                base_dtype_decref(e->dst_cat_dt);
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
    assignment_kernel k;
    make_assignment_kernel(&k, 0,
                    udtype, categories.get_ndo_meta() + sizeof(strided_dim_dtype_metadata),
                    udtype, metadata,
                    kernel_request_single, assign_error_default,
                    &eval::default_eval_context);

    intptr_t stride = reinterpret_cast<const strided_dim_dtype_metadata *>(categories.get_ndo_meta())->stride;
    char *dst_ptr = categories.get_readwrite_originptr();
    for (set<const char *, cmp>::const_iterator it = uniques.begin(); it != uniques.end(); ++it) {
        k(dst_ptr, *it);
        dst_ptr += stride;
    }
    categories.get_dtype().extended()->metadata_finalize_buffers(categories.get_ndo_meta());
    categories.flag_as_immutable();

    return categories;
}

categorical_dtype::categorical_dtype(const ndobject& categories, bool presorted)
    : base_dtype(categorical_type_id, custom_kind, 4, 4, dtype_flag_scalar, 0, 0)
{
    intptr_t category_count;
    if (presorted) {
        // This is construction shortcut, for the case when the categories are already
        // sorted. No validation of this is done, the caller should have ensured it
        // was correct already, typically by construction.
        m_categories = categories.eval_immutable();
        m_category_dtype = m_categories.get_dtype().at(0);

        category_count = categories.get_dim_size();
        m_value_to_category_index.resize(category_count);
        m_category_index_to_value.resize(category_count);
        for (size_t i = 0; i != (size_t)category_count; ++i) {
            m_value_to_category_index[i] = i;
            m_category_index_to_value[i] = i;
        }

    } else {
        // Process the categories array to make sure it's valid
        const dtype& cdt = categories.get_dtype();
        if (cdt.get_type_id() != strided_dim_type_id) {
            throw runtime_error("categorical_dtype only supports construction from a strided array of categories");
        }
        m_category_dtype = categories.get_dtype().at(0);
        if (!m_category_dtype.is_scalar()) {
            throw runtime_error("categorical_dtype only supports construction from a 1-dimensional strided array of categories");
        }

        category_count = categories.get_dim_size();
        intptr_t categories_stride = reinterpret_cast<const strided_dim_dtype_metadata *>(categories.get_ndo_meta())->stride;

        const char *categories_element_metadata = categories.get_ndo_meta() + sizeof(strided_dim_dtype_metadata);
        comparison_kernel k;
        ::make_comparison_kernel(&k, 0,
                        m_category_dtype, categories_element_metadata,
                        m_category_dtype, categories_element_metadata,
                        comparison_type_sorting_less, &eval::default_eval_context);

        cmp less(k.get_function(), k.get());
        set<const char *, cmp> uniques(less);

        m_value_to_category_index.resize(category_count);
        m_category_index_to_value.resize(category_count);

        // create the mapping from indices of (to be lexicographically sorted) categories to values
        for (size_t i = 0; i != (size_t)category_count; ++i) {
            m_category_index_to_value[i] = i;
            const char *category_value = categories.get_readonly_originptr() +
                            i * categories_stride;

            if (uniques.find(category_value) == uniques.end()) {
                uniques.insert(category_value);
            } else {
                stringstream ss;
                ss << "categories must be unique: category value ";
                m_category_dtype.print_data(ss, categories_element_metadata, category_value);
                ss << " appears more than once";
                throw std::runtime_error(ss.str());
            }
        }
        // TODO: Putting everything in a set already caused a sort operation to occur,
        //       there's no reason we should need a second sort.
        std::sort(m_category_index_to_value.begin(), m_category_index_to_value.end(),
                        sorter(categories.get_readonly_originptr(), categories_stride,
                            k.get_function(), k.get()));

        // invert the m_category_index_to_value permutation
        for (uint32_t i = 0; i < m_category_index_to_value.size(); ++i) {
            m_value_to_category_index[m_category_index_to_value[i]] = i;
        }

        m_categories = make_sorted_categories(uniques, m_category_dtype,
                        categories_element_metadata);
    }

    // Use the number of categories to set which underlying integer storage to use
    if (category_count <= 256) {
        m_category_int_dtype = make_dtype<uint8_t>();
    } else if (category_count <= 32768) {
        m_category_int_dtype = make_dtype<uint16_t>();
    } else {
        m_category_int_dtype = make_dtype<uint32_t>();
    }
    m_members.data_size = m_category_int_dtype.get_data_size();
    m_members.alignment = (uint8_t)m_category_int_dtype.get_alignment();
}

void categorical_dtype::print_data(std::ostream& o, const char *metadata, const char *data) const
{
    uint32_t value;
    switch (m_category_int_dtype.get_type_id()) {
        case uint8_type_id:
            value = *reinterpret_cast<const uint8_t*>(data);
            break;
        case uint16_type_id:
            value = *reinterpret_cast<const uint16_t*>(data);
            break;
        case uint32_type_id:
            value = *reinterpret_cast<const uint32_t*>(data);
            break;
        default:
            throw runtime_error("internal error in categorical_dtype::print_data");
    }
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
    const char *metadata = m_categories.get_ndo_meta() + sizeof(strided_dim_dtype_metadata);

    o << "categorical<" << m_category_dtype;
    o << ", [";
    m_category_dtype.print_data(o, metadata, get_category_data_from_value(0));
    for (size_t i = 1; i != category_count; ++i) {
        o << ", ";
        m_category_dtype.print_data(o, metadata, get_category_data_from_value((uint32_t)i));
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
        ss << "Unrecognized category value ";
        m_category_dtype.print_data(ss, category_metadata, category_data);
        ss << " assigning to dtype " << dtype(this, true);
        throw std::runtime_error(ss.str());
    } else {
        return (uint32_t)m_category_index_to_value[i];
    }
}

uint32_t categorical_dtype::get_value_from_category(const ndobject& category) const
{
    if (category.get_dtype() == m_category_dtype) {
        // If the dtype is right, get the category value directly
        return get_value_from_category(category.get_ndo_meta(), category.get_readonly_originptr());
    } else {
        // Otherwise convert to the correct dtype, then get the category value
        ndobject c = empty(m_category_dtype);
        c.val_assign(category);
        return get_value_from_category(c.get_ndo_meta(), c.get_readonly_originptr());
    }
}

const char *categorical_dtype::get_category_metadata() const
{
    const char *metadata = m_categories.get_ndo_meta();
    m_categories.get_dtype().extended()->at_single(0, &metadata, NULL);
    return metadata;
}

ndobject categorical_dtype::get_categories() const
{
    // TODO: store categories in their original order
    //       so this is simply "return m_categories".
    ndobject categories = make_strided_ndobject(get_category_count(), m_category_dtype);
    ndobject_iter<1,0> iter(categories);
    assignment_kernel k;
    ::make_assignment_kernel(&k, 0, iter.get_uniform_dtype(), iter.metadata(),
                    m_category_dtype, get_category_metadata(),
                    kernel_request_single, assign_error_default, &eval::default_eval_context);
    if (!iter.empty()) {
        uint32_t i = 0;
        do {
            k(iter.data(), get_category_data_from_value(i));
            ++i;
        } while(iter.next());
    }
    return categories;
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

size_t categorical_dtype::make_assignment_kernel(
                hierarchical_kernel *out, size_t offset_out,
                const dtype& dst_dt, const char *dst_metadata,
                const dtype& src_dt, const char *src_metadata,
                kernel_request_t kernreq, assign_error_mode errmode,
                const eval::eval_context *ectx) const
{
    if (this == dst_dt.extended()) {
        if (this == src_dt.extended()) {
            // When assigning identical types, just use a POD copy
            return make_pod_dtype_assignment_kernel(out, offset_out,
                            get_data_size(), get_alignment(), kernreq);
        }
        // try to assign from another categorical dtype if it can be mapped
        else if (src_dt.get_type_id() == categorical_type_id) {
            //out_kernel.specializations = assign_from_commensurate_category_specializations;
            // TODO auxdata
            throw std::runtime_error("assignment between different categorical dtypes isn't supported yet");
        }
        // assign from the same category value dtype
        else if (src_dt == m_category_dtype) {
            offset_out = make_kernreq_to_single_kernel_adapter(out, offset_out, kernreq);
            out->ensure_capacity_leaf(offset_out + sizeof(category_to_categorical_kernel_extra));
            category_to_categorical_kernel_extra *e =
                            out->get_at<category_to_categorical_kernel_extra>(offset_out);
            switch (m_category_int_dtype.get_type_id()) {
                case uint8_type_id:
                    e->base.set_function<unary_single_operation_t>(
                                    &category_to_categorical_kernel_extra::single_uint8);
                    break;
                case uint16_type_id:
                    e->base.set_function<unary_single_operation_t>(
                                    &category_to_categorical_kernel_extra::single_uint16);
                    break;
                case uint32_type_id:
                    e->base.set_function<unary_single_operation_t>(
                                    &category_to_categorical_kernel_extra::single_uint32);
                    break;
                default:
                    throw runtime_error("internal error in categorical_dtype::make_assignment_kernel");
            }
            e->base.destructor = &category_to_categorical_kernel_extra::destruct;
            // The kernel dtype owns a reference to this dtype
            e->dst_cat_dt = static_cast<const categorical_dtype *>(dtype(dst_dt).release());
            e->src_metadata = src_metadata;
            return offset_out + sizeof(category_to_categorical_kernel_extra);
        } else if (src_dt.value_dtype() != m_category_dtype &&
                        src_dt.value_dtype().get_type_id() != categorical_type_id) {
            // Make a convert dtype to the category dtype, and have it do the chaining
            dtype src_cvt_dt = make_convert_dtype(m_category_dtype, src_dt);
            return src_cvt_dt.extended()->make_assignment_kernel(out, offset_out,
                            dst_dt, dst_metadata,
                            src_cvt_dt, src_metadata,
                            kernreq, errmode, ectx);
        } else {
            // Let the src_dt handle it
            return src_dt.extended()->make_assignment_kernel(out, offset_out,
                            dst_dt, dst_metadata,
                            src_dt, src_metadata,
                            kernreq, errmode, ectx);
        }
    }
    else {
        if (dst_dt.value_dtype().get_type_id() != categorical_type_id) {
            offset_out = make_kernreq_to_single_kernel_adapter(out, offset_out, kernreq);
            out->ensure_capacity(offset_out + sizeof(categorical_to_other_kernel_extra));
            categorical_to_other_kernel_extra *e = out->get_at<categorical_to_other_kernel_extra>(offset_out);
            switch (m_category_int_dtype.get_type_id()) {
                case uint8_type_id:
                    e->base.set_function<unary_single_operation_t>(&categorical_to_other_kernel_extra::single_uint8);
                    break;
                case uint16_type_id:
                    e->base.set_function<unary_single_operation_t>(&categorical_to_other_kernel_extra::single_uint16);
                    break;
                case uint32_type_id:
                    e->base.set_function<unary_single_operation_t>(&categorical_to_other_kernel_extra::single_uint32);
                    break;
                default:
                    throw runtime_error("internal error in categorical_dtype::make_assignment_kernel");
            }
            e->base.destructor = &categorical_to_other_kernel_extra::destruct;
            // The kernel dtype owns a reference to this dtype
            e->src_cat_dt = static_cast<const categorical_dtype *>(dtype(src_dt).release());
            return ::make_assignment_kernel(out, offset_out + sizeof(categorical_to_other_kernel_extra),
                            dst_dt, dst_metadata,
                            get_category_dtype(), get_category_metadata(),
                            kernel_request_single, errmode, ectx);
        }
        else {
            stringstream ss;
            ss << "Cannot assign from " << src_dt << " to " << dst_dt;
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

void categorical_dtype::metadata_default_construct(char *DYND_UNUSED(metadata),
                size_t DYND_UNUSED(ndim), const intptr_t* DYND_UNUSED(shape)) const
{
    // Data is stored as uint##, no metadata to process
}

void categorical_dtype::metadata_copy_construct(char *DYND_UNUSED(dst_metadata),
                const char *DYND_UNUSED(src_metadata),
                memory_block_data *DYND_UNUSED(embedded_reference)) const
{
    // Data is stored as uint##, no metadata to process
}

void categorical_dtype::metadata_destruct(char *DYND_UNUSED(metadata)) const
{
    // Data is stored as uint##, no metadata to process
}

void categorical_dtype::metadata_debug_print(const char *DYND_UNUSED(metadata),
                std::ostream& DYND_UNUSED(o), const std::string& DYND_UNUSED(indent)) const
{
    // Data is stored as uint##, no metadata to process
}

dtype dynd::factor_categorical_dtype(const ndobject& values)
{
    ndobject_iter<0, 1> iter(values);

    comparison_kernel k;
    ::make_comparison_kernel(&k, 0,
                    iter.get_uniform_dtype(), iter.metadata(),
                    iter.get_uniform_dtype(), iter.metadata(),
                    comparison_type_sorting_less, &eval::default_eval_context);

    cmp less(k.get_function(), k.get());
    set<const char *, cmp> uniques(less);

    if (!iter.empty()) {
        do {
            if (uniques.find(iter.data()) == uniques.end()) {
                uniques.insert(iter.data());
            }
        } while (iter.next());
    }

    // Copy the values (now sorted and unique) into a new ndobject
    ndobject categories = make_sorted_categories(uniques,
                    iter.get_uniform_dtype(), iter.metadata());

    return dtype(new categorical_dtype(categories, true), false);
}

static ndobject property_ndo_get_category_ints(const ndobject& n) {
    dtype udt = n.get_udtype().value_dtype();
    const categorical_dtype *cd = static_cast<const categorical_dtype *>(udt.extended());
    return n.view_scalars(cd->get_category_int_dtype());
}

static pair<string, gfunc::callable> categorical_ndobject_properties[] = {
    pair<string, gfunc::callable>("category_ints",
                    gfunc::make_callable(&property_ndo_get_category_ints, "self"))
};

void categorical_dtype::get_dynamic_ndobject_properties(
                const std::pair<std::string, gfunc::callable> **out_properties,
                size_t *out_count) const
{
    *out_properties = categorical_ndobject_properties;
    *out_count = sizeof(categorical_ndobject_properties) / sizeof(categorical_ndobject_properties[0]);
}

static ndobject property_dtype_get_categories(const dtype& d) {
    const categorical_dtype *cd = static_cast<const categorical_dtype *>(d.extended());
    return cd->get_categories();
}

static pair<string, gfunc::callable> categorical_dtype_properties[] = {
    pair<string, gfunc::callable>("categories",
                    gfunc::make_callable(&property_dtype_get_categories, "self"))
};

void categorical_dtype::get_dynamic_dtype_properties(
                const std::pair<std::string, gfunc::callable> **out_properties,
                size_t *out_count) const
{
    *out_properties = categorical_dtype_properties;
    *out_count = sizeof(categorical_dtype_properties) / sizeof(categorical_dtype_properties[0]);
}

