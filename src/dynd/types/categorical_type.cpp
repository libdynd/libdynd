//
// Copyright (C) 2011-13 Mark Wiebe, DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <cstring>
#include <set>

#include <dynd/auxiliary_data.hpp>
#include <dynd/array_iter.hpp>
#include <dynd/types/categorical_type.hpp>
#include <dynd/kernels/assignment_kernels.hpp>
#include <dynd/kernels/comparison_kernels.hpp>
#include <dynd/types/strided_dim_type.hpp>
#include <dynd/types/convert_type.hpp>
#include <dynd/gfunc/make_callable.hpp>

using namespace dynd;
using namespace std;

namespace {

    class sorter {
        const char *m_originptr;
        intptr_t m_stride;
        const binary_single_predicate_t m_less;
        ckernel_prefix *m_extra;
    public:
        sorter(const char *originptr, intptr_t stride,
                        const binary_single_predicate_t less, ckernel_prefix *extra) :
            m_originptr(originptr), m_stride(stride), m_less(less), m_extra(extra) {}
        bool operator()(intptr_t i, intptr_t j) const {
            return m_less(m_originptr + i * m_stride, m_originptr + j * m_stride, m_extra) != 0;
        }
    };

    class cmp {
        const binary_single_predicate_t m_less;
        ckernel_prefix *m_extra;
    public:
        cmp(const binary_single_predicate_t less, ckernel_prefix *extra) :
            m_less(less), m_extra(extra) {}
        bool operator()(const char *a, const char *b) const {
            bool result = m_less(a, b, m_extra) != 0;
            return result;
        }
    };

    // Assign from a categorical type to some other type
    struct categorical_to_other_kernel_extra {
        typedef categorical_to_other_kernel_extra extra_type;

        ckernel_prefix base;
        const categorical_type *src_cat_tp;

        template<typename UIntType>
        inline static void single(char *dst, const char *src, ckernel_prefix *extra)
        {
            extra_type *e = reinterpret_cast<extra_type *>(extra);
            ckernel_prefix *echild = &(e + 1)->base;
            unary_single_operation_t opchild = echild->get_function<unary_single_operation_t>();

            uint32_t value = *reinterpret_cast<const UIntType *>(src);
            const char *src_val = e->src_cat_tp->get_category_data_from_value(value);
            opchild(dst, src_val, echild);
        }

        // Some compilers are finicky about getting single<T> as a function pointer, so this...
        static void single_uint8(char *dst, const char *src, ckernel_prefix *extra) {
            single<uint8_t>(dst, src, extra);
        }
        static void single_uint16(char *dst, const char *src, ckernel_prefix *extra) {
            single<uint16_t>(dst, src, extra);
        }
        static void single_uint32(char *dst, const char *src, ckernel_prefix *extra) {
            single<uint32_t>(dst, src, extra);
        }

        static void destruct(ckernel_prefix *extra)
        {
            extra_type *e = reinterpret_cast<extra_type *>(extra);
            if (e->src_cat_tp != NULL) {
                base_type_decref(e->src_cat_tp);
            }
            ckernel_prefix *echild = &(e + 1)->base;
            if (echild->destructor) {
                echild->destructor(echild);
            }
        }
    };

    struct category_to_categorical_kernel_extra {
        typedef category_to_categorical_kernel_extra extra_type;

        ckernel_prefix base;
        const categorical_type *dst_cat_tp;
        const char *src_metadata;

        // Assign from an input matching the category type to a categorical type
        template<typename UIntType>
        inline static void single(char *dst, const char *src, ckernel_prefix *extra)
        {
            extra_type *e = reinterpret_cast<extra_type *>(extra);
            uint32_t src_val = e->dst_cat_tp->get_value_from_category(e->src_metadata, src);
            *reinterpret_cast<UIntType *>(dst) = src_val;
        }

        // Some compilers are finicky about getting single<T> as a function pointer, so this...
        static void single_uint8(char *dst, const char *src, ckernel_prefix *extra) {
            single<uint8_t>(dst, src, extra);
        }
        static void single_uint16(char *dst, const char *src, ckernel_prefix *extra) {
            single<uint16_t>(dst, src, extra);
        }
        static void single_uint32(char *dst, const char *src, ckernel_prefix *extra) {
            single<uint32_t>(dst, src, extra);
        }

        static void destruct(ckernel_prefix *extra)
        {
            extra_type *e = reinterpret_cast<extra_type *>(extra);
            if (e->dst_cat_tp != NULL) {
                base_type_decref(e->dst_cat_tp);
            }
        }
    };

    // struct assign_from_commensurate_category {
    //     static void general_kernel(char *dst, intptr_t dst_stride, const char *src, intptr_t src_stride,
    //                         intptr_t count, const AuxDataBase *auxdata)
    //     {
    //         categorical_type *cat = reinterpret_cast<categorical_type *>(
    //             get_raw_auxiliary_data(auxdata)&~1
    //         );
    //     }

    //     static void scalar_kernel(char *dst, intptr_t DYND_UNUSED(dst_stride), const char *src, intptr_t DYND_UNUSED(src_stride),
    //                         intptr_t, const AuxDataBase *auxdata)
    //     {
    //         categorical_type *cat = reinterpret_cast<categorical_type *>(
    //             get_raw_auxiliary_data(auxdata)&~1
    //         );
    //     }

    //     static void contiguous_kernel(char *dst, intptr_t DYND_UNUSED(dst_stride), const char *src, intptr_t DYND_UNUSED(src_stride),
    //                         intptr_t count, const AuxDataBase *auxdata)
    //     {
    //         categorical_type *cat = reinterpret_cast<categorical_type *>(
    //             get_raw_auxiliary_data(auxdata)&~1
    //         );
    //     }

    //     static void scalar_to_contiguous_kernel(char *dst, intptr_t DYND_UNUSED(dst_stride), const char *src, intptr_t DYND_UNUSED(src_stride),
    //                         intptr_t count, const AuxDataBase *auxdata)
    //     {
    //         categorical_type *cat = reinterpret_cast<categorical_type *>(
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

/** This function converts the set of char* pointers into a strided immutable nd::array of the categories */
static nd::array make_sorted_categories(const set<const char *, cmp>& uniques,
                const ndt::type& element_tp, const char *metadata)
{
    nd::array categories = nd::make_strided_array(uniques.size(), element_tp);
    assignment_ckernel_builder k;
    make_assignment_kernel(&k, 0,
                    element_tp, categories.get_ndo_meta() + sizeof(strided_dim_type_metadata),
                    element_tp, metadata,
                    kernel_request_single, assign_error_default,
                    &eval::default_eval_context);

    intptr_t stride = reinterpret_cast<const strided_dim_type_metadata *>(categories.get_ndo_meta())->stride;
    char *dst_ptr = categories.get_readwrite_originptr();
    for (set<const char *, cmp>::const_iterator it = uniques.begin(); it != uniques.end(); ++it) {
        k(dst_ptr, *it);
        dst_ptr += stride;
    }
    categories.get_type().extended()->metadata_finalize_buffers(categories.get_ndo_meta());
    categories.flag_as_immutable();

    return categories;
}

categorical_type::categorical_type(const nd::array& categories, bool presorted)
    : base_type(categorical_type_id, custom_kind, 4, 4, type_flag_scalar, 0, 0)
{
    intptr_t category_count;
    if (presorted) {
        // This is construction shortcut, for the case when the categories are already
        // sorted. No validation of this is done, the caller should have ensured it
        // was correct already, typically by construction.
        m_categories = categories.eval_immutable();
        m_category_tp = m_categories.get_type().at(0);

        category_count = categories.get_dim_size();
        m_value_to_category_index.resize(category_count);
        m_category_index_to_value.resize(category_count);
        for (size_t i = 0; i != (size_t)category_count; ++i) {
            m_value_to_category_index[i] = i;
            m_category_index_to_value[i] = i;
        }

    } else {
        // Process the categories array to make sure it's valid
        const ndt::type& cdt = categories.get_type();
        if (cdt.get_type_id() != strided_dim_type_id) {
            throw runtime_error("categorical_type only supports construction from a strided array of categories");
        }
        m_category_tp = categories.get_type().at(0);
        if (!m_category_tp.is_scalar()) {
            throw runtime_error("categorical_type only supports construction from a 1-dimensional strided array of categories");
        }

        category_count = categories.get_dim_size();
        intptr_t categories_stride = reinterpret_cast<const strided_dim_type_metadata *>(categories.get_ndo_meta())->stride;

        const char *categories_element_metadata = categories.get_ndo_meta() + sizeof(strided_dim_type_metadata);
        comparison_ckernel_builder k;
        ::make_comparison_kernel(&k, 0,
                        m_category_tp, categories_element_metadata,
                        m_category_tp, categories_element_metadata,
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
                m_category_tp.print_data(ss, categories_element_metadata, category_value);
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

        m_categories = make_sorted_categories(uniques, m_category_tp,
                        categories_element_metadata);
    }

    // Use the number of categories to set which underlying integer storage to use
    if (category_count <= 256) {
        m_storage_type = ndt::make_type<uint8_t>();
    } else if (category_count <= 65536) {
        m_storage_type = ndt::make_type<uint16_t>();
    } else {
        m_storage_type = ndt::make_type<uint32_t>();
    }
    m_members.data_size = m_storage_type.get_data_size();
    m_members.data_alignment = (uint8_t)m_storage_type.get_data_alignment();
}

void categorical_type::print_data(std::ostream& o, const char *metadata, const char *data) const
{
    uint32_t value;
    switch (m_storage_type.get_type_id()) {
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
            throw runtime_error("internal error in categorical_type::print_data");
    }
    if (value < m_value_to_category_index.size()) {
        m_category_tp.print_data(o, metadata, get_category_data_from_value(value));
    }
    else {
        o << "UNK"; // TODO better outpout?
    }
}


void categorical_type::print_type(std::ostream& o) const
{
    size_t category_count = get_category_count();
    const char *metadata = m_categories.get_ndo_meta() + sizeof(strided_dim_type_metadata);

    o << "categorical<" << m_category_tp;
    o << ", [";
    m_category_tp.print_data(o, metadata, get_category_data_from_value(0));
    for (size_t i = 1; i != category_count; ++i) {
        o << ", ";
        m_category_tp.print_data(o, metadata, get_category_data_from_value((uint32_t)i));
    }
    o << "]>";
}

void dynd::categorical_type::get_shape(intptr_t ndim, intptr_t i, intptr_t *out_shape,
                const char *DYND_UNUSED(metadata), const char *DYND_UNUSED(data)) const
{
    const ndt::type& cd = get_category_type();
    if (!cd.is_builtin()) {
        cd.extended()->get_shape(ndim, i, out_shape, get_category_metadata(), NULL);
    } else {
        stringstream ss;
        ss << "requested too many dimensions from type " << ndt::type(this, true);
        throw runtime_error(ss.str());
    }
}

uint32_t categorical_type::get_value_from_category(const char *category_metadata, const char *category_data) const
{
    intptr_t i = nd::binary_search(m_categories, category_metadata, category_data);
    if (i < 0) {
        stringstream ss;
        ss << "Unrecognized category value ";
        m_category_tp.print_data(ss, category_metadata, category_data);
        ss << " assigning to dynd type " << ndt::type(this, true);
        throw std::runtime_error(ss.str());
    } else {
        return (uint32_t)m_category_index_to_value[i];
    }
}

uint32_t categorical_type::get_value_from_category(const nd::array& category) const
{
    if (category.get_type() == m_category_tp) {
        // If the type is right, get the category value directly
        return get_value_from_category(category.get_ndo_meta(), category.get_readonly_originptr());
    } else {
        // Otherwise convert to the correct type, then get the category value
        nd::array c = nd::empty(m_category_tp);
        c.val_assign(category);
        return get_value_from_category(c.get_ndo_meta(), c.get_readonly_originptr());
    }
}

const char *categorical_type::get_category_metadata() const
{
    const char *metadata = m_categories.get_ndo_meta();
    m_categories.get_type().extended()->at_single(0, &metadata, NULL);
    return metadata;
}

nd::array categorical_type::get_categories() const
{
    // TODO: store categories in their original order
    //       so this is simply "return m_categories".
    nd::array categories = nd::make_strided_array(get_category_count(), m_category_tp);
    array_iter<1,0> iter(categories);
    assignment_ckernel_builder k;
    ::make_assignment_kernel(&k, 0, iter.get_uniform_dtype(), iter.metadata(),
                    m_category_tp, get_category_metadata(),
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


bool categorical_type::is_lossless_assignment(const ndt::type& dst_tp, const ndt::type& src_tp) const
{
    if (dst_tp.extended() == this) {
        if (src_tp.extended() == this) {
            // Casting from identical types
            return true;
        } else  {
            return false; // TODO
        }

    } else {
        return ::is_lossless_assignment(dst_tp, m_category_tp); // TODO
    }
}

size_t categorical_type::make_assignment_kernel(
                ckernel_builder *out, size_t offset_out,
                const ndt::type& dst_tp, const char *dst_metadata,
                const ndt::type& src_tp, const char *src_metadata,
                kernel_request_t kernreq, assign_error_mode errmode,
                const eval::eval_context *ectx) const
{
    if (this == dst_tp.extended()) {
        if (this == src_tp.extended()) {
            // When assigning identical types, just use a POD copy
            return make_pod_typed_data_assignment_kernel(out, offset_out,
                            get_data_size(), get_data_alignment(), kernreq);
        }
        // try to assign from another categorical type if it can be mapped
        else if (src_tp.get_type_id() == categorical_type_id) {
            //out_kernel.specializations = assign_from_commensurate_category_specializations;
            // TODO auxdata
            throw std::runtime_error("assignment between different categorical types isn't supported yet");
        }
        // assign from the same category value type
        else if (src_tp == m_category_tp) {
            offset_out = make_kernreq_to_single_kernel_adapter(out, offset_out, kernreq);
            out->ensure_capacity_leaf(offset_out + sizeof(category_to_categorical_kernel_extra));
            category_to_categorical_kernel_extra *e =
                            out->get_at<category_to_categorical_kernel_extra>(offset_out);
            switch (m_storage_type.get_type_id()) {
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
                    throw runtime_error("internal error in categorical_type::make_assignment_kernel");
            }
            e->base.destructor = &category_to_categorical_kernel_extra::destruct;
            // The kernel type owns a reference to this type
            e->dst_cat_tp = static_cast<const categorical_type *>(ndt::type(dst_tp).release());
            e->src_metadata = src_metadata;
            return offset_out + sizeof(category_to_categorical_kernel_extra);
        } else if (src_tp.value_type() != m_category_tp &&
                        src_tp.value_type().get_type_id() != categorical_type_id) {
            // Make a convert type to the category type, and have it do the chaining
            ndt::type src_cvt_tp = ndt::make_convert(m_category_tp, src_tp);
            return src_cvt_tp.extended()->make_assignment_kernel(out, offset_out,
                            dst_tp, dst_metadata,
                            src_cvt_tp, src_metadata,
                            kernreq, errmode, ectx);
        } else {
            // Let the src_tp handle it
            return src_tp.extended()->make_assignment_kernel(out, offset_out,
                            dst_tp, dst_metadata,
                            src_tp, src_metadata,
                            kernreq, errmode, ectx);
        }
    }
    else {
        if (dst_tp.value_type().get_type_id() != categorical_type_id) {
            offset_out = make_kernreq_to_single_kernel_adapter(out, offset_out, kernreq);
            out->ensure_capacity(offset_out + sizeof(categorical_to_other_kernel_extra));
            categorical_to_other_kernel_extra *e = out->get_at<categorical_to_other_kernel_extra>(offset_out);
            switch (m_storage_type.get_type_id()) {
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
                    throw runtime_error("internal error in categorical_type::make_assignment_kernel");
            }
            e->base.destructor = &categorical_to_other_kernel_extra::destruct;
            // The kernel type owns a reference to this type
            e->src_cat_tp = static_cast<const categorical_type *>(ndt::type(src_tp).release());
            return ::make_assignment_kernel(out, offset_out + sizeof(categorical_to_other_kernel_extra),
                            dst_tp, dst_metadata,
                            get_category_type(), get_category_metadata(),
                            kernel_request_single, errmode, ectx);
        }
        else {
            stringstream ss;
            ss << "Cannot assign from " << src_tp << " to " << dst_tp;
            throw runtime_error(ss.str());
        }

    }

}

bool categorical_type::operator==(const base_type& rhs) const
{
    if (this == &rhs)
        return true;
    if (rhs.get_type_id() != categorical_type_id)
        return false;
    if (!m_categories.equals_exact(static_cast<const categorical_type&>(rhs).m_categories))
        return false;
    if (static_cast<const categorical_type&>(rhs).m_category_index_to_value != m_category_index_to_value)
        return false;
    if (static_cast<const categorical_type&>(rhs).m_value_to_category_index != m_value_to_category_index)
        return false;

    return true;

}

void categorical_type::metadata_default_construct(char *DYND_UNUSED(metadata),
                intptr_t DYND_UNUSED(ndim), const intptr_t* DYND_UNUSED(shape)) const
{
    // Data is stored as uint##, no metadata to process
}

void categorical_type::metadata_copy_construct(char *DYND_UNUSED(dst_metadata),
                const char *DYND_UNUSED(src_metadata),
                memory_block_data *DYND_UNUSED(embedded_reference)) const
{
    // Data is stored as uint##, no metadata to process
}

void categorical_type::metadata_destruct(char *DYND_UNUSED(metadata)) const
{
    // Data is stored as uint##, no metadata to process
}

void categorical_type::metadata_debug_print(const char *DYND_UNUSED(metadata),
                std::ostream& DYND_UNUSED(o), const std::string& DYND_UNUSED(indent)) const
{
    // Data is stored as uint##, no metadata to process
}

ndt::type dynd::ndt::factor_categorical(const nd::array& values)
{
    // Do the factor operation on a concrete version of the values
    // TODO: Some cases where we don't want to do this?
    nd::array values_eval = values.eval();
    array_iter<0, 1> iter(values_eval);

    comparison_ckernel_builder k;
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

    // Copy the values (now sorted and unique) into a new nd::array
    nd::array categories = make_sorted_categories(uniques,
                    iter.get_uniform_dtype(), iter.metadata());

    return ndt::type(new categorical_type(categories, true), false);
}

static nd::array property_ndo_get_ints(const nd::array& n) {
    ndt::type udt = n.get_dtype().value_type();
    const categorical_type *cd = static_cast<const categorical_type *>(udt.extended());
    return n.view_scalars(cd->get_storage_type());
}

static pair<string, gfunc::callable> categorical_array_properties[] = {
    pair<string, gfunc::callable>("ints",
                    gfunc::make_callable(&property_ndo_get_ints, "self"))
};

void categorical_type::get_dynamic_array_properties(
                const std::pair<std::string, gfunc::callable> **out_properties,
                size_t *out_count) const
{
    *out_properties = categorical_array_properties;
    *out_count = sizeof(categorical_array_properties) / sizeof(categorical_array_properties[0]);
}

static nd::array property_type_get_categories(const ndt::type& d) {
    const categorical_type *cd = static_cast<const categorical_type *>(d.extended());
    return cd->get_categories();
}

static ndt::type property_type_get_storage_type(const ndt::type& d) {
    const categorical_type *cd = static_cast<const categorical_type *>(d.extended());
    return cd->get_storage_type();
}

static ndt::type property_type_get_category_type(const ndt::type& d) {
    const categorical_type *cd = static_cast<const categorical_type *>(d.extended());
    return cd->get_category_type();
}

static pair<string, gfunc::callable> categorical_type_properties[] = {
    pair<string, gfunc::callable>("categories",
                    gfunc::make_callable(&property_type_get_categories, "self")),
    pair<string, gfunc::callable>("storage_type",
                    gfunc::make_callable(&property_type_get_storage_type, "self")),
    pair<string, gfunc::callable>("category_type",
                    gfunc::make_callable(&property_type_get_category_type, "self"))
};

void categorical_type::get_dynamic_type_properties(
                const std::pair<std::string, gfunc::callable> **out_properties,
                size_t *out_count) const
{
    *out_properties = categorical_type_properties;
    *out_count = sizeof(categorical_type_properties) / sizeof(categorical_type_properties[0]);
}

