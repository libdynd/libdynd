//
// Copyright (C) 2011-14 Mark Wiebe, DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <dynd/types/cstruct_type.hpp>
#include <dynd/types/struct_type.hpp>
#include <dynd/types/type_alignment.hpp>
#include <dynd/types/property_type.hpp>
#include <dynd/shape_tools.hpp>
#include <dynd/exceptions.hpp>
#include <dynd/func/make_callable.hpp>
#include <dynd/kernels/assignment_kernels.hpp>
#include <dynd/kernels/struct_assignment_kernels.hpp>
#include <dynd/kernels/struct_comparison_kernels.hpp>

using namespace std;
using namespace dynd;

cstruct_type::cstruct_type(const nd::array &field_names,
                           const nd::array &field_types)
    : base_struct_type(cstruct_type_id, field_names, field_types,
                       type_flag_none, false),
      m_data_offsets(nd::empty(m_field_count, ndt::make_type<uintptr_t>()))
{
    uintptr_t *data_offsets = reinterpret_cast<uintptr_t *>(
        m_data_offsets.get_readwrite_originptr());

    size_t offs = 0;
    for (intptr_t i = 0; i < m_field_count; ++i) {
        const ndt::type& field_tp = get_field_type(i);
        offs = inc_to_alignment(offs, field_tp.get_data_alignment());
        data_offsets[i] = offs;
        size_t field_element_size = field_tp.get_data_size();
        if (field_element_size == 0) {
            stringstream ss;
            ss << "Cannot create dynd ctuple type with type " << field_tp;
            ss << " for field at index " << i << ", as it does not have a fixed size";
            throw runtime_error(ss.str());
        }
        offs += field_element_size;
    }
    m_members.data_size = inc_to_alignment(offs, m_members.data_alignment);

    m_data_offsets.flag_as_immutable();

    create_array_properties();
}

cstruct_type::~cstruct_type()
{
}

static bool is_simple_identifier_name(const char *begin, const char *end)
{
    if (begin == end) {
        return false;
    } else {
        char c = *begin++;
        if (!(('a' <= c && c <= 'z') || ('A' <= c && c <= 'Z') || c == '_')) {
            return false;
        }
        while (begin < end) {
            c = *begin++;
            if (!(('0' <= c && c <= '9') || ('a' <= c && c <= 'z')
                            || ('A' <= c && c <= 'Z') || c == '_')) {
                return false;
            }
        }
        return true;
    }
}

void cstruct_type::print_type(std::ostream& o) const
{
    // Use the record datashape syntax prefixed with a "c"
    o << "c{";
    for (intptr_t i = 0, i_end = m_field_count; i != i_end; ++i) {
        if (i != 0) {
            o << ", ";
        }
        const string_type_data& fn = get_field_name_raw(i);
        if (is_simple_identifier_name(fn.begin, fn.end)) {
            o.write(fn.begin, fn.end - fn.begin);
        } else {
            print_escaped_utf8_string(o, fn.begin, fn.end, true);
        }
        o << " : " << get_field_type(i);
    }
    o << "}";
}

void cstruct_type::transform_child_types(type_transform_fn_t transform_fn, void *extra,
                ndt::type& out_transformed_tp, bool& out_was_transformed) const
{
    nd::array tmp_field_types(
        nd::typed_empty(1, &m_field_count, ndt::make_strided_of_type()));
    ndt::type *tmp_field_types_raw = reinterpret_cast<ndt::type *>(
        tmp_field_types.get_readwrite_originptr());

    bool switch_to_struct = false;
    bool was_any_transformed = false;
    for (intptr_t i = 0, i_end = m_field_count; i != i_end; ++i) {
        bool was_transformed = false;
        transform_fn(get_field_type(i), extra, tmp_field_types_raw[i],
                     was_transformed);
        if (was_transformed) {
            // If the type turned into one without fixed size, have to use struct instead of cstruct
            if (tmp_field_types_raw[i].get_data_size() == 0) {
                switch_to_struct = true;
            }
            was_any_transformed = true;
        }
    }
    if (was_any_transformed) {
        tmp_field_types.flag_as_immutable();
        if (!switch_to_struct) {
            out_transformed_tp = ndt::make_cstruct(m_field_names, tmp_field_types);
        } else {
            out_transformed_tp = ndt::make_struct(m_field_names, tmp_field_types);
        }
        out_was_transformed = true;
    } else {
        out_transformed_tp = ndt::type(this, true);
    }
}

ndt::type cstruct_type::get_canonical_type() const
{
    nd::array tmp_field_types(
        nd::typed_empty(1, &m_field_count, ndt::make_strided_of_type()));
    ndt::type *tmp_field_types_raw = reinterpret_cast<ndt::type *>(
        tmp_field_types.get_readwrite_originptr());

    for (intptr_t i = 0, i_end = m_field_count; i != i_end; ++i) {
        tmp_field_types_raw[i] = get_field_type(i).get_canonical_type();
    }

    tmp_field_types.flag_as_immutable();
    return ndt::make_cstruct(m_field_names, tmp_field_types);
}

ndt::type cstruct_type::at_single(intptr_t i0,
                const char **inout_arrmeta, const char **inout_data) const
{
    // Bounds-checking of the index
    i0 = apply_single_index(i0, m_field_count, NULL);
    if (inout_arrmeta) {
        // Modify the arrmeta
        *inout_arrmeta += get_arrmeta_offsets_raw()[i0];
        // If requested, modify the data
        if (inout_data) {
            *inout_data += get_data_offsets_raw()[i0];
        }
    }
    return get_field_type(i0);
}

bool cstruct_type::is_lossless_assignment(const ndt::type& dst_tp, const ndt::type& src_tp) const
{
    if (dst_tp.extended() == this) {
        if (src_tp.extended() == this) {
            return true;
        } else if (src_tp.get_type_id() == cstruct_type_id) {
            return *dst_tp.extended() == *src_tp.extended();
        }
    }

    return false;
}

size_t cstruct_type::make_assignment_kernel(
    ckernel_builder *ckb, intptr_t ckb_offset, const ndt::type &dst_tp,
    const char *dst_arrmeta, const ndt::type &src_tp, const char *src_arrmeta,
    kernel_request_t kernreq, const eval::eval_context *ectx) const
{
    if (this == dst_tp.extended()) {
        if (*this == *src_tp.extended()) {
            return make_struct_identical_assignment_kernel(
                ckb, ckb_offset, dst_tp, dst_arrmeta, src_arrmeta, kernreq,
                ectx);
        } else if (src_tp.get_kind() == struct_kind) {
            return make_struct_assignment_kernel(ckb, ckb_offset, dst_tp,
                                                 dst_arrmeta, src_tp,
                                                 src_arrmeta, kernreq, ectx);
        } else if (src_tp.is_builtin()) {
            return make_broadcast_to_struct_assignment_kernel(
                ckb, ckb_offset, dst_tp, dst_arrmeta, src_tp, src_arrmeta,
                kernreq, ectx);
        } else {
            return src_tp.extended()->make_assignment_kernel(
                ckb, ckb_offset, dst_tp, dst_arrmeta, src_tp, src_arrmeta,
                kernreq, ectx);
        }
    }

    stringstream ss;
    ss << "Cannot assign from " << src_tp << " to " << dst_tp;
    throw dynd::type_error(ss.str());
}

size_t cstruct_type::make_comparison_kernel(
    ckernel_builder *ckb, intptr_t ckb_offset, const ndt::type &src0_tp,
    const char *src0_arrmeta, const ndt::type &src1_tp,
    const char *src1_arrmeta, comparison_type_t comptype,
    const eval::eval_context *ectx) const
{
    if (this == src0_tp.extended()) {
        if (*this == *src1_tp.extended()) {
            return make_struct_comparison_kernel(ckb, ckb_offset,
                            src0_tp, src0_arrmeta, src1_arrmeta,
                            comptype, ectx);
        } else if (src1_tp.get_kind() == struct_kind) {
            return make_general_struct_comparison_kernel(ckb, ckb_offset,
                            src0_tp, src0_arrmeta,
                            src1_tp, src1_arrmeta,
                            comptype, ectx);
        }
    }

    throw not_comparable_error(src0_tp, src1_tp, comptype);
}

bool cstruct_type::operator==(const base_type& rhs) const
{
    if (this == &rhs) {
        return true;
    } else if (rhs.get_type_id() != cstruct_type_id) {
        return false;
    } else {
        const cstruct_type *dt = static_cast<const cstruct_type*>(&rhs);
        return get_data_alignment() == dt->get_data_alignment() &&
                m_field_types.equals_exact(dt->m_field_types) &&
                m_field_names.equals_exact(dt->m_field_names);
    }
}

void cstruct_type::arrmeta_debug_print(const char *arrmeta, std::ostream &o,
                                       const std::string &indent) const
{
    o << indent << "cstruct arrmeta\n";
    const uintptr_t *arrmeta_offsets = get_arrmeta_offsets_raw();
    for (intptr_t i = 0; i < m_field_count; ++i) {
        const ndt::type& field_dt = get_field_type(i);
        if (!field_dt.is_builtin() && field_dt.extended()->get_arrmeta_size() > 0) {
            o << indent << " field " << i << " (";
            const string_type_data& fnr = get_field_name_raw(i);
            o.write(fnr.begin, fnr.end - fnr.begin);
            o << ") arrmeta:\n";
            field_dt.extended()->arrmeta_debug_print(arrmeta + arrmeta_offsets[i], o, indent + "  ");
        }
    }
}

///////// properties on the type

static nd::array property_get_field_names(const ndt::type& tp) {
    return tp.tcast<cstruct_type>()->get_field_names();
}

static nd::array property_get_field_types(const ndt::type& tp) {
    return tp.tcast<cstruct_type>()->get_field_types();
}

static nd::array property_get_data_offsets(const ndt::type& tp) {
    return tp.tcast<cstruct_type>()->get_data_offsets();
}

static nd::array property_get_arrmeta_offsets(const ndt::type& tp) {
    return tp.tcast<cstruct_type>()->get_arrmeta_offsets();
}

void cstruct_type::get_dynamic_type_properties(const std::pair<std::string, gfunc::callable> **out_properties, size_t *out_count) const
{
    static pair<string, gfunc::callable> type_properties[] = {
        pair<string, gfunc::callable>(
            "field_names",
            gfunc::make_callable(&property_get_field_names, "self")),
        pair<string, gfunc::callable>(
            "field_types",
            gfunc::make_callable(&property_get_field_types, "self")),
        pair<string, gfunc::callable>(
            "data_offsets",
            gfunc::make_callable(&property_get_data_offsets, "self")),
        pair<string, gfunc::callable>(
            "arrmeta_offsets",
            gfunc::make_callable(&property_get_arrmeta_offsets, "self"))};

    *out_properties = type_properties;
    *out_count = sizeof(type_properties) / sizeof(type_properties[0]);
}

///////// properties on the nd::array

static nd::array make_self_names()
{
    const char *selfname = "self";
    return nd::make_strided_string_array(&selfname, 1);
}

static nd::array make_self_types()
{
  intptr_t one = 1;
  nd::array result = nd::typed_empty(1, &one, ndt::make_strided_of_type());
  unchecked_strided_dim_get_rw<ndt::type>(result, 0) = ndt::make_ndarrayarg();
  result.flag_as_immutable();
  return result;
}

cstruct_type::cstruct_type(int, int)
    : base_struct_type(cstruct_type_id, make_self_names(), make_self_types(),
                       type_flag_none, false)
{
    // Equivalent to ndt::make_cstruct(ndt::make_ndarrayarg(), "self");
    // but hardcoded to break the dependency of cstruct_type::array_parameters_type
    uintptr_t metaoff[1] = {0};
    m_arrmeta_offsets = nd::array(metaoff);
    // The data offsets also consist of one zero
    m_data_offsets = m_arrmeta_offsets;
    // Inherit any operand flags from the fields
    m_members.flags |=
        (ndt::make_ndarrayarg().get_flags() & type_flags_operand_inherited);
    m_members.data_alignment = sizeof(void *);
    m_members.arrmeta_size = 0;
    m_members.data_size = sizeof(void *);
    // Leave m_array_properties so there is no reference loop
}

static array_preamble *property_get_array_field(const array_preamble *params, void *extra)
{
    // Get the nd::array 'self' parameter
    nd::array n = nd::array(*(array_preamble **)params->m_data_pointer, true);
    intptr_t i = reinterpret_cast<intptr_t>(extra);
    intptr_t undim = n.get_ndim();
    ndt::type udt = n.get_dtype();
    if (udt.get_kind() == expr_kind) {
        string field_name =
            udt.value_type().tcast<struct_type>()->get_field_name(i);
        return n.replace_dtype(ndt::make_property(udt, field_name, i))
            .release();
    } else {
        if (undim == 0) {
            return n(i).release();
        } else {
            shortvector<irange> idx(undim + 1);
            idx[undim] = irange(i);
            return n.at_array(undim + 1, idx.get()).release();
        }
    }
}

void cstruct_type::create_array_properties()
{
    ndt::type array_parameters_type(new cstruct_type(0, 0), false);

    m_array_properties.resize(m_field_count);
    for (intptr_t i = 0, i_end = m_field_count; i != i_end; ++i) {
        // TODO: Transform the name into a valid Python symbol?
        m_array_properties[i].first = get_field_name(i);
        m_array_properties[i].second.set(array_parameters_type, &property_get_array_field, (void *)i);
    }
}

void cstruct_type::get_dynamic_array_properties(const std::pair<std::string, gfunc::callable> **out_properties, size_t *out_count) const
{
    *out_properties = m_array_properties.empty() ? NULL : &m_array_properties[0];
    *out_count = (int)m_array_properties.size();
}
