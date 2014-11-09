//
// Copyright (C) 2011-14 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <dynd/types/arrfunc_type.hpp>
#include <dynd/types/fixed_dim_type.hpp>
#include <dynd/func/make_callable.hpp>
#include <dynd/ensure_immutable_contig.hpp>
#include <dynd/types/typevar_type.hpp>

using namespace std;
using namespace dynd;

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

arrfunc_type::arrfunc_type(const nd::array &src_names,
                               const nd::array &aux_names,
                               const nd::array &arg_types,
                               const ndt::type &return_type,
                               intptr_t naux)
    : base_type(arrfunc_type_id, symbolic_kind, 0, 1, type_flag_none, 0, 0,
                0),
      m_src_names(src_names), m_aux_names(aux_names), m_arg_types(arg_types),
      m_narg(arg_types.get_dim_size()), m_nsrc(m_narg - naux), m_naux(naux),
      m_return_type(return_type)
{
    if (!m_src_names.is_null()) {
        if (!nd::ensure_immutable_contig<nd::string>(m_src_names)) {
            stringstream ss;
            ss << "dynd funcproto src names requires an array of strings, got an "
                  "array with type " << m_src_names.get_type();
            throw invalid_argument(ss.str());
        }
        if (m_src_names.get_dim_size() > m_nsrc) {
            stringstream ss;
            ss << "dynd funcproto src names is larger than src types";
            throw invalid_argument(ss.str());
        }
    }

    if (!m_aux_names.is_null()) {
        if (!nd::ensure_immutable_contig<nd::string>(m_aux_names)) {
            stringstream ss;
            ss << "dynd funcproto aux names requires an array of strings, got an "
                  "array with type " << m_aux_names.get_type();
            throw invalid_argument(ss.str());
        }
        if (m_aux_names.get_dim_size() > m_naux) {
            stringstream ss;
            ss << "dynd funcproto aux names is larger than aux types";
            throw invalid_argument(ss.str());
        }
    }

    if (!nd::ensure_immutable_contig<ndt::type>(m_arg_types)) {
        stringstream ss;
        ss << "dynd funcproto arg types requires an array of types, got an "
              "array with type " << m_arg_types.get_type();
        throw invalid_argument(ss.str());
    }

    m_members.flags |= return_type.get_flags() & type_flags_value_inherited;
    for (intptr_t i = 0; i != m_narg; ++i) {
        m_members.flags |=
            get_arg_type(i).get_flags() & type_flags_value_inherited;
    }
}

void arrfunc_type::print_data(std::ostream &DYND_UNUSED(o),
                                const char *DYND_UNUSED(arrmeta),
                                const char *DYND_UNUSED(data)) const
{
    throw type_error("Cannot store data of funcproto type");
}

void arrfunc_type::print_type(std::ostream& o) const
{
    const ndt::type *arg_types = get_arg_types_raw();

    o << "(";

    intptr_t i_kwds = m_src_names.is_null() ? m_nsrc : (m_nsrc - m_src_names.get_dim_size());
    for (intptr_t i = 0, i_end = m_nsrc; i != i_end; ++i) {
        if (i > 0) {
            o << ", ";
        }

        if (i >= i_kwds) {
            const string_type_data& an = get_src_name_raw(i - i_kwds);
            if (is_simple_identifier_name(an.begin, an.end)) {
                o.write(an.begin, an.end - an.begin);
            } else {
                print_escaped_utf8_string(o, an.begin, an.end, true);
            }
            o << ": ";
        }
        o << arg_types[i];
    }

    if (m_naux > 0) {
        o << "; ";
    }

    i_kwds = m_aux_names.is_null() ? m_narg : (m_narg - m_aux_names.get_dim_size());
    for (intptr_t i = m_nsrc, i_end = m_narg; i != i_end; ++i) {
        if (i > m_nsrc) {
            o << ", ";
        }

        if (i >= i_kwds) {
            const string_type_data& an = get_aux_name_raw(i - i_kwds);
            if (is_simple_identifier_name(an.begin, an.end)) {
                o.write(an.begin, an.end - an.begin);
            } else {
                print_escaped_utf8_string(o, an.begin, an.end, true);
            }
            o << ": ";
        }
        o << arg_types[i];
    }

    o << ") -> " << m_return_type;
}

void arrfunc_type::transform_child_types(type_transform_fn_t transform_fn,
                                           intptr_t arrmeta_offset, void *extra,
                                           ndt::type &out_transformed_tp,
                                           bool &out_was_transformed) const
{
    const ndt::type *arg_types = get_arg_types_raw();
    std::vector<ndt::type> tmp_arg_types(m_narg);
    ndt::type tmp_return_type;

    bool was_transformed = false;
    for (size_t i = 0, i_end = m_narg; i != i_end; ++i) {
      transform_fn(arg_types[i], arrmeta_offset, extra, tmp_arg_types[i],
                   was_transformed);
    }
    transform_fn(m_return_type, arrmeta_offset, extra, tmp_return_type,
                 was_transformed);
    if (was_transformed) {
        out_transformed_tp =
            ndt::make_funcproto(tmp_arg_types, tmp_return_type);
        out_was_transformed = true;
    } else {
        out_transformed_tp = ndt::type(this, true);
    }
}

ndt::type arrfunc_type::get_canonical_type() const
{
    const ndt::type *arg_types = get_arg_types_raw();
    std::vector<ndt::type> tmp_arg_types(m_narg);
    ndt::type return_type;

    for (size_t i = 0, i_end = m_narg; i != i_end; ++i) {
        tmp_arg_types[i] = arg_types[i].get_canonical_type();
    }
    return_type = m_return_type.get_canonical_type();

    return ndt::make_funcproto(tmp_arg_types, return_type);
}

ndt::type arrfunc_type::apply_linear_index(
    intptr_t DYND_UNUSED(nindices), const irange *DYND_UNUSED(indices),
    size_t DYND_UNUSED(current_i), const ndt::type &DYND_UNUSED(root_tp),
    bool DYND_UNUSED(leading_dimension)) const
{
    throw type_error("Cannot store data of funcproto type");
}

intptr_t arrfunc_type::apply_linear_index(
    intptr_t DYND_UNUSED(nindices), const irange *DYND_UNUSED(indices),
    const char *DYND_UNUSED(arrmeta), const ndt::type &DYND_UNUSED(result_tp),
    char *DYND_UNUSED(out_arrmeta), memory_block_data *DYND_UNUSED(embedded_reference), size_t DYND_UNUSED(current_i),
    const ndt::type &DYND_UNUSED(root_tp), bool DYND_UNUSED(leading_dimension), char **DYND_UNUSED(inout_data),
    memory_block_data **DYND_UNUSED(inout_dataref)) const
{
    throw type_error("Cannot store data of funcproto type");
}

bool arrfunc_type::is_lossless_assignment(const ndt::type& dst_tp, const ndt::type& src_tp) const
{
    if (dst_tp.extended() == this) {
        if (src_tp.extended() == this) {
            return true;
        } else if (src_tp.get_type_id() == arrfunc_type_id) {
            return *dst_tp.extended() == *src_tp.extended();
        }
    }

    return false;
}

/*
size_t arrfunc_type::make_assignment_kernel(
                ckernel_builder *DYND_UNUSED(ckb), size_t DYND_UNUSED(ckb_offset),
                const ndt::type& dst_tp, const char *DYND_UNUSED(dst_arrmeta),
                const ndt::type& src_tp, const char *DYND_UNUSED(src_arrmeta),
                kernel_request_t DYND_UNUSED(kernreq), assign_error_mode DYND_UNUSED(errmode),
                const eval::eval_context *DYND_UNUSED(ectx)) const
{
    throw type_error("Cannot store data of funcproto type");
}

size_t arrfunc_type::make_comparison_kernel(
                ckernel_builder *DYND_UNUSED(ckb), intptr_t DYND_UNUSED(ckb_offset),
                const ndt::type& src0_tp, const char *DYND_UNUSED(src0_arrmeta),
                const ndt::type& src1_tp, const char *DYND_UNUSED(src1_arrmeta),
                comparison_type_t comptype,
                const eval::eval_context *DYND_UNUSED(ectx)) const
{
    throw type_error("Cannot store data of funcproto type");
}
*/

bool arrfunc_type::operator==(const base_type& rhs) const
{
    if (this == &rhs) {
        return true;
    } else if (rhs.get_type_id() != arrfunc_type_id) {
        return false;
    } else {
        const arrfunc_type *fpt = static_cast<const arrfunc_type *>(&rhs);
        return m_arg_types.equals_exact(fpt->m_arg_types) && m_return_type == fpt->m_return_type
            && m_naux == fpt->m_naux;
    }
}

void arrfunc_type::arrmeta_default_construct(
    char *DYND_UNUSED(arrmeta), bool DYND_UNUSED(blockref_alloc)) const
{
  throw type_error("Cannot store data of funcproto type");
}

void arrfunc_type::arrmeta_copy_construct(
    char *DYND_UNUSED(dst_arrmeta), const char *DYND_UNUSED(src_arrmeta),
    memory_block_data *DYND_UNUSED(embedded_reference)) const
{
    throw type_error("Cannot store data of funcproto type");
}

void arrfunc_type::arrmeta_destruct(char *DYND_UNUSED(arrmeta)) const
{
    throw type_error("Cannot store data of funcproto type");
}

static nd::array property_get_arg_types(const ndt::type& dt) {
    return dt.extended<arrfunc_type>()->get_arg_types();
}

static nd::array property_get_return_type(const ndt::type& dt) {
    return dt.extended<arrfunc_type>()->get_return_type();
}

void arrfunc_type::get_dynamic_type_properties(const std::pair<std::string, gfunc::callable> **out_properties, size_t *out_count) const
{
    static pair<string, gfunc::callable> type_properties[] = {
        pair<string, gfunc::callable>(
            "arg_types",
            gfunc::make_callable(&property_get_arg_types, "self")),
        pair<string, gfunc::callable>(
            "return_type",
            gfunc::make_callable(&property_get_return_type, "self"))};

    *out_properties = type_properties;
    *out_count = sizeof(type_properties) / sizeof(type_properties[0]);
}

ndt::type ndt::make_generic_funcproto(intptr_t nargs)
{
  vector<ndt::type> args;
  ndt::make_typevar_range("T", nargs, args);
  ndt::type ret = ndt::make_typevar("R");
  return ndt::make_funcproto(args, ret);
}
