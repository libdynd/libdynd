//
// Copyright (C) 2011-14 Mark Wiebe, DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <dynd/types/funcproto_type.hpp>
#include <dynd/types/strided_dim_type.hpp>
#include <dynd/func/make_callable.hpp>
#include <dynd/ensure_immutable_contig.hpp>

using namespace std;
using namespace dynd;

funcproto_type::funcproto_type(const nd::array &param_types,
                               const ndt::type &return_type)
    : base_type(funcproto_type_id, symbolic_kind, 0, 1, type_flag_none, 0, 0,
                0),
      m_param_types(param_types), m_return_type(return_type)
{
    if (!nd::ensure_immutable_contig<ndt::type>(m_param_types)) {
        stringstream ss;
        ss << "dynd funcproto param types requires an array of types, got an "
              "array with type " << m_param_types.get_type();
        throw invalid_argument(ss.str());
    }
    m_param_count = reinterpret_cast<const strided_dim_type_arrmeta *>(
                        m_param_types.get_arrmeta())->dim_size;
    m_members.flags |= return_type.get_flags() & type_flags_value_inherited;
    for (intptr_t i = 0; i != m_param_count; ++i) {
        m_members.flags |=
            get_param_type(i).get_flags() & type_flags_value_inherited;
    }
}

void funcproto_type::print_data(std::ostream &DYND_UNUSED(o),
                                const char *DYND_UNUSED(arrmeta),
                                const char *DYND_UNUSED(data)) const
{
    throw type_error("Cannot store data of funcproto type");
}

void funcproto_type::print_type(std::ostream& o) const
{
    const ndt::type *param_types = get_param_types_raw();
    // Use the function prototype datashape syntax
    o << "(";
    for (size_t i = 0, i_end = m_param_count; i != i_end; ++i) {
        if (i != 0) {
            o << ", ";
        }
        o << param_types[i];
    }
    o << ") -> " << m_return_type;
}

void funcproto_type::transform_child_types(type_transform_fn_t transform_fn, void *extra,
                ndt::type& out_transformed_tp, bool& out_was_transformed) const
{
    const ndt::type *param_types = get_param_types_raw();
    std::vector<ndt::type> tmp_param_types(m_param_count);
    ndt::type tmp_return_type;

    bool was_transformed = false;
    for (size_t i = 0, i_end = m_param_count; i != i_end; ++i) {
        transform_fn(param_types[i], extra, tmp_param_types[i], was_transformed);
    }
    transform_fn(m_return_type, extra, tmp_return_type, was_transformed);
    if (was_transformed) {
        out_transformed_tp =
            ndt::make_funcproto(tmp_param_types, tmp_return_type);
        out_was_transformed = true;
    } else {
        out_transformed_tp = ndt::type(this, true);
    }
}

ndt::type funcproto_type::get_canonical_type() const
{
    const ndt::type *param_types = get_param_types_raw();
    std::vector<ndt::type> tmp_param_types(m_param_count);
    ndt::type return_type;

    for (size_t i = 0, i_end = m_param_count; i != i_end; ++i) {
        tmp_param_types[i] = param_types[i].get_canonical_type();
    }
    return_type = m_return_type.get_canonical_type();

    return ndt::make_funcproto(tmp_param_types, return_type);
}

ndt::type funcproto_type::apply_linear_index(
    intptr_t DYND_UNUSED(nindices), const irange *DYND_UNUSED(indices),
    size_t DYND_UNUSED(current_i), const ndt::type &DYND_UNUSED(root_tp),
    bool DYND_UNUSED(leading_dimension)) const
{
    throw type_error("Cannot store data of funcproto type");
}

intptr_t funcproto_type::apply_linear_index(
    intptr_t DYND_UNUSED(nindices), const irange *DYND_UNUSED(indices),
    const char *DYND_UNUSED(arrmeta), const ndt::type &DYND_UNUSED(result_tp),
    char *DYND_UNUSED(out_arrmeta), memory_block_data *DYND_UNUSED(embedded_reference), size_t DYND_UNUSED(current_i),
    const ndt::type &DYND_UNUSED(root_tp), bool DYND_UNUSED(leading_dimension), char **DYND_UNUSED(inout_data),
    memory_block_data **DYND_UNUSED(inout_dataref)) const
{
    throw type_error("Cannot store data of funcproto type");
}

bool funcproto_type::is_lossless_assignment(const ndt::type& dst_tp, const ndt::type& src_tp) const
{
    if (dst_tp.extended() == this) {
        if (src_tp.extended() == this) {
            return true;
        } else if (src_tp.get_type_id() == funcproto_type_id) {
            return *dst_tp.extended() == *src_tp.extended();
        }
    }

    return false;
}

/*
size_t funcproto_type::make_assignment_kernel(
                ckernel_builder *DYND_UNUSED(ckb), size_t DYND_UNUSED(ckb_offset),
                const ndt::type& dst_tp, const char *DYND_UNUSED(dst_arrmeta),
                const ndt::type& src_tp, const char *DYND_UNUSED(src_arrmeta),
                kernel_request_t DYND_UNUSED(kernreq), assign_error_mode DYND_UNUSED(errmode),
                const eval::eval_context *DYND_UNUSED(ectx)) const
{
    throw type_error("Cannot store data of funcproto type");
}

size_t funcproto_type::make_comparison_kernel(
                ckernel_builder *DYND_UNUSED(ckb), intptr_t DYND_UNUSED(ckb_offset),
                const ndt::type& src0_tp, const char *DYND_UNUSED(src0_arrmeta),
                const ndt::type& src1_tp, const char *DYND_UNUSED(src1_arrmeta),
                comparison_type_t comptype,
                const eval::eval_context *DYND_UNUSED(ectx)) const
{
    throw type_error("Cannot store data of funcproto type");
}
*/

bool funcproto_type::operator==(const base_type& rhs) const
{
    if (this == &rhs) {
        return true;
    } else if (rhs.get_type_id() != funcproto_type_id) {
        return false;
    } else {
        const funcproto_type *fpt = static_cast<const funcproto_type *>(&rhs);
        return m_param_types.equals_exact(fpt->m_param_types) && m_return_type == fpt->m_return_type;
    }
}

void funcproto_type::arrmeta_default_construct(
    char *DYND_UNUSED(arrmeta), intptr_t DYND_UNUSED(ndim),
    const intptr_t *DYND_UNUSED(shape)) const
{
    throw type_error("Cannot store data of funcproto type");
}

void funcproto_type::arrmeta_copy_construct(
    char *DYND_UNUSED(dst_arrmeta), const char *DYND_UNUSED(src_arrmeta),
    memory_block_data *DYND_UNUSED(embedded_reference)) const
{
    throw type_error("Cannot store data of funcproto type");
}

void funcproto_type::arrmeta_destruct(char *DYND_UNUSED(arrmeta)) const
{
    throw type_error("Cannot store data of funcproto type");
}

static nd::array property_get_param_types(const ndt::type& dt) {
    return dt.tcast<funcproto_type>()->get_param_types();
}

static nd::array property_get_return_type(const ndt::type& dt) {
    return dt.tcast<funcproto_type>()->get_return_type();
}

void funcproto_type::get_dynamic_type_properties(const std::pair<std::string, gfunc::callable> **out_properties, size_t *out_count) const
{
    static pair<string, gfunc::callable> type_properties[] = {
        pair<string, gfunc::callable>(
            "param_types",
            gfunc::make_callable(&property_get_param_types, "self")),
        pair<string, gfunc::callable>(
            "return_type",
            gfunc::make_callable(&property_get_return_type, "self"))};

    *out_properties = type_properties;
    *out_count = sizeof(type_properties) / sizeof(type_properties[0]);
}
