//
// Copyright (C) 2011-14 Mark Wiebe, DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <dynd/types/ellipsis_dim_type.hpp>
#include <dynd/types/typevar_type.hpp>
#include <dynd/func/make_callable.hpp>

using namespace std;
using namespace dynd;

ellipsis_dim_type::ellipsis_dim_type(const nd::string &name,
                                     const ndt::type &element_type)
    : base_dim_type(ellipsis_dim_type_id, element_type, 0, 1, 0,
                            type_flag_symbolic, false),
      m_name(name)
{
    if (!m_name.is_null()) {
        // Make sure name begins with a capital letter, and is an identifier
        const char *begin = m_name.begin(), *end = m_name.end();
        if (end - begin == 0) {
            // Convert empty string into NULL
            m_name = nd::string();
        } else if (!is_valid_typevar_name(begin, end)) {
            stringstream ss;
            ss << "dynd ellipsis name \"";
            print_escaped_utf8_string(ss, begin, end);
            ss << "\" is not valid, it must be alphanumeric and begin with a capital";
            throw type_error(ss.str());
        }
    }
}

void ellipsis_dim_type::print_data(std::ostream &DYND_UNUSED(o),
                                const char *DYND_UNUSED(arrmeta),
                                const char *DYND_UNUSED(data)) const
{
    throw type_error("Cannot store data of ellipsis type");
}

void ellipsis_dim_type::print_type(std::ostream& o) const
{
    // Type variables are barewords starting with a capital letter
    if (!m_name.is_null()) {
        o << m_name.str();
    }
    o << "... * " << get_element_type();
}

ndt::type ellipsis_dim_type::apply_linear_index(
    intptr_t DYND_UNUSED(nindices), const irange *DYND_UNUSED(indices),
    size_t DYND_UNUSED(current_i), const ndt::type &DYND_UNUSED(root_tp),
    bool DYND_UNUSED(leading_dimension)) const
{
    throw type_error("Cannot store data of ellipsis type");
}

intptr_t ellipsis_dim_type::apply_linear_index(
    intptr_t DYND_UNUSED(nindices), const irange *DYND_UNUSED(indices),
    const char *DYND_UNUSED(arrmeta), const ndt::type &DYND_UNUSED(result_tp),
    char *DYND_UNUSED(out_arrmeta), memory_block_data *DYND_UNUSED(embedded_reference), size_t DYND_UNUSED(current_i),
    const ndt::type &DYND_UNUSED(root_tp), bool DYND_UNUSED(leading_dimension), char **DYND_UNUSED(inout_data),
    memory_block_data **DYND_UNUSED(inout_dataref)) const
{
    throw type_error("Cannot store data of ellipsis type");
}

intptr_t ellipsis_dim_type::get_dim_size(const char *DYND_UNUSED(arrmeta),
                                        const char *DYND_UNUSED(data)) const
{
    return -1;
}

bool ellipsis_dim_type::is_lossless_assignment(const ndt::type& dst_tp, const ndt::type& src_tp) const
{
    if (dst_tp.extended() == this) {
        if (src_tp.extended() == this) {
            return true;
        } else if (src_tp.get_type_id() == ellipsis_dim_type_id) {
            return *dst_tp.extended() == *src_tp.extended();
        }
    }

    return false;
}

bool ellipsis_dim_type::operator==(const base_type& rhs) const
{
    if (this == &rhs) {
        return true;
    } else if (rhs.get_type_id() != ellipsis_dim_type_id) {
        return false;
    } else {
        const ellipsis_dim_type *tvt =
            static_cast<const ellipsis_dim_type *>(&rhs);
        return m_name == tvt->m_name &&
               m_element_tp == tvt->m_element_tp;
    }
}

void ellipsis_dim_type::arrmeta_default_construct(
    char *DYND_UNUSED(arrmeta), intptr_t DYND_UNUSED(ndim),
    const intptr_t *DYND_UNUSED(shape)) const
{
    throw type_error("Cannot store data of ellipsis type");
}

void ellipsis_dim_type::arrmeta_copy_construct(
    char *DYND_UNUSED(dst_arrmeta), const char *DYND_UNUSED(src_arrmeta),
    memory_block_data *DYND_UNUSED(embedded_reference)) const
{
    throw type_error("Cannot store data of ellipsis type");
}

size_t ellipsis_dim_type::arrmeta_copy_construct_onedim(
    char *DYND_UNUSED(dst_arrmeta), const char *DYND_UNUSED(src_arrmeta),
    memory_block_data *DYND_UNUSED(embedded_reference)) const
{
    throw type_error("Cannot store data of ellipsis type");
}

void ellipsis_dim_type::arrmeta_destruct(char *DYND_UNUSED(arrmeta)) const
{
    throw type_error("Cannot store data of ellipsis type");
}

static nd::array property_get_name(const ndt::type& tp) {
    return tp.tcast<ellipsis_dim_type>()->get_name();
}

static ndt::type property_get_element_type(const ndt::type& dt) {
    return dt.tcast<ellipsis_dim_type>()->get_element_type();
}

void ellipsis_dim_type::get_dynamic_type_properties(
    const std::pair<std::string, gfunc::callable> **out_properties,
    size_t *out_count) const
{
    static pair<string, gfunc::callable> type_properties[] = {
        pair<string, gfunc::callable>(
            "name", gfunc::make_callable(&property_get_name, "self")),
        pair<string, gfunc::callable>(
            "element_type",
            gfunc::make_callable(&property_get_element_type, "self")), };

    *out_properties = type_properties;
    *out_count = sizeof(type_properties) / sizeof(type_properties[0]);
}
