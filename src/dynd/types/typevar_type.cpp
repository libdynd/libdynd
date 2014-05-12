//
// Copyright (C) 2011-14 Mark Wiebe, DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <dynd/types/typevar_type.hpp>
#include <dynd/gfunc/make_callable.hpp>

using namespace std;
using namespace dynd;

typevar_type::typevar_type(const nd::array& name)
    : base_type(typevar_type_id, symbolic_kind, 0, 1, type_flag_none, 0, 0),
      m_name(name)
{
    if (!(m_name.is_immutable() &&
          m_name.get_type().get_type_id() == string_type_id &&
          m_name.get_type().tcast<string_type>()->get_encoding() ==
              string_encoding_utf_8)) {
        // Make sure name is an immutable utf8 string
        if (m_name.get_type().value_type().get_kind() == string_kind) {
            m_name = m_name.ucast(ndt::make_string()).eval_immutable();
        } else {
            stringstream ss;
            ss << "dynd typevar requires a string as the "
                  "typevar name, got type " << m_name.get_type();
            throw type_error(ss.str());
        }
    }
    // Make sure name begins with a capital letter, and is an identifier
    const string_type_data *std =
        reinterpret_cast<const string_type_data *>(
            m_name.get_readonly_originptr());
    const char *begin = std->begin, *end = std->end;
    if (end - begin == 0) {
        throw type_error("dynd typevar requires a non-empty name");
    }
    if (*begin < 'A' || *begin > 'Z') {
        stringstream ss;
        ss << "dynd typevar name \"";
        print_escaped_utf8_string(ss, std->begin, end);
        ss << "\" is invalid, it does not begin with a capital letter";
        throw type_error(ss.str());
    }
    ++begin;
    while (begin < end) {
        char c = *begin;
        if ((c < 'a' || c > 'z') && (c < 'A' || c > 'Z') &&
                (c < '0' || c > '9') && c != '_') {
            stringstream ss;
            ss << "dynd typevar name \"";
            print_escaped_utf8_string(ss, std->begin, end);
            ss << "\" is invalid, it has a character which is not "
                    "alphanumeric or underscore";
            throw type_error(ss.str());
        }
        ++begin;
    }
}

void typevar_type::print_data(std::ostream &DYND_UNUSED(o),
                                const char *DYND_UNUSED(metadata),
                                const char *DYND_UNUSED(data)) const
{
    throw type_error("Cannot store data of typevar type");
}

void typevar_type::print_type(std::ostream& o) const
{
    // Type variables are barewords starting with a capital letter
    o << m_name.as<string>();
}

ndt::type typevar_type::apply_linear_index(
    intptr_t DYND_UNUSED(nindices), const irange *DYND_UNUSED(indices),
    size_t DYND_UNUSED(current_i), const ndt::type &DYND_UNUSED(root_tp),
    bool DYND_UNUSED(leading_dimension)) const
{
    throw type_error("Cannot store data of typevar type");
}

intptr_t typevar_type::apply_linear_index(
    intptr_t DYND_UNUSED(nindices), const irange *DYND_UNUSED(indices),
    const char *DYND_UNUSED(metadata), const ndt::type &result_tp,
    char *DYND_UNUSED(out_metadata), memory_block_data *DYND_UNUSED(embedded_reference), size_t DYND_UNUSED(current_i),
    const ndt::type &DYND_UNUSED(root_tp), bool DYND_UNUSED(leading_dimension), char **DYND_UNUSED(inout_data),
    memory_block_data **DYND_UNUSED(inout_dataref)) const
{
    throw type_error("Cannot store data of typevar type");
}

bool typevar_type::is_lossless_assignment(const ndt::type& dst_tp, const ndt::type& src_tp) const
{
    if (dst_tp.extended() == this) {
        if (src_tp.extended() == this) {
            return true;
        } else if (src_tp.get_type_id() == typevar_type_id) {
            return *dst_tp.extended() == *src_tp.extended();
        }
    }

    return false;
}

bool typevar_type::operator==(const base_type& rhs) const
{
    if (this == &rhs) {
        return true;
    } else if (rhs.get_type_id() != typevar_type_id) {
        return false;
    } else {
        const typevar_type *tvt = static_cast<const typevar_type *>(&rhs);
        return m_name.equals_exact(tvt->m_name);
    }
}

void typevar_type::metadata_default_construct(
    char *DYND_UNUSED(metadata), intptr_t DYND_UNUSED(ndim),
    const intptr_t *DYND_UNUSED(shape)) const
{
    throw type_error("Cannot store data of typevar type");
}

void typevar_type::metadata_copy_construct(
    char *DYND_UNUSED(dst_metadata), const char *DYND_UNUSED(src_metadata),
    memory_block_data *DYND_UNUSED(embedded_reference)) const
{
    throw type_error("Cannot store data of typevar type");
}

void typevar_type::metadata_destruct(char *DYND_UNUSED(metadata)) const
{
    throw type_error("Cannot store data of typevar type");
}

static nd::array property_get_name(const ndt::type& tp) {
    return tp.tcast<typevar_type>()->get_name();
}

static pair<string, gfunc::callable> type_properties[] = {
    pair<string, gfunc::callable>("name", gfunc::make_callable(&property_get_name, "self")),
};

void typevar_type::get_dynamic_type_properties(const std::pair<std::string, gfunc::callable> **out_properties, size_t *out_count) const
{
    *out_properties = type_properties;
    *out_count = sizeof(type_properties) / sizeof(type_properties[0]);
}
