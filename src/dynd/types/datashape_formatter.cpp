//
// Copyright (C) 2011-14 Mark Wiebe, DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <dynd/types/datashape_formatter.hpp>
#include <dynd/types/base_struct_type.hpp>
#include <dynd/types/strided_dim_type.hpp>
#include <dynd/types/fixed_dim_type.hpp>
#include <dynd/types/cfixed_dim_type.hpp>
#include <dynd/types/var_dim_type.hpp>
#include <dynd/types/string_type.hpp>
#include <dynd/types/fixedstring_type.hpp>

using namespace std;
using namespace dynd;

static void format_datashape(std::ostream& o, const ndt::type& dt, const char *arrmeta, const char *data,
                const std::string& indent, bool multiline);

static void format_struct_datashape(std::ostream &o, const ndt::type &dt,
                                    const char *arrmeta, const char *data,
                                    const std::string &indent, bool multiline)
{
    // The data requires arrmeta
    if (arrmeta == NULL) {
        data = NULL;
    }
    const base_struct_type *bsd = dt.tcast<base_struct_type>();
    size_t field_count = bsd->get_field_count();
    const uintptr_t *arrmeta_offsets = bsd->get_arrmeta_offsets_raw();
    const uintptr_t *data_offsets = NULL;
    if (data != NULL) {
        data_offsets = bsd->get_data_offsets(arrmeta);
    }
    o << (multiline ? "{\n" : "{");
    for (size_t i = 0; i < field_count; ++i) {
        if (multiline) {
            o << indent << "  ";
        }
        o << bsd->get_field_name(i) << ": ";
        format_datashape(o, bsd->get_field_type(i), arrmeta ? (arrmeta + arrmeta_offsets[i]) : NULL,
                        data ? (data + data_offsets[i]) : NULL,
                        multiline ? (indent + "  ") : indent, multiline);
        if (multiline) {
            o << ",\n";
        } else if (i != field_count - 1) {
            o << ", ";
        }
    }
    o << indent << "}";
}

static void format_uniform_dim_datashape(std::ostream& o,
                const ndt::type& dt, const char *arrmeta, const char *data,
                const std::string& indent, bool multiline)
{
    switch (dt.get_type_id()) {
        case strided_dim_type_id: {
            const strided_dim_type *sad = dt.tcast<strided_dim_type>();
            if (arrmeta) {
                // If arrmeta is provided, use the actual dimension size
                const strided_dim_type_arrmeta *md =
                                reinterpret_cast<const strided_dim_type_arrmeta *>(arrmeta);
                o << md->dim_size << " * ";
                // Allow data to keep going only if the dimension size is 1
                if (md->dim_size != 1) {
                    data = NULL;
                }
                format_datashape(o, sad->get_element_type(),
                                arrmeta + sizeof(strided_dim_type_arrmeta), data,
                                indent, multiline);
            } else {
                // If no arrmeta, use "strided"
                o << "strided * ";
                format_datashape(o, sad->get_element_type(), NULL, NULL,
                                indent, multiline);
            }
            break;
        }
        case fixed_dim_type_id: {
            const fixed_dim_type *fad = dt.tcast<fixed_dim_type>();
            intptr_t dim_size = fad->get_fixed_dim_size();
            o << dim_size << " * ";
            // Allow data to keep going only if the dimension size is 1
            if (dim_size != 1) {
                data = NULL;
            }
            format_datashape(o, fad->get_element_type(),
                             arrmeta +
                                 (arrmeta ? sizeof(fixed_dim_type_arrmeta) : 0),
                             data, indent, multiline);
            break;
        }
        case cfixed_dim_type_id: {
            const cfixed_dim_type *fad = dt.tcast<cfixed_dim_type>();
            intptr_t dim_size = fad->get_fixed_dim_size();
            o << dim_size << " * ";
            // Allow data to keep going only if the dimension size is 1
            if (dim_size != 1) {
                data = NULL;
            }
            format_datashape(
                o, fad->get_element_type(),
                arrmeta + (arrmeta ? sizeof(cfixed_dim_type_arrmeta) : 0), data,
                indent, multiline);
            break;
        }
        case var_dim_type_id: {
            const var_dim_type *vad = dt.tcast<var_dim_type>();
            const char *child_data = NULL;
            if (data == NULL || arrmeta == NULL) {
                o << "var * ";
            } else {
                const var_dim_type_data *d = reinterpret_cast<const var_dim_type_data *>(data);
                if (d->begin == NULL) {
                    o << "var * ";
                } else {
                    o << d->size << " * ";
                    if (d->size == 1) {
                        const var_dim_type_arrmeta *md = reinterpret_cast<const var_dim_type_arrmeta *>(arrmeta);
                        child_data = d->begin + md->offset;
                    }
                }
            }
            format_datashape(o, vad->get_element_type(),
                            arrmeta ? (arrmeta + sizeof(var_dim_type_arrmeta)) : NULL,
                            child_data,
                            indent, multiline);
            break;
        }
        default: {
            stringstream ss;
            ss << "Datashape formatting for dynd type " << dt << " is not yet implemented";
            throw runtime_error(ss.str());
        }
    }
}

static void format_string_datashape(std::ostream& o, const ndt::type& dt)
{
    switch (dt.get_type_id()) {
        case string_type_id:
        case fixedstring_type_id:
            // data shape only has one kind of string
            o << "string";
            break;
        case json_type_id: {
            o << "json";
            break;
        }
        default: {
            stringstream ss;
            ss << "unrecognized string dynd type " << dt << " while formatting datashape";
            throw dynd::type_error(ss.str());
        }
    }
}

static void format_complex_datashape(std::ostream& o, const ndt::type& dt)
{
    switch (dt.get_type_id()) {
        case complex_float32_type_id:
            o << "complex[float32]";
            break;
        case complex_float64_type_id:
            o << "complex[float64]";
            break;
        default: {
            stringstream ss;
            ss << "unrecognized string complex type " << dt << " while formatting datashape";
            throw dynd::type_error(ss.str());
        }
    }
}

static void format_datashape(std::ostream& o, const ndt::type& dt, const char *arrmeta, const char *data,
                const std::string& indent, bool multiline)
{
    switch (dt.get_kind()) {
        case struct_kind:
            format_struct_datashape(o, dt, arrmeta, data, indent, multiline);
            break;
        case uniform_dim_kind:
            format_uniform_dim_datashape(o, dt, arrmeta, data, indent, multiline);
            break;
        case string_kind:
            format_string_datashape(o, dt);
            break;
        case complex_kind:
            format_complex_datashape(o, dt);
            break;
        case expression_kind:
            format_datashape(o, dt.value_type(), NULL, NULL, indent, multiline);
            break;
        default:
            o << dt;
            break;
    }
}

void dynd::format_datashape(std::ostream& o, const ndt::type& dt, const char *arrmeta, const char *data,
                bool multiline)
{
    ::format_datashape(o, dt, arrmeta, data, "", multiline);
}

string dynd::format_datashape(const nd::array& n,
                const std::string& prefix, bool multiline)
{
    stringstream ss;
    ss << prefix;
    ::format_datashape(ss, n.get_type(), n.get_arrmeta(),
                    n.get_readonly_originptr(), "", multiline);
    return ss.str();
}

string dynd::format_datashape(const ndt::type& d,
                const std::string& prefix, bool multiline)
{
    stringstream ss;
    ss << prefix;
    ::format_datashape(ss, d, NULL, NULL, "", multiline);
    return ss.str();
}
