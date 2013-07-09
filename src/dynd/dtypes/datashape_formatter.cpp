//
// Copyright (C) 2011-13 Mark Wiebe, DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <dynd/dtypes/datashape_formatter.hpp>
#include <dynd/dtypes/base_struct_dtype.hpp>
#include <dynd/dtypes/strided_dim_dtype.hpp>
#include <dynd/dtypes/fixed_dim_dtype.hpp>
#include <dynd/dtypes/var_dim_dtype.hpp>
#include <dynd/dtypes/string_dtype.hpp>
#include <dynd/dtypes/fixedstring_dtype.hpp>

using namespace std;
using namespace dynd;

static void format_datashape(std::ostream& o, const ndt::type& dt, const char *metadata, const char *data,
                const std::string& indent, bool multiline, int &identifier);

static void format_identifier_string(std::ostream& o, int &identifier)
{
    if (identifier < 26) {
        string result("A");
        result[0] += identifier;
        o << result;
    } else {
        o << "X" << (identifier - 26);
    }
    ++identifier;
}

static void format_struct_datashape(std::ostream& o, const ndt::type& dt, const char *metadata, const char *data,
                const std::string& indent, bool multiline, int &identifier)
{
    // The data requires metadata
    if (metadata == NULL) {
        data = NULL;
    }
    const base_struct_dtype *bsd = static_cast<const base_struct_dtype *>(dt.extended());
    size_t field_count = bsd->get_field_count();
    const string *field_names = bsd->get_field_names();
    const ndt::type *field_types = bsd->get_field_types();
    const size_t *metadata_offsets = bsd->get_metadata_offsets();
    const size_t *data_offsets = NULL;
    if (data != NULL) {
        data_offsets = bsd->get_data_offsets(metadata);
    }
    o << (multiline ? "{\n" : "{");
    for (size_t i = 0; i < field_count; ++i) {
        if (multiline) {
            o << indent << "  ";
        }
        o << field_names[i] << ": ";
        format_datashape(o, field_types[i], metadata ? (metadata + metadata_offsets[i]) : NULL,
                        data ? (data + data_offsets[i]) : NULL,
                        multiline ? (indent + "  ") : indent, multiline, identifier);
        if (multiline) {
            o << ";\n";
        } else if (i != field_count - 1) {
            o << "; ";
        }
    }
    o << indent << "}";
}

static void format_uniform_dim_datashape(std::ostream& o,
                const ndt::type& dt, const char *metadata, const char *data,
                const std::string& indent, bool multiline, int &identifier)
{
    switch (dt.get_type_id()) {
        case strided_dim_type_id: {
            const strided_dim_dtype *sad = static_cast<const strided_dim_dtype *>(dt.extended());
            if (metadata) {
                // If metadata is provided, use the actual dimension size
                const strided_dim_dtype_metadata *md =
                                reinterpret_cast<const strided_dim_dtype_metadata *>(metadata);
                o << md->size << ", ";
                // Allow data to keep going only if the dimension size is 1
                if (md->size != 1) {
                    data = NULL;
                }
                format_datashape(o, sad->get_element_type(),
                                metadata + sizeof(strided_dim_dtype_metadata), data,
                                indent, multiline, identifier);
            } else {
                // If no metadata, use a symbol
                format_identifier_string(o, identifier);
                o << ", ";
                format_datashape(o, sad->get_element_type(), NULL, NULL,
                                indent, multiline, identifier);
            }
            break;
        }
        case fixed_dim_type_id: {
            const fixed_dim_dtype *fad = static_cast<const fixed_dim_dtype *>(dt.extended());
            size_t dim_size = fad->get_fixed_dim_size();
            o << dim_size << ", ";
            // Allow data to keep going only if the dimension size is 1
            if (dim_size != 1) {
                data = NULL;
            }
            format_datashape(o, fad->get_element_type(), metadata, data, indent, multiline, identifier);
            break;
        }
        case var_dim_type_id: {
            const var_dim_dtype *vad = static_cast<const var_dim_dtype *>(dt.extended());
            if (data == NULL) {
                o << "var, ";
            } else {
                const var_dim_dtype_data *d = reinterpret_cast<const var_dim_dtype_data *>(data);
                if (d->begin == NULL) {
                    o << "var, ";
                } else {
                    o << d->size << ", ";
                }
            }
            format_datashape(o, vad->get_element_type(),
                            metadata ? (metadata + sizeof(var_dim_dtype_metadata)) : NULL, NULL,
                            indent, multiline, identifier);
            break;
        }
        default: {
            stringstream ss;
            ss << "Datashape formatting for dtype " << dt << " is not yet implemented";
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
            throw runtime_error(ss.str());
        }
    }
}

static void format_complex_datashape(std::ostream& o, const ndt::type& dt)
{
    switch (dt.get_type_id()) {
        case complex_float32_type_id:
            o << "cfloat32";
            break;
        case complex_float64_type_id:
            o << "cfloat64";
            break;
        default: {
            stringstream ss;
            ss << "unrecognized string complex type " << dt << " while formatting datashape";
            throw runtime_error(ss.str());
        }
    }
}

static void format_datashape(std::ostream& o, const ndt::type& dt, const char *metadata, const char *data,
                const std::string& indent, bool multiline, int &identifier)
{
    switch (dt.get_kind()) {
        case struct_kind:
            format_struct_datashape(o, dt, metadata, data, indent, multiline, identifier);
            break;
        case uniform_dim_kind:
            format_uniform_dim_datashape(o, dt, metadata, data, indent, multiline, identifier);
            break;
        case string_kind:
            format_string_datashape(o, dt);
            break;
        case complex_kind:
            format_complex_datashape(o, dt);
            break;
        case expression_kind:
            format_datashape(o, dt.value_type(), NULL, NULL, indent, multiline, identifier);
            break;
        default:
            o << dt;
            break;
    }
}

void dynd::format_datashape(std::ostream& o, const ndt::type& dt, const char *metadata, const char *data,
                bool multiline)
{
    int identifier = 0;
    ::format_datashape(o, dt, metadata, data, "", multiline, identifier);
}

string dynd::format_datashape(const nd::array& n,
                const std::string& prefix, bool multiline)
{
    stringstream ss;
    ss << prefix;
    int identifier = 0;
    ::format_datashape(ss, n.get_dtype(), n.get_ndo_meta(),
                    n.get_readonly_originptr(), "", multiline, identifier);
    return ss.str();
}

string dynd::format_datashape(const ndt::type& d,
                const std::string& prefix, bool multiline)
{
    stringstream ss;
    ss << prefix;
    int identifier = 0;
    ::format_datashape(ss, d, NULL, NULL, "", multiline, identifier);
    return ss.str();
}
