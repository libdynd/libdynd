//
// Copyright (C) 2011-13, DyND Developers
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

static void format_datashape(std::ostream& o, const dtype& dt, const char *metadata,
                const std::string& indent, bool multiline, int *identifier);

static void format_identifier_string(std::ostream& o, int *identifier)
{
    if (*identifier < 26) {
        string result("A");
        result[0] += *identifier;
        o << result;
    } else {
        o << "X" << identifier - 26;
    }
    ++(*identifier);
}

static void format_struct_datashape(std::ostream& o, const dtype& dt, const char *metadata,
                const std::string& indent, bool multiline, int *identifier)
{
    const base_struct_dtype *bsd = static_cast<const base_struct_dtype *>(dt.extended());
    size_t field_count = bsd->get_field_count();
    const string *field_names = bsd->get_field_names();
    const dtype *field_types = bsd->get_field_types();
    const size_t *metadata_offsets = bsd->get_metadata_offsets();
    o << (multiline ? "{\n" : "{");
    for (size_t i = 0; i < field_count; ++i) {
        if (multiline) {
            o << indent << "  ";
        }
        o << field_names[i] << ": ";
        format_datashape(o, field_types[i], metadata ? (metadata + metadata_offsets[i]) : NULL,
                        multiline ? (indent + "  ") : indent, multiline, identifier);
        if (multiline) {
            o << ";\n";
        } else if (i != field_count - 1) {
            o << "; ";
        }
    }
    o << indent << "}";
}

static void format_uniform_array_datashape(std::ostream& o,
                const dtype& dt, const char *metadata,
                const std::string& indent, bool multiline, int *identifier)
{
    switch (dt.get_type_id()) {
        case strided_dim_type_id: {
            const strided_dim_dtype *sad = static_cast<const strided_dim_dtype *>(dt.extended());
            if (metadata) {
                // If metadata is provided, use the actual dimension size
                const strided_dim_dtype_metadata *md =
                                reinterpret_cast<const strided_dim_dtype_metadata *>(metadata);
                o << md->size << ", ";
                format_datashape(o, sad->get_element_dtype(),
                                metadata + sizeof(strided_dim_dtype_metadata),
                                indent, multiline, identifier);
            } else {
                // If no metadata, use a symbol
                format_identifier_string(o, identifier);
                o << ", ";
                format_datashape(o, sad->get_element_dtype(), NULL,
                                indent, multiline, identifier);
            }
            break;
        }
        case fixed_dim_type_id: {
            const fixed_dim_dtype *fad = static_cast<const fixed_dim_dtype *>(dt.extended());
            o << fad->get_fixed_dim_size() << ", ";
            format_datashape(o, fad->get_element_dtype(), metadata, indent, multiline, identifier);
            break;
        }
        case var_dim_type_id: {
            const var_dim_dtype *vad = static_cast<const var_dim_dtype *>(dt.extended());
            o << "VarDim, ";
            format_datashape(o, vad->get_element_dtype(),
                            metadata ? (metadata + sizeof(var_dim_dtype_metadata)) : NULL,
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

static void format_string_encoding(std::ostream& o, string_encoding_t enc)
{
    switch (enc) {
        case string_encoding_ascii:
            o << "'A'";
            break;
        case string_encoding_utf_8:
            o << "'U8'";
            break;
        case string_encoding_ucs_2:
            o << "'ucs2'";
            break;
        case string_encoding_utf_16:
            o << "'U16'";
            break;
        case string_encoding_utf_32:
            o << "'U32'";
            break;
        default: {
            stringstream ss;
            ss << "unrecognized string encoding " << enc << " while formatting datashape";
            throw runtime_error(ss.str());
        }
    }
}

static void format_string_datashape(std::ostream& o, const dtype& dt)
{
    switch (dt.get_type_id()) {
        case string_type_id: {
            const string_dtype *sd = static_cast<const string_dtype *>(dt.extended());
            o << "string";
            string_encoding_t enc = sd->get_encoding();
            if (enc != string_encoding_utf_8) {
                o << "(";
                format_string_encoding(o, enc);
                o << ")";
            }
            break;
        }
        case fixedstring_type_id: {
            const fixedstring_dtype *fsd = static_cast<const fixedstring_dtype *>(dt.extended());
            o << "string(";
            o << fsd->get_data_size() / fsd->get_alignment();
            string_encoding_t enc = fsd->get_encoding();
            if (enc != string_encoding_utf_8) {
                o << ",";
                format_string_encoding(o, enc);
            }
            o << ")";
            break;
        }
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

static void format_complex_datashape(std::ostream& o, const dtype& dt)
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

static void format_datashape(std::ostream& o, const dtype& dt, const char *metadata,
                const std::string& indent, bool multiline, int *identifier)
{
    switch (dt.get_kind()) {
        case struct_kind:
            format_struct_datashape(o, dt, metadata, indent, multiline, identifier);
            break;
        case uniform_dim_kind:
            format_uniform_array_datashape(o, dt, metadata, indent, multiline, identifier);
            break;
        case string_kind:
            format_string_datashape(o, dt);
            break;
        case complex_kind:
            format_complex_datashape(o, dt);
            break;
        default:
            o << dt;
            break;
    }
}

void dynd::format_datashape(std::ostream& o, const dtype& dt, const char *metadata,
                bool multiline)
{
    ::format_datashape(o, dt, metadata, "", multiline, 0);
}

string dynd::format_datashape(const ndobject& n,
                const std::string& prefix, bool multiline)
{
    stringstream ss;
    ss << prefix;
    int identifier = 0;
    ::format_datashape(ss, n.get_dtype(), n.get_ndo_meta(), "", multiline, &identifier);
    return ss.str();
}

string dynd::format_datashape(const dtype& d,
                const std::string& prefix, bool multiline)
{
    stringstream ss;
    ss << prefix;
    int identifier = 0;
    ::format_datashape(ss, d, NULL, "", multiline, &identifier);
    return ss.str();
}
