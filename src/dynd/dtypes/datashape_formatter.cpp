//
// Copyright (C) 2011-13, DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <dynd/dtypes/datashape_formatter.hpp>
#include <dynd/dtypes/base_struct_dtype.hpp>
#include <dynd/dtypes/strided_array_dtype.hpp>
#include <dynd/dtypes/fixedarray_dtype.hpp>
#include <dynd/dtypes/var_array_dtype.hpp>

using namespace std;
using namespace dynd;

static void format_datashape(std::ostream& o, const dtype& dt, const char *metadata, const std::string& indent);

static void format_struct_datashape(std::ostream& o, const dtype& dt, const char *metadata, const std::string& indent)
{
    const base_struct_dtype *bsd = static_cast<const base_struct_dtype *>(dt.extended());
    size_t field_count = bsd->get_field_count();
    const string *field_names = bsd->get_field_names();
    const dtype *field_types = bsd->get_field_types();
    const size_t *metadata_offsets = bsd->get_metadata_offsets();
    o << "{\n";
    for (size_t i = 0; i < field_count; ++i) {
        o << indent << "  " << field_names[i] << ": ";
        format_datashape(o, field_types[i], metadata + metadata_offsets[i], indent + "  ");
        o << ";\n";
    }
    o << indent << "}";
}

static void format_uniform_array_datashape(std::ostream& o, const dtype& dt, const char *metadata, const std::string& indent)
{
    switch (dt.get_type_id()) {
        case strided_array_type_id: {
            const strided_array_dtype *sad = static_cast<const strided_array_dtype *>(dt.extended());
            const strided_array_dtype_metadata *md = reinterpret_cast<const strided_array_dtype_metadata *>(metadata);
            o << md->size << ", ";
            format_datashape(o, sad->get_element_dtype(),
                            metadata + sizeof(strided_array_dtype_metadata), indent);
            break;
        }
        case fixedarray_type_id: {
            const fixedarray_dtype *fad = static_cast<const fixedarray_dtype *>(dt.extended());
            o << fad->get_fixed_dim_size() << ", ";
            format_datashape(o, fad->get_element_dtype(), metadata, indent);
            break;
        }
        case var_array_type_id: {
            const var_array_dtype *vad = static_cast<const var_array_dtype *>(dt.extended());
            o << "VarDim, ";
            format_datashape(o, vad->get_element_dtype(),
                            metadata + sizeof(var_array_dtype_metadata), indent);
            break;
        }
        default: {
            stringstream ss;
            ss << "Datashape formatting for dtype " << dt << " is not yet implemented";
            throw runtime_error(ss.str());
        }
    }
}

static void format_datashape(std::ostream& o, const dtype& dt, const char *metadata, const std::string& indent)
{
    switch (dt.get_kind()) {
        case struct_kind:
            format_struct_datashape(o, dt, metadata, indent);
            break;
        case uniform_array_kind:
            format_uniform_array_datashape(o, dt, metadata, indent);
            break;
        default:
            o << dt;
            break;
    }
}

string dynd::format_datashape(const ndobject& n)
{
    stringstream ss;
    ss << "type BlazeDataShape = ";
    ::format_datashape(ss, n.get_dtype(), n.get_ndo_meta(), "");
    return ss.str();
}