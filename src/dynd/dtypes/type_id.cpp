//
// Copyright (C) 2011-13, DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <dynd/dtypes/type_id.hpp>

using namespace std;
using namespace dynd;

std::ostream& dynd::operator<<(std::ostream& o, dtype_kind_t kind)
{
    switch (kind) {
        case bool_kind:
            return (o << "bool");
        case int_kind:
            return (o << "int");
        case uint_kind:
            return (o << "uint");
        case real_kind:
            return (o << "real");
        case complex_kind:
            return (o << "complex");
        case string_kind:
            return (o << "string");
        case bytes_kind:
            return (o << "bytes");
        case void_kind:
            return (o << "void");
        case datetime_kind:
            return (o << "datetime");
        case uniform_dim_kind:
            return (o << "uniform_dim");
        case struct_kind:
            return (o << "struct");
        case expression_kind:
            return (o << "expression");
        case pattern_kind:
            return (o << "pattern");
        case custom_kind:
            return (o << "custom");
        default:
            return (o << "(unknown kind " << (int)kind << ")");
    }
}

std::ostream& dynd::operator<<(std::ostream& o, type_id_t tid)
{
    switch (tid) {
        case uninitialized_type_id:
            return (o << "uninitialized");
        case bool_type_id:
            return (o << "bool");
        case int8_type_id:
            return (o << "int8");
        case int16_type_id:
            return (o << "int16");
        case int32_type_id:
            return (o << "int32");
        case int64_type_id:
            return (o << "int64");
        case uint8_type_id:
            return (o << "uint8");
        case uint16_type_id:
            return (o << "uint16");
        case uint32_type_id:
            return (o << "uint32");
        case uint64_type_id:
            return (o << "uint64");
        case float32_type_id:
            return (o << "float32");
        case float64_type_id:
            return (o << "float64");
        case complex_float32_type_id:
            return (o << "complex_float32");
        case complex_float64_type_id:
            return (o << "complex_float64");
        case void_type_id:
            return (o << "void");
        case void_pointer_type_id:
            return (o << "void_pointer");
        case pointer_type_id:
            return (o << "pointer");
        case bytes_type_id:
            return (o << "bytes");
        case fixedbytes_type_id:
            return (o << "fixedbytes");
        case string_type_id:
            return (o << "string");
        case fixedstring_type_id:
            return (o << "fixedstring");
        case categorical_type_id:
            return (o << "categorical");
        case date_type_id:
            return (o << "date");
        case busdate_type_id:
            return (o << "busdate");
        case json_type_id:
            return (o << "json");
        case strided_dim_type_id:
            return (o << "strided_dim");
        case fixed_dim_type_id:
            return (o << "fixed_dim");
        case var_dim_type_id:
            return (o << "var_dim");
        case struct_type_id:
            return (o << "struct");
        case fixedstruct_type_id:
            return (o << "fixedstruct");
        case tuple_type_id:
            return (o << "tuple");
        case ndobject_type_id:
            return (o << "ndobject");
        case convert_type_id:
            return (o << "convert");
        case byteswap_type_id:
            return (o << "byteswap");
        case view_type_id:
            return (o << "view");
        case property_type_id:
            return (o << "property");
        case groupby_type_id:
            return (o << "groupby");
        default:
            return (o << "(unknown type id " << (int)tid << ")");
    }
}
