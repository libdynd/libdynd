//
// Copyright (C) 2011-13, DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <dynd/json_parser.hpp>
#include <dynd/dtypes/base_bytes_dtype.hpp>
#include <dynd/dtypes/string_dtype.hpp>

using namespace std;
using namespace dynd;

ndobject dynd::parse_json(const dtype& dt, const ndobject& json)
{
    // Check the dtype of 'json', and get pointers to the begin/end of a UTF-8 buffer
    ndobject json_tmp;
    const char *json_begin = NULL, *json_end = NULL;

    dtype json_dtype = json.get_dtype();
    switch (json_dtype.get_kind()) {
        case string_kind: {
            const base_string_dtype *sdt = static_cast<const base_string_dtype *>(json_dtype.extended());
            switch (sdt->get_encoding()) {
                case string_encoding_ascii:
                case string_encoding_utf_8:
                    // The data is already UTF-8, so use the buffer directly
                    sdt->get_string_range(&json_begin, &json_end, json.get_ndo_meta(), json.get_readonly_originptr());
                    break;
                default:
                    // The data needs to be converted to UTF-8 before parsing
                    json_dtype = make_string_dtype(string_encoding_utf_8);
                    json_tmp = json.cast_scalars(json_dtype);
                    sdt = static_cast<const base_string_dtype *>(json_dtype.extended());
                    sdt->get_string_range(&json_begin, &json_end, json_tmp.get_ndo_meta(), json_tmp.get_readonly_originptr());
                    break;
            }
            break;
        }
        case bytes_kind: {
            const base_bytes_dtype *bdt = static_cast<const base_bytes_dtype *>(json_dtype.extended());
            bdt->get_bytes_range(&json_begin, &json_end, json.get_ndo_meta(), json.get_readonly_originptr());
        }
        default: {
            stringstream ss;
            ss << "Input for JSON parsing must be either bytes (interpreted as UTF-8) or a string, not ";
            ss << json_dtype;
            throw runtime_error(ss.str());
            break;
        }
    }

    return parse_json(dt, json_begin, json_end);
}

ndobject dynd::parse_json(const dtype& dt, const char *json_begin, const char *json_end)
{
    ndobject result;
    if (dt.get_data_size() != 0) {
        result = ndobject(dt);
    } else {
        stringstream ss;
        ss << "The dtype provided to parse_json, " << dt << ", cannot be used because it requires additional shape information";
        throw runtime_error(ss.str());
    }

    throw runtime_error("TODO: finish parse_json");
}
