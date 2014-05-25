//
// Copyright (C) 2011-14 Mark Wiebe, DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <dynd/json_parser.hpp>
#include <dynd/types/base_bytes_type.hpp>
#include <dynd/types/string_type.hpp>
#include <dynd/types/json_type.hpp>
#include <dynd/types/fixed_dim_type.hpp>
#include <dynd/types/cfixed_dim_type.hpp>
#include <dynd/types/var_dim_type.hpp>
#include <dynd/types/cstruct_type.hpp>
#include <dynd/types/date_type.hpp>
#include <dynd/types/datetime_type.hpp>
#include <dynd/types/time_type.hpp>
#include <dynd/kernels/string_numeric_assignment_kernels.hpp>
#include <dynd/parser_util.hpp>

using namespace std;
using namespace dynd;

namespace {
    class json_parse_error : public parse::parse_error {
        ndt::type m_type;
    public:
        json_parse_error(const char *position, const std::string& message, const ndt::type& tp)
            : parse::parse_error(position, message), m_type(tp) {
        }
        virtual ~json_parse_error() throw () {
        }
        const ndt::type& get_type() const {
            return m_type;
        }
    };
} // anonymous namespace

static void json_as_buffer(const nd::array& json, nd::array& out_tmp_ref, const char *&begin, const char *&end)
{
    // Check the type of 'json', and get pointers to the begin/end of a UTF-8 buffer
    ndt::type json_type = json.get_type().value_type();
    switch (json_type.get_kind()) {
        case string_kind: {
            const base_string_type *sdt = json_type.tcast<base_string_type>();
            switch (sdt->get_encoding()) {
                case string_encoding_ascii:
                case string_encoding_utf_8:
                    out_tmp_ref = json.eval();
                    // The data is already UTF-8, so use the buffer directly
                    sdt->get_string_range(&begin, &end,
                                    out_tmp_ref.get_arrmeta(), out_tmp_ref.get_readonly_originptr());
                    break;
                default: {
                    // The data needs to be converted to UTF-8 before parsing
                    ndt::type utf8_tp = ndt::make_string(string_encoding_utf_8);
                    out_tmp_ref = json.ucast(utf8_tp).eval();
                    sdt = static_cast<const base_string_type *>(utf8_tp.extended());
                    sdt->get_string_range(&begin, &end,
                                    out_tmp_ref.get_arrmeta(), out_tmp_ref.get_readonly_originptr());
                    break;
                }
            }
            break;
        }
        case bytes_kind: {
            out_tmp_ref = json.eval();
            const base_bytes_type *bdt = json_type.tcast<base_bytes_type>();
            bdt->get_bytes_range(&begin, &end,
                            out_tmp_ref.get_arrmeta(), out_tmp_ref.get_readonly_originptr());
            break;
        }
        default: {
            stringstream ss;
            ss << "Input for JSON parsing must be either bytes (interpreted as UTF-8) or a string, not ";
            ss << json_type;
            throw runtime_error(ss.str());
            break;
        }
    }
}

void dynd::parse_json(nd::array &out, const nd::array &json,
                      const eval::eval_context *ectx)
{
    const char *json_begin = NULL, *json_end = NULL;
    nd::array tmp_ref;
    json_as_buffer(json, tmp_ref, json_begin, json_end);
    parse_json(out, json_begin, json_end, ectx);
}

nd::array dynd::parse_json(const ndt::type &tp, const nd::array &json,
                           const eval::eval_context *ectx)
{
    const char *json_begin = NULL, *json_end = NULL;
    nd::array tmp_ref;
    json_as_buffer(json, tmp_ref, json_begin, json_end);
    return parse_json(tp, json_begin, json_end, ectx);
}

static void parse_json(const ndt::type &tp, const char *metadata,
                       char *out_data, const char *&json_begin,
                       const char *json_end, const eval::eval_context *ectx);

static const char *skip_whitespace(const char *begin, const char *end)
{
    while (begin < end && isspace(*begin)) {
        ++begin;
    }

    return begin;
}

template <int N>
static bool parse_token(const char *&begin, const char *end, const char (&token)[N])
{
    const char *begin_skipws = skip_whitespace(begin, end);
    if (N-1 <= end - begin_skipws && memcmp(begin_skipws, token, N-1) == 0) {
        begin = begin_skipws + N-1;
        return true;
    } else {
        return false;
    }
}

static void skip_json_value(const char *&begin, const char *end)
{
    begin = skip_whitespace(begin, end);
    if (begin == end) {
        throw parse::parse_error(begin, "malformed JSON, expecting an element");
    }
    char c = *begin;
    switch (c) {
        // Object
        case '{':
            ++begin;
            if (!parse_token(begin, end, "}")) {
                for (;;) {
                    const char *strbegin, *strend;
                    bool escaped;
                    begin = skip_whitespace(begin, end);
                    if (!parse::parse_doublequote_string_no_ws(
                            begin, end, strbegin, strend, escaped)) {
                        throw parse::parse_error(
                            begin, "expected string for name in object dict");
                    }
                    if (!parse_token(begin, end, ":")) {
                        throw parse::parse_error(begin, "expected ':' separating name from value in object dict");
                    }
                    skip_json_value(begin, end);
                    if (!parse_token(begin, end, ",")) {
                        break;
                    }
                }
                if (!parse_token(begin, end, "}")) {
                    throw parse::parse_error(begin, "expected object separator ',' or terminator '}'");
                }
            }
            break;
        // Array
        case '[':
            ++begin;
            if (!parse_token(begin, end, "]")) {
                for (;;) {
                    skip_json_value(begin, end);
                    if (!parse_token(begin, end, ",")) {
                        break;
                    }
                }
                if (!parse_token(begin, end, "]")) {
                    throw parse::parse_error(begin, "expected array separator ',' or terminator ']'");
                }
            }
            break;
        case '"': {
            const char *strbegin, *strend;
            bool escaped;
            if (!parse::parse_doublequote_string_no_ws(begin, end, strbegin,
                                                       strend, escaped)) {
                throw parse::parse_error(begin, "invalid string");
            }
            break;
        }
        case 't':
            if (!parse_token(begin, end, "true")) {
                throw parse::parse_error(begin, "invalid json value");
            }
            break;
        case 'f':
            if (!parse_token(begin, end, "false")) {
                throw parse::parse_error(begin, "invalid json value");
            }
            break;
        case 'n':
            if (!parse_token(begin, end, "null")) {
                throw parse::parse_error(begin, "invalid json value");
            }
            break;
        default:
            if (c == '-' || ('0' <= c && c <= '9')) {
                const char *nbegin = NULL, *nend = NULL;
                if (!parse::parse_json_number_no_ws(begin, end, nbegin, nend)) {
                    throw parse::parse_error(begin, "invalid number");
                }
            } else {
                throw parse::parse_error(begin, "invalid json value");
            }
    }
}

static void parse_fixed_dim_json(const ndt::type& tp, const char *metadata, char *out_data,
                const char *&begin, const char *end, const eval::eval_context *ectx)
{
    const fixed_dim_type *fad = tp.tcast<fixed_dim_type>();
    const fixed_dim_type_metadata *md = reinterpret_cast<const fixed_dim_type_metadata *>(metadata);
    intptr_t size = fad->get_fixed_dim_size();
    intptr_t stride = md->stride;

    if (!parse_token(begin, end, "[")) {
        throw json_parse_error(begin, "expected list starting with '['", tp);
    }
    for (intptr_t i = 0; i < size; ++i) {
        parse_json(fad->get_element_type(), metadata + sizeof(fixed_dim_type_metadata), out_data + i * stride, begin, end, ectx);
        if (i < size-1 && !parse_token(begin, end, ",")) {
            throw json_parse_error(begin, "array is too short, expected ',' list item separator", tp);
        }
    }
    if (!parse_token(begin, end, "]")) {
        throw json_parse_error(begin, "array is too long, expected list terminator ']'", tp);
    }
}

static void parse_cfixed_dim_json(const ndt::type& tp, const char *metadata, char *out_data,
                const char *&begin, const char *end, const eval::eval_context *ectx)
{
    const cfixed_dim_type *fad = tp.tcast<cfixed_dim_type>();
    intptr_t size = fad->get_fixed_dim_size();
    intptr_t stride = fad->get_fixed_stride();

    if (!parse_token(begin, end, "[")) {
        throw json_parse_error(begin, "expected list starting with '['", tp);
    }
    for (intptr_t i = 0; i < size; ++i) {
        parse_json(fad->get_element_type(), metadata, out_data + i * stride, begin, end, ectx);
        if (i < size-1 && !parse_token(begin, end, ",")) {
            throw json_parse_error(begin, "array is too short, expected ',' list item separator", tp);
        }
    }
    if (!parse_token(begin, end, "]")) {
        throw json_parse_error(begin, "array is too long, expected list terminator ']'", tp);
    }
}

static void parse_var_dim_json(const ndt::type& tp, const char *metadata, char *out_data,
                const char *&begin, const char *end, const eval::eval_context *ectx)
{
    const var_dim_type *vad = tp.tcast<var_dim_type>();
    const var_dim_type_metadata *md = reinterpret_cast<const var_dim_type_metadata *>(metadata);
    intptr_t stride = md->stride;
    const ndt::type& element_tp = vad->get_element_type();

    var_dim_type_data *out = reinterpret_cast<var_dim_type_data *>(out_data);
    char *out_end = NULL;

    memory_block_pod_allocator_api *allocator = get_memory_block_pod_allocator_api(md->blockref);
    intptr_t size = 0, allocated_size = 8;
    allocator->allocate(md->blockref, allocated_size * stride,
                    element_tp.get_data_alignment(), &out->begin, &out_end);

    if (!parse_token(begin, end, "[")) {
        throw json_parse_error(begin, "expected array starting with '['", tp);
    }
    // If it's not an empty list, start the loop parsing the elements
    if (!parse_token(begin, end, "]")) {
        for (;;) {
            // Increase the allocated array size if necessary
            if (size == allocated_size) {
                allocated_size *= 2;
                allocator->resize(md->blockref, allocated_size * stride, &out->begin, &out_end);
            }
            ++size;
            out->size = size;
            parse_json(element_tp, metadata + sizeof(var_dim_type_metadata),
                            out->begin + (size-1) * stride, begin, end, ectx);
            if (!parse_token(begin, end, ",")) {
                break;
            }
        }
        if (!parse_token(begin, end, "]")) {
            throw json_parse_error(begin, "expected array separator ',' or terminator ']'", tp);
        }
    }

    // Shrink-wrap the memory to just fit the string
    allocator->resize(md->blockref, size * stride, &out->begin, &out_end);
    out->size = size;
}

static bool parse_struct_json_from_object(const ndt::type &tp,
                                          const char *metadata, char *out_data,
                                          const char *&begin, const char *end,
                                          const eval::eval_context *ectx)
{
    const char *saved_begin = begin;
    if (!parse_token(begin, end, "{")) {
        return false;
    }

    const base_struct_type *fsd = tp.tcast<base_struct_type>();
    size_t field_count = fsd->get_field_count();
    const size_t *data_offsets = fsd->get_data_offsets(metadata);
    const size_t *arrmeta_offsets = fsd->get_arrmeta_offsets_raw();

    // Keep track of which fields we've seen
    shortvector<bool> populated_fields(field_count);
    memset(populated_fields.get(), 0, sizeof(bool) * field_count);

    // If it's not an empty object, start the loop parsing the elements
    if (!parse_token(begin, end, "}")) {
        for (;;) {
            const char *strbegin, *strend;
            bool escaped;
            begin = skip_whitespace(begin, end);
            if (!parse::parse_doublequote_string_no_ws(begin, end, strbegin,
                                                       strend, escaped)) {
                throw json_parse_error(begin, "expected string for name in object dict", tp);
            }
            if (!parse_token(begin, end, ":")) {
                throw json_parse_error(begin, "expected ':' separating name from value in object dict", tp);
            }
            intptr_t i;
            if (escaped) {
                string name;
                parse::unescape_string(strbegin, strend, name);
                i = fsd->get_field_index(name);
            } else {
                i = fsd->get_field_index(strbegin, strend);
            }
            if (i == -1) {
                // TODO: Add an error policy to this parser of whether to throw an error
                //       or not. For now, just throw away fields not in the destination.
                skip_json_value(begin, end);
            } else {
                parse_json(fsd->get_field_type(i), metadata + arrmeta_offsets[i],
                           out_data + data_offsets[i], begin, end, ectx);
                populated_fields[i] = true;
            }
            if (!parse_token(begin, end, ",")) {
                break;
            }
        }
        if (!parse_token(begin, end, "}")) {
            throw json_parse_error(begin, "expected object dict separator ',' or terminator '}'", tp);
        }
    }

    for (size_t i = 0; i < field_count; ++i) {
        if (!populated_fields[i]) {
            stringstream ss;
            ss << "object dict does not contain the field ";
            print_escaped_utf8_string(ss, fsd->get_field_name(i));
            ss << " as required by the data type";
            throw json_parse_error(skip_whitespace(saved_begin, end), ss.str(), tp);
        }
    }

    return true;
}

static bool parse_struct_json_from_list(const ndt::type &tp,
                                        const char *metadata, char *out_data,
                                        const char *&begin, const char *end,
                                        const eval::eval_context *ectx)
{
    if (!parse_token(begin, end, "[")) {
        return false;
    }

    const base_struct_type *fsd = tp.tcast<base_struct_type>();
    size_t field_count = fsd->get_field_count();
    const size_t *data_offsets = fsd->get_data_offsets(metadata);
    const size_t *arrmeta_offsets = fsd->get_arrmeta_offsets_raw();

    // Loop through all the fields
    for (size_t i = 0; i != field_count; ++i) {
        begin = skip_whitespace(begin, end);
        parse_json(fsd->get_field_type(i), metadata + arrmeta_offsets[i],
                   out_data + data_offsets[i], begin, end, ectx);
        if (i != field_count - 1 && !parse_token(begin, end, ",")) {
            throw json_parse_error(begin, "expected list item separator ','",
                                   tp);
        }
    }

    if (!parse_token(begin, end, "]")) {
        throw json_parse_error(begin, "expected end of list ']'", tp);
    }

    return true;
}

static void parse_struct_json(const ndt::type& tp, const char *metadata, char *out_data,
                const char *&begin, const char *end, const eval::eval_context *ectx)
{
    if (parse_struct_json_from_object(tp, metadata, out_data, begin, end, ectx)) {
    } else if (parse_struct_json_from_list(tp, metadata, out_data, begin, end, ectx)) {
    } else {
        throw json_parse_error(
            begin, "expected object dict starting with '{' or list with '['",
            tp);
    }
}

static void parse_bool_json(const ndt::type& tp, const char *metadata, char *out_data,
                const char *&begin, const char *end)
{
    // TODO: allow more general input (strings) with a boolean parsing policy
    char value = 2;
    const char *nbegin, *nend;
    if (parse_token(begin, end, "true")) {
        value = 1;
    } else if (parse_token(begin, end, "false")) {
        value = 0;
    } else if (parse_token(begin, end, "null")) {
        // TODO: error handling policy for NULL in this case
        value = 0;
    } else if (parse::parse_json_number_no_ws(begin, end, nbegin, nend)) {
        if (nend - nbegin == 1) {
            if (*nbegin == '0') {
                value = 0;
            } else if (*nbegin == '1') {
                value = 1;
            }
        }
    }

    if (value != 2) {
        if (tp.get_type_id() == bool_type_id) {
            *out_data = value;
        } else {
            typed_data_assign(tp, metadata, out_data, ndt::make_type<dynd_bool>(), NULL, &value);
        }
    } else {
        throw json_parse_error(begin, "expected a boolean true or false", tp);
    }
}

static void parse_dynd_builtin_json(const ndt::type& tp, const char *DYND_UNUSED(metadata), char *out_data,
                const char *&rbegin, const char *end)
{
    const char *begin = rbegin;
    const char *strbegin = NULL, *strend = NULL;
    bool escaped;
    begin = skip_whitespace(begin, end);
    if (parse::parse_json_number_no_ws(begin, end, strbegin, strend)) {
        try {
            assign_utf8_string_to_builtin(tp.get_type_id(), out_data, strbegin, strend);
        } catch (const std::exception& e) {
            throw json_parse_error(skip_whitespace(rbegin, begin), e.what(), tp);
        }
    } else if (parse::parse_doublequote_string_no_ws(begin, end, strbegin, strend, escaped)) {
        try {
            if (!escaped) {
                assign_utf8_string_to_builtin(tp.get_type_id(), out_data, strbegin, strend);
            } else {
                string val;
                parse::unescape_string(strbegin, strend, val);
                assign_utf8_string_to_builtin(tp.get_type_id(), out_data, val.data(), val.data() + val.size());
            }
        } catch (const std::exception& e) {
            throw json_parse_error(skip_whitespace(rbegin, begin), e.what(), tp);
        }
    } else {
        throw json_parse_error(begin, "invalid input", tp);
    }
    rbegin = begin;
}

static void parse_integer_json(const ndt::type& tp, const char *DYND_UNUSED(metadata), char *out_data,
                const char *&begin, const char *end, assign_error_mode errmode)
{
    const char *nbegin, *nend;
    if (parse::parse_json_number_no_ws(begin, end, nbegin, nend)) {
        parse::string_to_int(out_data, tp.get_type_id(), nbegin, nend, errmode);
    } else {
        throw json_parse_error(begin, "expected a number", tp);
    }
}

static void parse_real_json(const ndt::type& tp, const char *metadata, char *out_data,
                const char *&begin, const char *end)
{
    // TODO: Parsing policy for how to handle reals
    parse_dynd_builtin_json(tp, metadata, out_data, begin, end);
}

static void parse_complex_json(const ndt::type& tp, const char *metadata, char *out_data,
                const char *&begin, const char *end)
{
    // TODO: Parsing policy for how to handle complex
    parse_dynd_builtin_json(tp, metadata, out_data, begin, end);
}

static void parse_jsonstring_json(const ndt::type& tp, const char *metadata, char *out_data,
                const char *&begin, const char *end)
{
    const char *saved_begin = skip_whitespace(begin, end);
    skip_json_value(begin, end);
    const base_string_type *bsd = tp.tcast<base_string_type>();
    // The skipped JSON value gets copied verbatim into the json string
    bsd->set_utf8_string(metadata, out_data, assign_error_none,
            saved_begin, begin);
}

static void parse_string_json(const ndt::type& tp, const char *metadata, char *out_data,
                const char *&rbegin, const char *end)
{
    const char *begin = rbegin;
    begin = skip_whitespace(begin, end);
    const char *strbegin, *strend;
    bool escaped;
    if (parse::parse_doublequote_string_no_ws(begin, end, strbegin, strend,
                                              escaped)) {
        const base_string_type *bsd = tp.tcast<base_string_type>();
        try {
            if (!escaped) {
                bsd->set_utf8_string(metadata, out_data, assign_error_fractional, strbegin, strend);
            } else {
                string val;
                parse::unescape_string(strbegin, strend, val);
                bsd->set_utf8_string(metadata, out_data, assign_error_fractional, val);
            }
        } catch (const std::exception& e) {
            throw json_parse_error(skip_whitespace(rbegin, begin), e.what(), tp);
        }
    } else {
        throw json_parse_error(begin, "expected a string", tp);
    }
    rbegin = begin;
}

static void parse_datetime_json(const ndt::type& tp, const char *metadata, char *out_data,
                const char *&rbegin, const char *end, const eval::eval_context *ectx)
{
    const char *begin = rbegin;
    begin = skip_whitespace(begin, end);
    const char *strbegin, *strend;
    bool escaped;
    if (parse::parse_doublequote_string_no_ws(begin, end, strbegin, strend,
                                              escaped)) {
        string val;
        if (!escaped) {
            val.assign(strbegin, strend);
        } else {
            parse::unescape_string(strbegin, strend, val);
        }
        if (tp.get_type_id() == date_type_id) {
            const date_type *dd = tp.tcast<date_type>();
            try {
                dd->set_utf8_string(metadata, out_data, assign_error_fractional, val, ectx);
            } catch (const std::exception& e) {
                throw json_parse_error(skip_whitespace(rbegin, begin), e.what(), tp);
            }
        } else if (tp.get_type_id() == datetime_type_id) {
            const datetime_type *dt = tp.tcast<datetime_type>();
            try {
                dt->set_utf8_string(metadata, out_data, assign_error_fractional, val, ectx);
            } catch (const std::exception& e) {
                throw json_parse_error(skip_whitespace(rbegin, begin), e.what(), tp);
            }
        } else if (tp.get_type_id() == time_type_id) {
            const time_type *tt = tp.tcast<time_type>();
            try {
                tt->set_utf8_string(metadata, out_data, assign_error_fractional, val);
            } catch (const std::exception& e) {
                throw json_parse_error(skip_whitespace(rbegin, begin), e.what(), tp);
            }
        } else {
            stringstream ss;
            ss << "nd::parse_json: unexpected type " << tp;
            throw type_error(ss.str());
        }
    } else {
        throw json_parse_error(begin, "expected a string", tp);
    }
    rbegin = begin;
}

static void parse_uniform_dim_json(const ndt::type& tp, const char *metadata, char *out_data,
                const char *&begin, const char *end, const eval::eval_context *ectx)
{
    switch (tp.get_type_id()) {
        case fixed_dim_type_id:
            parse_fixed_dim_json(tp, metadata, out_data, begin, end, ectx);
            break;
        case cfixed_dim_type_id:
            parse_cfixed_dim_json(tp, metadata, out_data, begin, end, ectx);
            break;
        case var_dim_type_id:
            parse_var_dim_json(tp, metadata, out_data, begin, end, ectx);
            break;
        default: {
            stringstream ss;
            ss << "parse_json: unsupported dynd array type " << tp;
            throw runtime_error(ss.str());
        }
    }
}

static void parse_json(const ndt::type& tp, const char *metadata, char *out_data,
                const char *&json_begin, const char *json_end, const eval::eval_context *ectx)
{
    switch (tp.get_kind()) {
        case uniform_dim_kind:
            parse_uniform_dim_json(tp, metadata, out_data, json_begin, json_end, ectx);
            return;
        case struct_kind:
            parse_struct_json(tp, metadata, out_data, json_begin, json_end, ectx);
            return;
        case bool_kind:
            parse_bool_json(tp, metadata, out_data, json_begin, json_end);
            return;
        case int_kind:
        case uint_kind:
            try {
                json_begin = skip_whitespace(json_begin, json_end);
                parse_integer_json(tp, metadata, out_data, json_begin, json_end,
                                   ectx->default_errmode);
            } catch(const std::exception& e) {
                // Transform any exceptions into a parse error that includes
                // the location of the error
                throw json_parse_error(json_begin, e.what(), tp);
            }
            return;
        case real_kind:
            parse_real_json(tp, metadata, out_data, json_begin, json_end);
            return;
        case complex_kind:
            parse_complex_json(tp, metadata, out_data, json_begin, json_end);
            return;
        case string_kind:
            parse_string_json(tp, metadata, out_data, json_begin, json_end);
            return;
        case datetime_kind:
            parse_datetime_json(tp, metadata, out_data, json_begin, json_end, ectx);
            return;
        case dynamic_kind:
            if (tp.get_type_id() == json_type_id) {
                // The json type is a special string type that contains JSON directly
                // Copy the JSON verbatim in this case.
                parse_jsonstring_json(tp, metadata, out_data, json_begin, json_end);
                return;
            }
            break;
        default:
            break;
    }

    stringstream ss;
    ss << "parse_json: unsupported dynd type " << tp;
    throw runtime_error(ss.str());
}

/**
 * Returns the row/column where the error occured, as well as the current and previous
 * lines for printing some context.
 */
static void get_error_line_column(const char *begin, const char *end, const char *position,
        std::string& out_line_prev, std::string& out_line_cur, int& out_line, int& out_column)
{
    out_line_prev = "";
    out_line_cur = "";
    out_line = 1;
    while (begin < end) {
        const char *line_end = (const char *)memchr(begin, '\n', end - begin);
        out_line_prev.swap(out_line_cur);
        // If no \n was found
        if (line_end == NULL) {
            out_column = int(position - begin + 1);
            out_line_cur = string(begin, end);
            return;
        } else {
            out_line_cur = string(begin, line_end);
            ++line_end;
            if (position < line_end) {
                out_column = int(position - begin + 1);
                return;
            }
        }
        begin = line_end;
        ++out_line;
    }

    throw runtime_error("Cannot get line number of error, its position is out of range");
}

void print_json_parse_error_marker(std::ostream& o, const std::string& line_prev,
                const std::string& line_cur, int line, int column)
{
    if (line_cur.size() < 200) {
        // If the line is short, print the lines and indicate the error
        if (line > 1) {
            o << line_prev << "\n";
        }
        o << line_cur << "\n";
        for (int i = 0; i < column-1; ++i) {
            o << " ";
        }
        o << "^\n";
    } else {
        // If the line is long, print a part of the line and indicate the error
        if (column < 80) {
            o << line_cur.substr(0, 80) << " ...\n";
            for (int i = 0; i < column-1; ++i) {
                o << " ";
            }
            o << "^\n";
        } else {
            int start = column - 60;
            o << " ... " << line_cur.substr(start - 1, 80) << " ...\n";
            for (int i = 0; i < 65; ++i) {
                o << " ";
            }
            o << "^\n";
        }
    }
}

void dynd::validate_json(const char *json_begin, const char *json_end)
{
    try {
        const char *begin = json_begin, *end = json_end;
        ::skip_json_value(begin, end);
        begin = skip_whitespace(begin, end);
        if (begin != end) {
            throw parse::parse_error(begin, "unexpected trailing JSON text");
        }
    } catch (const parse::parse_error& e) {
        stringstream ss;
        string line_prev, line_cur;
        int line, column;
        get_error_line_column(json_begin, json_end, e.get_position(),
                        line_prev, line_cur, line, column);
        ss << "Error validating JSON at line " << line << ", column " << column << "\n";
        ss << "Message: " << e.what() << "\n";
        print_json_parse_error_marker(ss, line_prev, line_cur, line, column);
        throw invalid_argument(ss.str());
    }
}

void dynd::parse_json(nd::array &out, const char *json_begin,
                      const char *json_end, const eval::eval_context *ectx)
{
    try {
        const char *begin = json_begin, *end = json_end;
        ndt::type tp = out.get_type();
        ::parse_json(tp, out.get_arrmeta(), out.get_readwrite_originptr(), begin, end, ectx);
        begin = skip_whitespace(begin, end);
        if (begin != end) {
            throw json_parse_error(begin, "unexpected trailing JSON text", tp);
        }
    } catch (const json_parse_error& e) {
        stringstream ss;
        string line_prev, line_cur;
        int line, column;
        get_error_line_column(json_begin, json_end, e.get_position(),
                        line_prev, line_cur, line, column);
        ss << "Error parsing JSON at line " << line << ", column " << column << "\n";
        ss << "DType: " << e.get_type() << "\n";
        ss << "Message: " << e.what() << "\n";
        print_json_parse_error_marker(ss, line_prev, line_cur, line, column);
        throw invalid_argument(ss.str());
    } catch (const parse::parse_error& e) {
        stringstream ss;
        string line_prev, line_cur;
        int line, column;
        get_error_line_column(json_begin, json_end, e.get_position(),
                        line_prev, line_cur, line, column);
        ss << "Error parsing JSON at line " << line << ", column " << column << "\n";
        ss << "Message: " << e.what() << "\n";
        print_json_parse_error_marker(ss, line_prev, line_cur, line, column);
        throw invalid_argument(ss.str());
    }
}

nd::array dynd::parse_json(const ndt::type &tp, const char *json_begin,
                           const char *json_end, const eval::eval_context *ectx)
{
    nd::array result;
    result = nd::empty(tp);
    parse_json(result, json_begin, json_end, ectx);
    if (!tp.is_builtin()) {
        tp.extended()->metadata_finalize_buffers(result.get_arrmeta());
    }
    result.flag_as_immutable();
    return result;
}
