//
// Copyright (C) 2011-13, DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <dynd/json_parser.hpp>
#include <dynd/dtypes/base_bytes_dtype.hpp>
#include <dynd/dtypes/string_dtype.hpp>
#include <dynd/dtypes/fixedarray_dtype.hpp>
#include <dynd/dtypes/var_array_dtype.hpp>
#include <dynd/dtypes/fixedstruct_dtype.hpp>
#include <dynd/dtypes/date_dtype.hpp>
#include <dynd/kernels/string_numeric_assignment_kernels.hpp>

using namespace std;
using namespace dynd;

namespace {
    class json_parse_error {
        const char *m_position;
        string m_message;
        dtype m_dtype;
    public:
        json_parse_error(const char *position, const std::string& message, const dtype& dt)
            : m_position(position), m_message(message), m_dtype(dt) {
        }
        virtual ~json_parse_error() {
        }
        const char *get_position() const {
            return m_position;
        }
        const char *get_message() const {
            return m_message.c_str();
        }
        const dtype& get_dtype() const {
            return m_dtype;
        }
    };
} // anonymous namespace

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

static void parse_json(const dtype& dt, const char *metadata, char *out_data,
                const char *&json_begin, const char *json_end);

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

static bool parse_json_string(const char *&begin, const char *end, string& out_val)
{
    out_val = "";
    const char *saved_begin = begin;
    if (!parse_token(begin, end, "\"")) {
        return false;
    }
    for (;;) {
        if (begin == end) {
            throw json_parse_error(skip_whitespace(saved_begin, end), "string has no ending quote", dtype());
        }
        char c = *begin++;
        if (c == '\\') {
            if (begin == end) {
                throw json_parse_error(skip_whitespace(saved_begin, end), "string has no ending quote", dtype());
            }
            c = *begin++;
            switch (c) {
                case '"':
                case '\\':
                case '/':
                    out_val += c;
                    break;
                case 'b':
                    out_val += '\b';
                    break;
                case 'f':
                    out_val += '\f';
                    break;
                case 'n':
                    out_val += '\n';
                    break;
                case 'r':
                    out_val += '\r';
                    break;
                case 't':
                    out_val += '\t';
                    break;
                case 'u': {
                    if (end - begin < 4) {
                        throw json_parse_error(begin-2, "invalid unicode escape sequence in string", dtype());
                    }
                    uint32_t cp = 0;
                    for (int i = 0; i < 4; ++i) {
                        char c = *begin++;
                        cp *= 16;
                        if ('0' <= c && c <= '9') {
                            cp += c - '0';
                        } else if ('A' <= c && c <= 'F') {
                            cp += c - 'A' + 10;
                        } else if ('a' <= c && c <= 'f') {
                            cp += c - 'a' + 10;
                        } else {
                            throw json_parse_error(begin-1, "invalid unicode escape sequence in string", dtype());
                        }
                    }
                    append_utf8_codepoint(cp, out_val);
                    break;
                }
                default:
                    throw json_parse_error(begin-2, "invalid escape sequence in string", dtype());
            }
        } else if (c != '"') {
            out_val += c;
        } else {
            return true;
        }
    }
}

static bool parse_json_number(const char *&begin, const char *end, const char *&out_nbegin, const char *&out_nend)
{
    const char *saved_begin = skip_whitespace(begin, end);
    const char *pos = saved_begin;
    if (pos == end) {
        return false;
    }
    // Optional minus sign
    if (*pos == '-') {
        ++pos;
    }
    if (pos == end) {
        return false;
    }
    // Either '0' or a non-zero digit followed by digits
    if (*pos == '0') {
        ++pos;
    } else if ('1' <= *pos && *pos <= '9') {
        ++pos;
        while (pos < end && ('0' <= *pos && *pos <= '9')) {
            ++pos;
        }
    } else {
        return false;
    }
    // Optional decimal point, followed by one or more digits
    if (pos < end && *pos == '.') {
        if (++pos == end) {
            return false;
        }
        if (!('0' <= *pos && *pos <= '9')) {
            return false;
        }
        ++pos;
        while (pos < end && ('0' <= *pos && *pos <= '9')) {
            ++pos;
        }
    }
    // Optional exponent, followed by +/- and some digits
    if (pos < end && (*pos == 'e' || *pos == 'E')) {
        if (++pos == end) {
            return false;
        }
        // +/- is optional
        if (*pos == '+' || *pos == '-') {
            if (++pos == end) {
                return false;
            }
        }
        // At least one digit is required
        if (!('0' <= *pos && *pos <= '9')) {
            return false;
        }
        ++pos;
        while (pos < end && ('0' <= *pos && *pos <= '9')) {
            ++pos;
        }
    }
    out_nbegin = saved_begin;
    out_nend = pos;
    begin = pos;
    return true;
}

static void skip_json_element(const char *&begin, const char *end)
{
    begin = skip_whitespace(begin, end);
    if (begin == end) {
        throw json_parse_error(begin, "malformed JSON, expecting an element", dtype());
    }
    char c = *begin;
    switch (c) {
        // Object
        case '{':
            ++begin;
            if (!parse_token(begin, end, "}")) {
                for (;;) {
                    string name;
                    if (!parse_json_string(begin, end, name)) {
                        throw json_parse_error(begin, "expected string for name in object dict", dtype());
                    }
                    if (!parse_token(begin, end, ":")) {
                        throw json_parse_error(begin, "expected ':' separating name from value in object dict", dtype());
                    }
                    skip_json_element(begin, end);
                    if (!parse_token(begin, end, ",")) {
                        break;
                    }
                }
                if (!parse_token(begin, end, "}")) {
                    throw json_parse_error(begin, "expected object separator ',' or terminator '}'", dtype());
                }
            }
            break;
        // Array
        case '[':
            ++begin;
            if (!parse_token(begin, end, "]")) {
                for (;;) {
                    skip_json_element(begin, end);
                    if (!parse_token(begin, end, ",")) {
                        break;
                    }
                }
                if (!parse_token(begin, end, "]")) {
                    throw json_parse_error(begin, "expected array separator ',' or terminator ']'", dtype());
                }
            }
            break;
        case '"': {
            string s;
            if (!parse_json_string(begin, end, s)) {
                throw json_parse_error(begin, "invalid string", dtype());
            }
            break;
        }
        case 't':
            if (!parse_token(begin, end, "true")) {
                throw json_parse_error(begin, "invalid json value", dtype());
            }
            break;
        case 'f':
            if (!parse_token(begin, end, "false")) {
                throw json_parse_error(begin, "invalid json value", dtype());
            }
            break;
        case 'n':
            if (!parse_token(begin, end, "null")) {
                throw json_parse_error(begin, "invalid json value", dtype());
            }
            break;
        default:
            if (c == '-' || ('0' <= c && c <= '9')) {
                const char *nbegin = NULL, *nend = NULL;
                if (!parse_json_number(begin, end, nbegin, nend)) {
                    throw json_parse_error(begin, "invalid number", dtype());
                }
            } else {
                throw json_parse_error(begin, "invalid json value", dtype());
            }
    }
}

static void parse_fixedarray_json(const dtype& dt, const char *metadata, char *out_data,
                const char *&begin, const char *end)
{
    const fixedarray_dtype *fad = static_cast<const fixedarray_dtype *>(dt.extended());
    intptr_t size = fad->get_fixed_dim_size();
    intptr_t stride = fad->get_fixed_stride();

    if (!parse_token(begin, end, "[")) {
        throw json_parse_error(begin, "expected list starting with '['", dt);
    }
    for (intptr_t i = 0; i < size; ++i) {
        parse_json(fad->get_element_dtype(), metadata, out_data + i * stride, begin, end);
        if (i < size-1 && !parse_token(begin, end, ",")) {
            throw json_parse_error(begin, "array is too short, expected ',' list item separator", dt);
        }
    }
    if (!parse_token(begin, end, "]")) {
        throw json_parse_error(begin, "array is too long, expected list terminator ']'", dt);
    }
}

static void parse_var_array_json(const dtype& dt, const char *metadata, char *out_data,
                const char *&begin, const char *end)
{
    const var_array_dtype *vad = static_cast<const var_array_dtype *>(dt.extended());
    const var_array_dtype_metadata *md = reinterpret_cast<const var_array_dtype_metadata *>(metadata);
    intptr_t stride = md->stride;
    const dtype& element_dtype = vad->get_element_dtype();

    var_array_dtype_data *out = reinterpret_cast<var_array_dtype_data *>(out_data);
    char *out_end = NULL;

    memory_block_pod_allocator_api *allocator = get_memory_block_pod_allocator_api(md->blockref);
    intptr_t size = 0, allocated_size = 8;
    allocator->allocate(md->blockref, allocated_size * stride,
                    element_dtype.get_alignment(), &out->begin, &out_end);

    if (!parse_token(begin, end, "[")) {
        throw json_parse_error(begin, "expected array starting with '['", dt);
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
            parse_json(element_dtype, metadata + sizeof(var_array_dtype_metadata),
                            out->begin + (size-1) * stride, begin, end);
            if (!parse_token(begin, end, ",")) {
                break;
            }
        }
        if (!parse_token(begin, end, "]")) {
            throw json_parse_error(begin, "expected array separator ',' or terminator ']'", dt);
        }
    }

    // Shrink-wrap the memory to just fit the string
    allocator->resize(md->blockref, size * stride, &out->begin, &out_end);
    out->size = size;
}

static void parse_fixedstruct_json(const dtype& dt, const char *metadata, char *out_data,
                const char *&begin, const char *end)
{
    const fixedstruct_dtype *fsd = static_cast<const fixedstruct_dtype *>(dt.extended());
    size_t field_count = fsd->get_field_count();
    const vector<string>& field_names = fsd->get_field_names();
    const vector<dtype>& field_types = fsd->get_field_types();
    const vector<size_t>& data_offsets = fsd->get_data_offsets();
    const vector<size_t>& metadata_offsets = fsd->get_metadata_offsets();

    // Keep track of which fields we've seen
    shortvector<bool> populated_fields(field_count);
    memset(populated_fields.get(), 0, sizeof(bool) * field_count);

    const char *saved_begin = begin;
    if (!parse_token(begin, end, "{")) {
        throw json_parse_error(begin, "expected object dict starting with '{'", dt);
    }
    // If it's not an empty object, start the loop parsing the elements
    if (!parse_token(begin, end, "}")) {
        for (;;) {
            string name;
            if (!parse_json_string(begin, end, name)) {
                throw json_parse_error(begin, "expected string for name in object dict", dt);
            }
            if (!parse_token(begin, end, ":")) {
                throw json_parse_error(begin, "expected ':' separating name from value in object dict", dt);
            }
            intptr_t i = fsd->get_field_index(name);
            if (i == -1) {
                // TODO: Add an error policy to this parser of whether to throw an error
                //       or not. For now, just throw away fields not in the destination.
                skip_json_element(begin, end);
            } else {
                parse_json(field_types[i], metadata + metadata_offsets[i], out_data + data_offsets[i], begin, end);
                populated_fields[i] = true;
            }
            if (!parse_token(begin, end, ",")) {
                break;
            }
        }
        if (!parse_token(begin, end, "}")) {
            throw json_parse_error(begin, "expected object dict separator ',' or terminator '}'", dt);
        }
    }

    for (size_t i = 0; i < field_count; ++i) {
        if (!populated_fields[i]) {
            stringstream ss;
            ss << "object dict does not contain the field " << field_names[i];
            ss << " as required by the data type";
            throw json_parse_error(skip_whitespace(saved_begin, end), ss.str(), dt);
        }
    }
}

static void parse_bool_json(const dtype& dt, const char *metadata, char *out_data,
                const char *&begin, const char *end)
{
    char value;
    if (parse_token(begin, end, "true")) {
        value = true;
    } else if (parse_token(begin, end, "false")) {
        value = false;
    } else if (parse_token(begin, end, "null")) {
        // TODO: error handling policy for NULL in this case
        value = false;
    } else {
        // TODO: allow more general input (strings, integers) with a boolean parsing policy
        throw json_parse_error(begin, "expected a boolean true or false", dt);
    }

    if (dt.get_type_id() == bool_type_id) {
        *out_data = value;
    } else {
        dtype_assign(dt, metadata, out_data, make_dtype<dynd_bool>(), NULL, &value);
    }
}

static void parse_dynd_builtin_json(const dtype& dt, const char *metadata, char *out_data,
                const char *&begin, const char *end)
{
    const char *saved_begin = begin;
    const char *nbegin = NULL, *nend = NULL;
    string val;
    if (parse_json_number(begin, end, nbegin, nend)) {
        try {
            assign_utf8_string_to_builtin(dt.get_type_id(), out_data, nbegin, nend);
        } catch (const std::exception& e) {
            throw json_parse_error(skip_whitespace(saved_begin, begin), e.what(), dt);
        }
    } else if (parse_json_string(begin, end, val)) {
        try {
            assign_utf8_string_to_builtin(dt.get_type_id(), out_data, val.data(), val.data() + val.size());
        } catch (const std::exception& e) {
            throw json_parse_error(skip_whitespace(saved_begin, begin), e.what(), dt);
        }
    } else {
        throw json_parse_error(begin, "invalid input", dt);
    }
}

static void parse_integer_json(const dtype& dt, const char *metadata, char *out_data,
                const char *&begin, const char *end)
{
    // TODO: Parsing policy for how to handle integers
    parse_dynd_builtin_json(dt, metadata, out_data, begin, end);
}

static void parse_real_json(const dtype& dt, const char *metadata, char *out_data,
                const char *&begin, const char *end)
{
    // TODO: Parsing policy for how to handle reals
    parse_dynd_builtin_json(dt, metadata, out_data, begin, end);
}

static void parse_complex_json(const dtype& dt, const char *metadata, char *out_data,
                const char *&begin, const char *end)
{
    // TODO: Parsing policy for how to handle complex
    parse_dynd_builtin_json(dt, metadata, out_data, begin, end);
}

static void parse_string_json(const dtype& dt, const char *metadata, char *out_data,
                const char *&begin, const char *end)
{
    const char *saved_begin = begin;
    string val;
    if (parse_json_string(begin, end, val)) {
        const base_string_dtype *bsd = static_cast<const base_string_dtype *>(dt.extended());
        try {
            bsd->set_utf8_string(metadata, out_data, assign_error_fractional, val);
        } catch (const std::exception& e) {
            throw json_parse_error(skip_whitespace(saved_begin, begin), e.what(), dt);
        }
    } else {
        throw json_parse_error(begin, "expected a string", dt);
    }
}

static void parse_datetime_json(const dtype& dt, const char *metadata, char *out_data,
                const char *&begin, const char *end)
{
    const char *saved_begin = begin;
    string val;
    if (parse_json_string(begin, end, val)) {
        if (dt.get_type_id() == date_type_id) {
            const date_dtype *dd = static_cast<const date_dtype *>(dt.extended());
            try {
                dd->set_utf8_string(metadata, out_data, assign_error_fractional, val);
            } catch (const std::exception& e) {
                throw json_parse_error(skip_whitespace(saved_begin, begin), e.what(), dt);
            }
        }
    } else {
        throw json_parse_error(begin, "expected a string", dt);
    }
}

static void parse_uniform_array_json(const dtype& dt, const char *metadata, char *out_data,
                const char *&begin, const char *end)
{
    switch (dt.get_type_id()) {
        case fixedarray_type_id:
            parse_fixedarray_json(dt, metadata, out_data, begin, end);
            break;
        case var_array_type_id:
            parse_var_array_json(dt, metadata, out_data, begin, end);
            break;
        default: {
            stringstream ss;
            ss << "parse_json: unsupported uniform array dtype " << dt;
            throw runtime_error(ss.str());
        }
    }
}

static void parse_struct_json(const dtype& dt, const char *metadata, char *out_data,
                const char *&json_begin, const char *json_end)
{
    switch (dt.get_type_id()) {
        case fixedstruct_type_id:
            parse_fixedstruct_json(dt, metadata, out_data, json_begin, json_end);
            break;
        default: {
            stringstream ss;
            ss << "parse_json: unsupported struct dtype " << dt;
            throw runtime_error(ss.str());
        }
    }
}


static void parse_json(const dtype& dt, const char *metadata, char *out_data,
                const char *&json_begin, const char *json_end)
{
    switch (dt.get_kind()) {
        case uniform_array_kind:
            parse_uniform_array_json(dt, metadata, out_data, json_begin, json_end);
            break;
        case struct_kind:
            parse_struct_json(dt, metadata, out_data, json_begin, json_end);
            break;
        case bool_kind:
            parse_bool_json(dt, metadata, out_data, json_begin, json_end);
            return;
        case int_kind:
        case uint_kind:
            parse_integer_json(dt, metadata, out_data, json_begin, json_end);
            return;
        case real_kind:
            parse_real_json(dt, metadata, out_data, json_begin, json_end);
            return;
        case complex_kind:
            parse_complex_json(dt, metadata, out_data, json_begin, json_end);
            return;
        case string_kind:
            parse_string_json(dt, metadata, out_data, json_begin, json_end);
            return;
        case datetime_kind:
            parse_datetime_json(dt, metadata, out_data, json_begin, json_end);
            return;
    }
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
            out_column = position - begin + 1;
            out_line_cur = string(begin, end);
            return;
        } else {
            out_line_cur = string(begin, line_end);
            ++line_end;
            if (position < line_end) {
                out_column = position - begin + 1;
                return;
            }
        }
        begin = line_end;
        ++out_line;
    }

    throw runtime_error("Cannot get line number of error, its position is out of range");
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

    try {
        const char *begin = json_begin, *end = json_end;
        ::parse_json(result.get_dtype(), result.get_ndo_meta(), result.get_ndo()->m_data_pointer, begin, end);
        if (!dt.is_builtin()) {
            dt.extended()->metadata_finalize_buffers(result.get_ndo_meta());
        }
        begin = skip_whitespace(begin, end);
        if (begin != end) {
            throw json_parse_error(begin, "unexpected trailing JSON text", dt);
        }
        return result;
    } catch (const json_parse_error& e) {
        stringstream ss;
        string line_prev, line_cur;
        int line, column;
        get_error_line_column(json_begin, json_end, e.get_position(),
                        line_prev, line_cur, line, column);
        ss << "Error parsing JSON at line " << line << ", column " << column << "\n";
        if (e.get_dtype().get_type_id() != uninitialized_type_id) {
            ss << "DType: " << e.get_dtype() << "\n";
        }
        ss << "Message: " << e.get_message() << "\n";
        if (line > 1) {
            ss << line_prev << "\n";
        }
        ss << line_cur << "\n";
        for (int i = 0; i < column-1; ++i) {
            ss << " ";
        }
        ss << "^\n";
        throw runtime_error(ss.str());
    }
}
