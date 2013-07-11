//
// Copyright (C) 2011-13 Mark Wiebe, DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <dynd/json_parser.hpp>
#include <dynd/dtypes/base_bytes_type.hpp>
#include <dynd/dtypes/string_type.hpp>
#include <dynd/dtypes/json_dtype.hpp>
#include <dynd/dtypes/fixed_dim_dtype.hpp>
#include <dynd/dtypes/var_dim_dtype.hpp>
#include <dynd/dtypes/cstruct_type.hpp>
#include <dynd/dtypes/date_dtype.hpp>
#include <dynd/kernels/string_numeric_assignment_kernels.hpp>

using namespace std;
using namespace dynd;

namespace {
    class json_parse_error {
        const char *m_position;
        string m_message;
        ndt::type m_dtype;
    public:
        json_parse_error(const char *position, const std::string& message, const ndt::type& dt)
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
        const ndt::type& get_dtype() const {
            return m_dtype;
        }
    };
} // anonymous namespace

static void json_as_buffer(const nd::array& json, nd::array& out_tmp_ref, const char *&begin, const char *&end)
{
    // Check the dtype of 'json', and get pointers to the begin/end of a UTF-8 buffer
    ndt::type json_dtype = json.get_dtype().value_type();
    switch (json_dtype.get_kind()) {
        case string_kind: {
            const base_string_type *sdt = static_cast<const base_string_type *>(json_dtype.extended());
            switch (sdt->get_encoding()) {
                case string_encoding_ascii:
                case string_encoding_utf_8:
                    out_tmp_ref = json.eval();
                    // The data is already UTF-8, so use the buffer directly
                    sdt->get_string_range(&begin, &end,
                                    out_tmp_ref.get_ndo_meta(), out_tmp_ref.get_readonly_originptr());
                    break;
                default: {
                    // The data needs to be converted to UTF-8 before parsing
                    ndt::type utf8_dt = make_string_type(string_encoding_utf_8);
                    out_tmp_ref = json.ucast(utf8_dt).eval();
                    sdt = static_cast<const base_string_type *>(utf8_dt.extended());
                    sdt->get_string_range(&begin, &end,
                                    out_tmp_ref.get_ndo_meta(), out_tmp_ref.get_readonly_originptr());
                    break;
                }
            }
            break;
        }
        case bytes_kind: {
            out_tmp_ref = json.eval();
            const base_bytes_type *bdt = static_cast<const base_bytes_type *>(json_dtype.extended());
            bdt->get_bytes_range(&begin, &end,
                            out_tmp_ref.get_ndo_meta(), out_tmp_ref.get_readonly_originptr());
            break;
        }
        default: {
            stringstream ss;
            ss << "Input for JSON parsing must be either bytes (interpreted as UTF-8) or a string, not ";
            ss << json_dtype;
            throw runtime_error(ss.str());
            break;
        }
    }
}

void dynd::parse_json(nd::array& out, const nd::array& json)
{
    const char *json_begin = NULL, *json_end = NULL;
    nd::array tmp_ref;
    json_as_buffer(json, tmp_ref, json_begin, json_end);
    parse_json(out, json_begin, json_end);
}

nd::array dynd::parse_json(const ndt::type& dt, const nd::array& json)
{
    const char *json_begin = NULL, *json_end = NULL;
    nd::array tmp_ref;
    json_as_buffer(json, tmp_ref, json_begin, json_end);
    return parse_json(dt, json_begin, json_end);
}

static void parse_json(const ndt::type& dt, const char *metadata, char *out_data,
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
            throw json_parse_error(skip_whitespace(saved_begin, end), "string has no ending quote", ndt::type());
        }
        char c = *begin++;
        if (c == '\\') {
            if (begin == end) {
                throw json_parse_error(skip_whitespace(saved_begin, end), "string has no ending quote", ndt::type());
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
                        throw json_parse_error(begin-2, "invalid unicode escape sequence in string", ndt::type());
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
                            throw json_parse_error(begin-1, "invalid unicode escape sequence in string", ndt::type());
                        }
                    }
                    append_utf8_codepoint(cp, out_val);
                    break;
                }
                default:
                    throw json_parse_error(begin-2, "invalid escape sequence in string", ndt::type());
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

static void skip_json_value(const char *&begin, const char *end)
{
    begin = skip_whitespace(begin, end);
    if (begin == end) {
        throw json_parse_error(begin, "malformed JSON, expecting an element", ndt::type());
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
                        throw json_parse_error(begin, "expected string for name in object dict", ndt::type());
                    }
                    if (!parse_token(begin, end, ":")) {
                        throw json_parse_error(begin, "expected ':' separating name from value in object dict", ndt::type());
                    }
                    skip_json_value(begin, end);
                    if (!parse_token(begin, end, ",")) {
                        break;
                    }
                }
                if (!parse_token(begin, end, "}")) {
                    throw json_parse_error(begin, "expected object separator ',' or terminator '}'", ndt::type());
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
                    throw json_parse_error(begin, "expected array separator ',' or terminator ']'", ndt::type());
                }
            }
            break;
        case '"': {
            string s;
            if (!parse_json_string(begin, end, s)) {
                throw json_parse_error(begin, "invalid string", ndt::type());
            }
            break;
        }
        case 't':
            if (!parse_token(begin, end, "true")) {
                throw json_parse_error(begin, "invalid json value", ndt::type());
            }
            break;
        case 'f':
            if (!parse_token(begin, end, "false")) {
                throw json_parse_error(begin, "invalid json value", ndt::type());
            }
            break;
        case 'n':
            if (!parse_token(begin, end, "null")) {
                throw json_parse_error(begin, "invalid json value", ndt::type());
            }
            break;
        default:
            if (c == '-' || ('0' <= c && c <= '9')) {
                const char *nbegin = NULL, *nend = NULL;
                if (!parse_json_number(begin, end, nbegin, nend)) {
                    throw json_parse_error(begin, "invalid number", ndt::type());
                }
            } else {
                throw json_parse_error(begin, "invalid json value", ndt::type());
            }
    }
}

static void parse_fixed_dim_json(const ndt::type& dt, const char *metadata, char *out_data,
                const char *&begin, const char *end)
{
    const fixed_dim_dtype *fad = static_cast<const fixed_dim_dtype *>(dt.extended());
    intptr_t size = fad->get_fixed_dim_size();
    intptr_t stride = fad->get_fixed_stride();

    if (!parse_token(begin, end, "[")) {
        throw json_parse_error(begin, "expected list starting with '['", dt);
    }
    for (intptr_t i = 0; i < size; ++i) {
        parse_json(fad->get_element_type(), metadata, out_data + i * stride, begin, end);
        if (i < size-1 && !parse_token(begin, end, ",")) {
            throw json_parse_error(begin, "array is too short, expected ',' list item separator", dt);
        }
    }
    if (!parse_token(begin, end, "]")) {
        throw json_parse_error(begin, "array is too long, expected list terminator ']'", dt);
    }
}

static void parse_var_dim_json(const ndt::type& dt, const char *metadata, char *out_data,
                const char *&begin, const char *end)
{
    const var_dim_dtype *vad = static_cast<const var_dim_dtype *>(dt.extended());
    const var_dim_dtype_metadata *md = reinterpret_cast<const var_dim_dtype_metadata *>(metadata);
    intptr_t stride = md->stride;
    const ndt::type& element_dtype = vad->get_element_type();

    var_dim_dtype_data *out = reinterpret_cast<var_dim_dtype_data *>(out_data);
    char *out_end = NULL;

    memory_block_pod_allocator_api *allocator = get_memory_block_pod_allocator_api(md->blockref);
    intptr_t size = 0, allocated_size = 8;
    allocator->allocate(md->blockref, allocated_size * stride,
                    element_dtype.get_data_alignment(), &out->begin, &out_end);

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
            parse_json(element_dtype, metadata + sizeof(var_dim_dtype_metadata),
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

static void parse_struct_json(const ndt::type& dt, const char *metadata, char *out_data,
                const char *&begin, const char *end)
{
    const base_struct_type *fsd = static_cast<const base_struct_type *>(dt.extended());
    size_t field_count = fsd->get_field_count();
    const string *field_names = fsd->get_field_names();
    const ndt::type *field_types = fsd->get_field_types();
    const size_t *data_offsets = fsd->get_data_offsets(metadata);
    const size_t *metadata_offsets = fsd->get_metadata_offsets();

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
                skip_json_value(begin, end);
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
            ss << "object dict does not contain the field ";
            print_escaped_utf8_string(ss, field_names[i]);
            ss << " as required by the data type";
            throw json_parse_error(skip_whitespace(saved_begin, end), ss.str(), dt);
        }
    }
}

static void parse_bool_json(const ndt::type& dt, const char *metadata, char *out_data,
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
        dtype_assign(dt, metadata, out_data, ndt::make_dtype<dynd_bool>(), NULL, &value);
    }
}

static void parse_dynd_builtin_json(const ndt::type& dt, const char *DYND_UNUSED(metadata), char *out_data,
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

static void parse_integer_json(const ndt::type& dt, const char *metadata, char *out_data,
                const char *&begin, const char *end)
{
    // TODO: Parsing policy for how to handle integers
    parse_dynd_builtin_json(dt, metadata, out_data, begin, end);
}

static void parse_real_json(const ndt::type& dt, const char *metadata, char *out_data,
                const char *&begin, const char *end)
{
    // TODO: Parsing policy for how to handle reals
    parse_dynd_builtin_json(dt, metadata, out_data, begin, end);
}

static void parse_complex_json(const ndt::type& dt, const char *metadata, char *out_data,
                const char *&begin, const char *end)
{
    // TODO: Parsing policy for how to handle complex
    parse_dynd_builtin_json(dt, metadata, out_data, begin, end);
}

static void parse_jsonstring_json(const ndt::type& dt, const char *metadata, char *out_data,
                const char *&begin, const char *end)
{
    const char *saved_begin = skip_whitespace(begin, end);
    skip_json_value(begin, end);
    const base_string_type *bsd = static_cast<const base_string_type *>(dt.extended());
    // The skipped JSON value gets copied verbatim into the json string
    bsd->set_utf8_string(metadata, out_data, assign_error_none,
            saved_begin, begin);
}

static void parse_string_json(const ndt::type& dt, const char *metadata, char *out_data,
                const char *&begin, const char *end)
{
    const char *saved_begin = begin;
    string val;
    if (parse_json_string(begin, end, val)) {
        const base_string_type *bsd = static_cast<const base_string_type *>(dt.extended());
        try {
            bsd->set_utf8_string(metadata, out_data, assign_error_fractional, val);
        } catch (const std::exception& e) {
            throw json_parse_error(skip_whitespace(saved_begin, begin), e.what(), dt);
        }
    } else {
        throw json_parse_error(begin, "expected a string", dt);
    }
}

static void parse_datetime_json(const ndt::type& dt, const char *metadata, char *out_data,
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

static void parse_uniform_dim_json(const ndt::type& dt, const char *metadata, char *out_data,
                const char *&begin, const char *end)
{
    switch (dt.get_type_id()) {
        case fixed_dim_type_id:
            parse_fixed_dim_json(dt, metadata, out_data, begin, end);
            break;
        case var_dim_type_id:
            parse_var_dim_json(dt, metadata, out_data, begin, end);
            break;
        default: {
            stringstream ss;
            ss << "parse_json: unsupported uniform array dtype " << dt;
            throw runtime_error(ss.str());
        }
    }
}

static void parse_json(const ndt::type& dt, const char *metadata, char *out_data,
                const char *&json_begin, const char *json_end)
{
    switch (dt.get_kind()) {
        case uniform_dim_kind:
            parse_uniform_dim_json(dt, metadata, out_data, json_begin, json_end);
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
            if (dt.get_type_id() == json_type_id) {
                // The json type is a special string type that contains JSON directly
                // Copy the JSON verbatim in this case.
                parse_jsonstring_json(dt, metadata, out_data, json_begin, json_end);
            } else {
                parse_string_json(dt, metadata, out_data, json_begin, json_end);
            }
            return;
        case datetime_kind:
            parse_datetime_json(dt, metadata, out_data, json_begin, json_end);
            return;
        default: {
            stringstream ss;
            ss << "parse_json: unsupported dtype " << dt;
            throw runtime_error(ss.str());
        }
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
            throw json_parse_error(begin, "unexpected trailing JSON text", ndt::type());
        }
    } catch (const json_parse_error& e) {
        stringstream ss;
        string line_prev, line_cur;
        int line, column;
        get_error_line_column(json_begin, json_end, e.get_position(),
                        line_prev, line_cur, line, column);
        ss << "Error validating JSON at line " << line << ", column " << column << "\n";
        ss << "Message: " << e.get_message() << "\n";
        print_json_parse_error_marker(ss, line_prev, line_cur, line, column);
        throw runtime_error(ss.str());
    }
}

void dynd::parse_json(nd::array& out, const char *json_begin, const char *json_end)
{
    try {
        const char *begin = json_begin, *end = json_end;
        ndt::type dt = out.get_dtype();
        ::parse_json(dt, out.get_ndo_meta(), out.get_readwrite_originptr(), begin, end);
        begin = skip_whitespace(begin, end);
        if (begin != end) {
            throw json_parse_error(begin, "unexpected trailing JSON text", dt);
        }
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
        print_json_parse_error_marker(ss, line_prev, line_cur, line, column);
        throw runtime_error(ss.str());
    }
}

nd::array dynd::parse_json(const ndt::type& dt, const char *json_begin, const char *json_end)
{
    nd::array result;
    if (dt.get_data_size() != 0) {
        result = nd::empty(dt);
        parse_json(result, json_begin, json_end);
        if (!dt.is_builtin()) {
            dt.extended()->metadata_finalize_buffers(result.get_ndo_meta());
        }
        result.flag_as_immutable();
        return result;
    } else {
        stringstream ss;
        ss << "The dtype provided to parse_json, " << dt << ", cannot be used because it requires additional shape information";
        throw runtime_error(ss.str());
    }
}
