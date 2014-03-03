//
// Copyright (C) 2011-14 Mark Wiebe, DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <map>
#include <set>

#include <dynd/types/datashape_parser.hpp>
#include <dynd/types/strided_dim_type.hpp>
#include <dynd/types/var_dim_type.hpp>
#include <dynd/types/fixed_dim_type.hpp>
#include <dynd/types/cstruct_type.hpp>
#include <dynd/types/string_type.hpp>
#include <dynd/types/fixedstring_type.hpp>
#include <dynd/types/json_type.hpp>
#include <dynd/types/date_type.hpp>
#include <dynd/types/datetime_type.hpp>
#include <dynd/types/bytes_type.hpp>
#include <dynd/types/type_type.hpp>
#include <dynd/types/ckernel_deferred_type.hpp>
#include <dynd/types/type_alignment.hpp>
#include <dynd/types/pointer_type.hpp>
#include <dynd/types/char_type.hpp>
#include <dynd/types/cuda_host_type.hpp>
#include <dynd/types/cuda_device_type.hpp>

using namespace std;
using namespace dynd;

namespace {
    class datashape_parse_error {
        const char *m_position;
        const char *m_message;
    public:
        datashape_parse_error(const char *position, const char *message)
            : m_position(position), m_message(message) {
        }
        virtual ~datashape_parse_error() {
        }
        const char *get_position() const {
            return m_position;
        }
        const char *get_message() const {
            return m_message;
        }
    };
} // anonymous namespace

// Simple recursive descent parser for a subset of the Blaze datashape grammar.
// (Blaze grammar modified slightly to work this way)

static ndt::type parse_rhs_expression(const char *&begin, const char *end, map<string, ndt::type>& symtable);

static const map<string, ndt::type>& get_builtin_types()
{
    static map<string, ndt::type> builtin_types;
    if (builtin_types.empty()) {
        builtin_types["void"] = ndt::make_type<void>();
        builtin_types["bool"] = ndt::make_type<dynd_bool>();
        builtin_types["int8"] = ndt::make_type<int8_t>();
        builtin_types["int16"] = ndt::make_type<int16_t>();
        builtin_types["int32"] = ndt::make_type<int32_t>();
        builtin_types["int64"] = ndt::make_type<int64_t>();
        builtin_types["int128"] = ndt::make_type<dynd_int128>();
        builtin_types["intptr"] = ndt::make_type<intptr_t>();
        builtin_types["uint8"] = ndt::make_type<uint8_t>();
        builtin_types["uint16"] = ndt::make_type<uint16_t>();
        builtin_types["uint32"] = ndt::make_type<uint32_t>();
        builtin_types["uint64"] = ndt::make_type<uint64_t>();
        builtin_types["uint128"] = ndt::make_type<dynd_uint128>();
        builtin_types["uintptr"] = ndt::make_type<uintptr_t>();
        builtin_types["float16"] = ndt::make_type<dynd_float16>();
        builtin_types["float32"] = ndt::make_type<float>();
        builtin_types["float64"] = ndt::make_type<double>();
        builtin_types["float128"] = ndt::make_type<dynd_float128>();
        builtin_types["complex64"] = ndt::make_type<dynd_complex<float> >();
        builtin_types["complex128"] = ndt::make_type<dynd_complex<double> >();
        builtin_types["json"] = ndt::make_json();
        builtin_types["date"] = ndt::make_date();
        builtin_types["bytes"] = ndt::make_bytes(1);
        builtin_types["type"] = ndt::make_type();
        builtin_types["ckernel_deferred"] = ndt::make_ckernel_deferred();
    }
    return builtin_types;
}
static const set<string>& get_reserved_typenames()
{
    static set<string> reserved_typenames;
    if (reserved_typenames.empty()) {
        const map<string, ndt::type>& builtin_types = get_builtin_types();
        for (map<string, ndt::type>::const_iterator i = builtin_types.begin();
                        i != builtin_types.end(); ++i) {
            reserved_typenames.insert(i->first);
        }
        reserved_typenames.insert("string");
        reserved_typenames.insert("char");
        reserved_typenames.insert("datetime");
        reserved_typenames.insert("unaligned");
        reserved_typenames.insert("pointer");
        reserved_typenames.insert("complex");
		reserved_typenames.insert("cuda_host");
		reserved_typenames.insert("cuda_device");
    }
    return reserved_typenames;
}

static const char *skip_whitespace(const char *begin, const char *end)
{
    while (begin < end && isspace(*begin)) {
        ++begin;
    }

    // Comments
    if (begin < end && *begin == '#') {
        const char *line_end = (const char *)memchr(begin, '\n', end - begin);
        if (line_end == NULL) {
            return end;
        } else {
            return skip_whitespace(line_end + 1, end);
        }
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

static bool parse_token(const char *&begin, const char *end, char token)
{
    const char *begin_skipws = skip_whitespace(begin, end);
    if (1 <= end - begin_skipws && *begin_skipws == token) {
        begin = begin_skipws + 1;
        return true;
    } else {
        return false;
    }
}

// NAME : [a-zA-Z_][a-zA-Z0-9_]*
static string parse_name(const char *&begin, const char *end)
{
    const char *begin_skipws, *pos;
    begin_skipws = pos = skip_whitespace(begin, end);
    if (pos == end) {
        return "";
    }
    if (('a' <= *pos && *pos <= 'z') ||
                    ('A' <= *pos && *pos <= 'Z') ||
                    *pos == '_') {
        ++pos;
    } else {
        return "";
    }
    while (pos < end && (('a' <= *pos && *pos <= 'z') ||
                    ('A' <= *pos && *pos <= 'Z') ||
                    ('0' <= *pos && *pos <= '9') ||
                    *pos == '_')) {
        ++pos;
    }
    begin = pos;
    return string(begin_skipws, pos);
}

static string parse_number(const char *&begin, const char *end)
{
    const char *begin_skipws = skip_whitespace(begin, end);
    const char *pos = begin_skipws;
    while (pos < end && ('0' <= *pos && *pos <= '9')) {
        ++pos;
    }
    if (pos > begin_skipws) {
        begin = pos;
        return string(begin_skipws, pos);
    } else {
        return string();
    }
}

static string parse_name_or_number(const char *&begin, const char *end)
{
    // NAME
    string result = parse_name(begin, end);
    if (result.empty()) {
        // NUMBER
        return parse_number(begin, end);
    }
    return result;
}

static bool parse_quoted_string(const char *&begin, const char *end, string& out_val)
{
    out_val = "";
    const char *saved_begin = begin;
    char beginning_quote = 0;
    if (parse_token(begin, end, '\'')) {
        beginning_quote = '\'';
    } else if (parse_token(begin, end, '"')) {
        beginning_quote = '"';
    } else {
        return false;
    }
    for (;;) {
        if (begin == end) {
            throw datashape_parse_error(skip_whitespace(saved_begin, end),
                            "string has no ending quote");
        }
        char c = *begin++;
        if (c == '\\') {
            if (begin == end) {
                throw datashape_parse_error(skip_whitespace(saved_begin, end),
                                "string has no ending quote");
            }
            c = *begin++;
            switch (c) {
                case '"':
                case '\'':
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
                        throw datashape_parse_error(begin-2,
                                        "invalid unicode escape sequence in string");
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
                            throw datashape_parse_error(begin-1,
                                            "invalid unicode escape sequence in string");
                        }
                    }
                    append_utf8_codepoint(cp, out_val);
                    break;
                }
                default:
                    throw datashape_parse_error(begin-2,
                                    "invalid escape sequence in string");
            }
        } else if (c != beginning_quote) {
            out_val += c;
        } else {
            return true;
        }
    }
}

static string_encoding_t string_to_encoding(const char *error_begin, const string& estr)
{
    if (estr == "A" || estr == "ascii" || estr == "us-ascii") {
        return string_encoding_ascii;
    } else if (estr == "U8" || estr == "utf8" || estr == "utf-8" || estr == "utf_8") {
        return string_encoding_utf_8;
    } else if (estr == "U16" || estr == "utf16" || estr == "utf-16" || estr == "utf_16") {
        return string_encoding_utf_16;
    } else if (estr == "U32" || estr == "utf32" || estr == "utf-32" || estr == "utf_32") {
        return string_encoding_utf_32;
    } else if (estr == "ucs2" || estr == "ucs-2" || estr == "ucs_2") {
        return string_encoding_ucs_2;
    } else {
        throw datashape_parse_error(error_begin, "unrecognized string encoding");
    }
}

// string_type : string |
//               string['encoding']
//               string[NUMBER] |
//               string[NUMBER,'encoding']
// This is called after 'string' is already matched
static ndt::type parse_string_parameters(const char *&begin, const char *end)
{
    if (parse_token(begin, end, '[')) {
        const char *saved_begin = begin;
        string value = parse_number(begin, end);
        string encoding_str;
        string_encoding_t encoding = string_encoding_utf_8;
        int string_size = 0;
        if (!value.empty()) {
            string_size = atoi(value.c_str());
            if (string_size == 0) {
                throw datashape_parse_error(saved_begin, "string size cannot be zero");
            }
            if (parse_token(begin, end, ',')) {
                saved_begin = begin;
                if (!parse_quoted_string(begin, end, encoding_str)) {
                    throw datashape_parse_error(saved_begin, "expected a string encoding");
                }
                encoding = string_to_encoding(saved_begin, encoding_str);
            }
        } else {
            if (!parse_quoted_string(begin, end, encoding_str)) {
                throw datashape_parse_error(saved_begin, "expected a size integer or string encoding");
            }
            encoding = string_to_encoding(saved_begin, encoding_str);
        }
        if (!parse_token(begin, end, ']')) {
            throw datashape_parse_error(begin, "expected closing ']'");
        }
        if (string_size != 0) {
            return ndt::make_fixedstring(string_size, encoding);
        } else {
            return ndt::make_string(encoding);
        }
    } else {
        return ndt::make_string(string_encoding_utf_8);
    }
}

// char_type : char | char[encoding]
// This is called after 'char' is already matched
static ndt::type parse_char_parameters(const char *&begin, const char *end)
{
    if (parse_token(begin, end, '[')) {
        const char *saved_begin = begin;
        string encoding_str = parse_name(begin, end);
        string_encoding_t encoding;
        if (!encoding_str.empty()) {
            encoding = string_to_encoding(saved_begin, encoding_str);
        } else {
            throw datashape_parse_error(saved_begin, "expected a string encoding");
        }
        if (!parse_token(begin, end, ']')) {
            throw datashape_parse_error(begin, "expected closing ']'");
        }
        return ndt::make_char(encoding);
    } else {
        return ndt::make_char();
    }
}

// complex_type : complex[float_type]
// This is called after 'complex' is already matched
static ndt::type parse_complex_parameters(const char *&begin, const char *end,
                map<string, ndt::type>& symtable)
{
    if (parse_token(begin, end, '[')) {
        const char *saved_begin = begin;
        ndt::type tp = parse_rhs_expression(begin, end, symtable);
        if (tp.get_type_id() == uninitialized_type_id) {
            throw datashape_parse_error(begin, "expected a type parameter");
        }
        if (!parse_token(begin, end, ']')) {
            throw datashape_parse_error(begin, "expected closing ']'");
        }
        if (tp.get_type_id() == float32_type_id) {
            return ndt::make_type<dynd_complex<float> >();
        } else if (tp.get_type_id() == float64_type_id) {
            return ndt::make_type<dynd_complex<double> >();
        } else {
            throw datashape_parse_error(saved_begin, "unsupported real type for complex numbers");
        }
    } else {
        throw datashape_parse_error(begin, "expected opening '['");
    }
}

// cuda_host_type : cuda_host[storage_type]
// This is called after 'cuda_host' is already matched
static ndt::type parse_cuda_host_parameters(const char *&begin, const char *end,
                map<string, ndt::type>& symtable)
{
    if (parse_token(begin, end, '[')) {
        ndt::type tp = parse_rhs_expression(begin, end, symtable);
        if (tp.get_type_id() == uninitialized_type_id) {
            throw datashape_parse_error(begin, "expected a type parameter");
        }
        if (!parse_token(begin, end, ']')) {
            throw datashape_parse_error(begin, "expected closing ']'");
        }
#ifdef DYND_CUDA
		return ndt::make_cuda_host(tp);
#else
        throw datashape_parse_error(begin, "type is not available");
#endif // DYND_CUDA
    } else {
        throw datashape_parse_error(begin, "expected opening '['");
    }
}

// cuda_device_type : cuda_device[storage_type]
// This is called after 'cuda_device' is already matched
static ndt::type parse_cuda_device_parameters(const char *&begin, const char *end,
                map<string, ndt::type>& symtable)
{
    if (parse_token(begin, end, '[')) {
        ndt::type tp = parse_rhs_expression(begin, end, symtable);
        if (tp.get_type_id() == uninitialized_type_id) {
            throw datashape_parse_error(begin, "expected a type parameter");
        }
        if (!parse_token(begin, end, ']')) {
            throw datashape_parse_error(begin, "expected closing ']'");
        }
#ifdef DYND_CUDA
		return ndt::make_cuda_device(tp);
#else
        throw datashape_parse_error(begin, "type is not available");
#endif // DYND_CUDA
    } else {
        throw datashape_parse_error(begin, "expected opening '['");
    }
}

// datetime_type : datetime['unit'] |
//               datetime['unit','timezone']
// This is called after 'datetime' is already matched
static ndt::type parse_datetime_parameters(const char *&begin, const char *end)
{
    if (parse_token(begin, end, '[')) {
        datetime_unit_t unit;
        datetime_tz_t timezone = tz_abstract;
        string unit_str;
        const char *saved_begin = begin;
        if (!parse_quoted_string(begin, end, unit_str)) {
            throw datashape_parse_error(begin, "expected a datetime unit string");
        }
        if (unit_str == "hour") {
            unit = datetime_unit_hour;
        } else if (unit_str == "min") {
            unit = datetime_unit_minute;
        } else if (unit_str == "sec") {
            unit = datetime_unit_second;
        } else if (unit_str == "msec") {
            unit = datetime_unit_msecond;
        } else if (unit_str == "usec") {
            unit = datetime_unit_usecond;
        } else if (unit_str == "nsec") {
            unit = datetime_unit_nsecond;
        } else {
            throw datashape_parse_error(saved_begin, "invalid datetime unit");
        }
        // Parse the timezone
        if (parse_token(begin, end, ',')) {
            string timezone_str;
            saved_begin = begin;
            if (!parse_quoted_string(begin, end, timezone_str)) {
                throw datashape_parse_error(begin, "expected a datetime timezone string");
            }
            if (timezone_str == "abstract") {
                timezone = tz_abstract;
            } else if (timezone_str == "UTC" || timezone_str == "utc") {
                timezone = tz_utc;
            } else {
                throw datashape_parse_error(saved_begin, "invalid datetime timezone");
            }
        }
        if (!parse_token(begin, end, ']')) {
            throw datashape_parse_error(begin, "expected closing ']'");
        }
         
        return ndt::make_datetime(unit, timezone);
    } else {
        throw datashape_parse_error(begin, "expected datetime parameters opening '['");
    }
}

static ndt::type parse_unaligned_parameters(const char *&begin, const char *end,
                map<string, ndt::type>& symtable)
{
    if (!parse_token(begin, end, '[')) {
        throw datashape_parse_error(begin, "expected opening '[' after 'unaligned'");
    }
    ndt::type tp = parse_rhs_expression(begin, end, symtable);
    if (tp.get_type_id() == uninitialized_type_id) {
        throw datashape_parse_error(begin, "expected a data type");
    }
    if (!parse_token(begin, end, ']')) {
        throw datashape_parse_error(begin, "expected closing ']'");
    }
    // TODO catch errors, convert them to datashape_parse_error so the position is shown
    return ndt::make_unaligned(tp);
}

static ndt::type parse_pointer_parameters(const char *&begin, const char *end,
                map<string, ndt::type>& symtable)
{
    if (!parse_token(begin, end, '[')) {
        throw datashape_parse_error(begin, "expected opening '[' after 'pointer'");
    }
    ndt::type tp = parse_rhs_expression(begin, end, symtable);
    if (tp.get_type_id() == uninitialized_type_id) {
        throw datashape_parse_error(begin, "expected a data type");
    }
    if (!parse_token(begin, end, ']')) {
        throw datashape_parse_error(begin, "expected closing ']'");
    }
    // TODO catch errors, convert them to datashape_parse_error so the position is shown
    return ndt::make_pointer(tp);
}

// record_item : NAME COLON rhs_expression
static bool parse_record_item(const char *&begin, const char *end, map<string, ndt::type>& symtable,
                string& out_field_name, ndt::type& out_field_type)
{
    out_field_name = parse_name(begin, end);
    if (out_field_name.empty()) {
        return false;
    }
    if (!parse_token(begin, end , ':')) {
        throw datashape_parse_error(begin, "expected ':' after record item name");
    }
    bool parens = false;
    if (parse_token(begin, end, '(')) {
        parens = true;
    }
    out_field_type = parse_rhs_expression(begin, end, symtable);
    if (out_field_type.get_type_id() == uninitialized_type_id) {
        throw datashape_parse_error(begin, "expected a data type");
    }
    if (parens && !parse_token(begin, end, ')')) {
        throw datashape_parse_error(begin, "expected closing ')'");
    }

    return true;
}

// record : LBRACE record_item record_item* RBRACE
static ndt::type parse_record(const char *&begin, const char *end, map<string, ndt::type>& symtable)
{
    vector<string> field_name_list;
    vector<ndt::type> field_type_list;
    string field_name;
    ndt::type field_type;

    if (!parse_token(begin, end, '{')) {
        return ndt::type(uninitialized_type_id);
    }
    for (;;) {
        if (parse_record_item(begin, end, symtable, field_name, field_type)) {
            field_name_list.push_back(field_name);
            field_type_list.push_back(field_type);
        } else {
            throw datashape_parse_error(begin, "expected a record item");
        }
        
        if (parse_token(begin, end, ',')) {
            if (!field_name_list.empty() && parse_token(begin, end, '}')) {
                break;
            }
        } else if (parse_token(begin, end, '}')) {
            break;
        } else {
            throw datashape_parse_error(begin, "expected ',' or '}'");
        }
    }

    return ndt::make_cstruct(field_type_list.size(),
                    &field_type_list[0], &field_name_list[0]);
}

/** This is what parses the main datashape grammar, excluding type aliases, etc. */
static ndt::type parse_rhs_expression(const char *&begin, const char *end, map<string, ndt::type>& symtable)
{
    const set<string>& reserved_typenames = get_reserved_typenames();
    ndt::type result;
    vector<intptr_t> shape;
    // rhs_expression : ((NAME | NUMBER) ASTERISK)* (record | NAME LPAREN rhs_expression RPAREN | NAME)
    for (;;) {
        const char *saved_begin = begin;
        string n = parse_name_or_number(begin, end); // NAME | NUMBER
        if (n.empty()) {
            break;
        } else if (!parse_token(begin, end, '*')) { // ASTERISK
            begin = saved_begin;
            break;
        } else {
            if ('0' <= n[0] && n[0] <= '9') {
                shape.push_back(atoi(n.c_str()));
            } else if (n == "var") {
                // Use -1 to signal a variable-length dimension
                shape.push_back(-1);
            } else if (reserved_typenames.find(n) == reserved_typenames.end() &&
                            symtable.find(n) == symtable.end()) {
                // Use -2 to signal a free dimension
                shape.push_back(-2);
            } else {
                throw datashape_parse_error(skip_whitespace(saved_begin, end),
                                "only free variables can be used for datashape dimensions");
            }
        }
    }
    // record
    result = parse_record(begin, end, symtable);
    if (result.get_type_id() == uninitialized_type_id) {
        const char *begin_saved = begin;
        // NAME
        string n = parse_name(begin, end);
        if (n.empty()) {
            if (shape.empty()) {
                return ndt::type(uninitialized_type_id);
            } else {
                throw datashape_parse_error(begin, "expected data type");
            }
        } else if (n == "string") {
            result = parse_string_parameters(begin, end);
        } else if (n == "complex") {
            result = parse_complex_parameters(begin, end, symtable);
        } else if (n == "datetime") {
            result = parse_datetime_parameters(begin, end);
        } else if (n == "unaligned") {
            result = parse_unaligned_parameters(begin, end, symtable);
        } else if (n == "pointer") {
            result = parse_pointer_parameters(begin, end, symtable);
        } else if (n == "char") {
            result = parse_char_parameters(begin, end);
        } else if (n == "cuda_host") {
            result = parse_cuda_host_parameters(begin, end, symtable);
        } else if (n == "cuda_device") {
            result = parse_cuda_device_parameters(begin, end, symtable);
        } else {
            const map<string, ndt::type>& builtin_types = get_builtin_types();
            map<string, ndt::type>::const_iterator i = builtin_types.find(n);
            if (i != builtin_types.end()) {
                result = i->second;
            } else {
                i = symtable.find(n);
                if (i != symtable.end()) {
                    result = i->second;
                } else {
                    // LPAREN rhs_expression RPAREN
                    const char *begin_tmp = begin;
                    if (parse_token(begin_tmp, end, '(')) {
                        throw datashape_parse_error(begin,
                                        "DyND does not support this kind of datashape parsing yet");
                    } else {
                        throw datashape_parse_error(skip_whitespace(begin_saved, end),
                                        "unrecognized data type");
                    }
                }
            }
        }
    }

    if (result.get_type_id() != uninitialized_type_id) {
        // Apply the shape
        if (!shape.empty()) {
            for (ptrdiff_t i = (ptrdiff_t)shape.size() - 1; i >= 0; --i) {
                if (shape[i] == -2) {
                    result = ndt::make_strided_dim(result);
                } else if (shape[i] == -1) {
                    result = ndt::make_var_dim(result);
                } else {
                    result = ndt::make_fixed_dim(shape[i], result);
                }
            }
        }
    }
    return result;
}

static ndt::type parse_stmt(const char *&begin, const char *end, map<string, ndt::type>& symtable)
{
    // stmt : TYPE name EQUALS rhs_expression
    // NOTE that this doesn't support parameterized lhs_expression, this is subset of Blaze datashape
    if (parse_token(begin, end, "type")) {
        const map<string, ndt::type>& builtin_types = get_builtin_types();
        const char *saved_begin = begin;
        string tname = parse_name(begin, end);
        if (tname.empty()) {
            if (skip_whitespace(begin, end) == end) {
                // If it's only "type" by itself, return the "type" type
                return builtin_types.find("type")->second;
            } else {
                throw datashape_parse_error(begin, "expected an identifier for a type name");
            }
        }
        if (!parse_token(begin, end, '=')) {
            throw datashape_parse_error(begin, "expected an '='");
        }
        ndt::type result = parse_rhs_expression(begin, end, symtable);
        if (result.get_type_id() == uninitialized_type_id) {
            throw datashape_parse_error(begin, "expected a data type");
        }
        // ACTION: Put the parsed type in the symbol table
        if (builtin_types.find(tname) != builtin_types.end()) {
            throw datashape_parse_error(skip_whitespace(saved_begin, end),
                            "cannot redefine a builtin type");
        }
        if (symtable.find(tname) != symtable.end()) {
            throw datashape_parse_error(skip_whitespace(saved_begin, end),
                            "type name already defined in datashape string");
        }
        symtable[tname] = result;
        return result;
    }
    // stmt : rhs_expression
    return parse_rhs_expression(begin, end, symtable);
}

// top : stmt stmt*
static ndt::type parse_top(const char *&begin, const char *end, map<string, ndt::type>& symtable)
{
    ndt::type result = parse_stmt(begin, end, symtable);
    if (result.get_type_id() == uninitialized_type_id) {
        throw datashape_parse_error(begin, "expected a datashape statement");
    }
    for (;;) {
        ndt::type next = parse_stmt(begin, end, symtable);
        if (next.get_type_id() == uninitialized_type_id) {
            begin = skip_whitespace(begin, end);
            if (begin != end) {
                throw datashape_parse_error(begin, "unexpected token in datashape");
            }
            return result;
        } else {
            result = next;
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

ndt::type dynd::type_from_datashape(const char *datashape_begin, const char *datashape_end)
{
    try {
        // Symbol table for intermediate types declared in the datashape
        map<string, ndt::type> symtable;
        // Parse the datashape and construct the type
        const char *begin = datashape_begin, *end = datashape_end;
        return parse_top(begin, end, symtable);
    } catch (const datashape_parse_error& e) {
        stringstream ss;
        string line_prev, line_cur;
        int line, column;
        get_error_line_column(datashape_begin, datashape_end, e.get_position(),
                        line_prev, line_cur, line, column);
        ss << "Error parsing datashape at line " << line << ", column " << column << "\n";
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

