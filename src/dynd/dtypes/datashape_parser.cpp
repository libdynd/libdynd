//
// Copyright (C) 2011-13, DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <map>
#include <set>

#include <dynd/dtypes/datashape_parser.hpp>
#include <dynd/dtypes/strided_dim_dtype.hpp>
#include <dynd/dtypes/var_dim_dtype.hpp>
#include <dynd/dtypes/fixed_dim_dtype.hpp>
#include <dynd/dtypes/fixedstruct_dtype.hpp>
#include <dynd/dtypes/string_dtype.hpp>
#include <dynd/dtypes/fixedstring_dtype.hpp>
#include <dynd/dtypes/json_dtype.hpp>
#include <dynd/dtypes/date_dtype.hpp>
#include <dynd/dtypes/bytes_dtype.hpp>
#include <dynd/dtypes/dtype_dtype.hpp>

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

static dtype parse_rhs_expression(const char *&begin, const char *end, map<string, dtype>& symtable);

static map<string, dtype> builtin_types;
static set<string> reserved_typenames;
namespace {
    struct init_bit {
        init_bit() {
            builtin_types["void"] = make_dtype<void>();
            builtin_types["bool"] = make_dtype<dynd_bool>();
            builtin_types["int8"] = make_dtype<int8_t>();
            builtin_types["int16"] = make_dtype<int16_t>();
            builtin_types["int32"] = make_dtype<int32_t>();
            builtin_types["int64"] = make_dtype<int64_t>();
            builtin_types["uint8"] = make_dtype<uint8_t>();
            builtin_types["uint16"] = make_dtype<uint16_t>();
            builtin_types["uint32"] = make_dtype<uint32_t>();
            builtin_types["uint64"] = make_dtype<uint64_t>();
            builtin_types["float32"] = make_dtype<float>();
            builtin_types["float64"] = make_dtype<double>();
            builtin_types["cfloat32"] = builtin_types["complex64"] = make_dtype<complex<float> >();
            builtin_types["cfloat64"] = builtin_types["complex128"] = make_dtype<complex<double> >();
            builtin_types["json"] = make_json_dtype();
            builtin_types["date"] = make_date_dtype();
            builtin_types["bytes"] = make_bytes_dtype(1);
            builtin_types["dtype"] = make_dtype_dtype();
            for (map<string, dtype>::iterator i = builtin_types.begin();
                            i != builtin_types.end(); ++i) {
                reserved_typenames.insert(i->first);
            }
            reserved_typenames.insert("string");
        }
    };
    static init_bit builtin_types_initializer;
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
//               string('encoding')
//               string(NUMBER) |
//               string(NUMBER,'encoding')
// This is called after 'string' is already matched
static dtype parse_string_parameters(const char *&begin, const char *end)
{
    if (parse_token(begin, end, '(')) {
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
        if (!parse_token(begin, end, ')')) {
            throw datashape_parse_error(begin, "expected closing ')'");
        }
        if (string_size != 0) {
            return make_fixedstring_dtype(string_size, encoding);
        } else {
            return make_string_dtype(encoding);
        }
    } else {
        return make_string_dtype(string_encoding_utf_8);
    }
}

// record_item : NAME COLON rhs_expression
static bool parse_record_item(const char *&begin, const char *end, map<string, dtype>& symtable,
                string& out_field_name, dtype& out_field_type)
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
static dtype parse_record(const char *&begin, const char *end, map<string, dtype>& symtable)
{
    vector<string> field_name_list;
    vector<dtype> field_type_list;
    string field_name;
    dtype field_type;

    if (!parse_token(begin, end, '{')) {
        return dtype(uninitialized_type_id);
    }
    for (;;) {
        if (parse_record_item(begin, end, symtable, field_name, field_type)) {
            field_name_list.push_back(field_name);
            field_type_list.push_back(field_type);
        } else {
            throw datashape_parse_error(begin, "expected a record item");
        }
        
        if (parse_token(begin, end, ';')) {
            if (!field_name_list.empty() && parse_token(begin, end, '}')) {
                break;
            }
        } else if (parse_token(begin, end, '}')) {
            break;
        } else {
            throw datashape_parse_error(begin, "expected ';' or '}'");
        }
    }

    return make_fixedstruct_dtype(field_type_list.size(),
                    &field_type_list[0], &field_name_list[0]);
}

static dtype parse_rhs_expression(const char *&begin, const char *end, map<string, dtype>& symtable)
{
    dtype result;
    vector<intptr_t> shape;
    // rhs_expression : ((NAME | NUMBER) COMMA)* (record | NAME LPAREN rhs_expression RPAREN | NAME)
    for (;;) {
        const char *saved_begin = begin;
        string n = parse_name_or_number(begin, end); // NAME | NUMBER
        if (n.empty()) {
            break;
        } else if (!parse_token(begin, end, ',')) { // COMMA
            begin = saved_begin;
            break;
        } else {
            if ('0' <= n[0] && n[0] <= '9') {
                shape.push_back(atoi(n.c_str()));
            } else if (n == "VarDim") { // TODO: This isn't in the Blaze datashape grammar
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
                return dtype(uninitialized_type_id);
            } else {
                throw datashape_parse_error(begin, "expected data type");
            }
        } else if (n == "string") {
            result = parse_string_parameters(begin, end);
        } else {
            map<string,dtype>::const_iterator i = builtin_types.find(n);
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
                    result = make_strided_dim_dtype(result);
                } else if (shape[i] == -1) {
                    result = make_var_dim_dtype(result);
                } else {
                    result = make_fixed_dim_dtype(shape[i], result);
                }
            }
        }
    }
    return result;
}

static dtype parse_stmt(const char *&begin, const char *end, map<string, dtype>& symtable)
{
    // stmt : TYPE name EQUALS rhs_expression
    // NOTE that this doesn't support parameterized lhs_expression, this is subset of Blaze datashape
    if (parse_token(begin, end, "type")) {
        const char *saved_begin = begin;
        string tname = parse_name(begin, end);
        if (tname.empty()) {
            throw datashape_parse_error(begin, "expected an identifier for a type name");
        }
        if (!parse_token(begin, end, '=')) {
            throw datashape_parse_error(begin, "expected an '='");
        }
        dtype result = parse_rhs_expression(begin, end, symtable);
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
static dtype parse_top(const char *&begin, const char *end, map<string, dtype>& symtable)
{
    dtype result = parse_stmt(begin, end, symtable);
    if (result.get_type_id() == uninitialized_type_id) {
        throw datashape_parse_error(begin, "expected a datashape statement");
    }
    for (;;) {
        dtype next = parse_stmt(begin, end, symtable);
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

dtype dynd::dtype_from_datashape(const char *datashape_begin, const char *datashape_end)
{
    try {
        // Symbol table for intermediate types declared in the datashape
        map<string, dtype> symtable;
        // Parse the datashape and construct the dtype
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

