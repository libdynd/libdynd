//
// Copyright (C) 2011-14 Mark Wiebe, DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <map>
#include <set>

#include <dynd/types/datashape_parser.hpp>
#include <dynd/parser_util.hpp>
#include <dynd/types/strided_dim_type.hpp>
#include <dynd/types/var_dim_type.hpp>
#include <dynd/types/fixed_dim_type.hpp>
#include <dynd/types/cfixed_dim_type.hpp>
#include <dynd/types/cstruct_type.hpp>
#include <dynd/types/struct_type.hpp>
#include <dynd/types/ctuple_type.hpp>
#include <dynd/types/tuple_type.hpp>
#include <dynd/types/string_type.hpp>
#include <dynd/types/fixedstring_type.hpp>
#include <dynd/types/json_type.hpp>
#include <dynd/types/date_type.hpp>
#include <dynd/types/time_type.hpp>
#include <dynd/types/datetime_type.hpp>
#include <dynd/types/fixedbytes_type.hpp>
#include <dynd/types/bytes_type.hpp>
#include <dynd/types/type_type.hpp>
#include <dynd/types/ckernel_deferred_type.hpp>
#include <dynd/types/type_alignment.hpp>
#include <dynd/types/pointer_type.hpp>
#include <dynd/types/char_type.hpp>
#include <dynd/types/byteswap_type.hpp>
#include <dynd/types/cuda_host_type.hpp>
#include <dynd/types/cuda_device_type.hpp>
#include <dynd/types/ndarrayarg_type.hpp>
#include <dynd/types/funcproto_type.hpp>
#include <dynd/types/typevar_type.hpp>
#include <dynd/types/typevar_dim_type.hpp>

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
        builtin_types["int"] = ndt::make_type<int32_t>();
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
        builtin_types["real"] = ndt::make_type<double>();
        builtin_types["float128"] = ndt::make_type<dynd_float128>();
        builtin_types["complex64"] = ndt::make_type<dynd_complex<float> >();
        builtin_types["complex128"] = ndt::make_type<dynd_complex<double> >();
        builtin_types["complex"] = ndt::make_type<dynd_complex<double> >();
        builtin_types["json"] = ndt::make_json();
        builtin_types["date"] = ndt::make_date();
        builtin_types["time"] = ndt::make_time(tz_abstract);
        builtin_types["datetime"] = ndt::make_datetime(tz_abstract);
        builtin_types["bytes"] = ndt::make_bytes(1);
        builtin_types["type"] = ndt::make_type();
        builtin_types["ckernel_deferred"] = ndt::make_ckernel_deferred();
        builtin_types["ndarrayarg"] = ndt::make_ndarrayarg();
    }
    return builtin_types;
}

template <int N>
inline bool parse_token_ds(const char *&rbegin, const char *end,
                        const char (&token)[N])
{
    const char *begin = rbegin;
    parse::skip_whitespace_and_pound_comments(begin, end);
    if (parse::parse_token_no_ws(begin, end, token)) {
        rbegin = begin;
        return true;
    } else {
        return false;
    }
}

inline bool parse_token_ds(const char *&rbegin, const char *end, char token)
{
    const char *begin = rbegin;
    parse::skip_whitespace_and_pound_comments(begin, end);
    if (parse::parse_token_no_ws(begin, end, token)) {
        rbegin = begin;
        return true;
    } else {
        return false;
    }
}

static string parse_name_or_number(const char *&rbegin, const char *end)
{
    const char *begin = rbegin;
    // NAME
    const char *result_begin, *result_end;
    parse::skip_whitespace_and_pound_comments(begin, end);
    if (!parse::parse_name_no_ws(begin, end, result_begin, result_end)) {
        // NUMBER
        if (!parse::parse_unsigned_int_no_ws(begin, end, result_begin,
                                             result_end)) {
            return string();
        }
    }
    rbegin = begin;
    return string(result_begin, result_end);
}

static string parse_number(const char *&rbegin, const char *end)
{
    const char *begin = rbegin;
    const char *result_begin, *result_end;
    // NUMBER
    parse::skip_whitespace_and_pound_comments(begin, end);
    if (!parse::parse_unsigned_int_no_ws(begin, end, result_begin,
                                            result_end)) {
        return string();
    }
    rbegin = begin;
    return string(result_begin, result_end);
}

static bool parse_quoted_string(const char *&rbegin, const char *end,
                                string &out_val)
{
    const char *begin = rbegin;
    char beginning_quote = 0;
    out_val = "";
    if (parse_token_ds(begin, end, '\'')) {
        beginning_quote = '\'';
    } else if (parse_token_ds(begin, end, '"')) {
        beginning_quote = '"';
    } else {
        return false;
    }
    for (;;) {
        if (begin == end) {
            begin = rbegin;
            parse::skip_whitespace_and_pound_comments(begin, end);
            throw datashape_parse_error(begin,
                            "string has no ending quote");
        }
        char c = *begin++;
        if (c == '\\') {
            if (begin == end) {
                begin = rbegin;
                parse::skip_whitespace_and_pound_comments(begin, end);
                throw datashape_parse_error(begin,
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
            rbegin = begin;
            return true;
        }
    }
}

// fixed_type : fixed[N] * rhs_expression
static ndt::type parse_fixed_dim_parameters(const char *&rbegin, const char *end,
                                            map<string, ndt::type> &symtable)
{
    const char *begin = rbegin;
    if (parse_token_ds(begin, end, '[')) {
        const char *saved_begin = begin;
        string dim_size_str = parse_number(begin, end);
        if (dim_size_str.empty()) {
            throw datashape_parse_error(saved_begin, "expected dimension size");
        }
        intptr_t dim_size = (intptr_t)DYND_ATOLL(dim_size_str.c_str());
        if (!parse_token_ds(begin, end, ']')) {
            throw datashape_parse_error(begin, "expected closing ']'");
        }
        if (!parse_token_ds(begin, end, '*')) {
            throw datashape_parse_error(begin, "expected dimension separator '*'");
        }
        ndt::type element_tp = parse_rhs_expression(begin, end, symtable);
        if (element_tp.get_type_id() == uninitialized_type_id) {
            throw datashape_parse_error(begin, "expected element type");
        }
        rbegin = begin;
        return ndt::make_fixed_dim(dim_size, element_tp);
    } else {
        throw datashape_parse_error(begin, "expected opening '['");
    }
}

// cfixed_type : cfixed[N] * rhs_expression
// cfixed_type : cfixed[N, stride=M] * rhs_expression
static ndt::type parse_cfixed_dim_parameters(const char *&rbegin,
                                             const char *end,
                                             map<string, ndt::type> &symtable)
{
    const char *begin = rbegin;
    if (parse_token_ds(begin, end, '[')) {
        const char *saved_begin = begin;
        string dim_size_str = parse_number(begin, end);
        intptr_t dim_size;
        intptr_t stride = numeric_limits<intptr_t>::min();
        if (!dim_size_str.empty()) {
            dim_size = (intptr_t)DYND_ATOLL(dim_size_str.c_str());
            if (dim_size < 0) {
                throw datashape_parse_error(saved_begin, "dim size cannot be negative");
            }
            if (parse_token_ds(begin, end, ',')) {
                saved_begin = begin;
                if (!parse_token_ds(begin, end, "stride")) {
                    throw datashape_parse_error(begin, "expected keyword parameter 'stride'");
                }
                // bytes type with an alignment
                if (!parse_token_ds(begin, end, '=')) {
                    throw datashape_parse_error(begin, "expected an =");
                }
                string stride_str = parse_number(begin, end);
                stride = (intptr_t)DYND_ATOLL(stride_str.c_str());
            }
            if (!parse_token_ds(begin, end, ']')) {
                throw datashape_parse_error(begin, "expected closing ']'");
            }
            if (!parse_token_ds(begin, end, '*')) {
                throw datashape_parse_error(begin, "expected dimension separator '*'");
            }
            ndt::type element_tp = parse_rhs_expression(begin, end, symtable);
            if (element_tp.get_type_id() == uninitialized_type_id) {
                throw datashape_parse_error(begin, "expected element type");
            }
            rbegin = begin;
            if (stride == numeric_limits<intptr_t>::min()) {
                return ndt::make_cfixed_dim(dim_size, element_tp);
            } else {
                return ndt::make_cfixed_dim(dim_size, element_tp, stride);
            }
        } else {
            throw datashape_parse_error(saved_begin, "expected dimension size");
        }
    } else {
        throw datashape_parse_error(begin, "expected opening '['");
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
static ndt::type parse_string_parameters(const char *&rbegin, const char *end)
{
    const char *begin = rbegin;
    if (parse_token_ds(begin, end, '[')) {
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
            if (parse_token_ds(begin, end, ',')) {
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
        if (!parse_token_ds(begin, end, ']')) {
            throw datashape_parse_error(begin, "expected closing ']'");
        }
        rbegin = begin;
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
static ndt::type parse_char_parameters(const char *&rbegin, const char *end)
{
    const char *begin = rbegin;
    if (parse_token_ds(begin, end, '[')) {
        const char *saved_begin = begin;
        string encoding_str;
        if (!parse_quoted_string(begin, end, encoding_str)) {
            throw datashape_parse_error(saved_begin, "expected a string encoding");
        }
        string_encoding_t encoding;
        if (!encoding_str.empty()) {
            encoding = string_to_encoding(saved_begin, encoding_str);
        } else {
            throw datashape_parse_error(begin, "expected string encoding");
        }
        if (!parse_token_ds(begin, end, ']')) {
            throw datashape_parse_error(begin, "expected closing ']'");
        }
        rbegin = begin;
        return ndt::make_char(encoding);
    } else {
        return ndt::make_char();
    }
}

// complex_type : complex[float_type]
// This is called after 'complex' is already matched
static ndt::type parse_complex_parameters(const char *&rbegin, const char *end,
                map<string, ndt::type>& symtable)
{
    const char *begin = rbegin;
    if (parse_token_ds(begin, end, '[')) {
        const char *saved_begin = begin;
        ndt::type tp = parse_rhs_expression(begin, end, symtable);
        if (tp.get_type_id() == uninitialized_type_id) {
            throw datashape_parse_error(begin, "expected a type parameter");
        }
        if (!parse_token_ds(begin, end, ']')) {
            throw datashape_parse_error(begin, "expected closing ']'");
        }
        if (tp.get_type_id() == float32_type_id) {
            rbegin = begin;
            return ndt::make_type<dynd_complex<float> >();
        } else if (tp.get_type_id() == float64_type_id) {
            rbegin = begin;
            return ndt::make_type<dynd_complex<double> >();
        } else {
            throw datashape_parse_error(saved_begin, "unsupported real type for complex numbers");
        }
    } else {
        // Default to complex[double] if no parameters are provided
        return ndt::make_type<dynd_complex<double> >();
    }
}

// byteswap_type : byteswap[type]
// This is called after 'byteswap' is already matched
static ndt::type parse_byteswap_parameters(const char *&rbegin, const char *end,
                map<string, ndt::type>& symtable)
{
    const char *begin = rbegin;
    if (parse_token_ds(begin, end, '[')) {
        ndt::type tp = parse_rhs_expression(begin, end, symtable);
        if (tp.get_type_id() == uninitialized_type_id) {
            throw datashape_parse_error(begin, "expected a type parameter");
        }
        if (!parse_token_ds(begin, end, ']')) {
            throw datashape_parse_error(begin, "expected closing ']'");
        }
        rbegin = begin;
        return ndt::make_byteswap(tp);
    } else {
        throw datashape_parse_error(begin, "expected opening '['");
    }
}

// byte_type : bytes[<size>] | bytes[align=<alignment>] | bytes[<size>, align=<alignment>]
// This is called after 'bytes' is already matched
static ndt::type parse_bytes_parameters(const char *&rbegin, const char *end)
{
    const char *begin = rbegin;
    if (parse_token_ds(begin, end, '[')) {
        if (parse_token_ds(begin, end, "align")) {
            // bytes type with an alignment
            if (!parse_token_ds(begin, end, '=')) {
                throw datashape_parse_error(begin, "expected an =");
            }
            string align_val = parse_number(begin, end);
            if (align_val.empty()) {
                throw datashape_parse_error(begin, "expected an integer");
            }
            if (!parse_token_ds(begin, end, ']')) {
                throw datashape_parse_error(begin, "expected closing ']'");
            }
            rbegin = begin;
            return ndt::make_bytes(atoi(align_val.c_str()));
        }
        string size_val = parse_number(begin, end);
        if (size_val.empty()) {
            throw datashape_parse_error(begin, "expected 'align' or an integer");
        }
        if (parse_token_ds(begin, end, ']')) {
            // Fixed bytes with just a size parameter
            rbegin = begin;
            return ndt::make_fixedbytes(atoi(size_val.c_str()), 1);
        }
        if (!parse_token_ds(begin, end, ',')) {
            throw datashape_parse_error(begin, "expected closing ']' or another argument");
        }
        if (!parse_token_ds(begin, end, "align")) {
            throw datashape_parse_error(begin, "expected align= parameter");
        }
        if (!parse_token_ds(begin, end, '=')) {
            throw datashape_parse_error(begin, "expected an =");
        }
        string align_val = parse_number(begin, end);
        if (align_val.empty()) {
            throw datashape_parse_error(begin, "expected an integer");
        }
        if (!parse_token_ds(begin, end, ']')) {
            throw datashape_parse_error(begin, "expected closing ']'");
        }
        rbegin = begin;
        return ndt::make_fixedbytes(atoi(size_val.c_str()), atoi(align_val.c_str()));
    } else {
        return ndt::make_bytes(1);
    }
}

// cuda_host_type : cuda_host[storage_type]
// This is called after 'cuda_host' is already matched
static ndt::type parse_cuda_host_parameters(const char *&rbegin, const char *end,
                map<string, ndt::type>& symtable)
{
    const char *begin = rbegin;
    if (parse_token_ds(begin, end, '[')) {
#ifdef DYND_CUDA
        ndt::type tp = parse_rhs_expression(begin, end, symtable);
        if (tp.get_type_id() == uninitialized_type_id) {
            throw datashape_parse_error(begin, "expected a type parameter");
        }
        if (!parse_token_ds(begin, end, ']')) {
            throw datashape_parse_error(begin, "expected closing ']'");
        }
        rbegin = begin;
		return ndt::make_cuda_host(tp);
#else
        // Silence the unused parameter warning
        symtable.empty();
        throw datashape_parse_error(begin, "cuda_host type is not available");
#endif // DYND_CUDA
    } else {
        throw datashape_parse_error(begin, "expected opening '['");
    }
}

// cuda_device_type : cuda_device[storage_type]
// This is called after 'cuda_device' is already matched
static ndt::type parse_cuda_device_parameters(const char *&rbegin, const char *end,
                map<string, ndt::type>& symtable)
{
    const char *begin = rbegin;
    if (parse_token_ds(begin, end, '[')) {
#ifdef DYND_CUDA
        ndt::type tp = parse_rhs_expression(begin, end, symtable);
        if (tp.get_type_id() == uninitialized_type_id) {
            throw datashape_parse_error(begin, "expected a type parameter");
        }
        if (!parse_token(begin, end, ']')) {
            throw datashape_parse_error(begin, "expected closing ']'");
        }
        rbegin = begin;
		return ndt::make_cuda_device(tp);
#else
        // Silence the unused parameter warning
        symtable.empty();
        throw datashape_parse_error(begin, "cuda_device type is not available");
#endif // DYND_CUDA
    } else {
        throw datashape_parse_error(begin, "expected opening '['");
    }
}

// datetime_type : datetime[tz='timezone']
// This is called after 'datetime' is already matched
static ndt::type parse_datetime_parameters(const char *&rbegin, const char *end)
{
    const char *begin = rbegin;
    if (parse_token_ds(begin, end, '[')) {
        datetime_tz_t timezone = tz_abstract;
        string unit_str;
        const char *saved_begin = begin;
        // Parse the timezone
        if (!parse_token_ds(begin, end, "tz")) {
            throw datashape_parse_error(begin, "expected tz= parameter");
        }
        if (!parse_token_ds(begin, end, '=')) {
            throw datashape_parse_error(begin, "expected '='");
        }
        string timezone_str;
        saved_begin = begin;
        if (!parse_quoted_string(begin, end, timezone_str)) {
            throw datashape_parse_error(begin, "expected a time zone string");
        }
        if (timezone_str == "abstract") {
            timezone = tz_abstract;
        } else if (timezone_str == "UTC") {
            timezone = tz_utc;
        } else {
            throw datashape_parse_error(saved_begin, "invalid time zone");
        }
        if (!parse_token_ds(begin, end, ']')) {
            throw datashape_parse_error(begin, "expected closing ']'");
        }

        rbegin = begin;
        return ndt::make_datetime(timezone);
    } else {
        return ndt::make_datetime(tz_abstract);
    }
}

// time_type : time[tz='timezone']
// This is called after 'datetime' is already matched
static ndt::type parse_time_parameters(const char *&rbegin, const char *end)
{
    const char *begin = rbegin;
    if (parse_token_ds(begin, end, '[')) {
        datetime_tz_t timezone = tz_abstract;
        string unit_str;
        const char *saved_begin = begin;
        // Parse the timezone
        if (!parse_token_ds(begin, end, "tz")) {
            throw datashape_parse_error(begin, "expected tz= parameter");
        }
        if (!parse_token_ds(begin, end, '=')) {
            throw datashape_parse_error(begin, "expected '='");
        }
        string timezone_str;
        saved_begin = begin;
        if (!parse_quoted_string(begin, end, timezone_str)) {
            throw datashape_parse_error(begin, "expected a time zone string");
        }
        if (timezone_str == "abstract") {
            timezone = tz_abstract;
        } else if (timezone_str == "UTC") {
            timezone = tz_utc;
        } else {
            throw datashape_parse_error(saved_begin, "invalid time zone");
        }
        if (!parse_token_ds(begin, end, ']')) {
            throw datashape_parse_error(begin, "expected closing ']'");
        }

        rbegin = begin;
        return ndt::make_time(timezone);
    } else {
        return ndt::make_time(tz_abstract);
    }
}

static ndt::type parse_unaligned_parameters(const char *&rbegin, const char *end,
                map<string, ndt::type>& symtable)
{
    const char *begin = rbegin;
    if (!parse_token_ds(begin, end, '[')) {
        throw datashape_parse_error(begin, "expected opening '[' after 'unaligned'");
    }
    ndt::type tp = parse_rhs_expression(begin, end, symtable);
    if (tp.get_type_id() == uninitialized_type_id) {
        throw datashape_parse_error(begin, "expected a data type");
    }
    if (!parse_token_ds(begin, end, ']')) {
        throw datashape_parse_error(begin, "expected closing ']'");
    }
    // TODO catch errors, convert them to datashape_parse_error so the position is shown
    rbegin = begin;
    return ndt::make_unaligned(tp);
}

static ndt::type parse_pointer_parameters(const char *&rbegin, const char *end,
                                          map<string, ndt::type> &symtable)
{
    const char *begin = rbegin;
    if (!parse_token_ds(begin, end, '[')) {
        throw datashape_parse_error(begin, "expected opening '[' after 'pointer'");
    }
    ndt::type tp = parse_rhs_expression(begin, end, symtable);
    if (tp.get_type_id() == uninitialized_type_id) {
        throw datashape_parse_error(begin, "expected a data type");
    }
    if (!parse_token_ds(begin, end, ']')) {
        throw datashape_parse_error(begin, "expected closing ']'");
    }
    // TODO catch errors, convert them to datashape_parse_error so the position is shown
    rbegin = begin;
    return ndt::make_pointer(tp);
}

// record_item : NAME COLON rhs_expression
static bool parse_struct_item(const char *&rbegin, const char *end,
                              map<string, ndt::type> &symtable,
                              string &out_field_name, ndt::type &out_field_type)
{
    const char *begin = rbegin;
    const char *field_name_begin, *field_name_end;
    parse::skip_whitespace_and_pound_comments(begin, end);
    if (!parse::parse_name_no_ws(begin, end, field_name_begin, field_name_end)) {
        return false;
    }
    if (!parse_token_ds(begin, end , ':')) {
        throw datashape_parse_error(begin, "expected ':' after record item name");
    }
    bool parens = false;
    if (parse_token_ds(begin, end, '(')) {
        parens = true;
    }
    out_field_type = parse_rhs_expression(begin, end, symtable);
    if (out_field_type.get_type_id() == uninitialized_type_id) {
        throw datashape_parse_error(begin, "expected a data type");
    }
    if (parens && !parse_token_ds(begin, end, ')')) {
        throw datashape_parse_error(begin, "expected closing ')'");
    }

    out_field_name.assign(field_name_begin, field_name_end);
    rbegin = begin;
    return true;
}

// struct : LBRACE record_item record_item* RBRACE
// cstruct : 'c{' record_item record_item* RBRACE
static ndt::type parse_struct(const char *&rbegin, const char *end,
                              map<string, ndt::type> &symtable)
{
    const char *begin = rbegin;
    vector<string> field_name_list;
    vector<ndt::type> field_type_list;
    string field_name;
    ndt::type field_type;
    bool cprefixed = false;

    if (!parse_token_ds(begin, end, '{')) {
        if (parse_token_ds(begin, end, "c{")) {
            cprefixed = true;
        } else {
            return ndt::type(uninitialized_type_id);
        }
    }
    for (;;) {
        const char *saved_begin = begin;
        parse::skip_whitespace_and_pound_comments(begin, end);
        if (parse_struct_item(begin, end, symtable, field_name, field_type)) {
            field_name_list.push_back(field_name);
            field_type_list.push_back(field_type);
        } else {
            throw datashape_parse_error(saved_begin, "expected a record item");
        }
        
        if (parse_token_ds(begin, end, ',')) {
            if (!field_name_list.empty() && parse_token_ds(begin, end, '}')) {
                break;
            }
        } else if (parse_token_ds(begin, end, '}')) {
            break;
        } else {
            throw datashape_parse_error(begin, "expected ',' or '}'");
        }
    }

    rbegin = begin;
    if (cprefixed) {
        return ndt::make_cstruct(field_type_list.size(), &field_type_list[0],
                                 &field_name_list[0]);
    } else {
        return ndt::make_struct(field_type_list.size(), &field_type_list[0],
                                 &field_name_list[0]);
    }
}

// tuple : LPAREN tuple_item tuple_item* RPAREN
// ctuple : 'c(' tuple_item tuple_item* RPAREN
// funcproto : tuple -> type
static ndt::type parse_tuple_or_funcproto(const char *&rbegin, const char *end, map<string, ndt::type>& symtable)
{
    const char *begin = rbegin;
    vector<ndt::type> field_type_list;
    bool cprefixed = false;

    if (!parse_token_ds(begin, end, '(')) {
        if (parse_token_ds(begin, end, "c(")) {
            cprefixed = true;
        } else {
            return ndt::type(uninitialized_type_id);
        }
    }
    for (;;) {
        ndt::type tp = parse_rhs_expression(begin, end, symtable);
        if (tp.get_type_id() != uninitialized_type_id) {
            field_type_list.push_back(tp);
        } else {
            throw datashape_parse_error(begin, "expected a type");
        }
        
        if (parse_token_ds(begin, end, ',')) {
            if (!field_type_list.empty() &&
                    parse_token_ds(begin, end, ')')) {
                break;
            }
        } else if (parse_token_ds(begin, end, ')')) {
            break;
        } else {
            throw datashape_parse_error(begin, "expected ',' or ')'");
        }
    }

    if (cprefixed) {
        rbegin = begin;
        return ndt::make_ctuple(field_type_list.size(), &field_type_list[0]);
    } else {
        // It might be a function prototype, check for the "->" token
        if (!parse_token_ds(begin, end, "->")) {
            rbegin = begin;
            return ndt::make_tuple(field_type_list.size(), &field_type_list[0]);
        }

        ndt::type return_type = parse_rhs_expression(begin, end, symtable);
        if (return_type.get_type_id() == uninitialized_type_id) {
            throw datashape_parse_error(begin, "expected function prototype return type");
        }
        rbegin = begin;
        return ndt::make_funcproto(field_type_list.size(), &field_type_list[0], return_type);
    }
}

/** This is what parses the main datashape grammar, excluding type aliases, etc. */
static ndt::type parse_rhs_expression(const char *&rbegin, const char *end, map<string, ndt::type>& symtable)
{
    const char *begin = rbegin;
    ndt::type result;
    vector<intptr_t> shape;
    // rhs_expression : ((NAME | NUMBER) ASTERISK)* (record | NAME LPAREN rhs_expression RPAREN | NAME)
    for (;;) {
        const char *saved_begin = begin;
        string n = parse_name_or_number(begin, end); // NAME | NUMBER
        if (n.empty()) {
            break;
        } else if (!parse_token_ds(begin, end, '*')) { // ASTERISK
            begin = saved_begin;
            break;
        } else {
            if ('0' <= n[0] && n[0] <= '9') {
                shape.push_back(atoi(n.c_str()));
            } else if (n == "var") {
                // Use -1 to signal a variable-length dimension
                shape.push_back(-1);
            } else if (n == "strided") {
                // Use -2 to signal a strided dimension
                shape.push_back(-2);
            } else if (isupper(n[0])) {
                ndt::type element_type = parse_rhs_expression(begin, end, symtable);
                if (element_type.get_type_id() == uninitialized_type_id) {
                    throw datashape_parse_error(begin, "expected a dynd type");
                }
                rbegin = begin;
                return ndt::make_typevar_dim(n, element_type);
            } else {
                parse::skip_whitespace_and_pound_comments(saved_begin, end);
                throw datashape_parse_error(saved_begin,
                                "unrecognized dimension type");
            }
        }
    }
    // struct
    result = parse_struct(begin, end, symtable);
    // tuple or funcproto
    if (result.get_type_id() == uninitialized_type_id) {
        result = parse_tuple_or_funcproto(begin, end, symtable);
    }
    if (result.get_type_id() == uninitialized_type_id) {
        const char *begin_saved = begin;
        // NAME
        const char *n_begin, *n_end;
        parse::skip_whitespace_and_pound_comments(begin, end);
        if (!parse::parse_name_no_ws(begin, end, n_begin, n_end)) {
            if (shape.empty()) {
                return ndt::type(uninitialized_type_id);
            } else {
                throw datashape_parse_error(begin, "expected data type");
            }
        } else if (parse::compare_range_to_literal(n_begin, n_end, "string")) {
            result = parse_string_parameters(begin, end);
        } else if (parse::compare_range_to_literal(n_begin, n_end, "complex")) {
            result = parse_complex_parameters(begin, end, symtable);
        } else if (parse::compare_range_to_literal(n_begin, n_end, "datetime")) {
            result = parse_datetime_parameters(begin, end);
        } else if (parse::compare_range_to_literal(n_begin, n_end, "time")) {
            result = parse_time_parameters(begin, end);
        } else if (parse::compare_range_to_literal(n_begin, n_end, "unaligned")) {
            result = parse_unaligned_parameters(begin, end, symtable);
        } else if (parse::compare_range_to_literal(n_begin, n_end, "pointer")) {
            result = parse_pointer_parameters(begin, end, symtable);
        } else if (parse::compare_range_to_literal(n_begin, n_end, "char")) {
            result = parse_char_parameters(begin, end);
        } else if (parse::compare_range_to_literal(n_begin, n_end, "byteswap")) {
            result = parse_byteswap_parameters(begin, end, symtable);
        } else if (parse::compare_range_to_literal(n_begin, n_end, "bytes")) {
            result = parse_bytes_parameters(begin, end);
        } else if (parse::compare_range_to_literal(n_begin, n_end, "cuda_host")) {
            result = parse_cuda_host_parameters(begin, end, symtable);
        } else if (parse::compare_range_to_literal(n_begin, n_end, "cuda_device")) {
            result = parse_cuda_device_parameters(begin, end, symtable);
        } else if (parse::compare_range_to_literal(n_begin, n_end, "cfixed")) {
            result = parse_cfixed_dim_parameters(begin, end, symtable);
        } else if (parse::compare_range_to_literal(n_begin, n_end, "fixed")) {
            result = parse_fixed_dim_parameters(begin, end, symtable);
        } else if (isupper(*n_begin)) {
            result = ndt::make_typevar(n_begin, n_end);
        } else {
            string n(n_begin, n_end);
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
                    if (parse_token_ds(begin_tmp, end, '(')) {
                        throw datashape_parse_error(
                            begin, "DyND does not support this kind of "
                                   "datashape parsing yet");
                    } else {
                        parse::skip_whitespace_and_pound_comments(begin_saved,
                                                                  end);
                        throw datashape_parse_error(begin_saved,
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
    rbegin = begin;
    return result;
}

static ndt::type parse_stmt(const char *&rbegin, const char *end, map<string, ndt::type>& symtable)
{
    const char *begin = rbegin;
    // stmt : TYPE name EQUALS rhs_expression
    // NOTE that this doesn't support parameterized lhs_expression, this is subset of Blaze datashape
    if (parse_token_ds(begin, end, "type")) {
        const map<string, ndt::type>& builtin_types = get_builtin_types();
        const char *saved_begin = begin;
        const char *tname_begin, *tname_end;
        if (!parse::skip_required_whitespace(begin, end)) {
            if (begin == end) {
                // If it's only "type" by itself, return the "type" type
                rbegin = begin;
                return builtin_types.find("type")->second;
            } else {
                return ndt::type();
            }
        }
        if (!parse::parse_name_no_ws(begin, end, tname_begin, tname_end)) {
            parse::skip_whitespace_and_pound_comments(begin, end);
            if (begin == end) {
                // If it's only "type" by itself, return the "type" type
                rbegin = begin;
                return builtin_types.find("type")->second;
            } else {
                throw datashape_parse_error(
                    saved_begin, "expected an identifier for a type name");
            }
        }
        if (!parse_token_ds(begin, end, '=')) {
            throw datashape_parse_error(begin, "expected an '='");
        }
        ndt::type result = parse_rhs_expression(begin, end, symtable);
        if (result.get_type_id() == uninitialized_type_id) {
            throw datashape_parse_error(begin, "expected a data type");
        }
        string tname(tname_begin, tname_end);
        // ACTION: Put the parsed type in the symbol table
        if (builtin_types.find(tname) != builtin_types.end()) {
            parse::skip_whitespace_and_pound_comments(saved_begin, end);
            throw datashape_parse_error(saved_begin,
                                        "cannot redefine a builtin type");
        }
        if (symtable.find(tname) != symtable.end()) {
            parse::skip_whitespace_and_pound_comments(saved_begin, end);
            throw datashape_parse_error(saved_begin,
                            "type name already defined in datashape string");
        }
        symtable[tname] = result;
        rbegin = begin;
        return result;
    } else {
        // stmt : rhs_expression
        return parse_rhs_expression(rbegin, end, symtable);
    }
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
            parse::skip_whitespace_and_pound_comments(begin, end);
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

