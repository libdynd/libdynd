//
// Copyright (C) 2011-14 Mark Wiebe, DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <string>

#include <dynd/parser_util.hpp>
#include <dynd/string_encodings.hpp>

using namespace std;
using namespace dynd;

// [a-zA-Z_][a-zA-Z0-9_]*
bool parse::parse_name_no_ws(const char *&rbegin, const char *end,
                             const char *&out_strbegin, const char *&out_strend)
{
    const char *begin = rbegin;
    if (begin == end) {
        return false;
    }
    if (('a' <= *begin && *begin <= 'z') || ('A' <= *begin && *begin <= 'Z') ||
            *begin == '_') {
        ++begin;
    } else {
        return false;
    }
    while (begin < end && (('a' <= *begin && *begin <= 'z') ||
                           ('A' <= *begin && *begin <= 'Z') ||
                           ('0' <= *begin && *begin <= '9') || *begin == '_')) {
        ++begin;
    }
    out_strbegin = rbegin;
    out_strend = begin;
    rbegin = begin;
    return true;
}
 
// [a-zA-Z]+
bool parse::parse_alpha_name_no_ws(const char *&rbegin, const char *end,
                                   const char *&out_strbegin,
                                   const char *&out_strend)
{
    const char *begin = rbegin;
    if (begin == end) {
        return false;
    }
    if (('a' <= *begin && *begin <= 'z') || ('A' <= *begin && *begin <= 'Z')) {
        ++begin;
    } else {
        return false;
    }
    while (begin < end && (('a' <= *begin && *begin <= 'z') ||
                           ('A' <= *begin && *begin <= 'Z'))) {
        ++begin;
    }
    out_strbegin = rbegin;
    out_strend = begin;
    rbegin = begin;
    return true;
}

bool parse::parse_doublequote_string_no_ws(const char *&rbegin, const char *end,
                                           const char *&out_strbegin,
                                           const char *&out_strend,
                                           bool &out_escaped)
{
    bool escaped = false;
    const char *begin = rbegin;
    if (!parse_token_no_ws(begin, end, '\"')) {
        return false;
    }
    for (;;) {
        if (begin == end) {
            throw parse::parse_error(rbegin, "string has no ending quote");
        }
        char c = *begin++;
        if (c == '\\') {
            escaped = true;
            if (begin == end) {
                throw parse::parse_error(rbegin, "string has no ending quote");
            }
            c = *begin++;
            switch (c) {
                case '"':
                case '\\':
                case '/':
                case 'b':
                case 'f':
                case 'n':
                case 'r':
                case 't':
                    break;
                case 'u': {
                    if (end - begin < 4) {
                        throw parse::parse_error(begin-2, "invalid unicode escape sequence in string");
                    }
                    for (int i = 0; i < 4; ++i) {
                        char c = *begin++;
                        if (!(('0' <= c && c <= '9') ||
                              ('A' <= c && c <= 'F') ||
                              ('a' <= c && c <= 'f'))) {
                            throw parse::parse_error(
                                begin - 1,
                                "invalid unicode escape sequence in string");
                        }
                    }
                    break;
                }
                default:
                    throw parse::parse_error(begin-2, "invalid escape sequence in string");
            }
        } else if (c == '"') {
            out_strbegin = rbegin + 1;
            out_strend = begin - 1;
            out_escaped = escaped;
            rbegin = begin;
            return true;
        }
    }
}

void parse::unescape_string(const char *strbegin, const char *strend,
                     std::string &out)
{
    out.resize(0);
    while (strbegin < strend) {
        char c = *strbegin++;
        if (c == '\\') {
            if (strbegin == strend) {
                return;
            }
            c = *strbegin++;
            switch (c) {
                case '"':
                case '\\':
                case '/':
                    out += c;
                    break;
                case 'b':
                    out += '\b';
                    break;
                case 'f':
                    out += '\f';
                    break;
                case 'n':
                    out += '\n';
                    break;
                case 'r':
                    out += '\r';
                    break;
                case 't':
                    out += '\t';
                    break;
                case 'u': {
                    if (strend - strbegin < 4) {
                        return;
                    }
                    uint32_t cp = 0;
                    for (int i = 0; i < 4; ++i) {
                        char c = *strbegin++;
                        cp *= 16;
                        if ('0' <= c && c <= '9') {
                            cp += c - '0';
                        } else if ('A' <= c && c <= 'F') {
                            cp += c - 'A' + 10;
                        } else if ('a' <= c && c <= 'f') {
                            cp += c - 'a' + 10;
                        } else {
                            cp = '?';
                        }
                    }
                    append_utf8_codepoint(cp, out);
                    break;
                }
                default:
                    out += '?';
            }
        } else {
            out += c;
        }
    }
}

bool parse::parse_json_number_no_ws(const char *&rbegin, const char *end,
                                    const char *&out_nbegin,
                                    const char *&out_nend)
{
    const char *begin = rbegin;
    if (begin == end) {
        return false;
    }
    // Optional minus sign
    if (*begin == '-') {
        ++begin;
    }
    if (begin == end) {
        return false;
    }
    // Either '0' or a non-zero digit followed by digits
    if (*begin == '0') {
        ++begin;
    } else if ('1' <= *begin && *begin <= '9') {
        ++begin;
        while (begin < end && ('0' <= *begin && *begin <= '9')) {
            ++begin;
        }
    } else {
        return false;
    }
    // Optional decimal point, followed by one or more digits
    if (begin < end && *begin == '.') {
        if (++begin == end) {
            return false;
        }
        if (!('0' <= *begin && *begin <= '9')) {
            return false;
        }
        ++begin;
        while (begin < end && ('0' <= *begin && *begin <= '9')) {
            ++begin;
        }
    }
    // Optional exponent, followed by +/- and some digits
    if (begin < end && (*begin == 'e' || *begin == 'E')) {
        if (++begin == end) {
            return false;
        }
        // +/- is optional
        if (*begin == '+' || *begin == '-') {
            if (++begin == end) {
                return false;
            }
        }
        // At least one digit is required
        if (!('0' <= *begin && *begin <= '9')) {
            return false;
        }
        ++begin;
        while (begin < end && ('0' <= *begin && *begin <= '9')) {
            ++begin;
        }
    }
    out_nbegin = rbegin;
    out_nend = begin;
    rbegin = begin;
    return true;
}

// [0-9][0-9]
bool parse::parse_2digit_int_no_ws(const char *&begin, const char *end,
                                   int &out_val)
{
    if (end - begin >= 2) {
        int d0 = begin[0], d1 = begin[1];
        if (d0 >= '0' && d0 <= '9' && d1 >= '0' && d1 <= '9') {
            begin += 2;
            out_val = (d0 - '0') * 10 + (d1 - '0');
            return true;
        }
    }
    return false;
}

// [0-9][0-9]?
bool parse::parse_1or2digit_int_no_ws(const char *&begin, const char *end,
                                      int &out_val)
{
    if (end - begin >= 2) {
        int d0 = begin[0], d1 = begin[1];
        if (d0 >= '0' && d0 <= '9') {
            if (d1 >= '0' && d1 <= '9') {
                begin += 2;
                out_val = (d0 - '0') * 10 + (d1 - '0');
                return true;
            } else {
                ++begin;
                out_val = (d0 - '0');
                return true;
            }
        }
    } else if (end - begin == 1) {
        int d0 = begin[0];
        if (d0 >= '0' && d0 <= '9') {
            ++begin;
            out_val = (d0 - '0');
            return true;
        }
    }
    return false;
}

// [0-9][0-9][0-9][0-9]
bool parse::parse_4digit_int_no_ws(const char *&begin, const char *end,
                                   int &out_val)
{
    if (end - begin >= 4) {
        int d0 = begin[0], d1 = begin[1], d2 = begin[2], d3 = begin[3];
        if (d0 >= '0' && d0 <= '9' && d1 >= '0' && d1 <= '9' && d2 >= '0' &&
                d2 <= '9' && d3 >= '0' && d3 <= '9') {
            begin += 4;
            out_val = (((d0 - '0') * 10 + (d1 - '0')) * 10 + (d2 - '0')) * 10 +
                   (d3 - '0');
            return true;
        }
    }
    return false;
}

// [0-9][0-9][0-9][0-9][0-9][0-9]
bool parse::parse_6digit_int_no_ws(const char *&begin, const char *end,
                                   int &out_val)
{
    if (end - begin >= 6) {
        int d0 = begin[0], d1 = begin[1], d2 = begin[2], d3 = begin[3],
            d4 = begin[4], d5 = begin[5];
        if (d0 >= '0' && d0 <= '9' && d1 >= '0' && d1 <= '9' && d2 >= '0' &&
                d2 <= '9' && d3 >= '0' && d3 <= '9' && d4 >= '0' &&
                d4 <= '9' && d5 >= '0' && d5 <= '9') {
            begin += 6;
            out_val = (((((d0 - '0') * 10 + (d1 - '0')) * 10 + (d2 - '0')) * 10 +
                     (d3 - '0')) * 10 +
                    (d4 - '0')) * 10 +
                   (d5 - '0');
            return true;
        }
    }
    return false;
}

template <class T>
inline static T checked_string_to_uint(const char *begin, const char *end,
                                       bool &out_overflow, bool &out_badparse)
{
    T result = 0, prev_result = 0;
    if (begin == end) {
        out_badparse = true;
    }
    while (begin < end) {
        char c = *begin;
        if ('0' <= c && c <= '9') {
            result = (result * 10u) + (uint32_t)(c - '0');
            if (result < prev_result) {
                out_overflow = true;
            }
        } else {
            if (c == '.') {
                // Accept ".", ".0" with trailing decimal zeros as well
                ++begin;
                while (begin < end && *begin == '0') {
                    ++begin;
                }
                if (begin == end) {
                    break;
                }
            } else if (c == 'e' || c == 'E') {
                // Accept "1e5", "1e+5" integers with a positive exponent,
                // a subset of floating point syntax. Note that "1.2e1"
                // is not accepted as the value 12 by this code.
                ++begin;
                if (begin < end && *begin == '+') {
                    ++begin;
                }
                if (begin < end) {
                    int exponent = 0;
                    // Accept any number of zeros followed by at most
                    // two digits. Anything greater would overflow.
                    while (begin < end && *begin == '0') {
                        ++begin;
                    }
                    if (begin < end && '0' <= *begin && *begin <= '9') {
                        exponent = *begin++ - '0';
                    }
                    if (begin < end && '0' <= *begin && *begin <= '9') {
                        exponent = (10 * exponent) + (*begin++ - '0');
                    }
                    if (begin == end) {
                        prev_result = result;
                        // Apply the exponent in a naive way, but with
                        // overflow checking
                        for (int i = 0; i < exponent; ++i) {
                            result = result * 10u;
                            if (result < prev_result) {
                                out_overflow = true;
                            }
                            prev_result = result;
                        }
                        return result;
                    }
                }
            }
            out_badparse = true;
            break;
        }
        ++begin;
        prev_result = result;
    }
    return result;
}

template <class T>
inline static T unchecked_string_to_uint(const char *begin, const char *end)
{
    T result = 0;
    while (begin < end) {
        char c = *begin;
        if ('0' <= c && c <= '9') {
            result = (result * 10u) + (uint32_t)(c - '0');
        } else if (c == 'e' || c == 'E') {
            // Accept "1e5", "1e+5" integers with a positive exponent,
            // a subset of floating point syntax. Note that "1.2e1"
            // is not accepted as the value 12 by this code.
            ++begin;
            if (begin < end && *begin == '+') {
                ++begin;
            }
            if (begin < end) {
                int exponent = 0;
                // Accept any number of zeros followed by at most
                // two digits. Anything greater would overflow.
                while (begin < end && *begin == '0') {
                    ++begin;
                }
                if (begin < end && '0' <= *begin && *begin <= '9') {
                    exponent = *begin++ - '0';
                }
                if (begin < end && '0' <= *begin && *begin <= '9') {
                    exponent = (10 * exponent) + (*begin++ - '0');
                }
                if (begin == end) {
                    // Apply the exponent in a naive way
                    for (int i = 0; i < exponent; ++i) {
                        result = result * 10u;
                    }
                }
            }
            break;
        } else {
            break;
        }
        ++begin;
    }
    return result;
}

uint64_t parse::checked_string_to_uint64(const char *begin, const char *end,
                                         bool &out_overflow, bool &out_badparse)
{
    return checked_string_to_uint<uint64_t>(begin, end, out_overflow, out_badparse);
}

dynd_uint128 parse::checked_string_to_uint128(const char *begin, const char *end,
                                         bool &out_overflow, bool &out_badparse)
{
    return checked_string_to_uint<dynd_uint128>(begin, end, out_overflow, out_badparse);
}

uint64_t parse::unchecked_string_to_uint64(const char *begin, const char *end)
{
    return unchecked_string_to_uint<uint64_t>(begin, end);
}

dynd_uint128 parse::unchecked_string_to_uint128(const char *begin, const char *end)
{
    return unchecked_string_to_uint<dynd_uint128>(begin, end);
}

namespace {
template <class T> struct overflow_check;
template <> struct overflow_check<int8_t> { inline static bool is_overflow(uint64_t value, bool negative) {
    return (value&~0x7fULL) != 0 && !(negative && value == 0x80ULL);
}};
template <> struct overflow_check<int16_t> { inline static bool is_overflow(uint64_t value, bool negative) {
    return (value&~0x7fffULL) != 0 && !(negative && value == 0x8000ULL);
}};
template <> struct overflow_check<int32_t> { inline static bool is_overflow(uint64_t value, bool negative) {
    return (value&~0x7fffffffULL) != 0 && !(negative && value == 0x80000000ULL);
}};
template <> struct overflow_check<int64_t> { inline static bool is_overflow(uint64_t value, bool negative) {
    return (value&~0x7fffffffffffffffULL) != 0 && !(negative && value == 0x8000000000000000ULL);
}};
template <> struct overflow_check<dynd_int128> { inline static bool is_overflow(const dynd_uint128 &value, bool negative) {
        return (value.m_hi & ~0x7fffffffffffffffULL) != 0 &&
               !(negative && value.m_hi == 0x8000000000000000ULL &&
                 value.m_lo == 0ULL);
}};
template <> struct overflow_check<uint8_t> { inline static bool is_overflow(uint64_t value) {
    return (value&~0xffULL) != 0;
}};
template <> struct overflow_check<uint16_t> { inline static bool is_overflow(uint64_t value) {
    return (value&~0xffffULL) != 0;
}};
template <> struct overflow_check<uint32_t> { inline static bool is_overflow(uint64_t value) {
    return (value&~0xffffffffULL) != 0;
}};
} // anonymous namespace

template <class T>
static inline void assign_signed_int_value(char *out_int, uint64_t uvalue,
                                    bool &negative, bool &overflow,
                                    bool &badparse)
{
    overflow = overflow || overflow_check<T>::is_overflow(uvalue, negative);
    if (!overflow && !badparse) {
        *reinterpret_cast<T *>(out_int) =
            static_cast<T>(negative ? -static_cast<int64_t>(uvalue)
                                    : static_cast<int64_t>(uvalue));
    }
}

static inline void assign_signed_int128_value(char *out_int, dynd_uint128 uvalue,
                                    bool &negative, bool &overflow,
                                    bool &badparse)
{
    overflow = overflow || overflow_check<dynd_int128>::is_overflow(uvalue, negative);
    if (!overflow && !badparse) {
        *reinterpret_cast<dynd_int128 *>(out_int) =
            negative ? -static_cast<dynd_int128>(uvalue)
                     : static_cast<dynd_int128>(uvalue);
    }
}

template <class T>
static inline void assign_unsigned_int_value(char *out_int, uint64_t uvalue,
                                    bool &negative, bool &overflow,
                                    bool &badparse)
{
    overflow = overflow || negative ||
               overflow_check<T>::is_overflow(uvalue);
    if (!overflow && !badparse) {
        *reinterpret_cast<T *>(out_int) = static_cast<T>(uvalue);
    }
}

void parse::string_to_int(char *out_int, type_id_t tid, const char *begin,
                          const char *end, assign_error_mode errmode)
{
    uint64_t uvalue;
    bool negative = false, overflow = false, badparse = false;
    if (begin < end && *begin == '-') {
        negative = true;
        ++begin;
    }
    if (errmode != assign_error_none) {
        switch (tid) {
            case int8_type_id:
                uvalue = parse::checked_string_to_uint64(begin, end, overflow,
                                                         badparse);
                assign_signed_int_value<int8_t>(out_int, uvalue, negative,
                                                overflow, badparse);
                break;
            case int16_type_id:
                uvalue = parse::checked_string_to_uint64(begin, end, overflow,
                                                         badparse);
                assign_signed_int_value<int16_t>(out_int, uvalue, negative,
                                                overflow, badparse);
                break;
            case int32_type_id:
                uvalue = parse::checked_string_to_uint64(begin, end, overflow,
                                                         badparse);
                assign_signed_int_value<int32_t>(out_int, uvalue, negative,
                                                overflow, badparse);
                break;
            case int64_type_id:
                uvalue = parse::checked_string_to_uint64(begin, end, overflow,
                                                         badparse);
                assign_signed_int_value<int64_t>(out_int, uvalue, negative,
                                                overflow, badparse);
                break;
            case int128_type_id: {
                dynd_uint128 buvalue = parse::checked_string_to_uint128(
                    begin, end, overflow, badparse);
                assign_signed_int128_value(out_int, buvalue, negative, overflow,
                                           badparse);
                break;
            }
            case uint8_type_id:
                uvalue = parse::checked_string_to_uint64(begin, end, overflow,
                                                         badparse);
                negative = negative && (uvalue != 0);
                assign_unsigned_int_value<uint8_t>(out_int, uvalue, negative,
                                                   overflow, badparse);
                break;
            case uint16_type_id:
                uvalue = parse::checked_string_to_uint64(begin, end, overflow,
                                                         badparse);
                negative = negative && (uvalue != 0);
                assign_unsigned_int_value<uint16_t>(out_int, uvalue, negative,
                                                   overflow, badparse);
                break;
            case uint32_type_id:
                uvalue = parse::checked_string_to_uint64(begin, end, overflow,
                                                         badparse);
                negative = negative && (uvalue != 0);
                assign_unsigned_int_value<uint32_t>(out_int, uvalue, negative,
                                                   overflow, badparse);
                break;
            case uint64_type_id:
                uvalue = parse::checked_string_to_uint64(begin, end, overflow,
                                                         badparse);
                negative = negative && (uvalue != 0);
                overflow = overflow || negative;
                if (!overflow && !badparse) {
                    *reinterpret_cast<uint64_t *>(out_int) = uvalue;
                }
                break;
            case uint128_type_id: {
                dynd_uint128 buvalue = parse::checked_string_to_uint128(
                    begin, end, overflow, badparse);
                negative = negative && (buvalue != 0);
                overflow = overflow || negative;
                if (!overflow && !badparse) {
                    *reinterpret_cast<dynd_uint128 *>(out_int) = buvalue;
                }
                break;
            }
            default: {
                stringstream ss;
                ss << "cannot parse integer as type id " << tid;
                throw runtime_error(ss.str());
            }
        }
        if (overflow) {
            stringstream ss;
            ss << "overflow converting string ";
            ss.write(begin, end-begin);
            ss << " to " << tid;
            throw overflow_error(ss.str());
        } else if (badparse) {
            stringstream ss;
            ss << "parse error converting string ";
            ss.write(begin, end-begin);
            ss << " to " << tid;
            throw invalid_argument(ss.str());
        }
    } else {
        // errmode == assign_error_none
        switch (tid) {
            case int8_type_id:
                uvalue = parse::unchecked_string_to_uint64(begin, end);
                *reinterpret_cast<int8_t *>(out_int) = static_cast<int8_t>(
                    negative ? -static_cast<int64_t>(uvalue)
                             : static_cast<int64_t>(uvalue));
                break;
            case int16_type_id:
                uvalue = parse::unchecked_string_to_uint64(begin, end);
                *reinterpret_cast<int16_t *>(out_int) = static_cast<int16_t>(
                    negative ? -static_cast<int64_t>(uvalue)
                             : static_cast<int64_t>(uvalue));
                break;
            case int32_type_id:
                uvalue = parse::unchecked_string_to_uint64(begin, end);
                *reinterpret_cast<int32_t *>(out_int) = static_cast<int32_t>(
                    negative ? -static_cast<int64_t>(uvalue)
                             : static_cast<int64_t>(uvalue));
                break;
            case int64_type_id:
                uvalue = parse::unchecked_string_to_uint64(begin, end);
                *reinterpret_cast<int64_t *>(out_int) =
                    negative ? -static_cast<int64_t>(uvalue)
                             : static_cast<int64_t>(uvalue);
                break;
            case int128_type_id: {
                dynd_uint128 buvalue =
                    parse::unchecked_string_to_uint128(begin, end);
                *reinterpret_cast<dynd_int128 *>(out_int) =
                    negative ? -static_cast<dynd_int128>(buvalue)
                             : static_cast<dynd_int128>(buvalue);
                break;
            }
            case uint8_type_id:
                uvalue = parse::unchecked_string_to_uint64(begin, end);
                *reinterpret_cast<uint8_t *>(out_int) =
                    static_cast<uint8_t>(negative ? 0 : uvalue);
                break;
            case uint16_type_id:
                uvalue = parse::unchecked_string_to_uint64(begin, end);
                *reinterpret_cast<uint16_t *>(out_int) =
                    static_cast<uint16_t>(negative ? 0 : uvalue);
                break;
            case uint32_type_id:
                uvalue = parse::unchecked_string_to_uint64(begin, end);
                *reinterpret_cast<uint32_t *>(out_int) =
                    static_cast<uint32_t>(negative ? 0 : uvalue);
                break;
            case uint64_type_id:
                uvalue = parse::unchecked_string_to_uint64(begin, end);
                *reinterpret_cast<uint64_t *>(out_int) = negative ? 0 : uvalue;
                break;
            case uint128_type_id: {
                dynd_uint128 buvalue =
                    parse::unchecked_string_to_uint128(begin, end);
                *reinterpret_cast<dynd_uint128 *>(out_int) =
                    negative ? static_cast<dynd_uint128>(0) : buvalue;
                break;
            }
            default: {
                stringstream ss;
                ss << "cannot parse integer as type id " << tid;
                throw runtime_error(ss.str());
            }
        }
    }
}
