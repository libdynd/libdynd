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

