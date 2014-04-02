//
// Copyright (C) 2011-14 Mark Wiebe, DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <string>

#include <dynd/parser_util.hpp>

using namespace std;
using namespace dynd;
 
// [a-zA-Z]+
bool parse::parse_alpha_name_no_ws(const char *&begin, const char *end,
                                   const char *&out_strbegin,
                                   const char *&out_strend)
{
    const char *pos = begin;
    if (pos == end) {
        return false;
    }
    if (('a' <= *pos && *pos <= 'z') || ('A' <= *pos && *pos <= 'Z')) {
        ++pos;
    } else {
        return false;
    }
    while (pos < end &&
           (('a' <= *pos && *pos <= 'z') || ('A' <= *pos && *pos <= 'Z'))) {
        ++pos;
    }
    out_strbegin = begin;
    out_strend = pos;
    begin = pos;
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

