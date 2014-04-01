//
// Copyright (C) 2011-14 Mark Wiebe, DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <string>

#include <dynd/types/date_parser.hpp>
#include <dynd/types/date_util.hpp>

using namespace std;
using namespace dynd;

static const char *skip_whitespace(const char *begin, const char *end)
{
    while (begin < end && isspace(*begin)) {
        ++begin;
    }

    return begin;
}

static bool skip_required_whitespace(const char *&begin, const char *end)
{
    if (begin < end && isspace(*begin)) {
        ++begin;
        while (begin < end && isspace(*begin)) {
            ++begin;
        }
        return true;
    } else {
        return false;
    }
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

template <int N>
static bool parse_token_no_ws(const char *&begin, const char *end, const char (&token)[N])
{
    if (N-1 <= end - begin && memcmp(begin, token, N-1) == 0) {
        begin += + N-1;
        return true;
    } else {
        return false;
    }
}

static bool parse_token_no_ws(const char *&begin, const char *end, char token)
{
    if (1 <= end - begin && *begin == token) {
        ++begin;
        return true;
    } else {
        return false;
    }
}

// NAME : [a-zA-Z]+
static bool parse_alpha_name_no_ws(const char *&begin, const char *end,
                                   const char *&out_strbegin,
                                   const char *&out_strend)
{
    const char *pos = begin;
    if (pos == end) {
        return false;
    }
    if (('a' <= *pos && *pos <= 'z') ||
                    ('A' <= *pos && *pos <= 'Z')) {
        ++pos;
    } else {
        return false;
    }
    while (pos < end && (('a' <= *pos && *pos <= 'z') ||
                    ('A' <= *pos && *pos <= 'Z'))) {
        ++pos;
    }
    out_strbegin = begin;
    out_strend = pos;
    begin = pos;
    return true;
}

// [0-9][0-9]
static bool parse_2digit_int(const char *&begin, const char *end, int &out_val)
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
static bool parse_1or2digit_int(const char *&begin, const char *end,
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
static bool parse_4digit_int(const char *&begin, const char *end, int &out_val)
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
static bool parse_6digit_int(const char *&begin, const char *end, int &out_val)
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

namespace {
    struct named_month {
        const char *name;
        int month;
        DYND_CONSTEXPR named_month(const char *name_, int month_)
            : name(name_), month(month_) {}
    };
    static named_month named_month_table[] = {
        named_month("jan", 1), named_month("january", 1),
        named_month("feb", 2), named_month("february", 2),
        named_month("mar", 3), named_month("march", 3),
        named_month("apr", 4), named_month("april", 4),
        named_month("may", 5),
        named_month("jun", 6), named_month("june", 6),
        named_month("jul", 7), named_month("july", 7),
        named_month("aug", 8), named_month("august", 8),
        named_month("sep", 9), named_month("sept", 9),
        named_month("september", 9),
        named_month("oct", 10), named_month("october", 10),
        named_month("nov", 11), named_month("november", 11),
        named_month("dec", 12), named_month("december", 12),
    };
} // anonymous namespace

// Parses a string month
static bool parse_str_month(const char *&begin, const char *end, int &out_month)
{
    const char *saved_begin = begin;
    const char *strbegin, *strend;
    if (!parse_alpha_name_no_ws(begin, end, strbegin, strend)) {
        return false;
    }
    string s(strbegin, strend);
    // Make it lower case
    for (size_t i = 0; i != s.size(); ++i) {
        s[i] = tolower(s[i]);
    }
    // Search through the named month table for a matching month name
    for (size_t i = 0;
         i != sizeof(named_month_table) / sizeof(named_month_table[0]); ++i) {
        if (s == named_month_table[i].name) {
            out_month = named_month_table[i].month;
            return true;
        }
    }
    begin = saved_begin;
    return false;
}

// sMMsDD for separator character 's'
// Returns true on success
static bool parse_md(const char *&begin, const char *end, char sep, int &out_month,
              int &out_day)
{
    const char *saved_begin = begin;
    // sMM
    if (!parse_token_no_ws(begin, end, sep)) {
        begin = saved_begin;
        return false;
    }
    if (!parse_2digit_int(begin, end, out_month)) {
        begin = saved_begin;
        return false;
    }
    // sDD
    if (!parse_token_no_ws(begin, end, sep)) {
        begin = saved_begin;
        return false;
    }
    if (!parse_2digit_int(begin, end, out_day)) {
        begin = saved_begin;
        return false;
    } else if (begin < end && isdigit(begin[0])) {
        // Don't match if the next character is another digit
        begin = saved_begin;
        return false;
    }
    return true;
}

// sMMMsDD for separator character 's' and string-based month
// Returns true on success
static bool parse_md_str_month(const char *&begin, const char *end, char sep, int &out_month,
              int &out_day)
{
    const char *saved_begin = begin;
    // sMMM
    if (!parse_token_no_ws(begin, end, sep)) {
        begin = saved_begin;
        return false;
    }
    if (!parse_str_month(begin, end, out_month)) {
        begin = saved_begin;
        return false;
    }
    // sDD
    if (!parse_token_no_ws(begin, end, sep)) {
        begin = saved_begin;
        return false;
    }
    if (!parse_2digit_int(begin, end, out_day)) {
        begin = saved_begin;
        return false;
    } else if (begin < end && isdigit(begin[0])) {
        // Don't match if the next character is another digit
        begin = saved_begin;
        return false;
    }
    return true;
}

// YYYYsMMsDD, YYYYsMMMsDD for separator character 's', where MM is
// numbers and MMM is a string.
// Returns true on success
static bool parse_ymd_sep_date(const char *&begin, const char *end, char sep,
                               date_ymd &out_ymd)
{
    const char *saved_begin = begin;
    // YYYY
    int year;
    if (!parse_4digit_int(begin, end, year)) {
        begin = saved_begin;
        return false;
    }
    // sMMsDD
    int month, day;
    if (!parse_md(begin, end, sep, month, day)) {
        // sMMMsDD with a string month
        if (!parse_md_str_month(begin, end, sep, month, day)) {
            begin = saved_begin;
            return false;
        }
    }
    // Validate and return the date
    if (!date_ymd::is_valid(year, month, day)) {
        begin = saved_begin;
        return false;
    }
    out_ymd.year = year;
    out_ymd.month = month;
    out_ymd.day = day;
    return true;
}

// DDsMMMsYYYY for separator character 's', where MM is
// numbers and MMM is a string.
// Returns true on success
static bool parse_dmy_str_month_sep_date(const char *&begin, const char *end,
                                         char sep, date_ymd &out_ymd)
{
    const char *saved_begin = begin;
    // DD
    int day;
    if (!parse_2digit_int(begin, end, day)) {
        begin = saved_begin;
        return false;
    }
    // sMMM string month
    if (!parse_token_no_ws(begin, end, sep)) {
        begin = saved_begin;
        return false;
    }
    int month;
    if (!parse_str_month(begin, end, month)) {
        begin = saved_begin;
        return false;
    }
    // sYYYY
    if (!parse_token_no_ws(begin, end, sep)) {
        begin = saved_begin;
        return false;
    }
    int year;
    if (!parse_4digit_int(begin, end, year)) {
        begin = saved_begin;
        return false;
    } else if (begin < end && isdigit(begin[0])) {
        // Don't match if the next character is another digit
        begin = saved_begin;
        return false;
    }
    // Validate and return the date
    if (!date_ymd::is_valid(year, month, day)) {
        begin = saved_begin;
        return false;
    }
    out_ymd.year = year;
    out_ymd.month = month;
    out_ymd.day = day;
    return true;
}

// YYYY-MM-DD, +YYYYYY-MM-DD, or -YYYYYY-MM-DD
// Returns true on success
static bool parse_iso8601_dashes_date(const char *&begin, const char *end,
                                        date_ymd &out_ymd)
{
    const char *saved_begin = begin;
    // YYYY, +YYYYYY, or -YYYYYY
    int year;
    if (parse_token_no_ws(begin, end, '-')) {
        if (!parse_6digit_int(begin, end, year)) {
            begin = saved_begin;
            return false;
        }
        year = -year;
    } else if (parse_token_no_ws(begin, end, '+')) {
        if (!parse_6digit_int(begin, end, year)) {
            begin = saved_begin;
            return false;
        }
    } else {
        if (!parse_4digit_int(begin, end, year)) {
            begin = saved_begin;
            return false;
        }
    }
    // -MM-DD
    int month, day;
    if (!parse_md(begin, end, '-', month, day)) {
        begin = saved_begin;
        return false;
    }
    // Validate and return the date
    if (!date_ymd::is_valid(year, month, day)) {
        begin = saved_begin;
        return false;
    }
    out_ymd.year = year;
    out_ymd.month = month;
    out_ymd.day = day;
    return true;
}

static bool parse_mdy_long_format_date(const char *&begin, const char *end, date_ymd& out_ymd)
{
    const char *saved_begin = begin;
    int month;
    if (!parse_str_month(begin, end, month)) {
        begin = saved_begin;
        return false;
    }
    if (!skip_required_whitespace(begin, end)) {
        begin = saved_begin;
        return false;
    }
    int day;
    if (!parse_1or2digit_int(begin, end, day)) {
        begin = saved_begin;
        return false;
    }
    if (!parse_token(begin, end, ',')) {
        begin = saved_begin;
        return false;
    }
    begin = skip_whitespace(begin, end);
    int year;
    if (!parse_4digit_int(begin, end, year)) {
        begin = saved_begin;
        return false;
    } else if (begin < end && isdigit(begin[0])) {
        // Don't match if the next character is another digit
        begin = saved_begin;
        return false;
    }
    // Validate and return the date
    if (!date_ymd::is_valid(year, month, day)) {
        begin = saved_begin;
        return false;
    }
    out_ymd.year = year;
    out_ymd.month = month;
    out_ymd.day = day;
    return true;
}

// Skips Z, +####, -####, +##, -##, +##:##, -##:##
static void skip_timezone(const char *&begin, const char *end)
{
    if (parse_token_no_ws(begin, end, 'Z')) {
        return;
    } else {
        if (!parse_token_no_ws(begin, end, '+')) {
            if (!parse_token_no_ws(begin, end, '-')) {
                return;
            }
        }
        int tzoffset;
        if (parse_4digit_int(begin, end, tzoffset)) {
            return;
        } else if (parse_2digit_int(begin, end, tzoffset)) {
            const char *saved_begin = begin;
            if (parse_token_no_ws(begin, end, ':')) {
                if (parse_2digit_int(begin, end, tzoffset)) {
                    return;
                } else {
                    begin = saved_begin;
                    return;
                }
            }
        }
    }
}

// Skips 00:00:00.0000, 00:00:00.0000Z, 00:00:00.0000-06:00, 00:00:00.0000-0600, 00:00:00.0000-06
static void skip_midnight_time(const char *&begin, const char *end)
{
    // Hour
    if (!parse_token_no_ws(begin, end, "00")) {
        return;
    }
    // Minute
    if (!parse_token_no_ws(begin, end, ":00")) {
        skip_timezone(begin, end);
        return;
    }
    // Second
    if (!parse_token_no_ws(begin, end, ":00")) {
        skip_timezone(begin, end);
        return;
    }
    // Second decimal
    if (!parse_token_no_ws(begin, end, ".0")) {
        skip_timezone(begin, end);
        return;
    }
    while (begin < end && *begin == '0') {
        ++begin;
    }
    skip_timezone(begin, end);
    return;
}

bool dynd::parse_date(const char *&begin, const char *end, date_ymd& out_ymd)
{
    if (parse_iso8601_dashes_date(begin, end, out_ymd)) {
        // 1979-03-22, +001979-03-22, -005999-01-01
        return true;
    } else if (parse_ymd_sep_date(begin, end, '/', out_ymd)) {
        // 1979/03/22 or 1979/Mar/22
        return true;
    } else if (parse_ymd_sep_date(begin, end, '-', out_ymd)) {
        // 1979-03-22 or 1979-Mar-22
        return true;
    } else if (parse_ymd_sep_date(begin, end, '.', out_ymd)) {
        // 1979.03.22 or 1979.Mar.22
        return true;
    } else if (parse_dmy_str_month_sep_date(begin, end, '/', out_ymd)) {
        // 22/Mar/1979
        return true;
    } else if (parse_dmy_str_month_sep_date(begin, end, '-', out_ymd)) {
        // 22/Mar/1979
        return true;
    } else if (parse_dmy_str_month_sep_date(begin, end, '.', out_ymd)) {
        // 22.Mar.1979
        return true;
    } else if (parse_mdy_long_format_date(begin, end, out_ymd)) {
        // March 22, 1979
        return true;
    } else {
        return false;
    }
}

bool dynd::string_to_date(const char *begin, const char *end, date_ymd& out_ymd)
{
    date_ymd ymd;
    begin = skip_whitespace(begin, end);
    if (!parse_date(begin, end, ymd)) {
        return false;
    }
    // Either a "T" or whitespace may separate a date and a time
    if (parse_token_no_ws(begin, end, 'T')) {
        skip_midnight_time(begin, end);
    } else if (skip_required_whitespace(begin, end)) {
        skip_midnight_time(begin, end);
    }
    begin = skip_whitespace(begin, end);
    if (begin == end) {
        out_ymd = ymd;
        return true;
    } else {
        return false;
    }
}
