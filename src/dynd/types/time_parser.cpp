// Copyright (C) 2011-14 Mark Wiebe, DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <string>

#include <dynd/parser_util.hpp>
#include <dynd/types/time_parser.hpp>
#include <dynd/types/time_util.hpp>

using namespace std;
using namespace dynd;
using namespace dynd::parse;

// Similar to ISO 8601, but allow 1-digit hour, AM/PM specifier
static bool parse_flex_time(const char *&begin, const char *end, time_hmst &out_hmst)
{
    saved_begin_state sbs(begin);

    // HH
    int hour;
    if (!parse_1or2digit_int_no_ws(begin, end, hour)) {
        return sbs.fail();
    }
    // :MM
    if (!parse_token_no_ws(begin, end, ':')) {
        // If there's no ':', fail
        return sbs.fail();
    }
    int minute;
    if (!parse_2digit_int_no_ws(begin, end, minute)) {
        return sbs.fail();
    }
    // :SS
    if (!parse_token_no_ws(begin, end, ':')) {
        // If there's no ':', stop here and match just the HH:MM
        parse_time_ampm(begin, end, hour);
        if (time_hmst::is_valid(hour, minute, 0, 0)) {
            out_hmst.hour = hour;
            out_hmst.minute = minute;
            out_hmst.second = 0;
            out_hmst.tick = 0;
            return sbs.succeed();
        } else {
            return sbs.fail();
        }
    }
    int second;
    if (!parse_2digit_int_no_ws(begin, end, second)) {
        return sbs.fail();
    }
    // .SS*
    if (!parse_token_no_ws(begin, end, '.')) {
        // If there's no '.', stop here and match just the HH:MM:SS
        parse_time_ampm(begin, end, hour);
        if (time_hmst::is_valid(hour, minute, second, 0)) {
            out_hmst.hour = hour;
            out_hmst.minute = minute;
            out_hmst.second = second;
            out_hmst.tick = 0;
            return sbs.succeed();
        } else {
            return sbs.fail();
        }
    }
    int tick = 0;
    if (!(begin < end && isdigit(*begin))) {
        // Require at least one digit after the decimal place
        return sbs.fail();
    }
    tick = (*begin - '0');
    ++begin;
    for (int i = 1; i < 7; ++i) {
        tick *= 10;
        if (begin < end && isdigit(*begin)) {
            tick += (*begin - '0');
            ++begin;
        }
    }
    // Swallow any additional decimal digits, truncating to ticks
    while (begin < end && isdigit(*begin)) {
        ++begin;
    }
    parse_time_ampm(begin, end, hour);
    if (time_hmst::is_valid(hour, minute, second, tick)) {
        out_hmst.hour = hour;
        out_hmst.minute = minute;
        out_hmst.second = second;
        out_hmst.tick = tick;
        return sbs.succeed();
    } else {
        return sbs.fail();
    }
}

// Matches various AM/PM indicators, and adjusts the hour value
bool parse::parse_time_ampm(const char *&begin, const char *end, int& inout_hour)
{
    saved_begin_state sbs(begin);

    skip_whitespace(begin, end);

    if (parse_token(begin, end, "AM") || parse_token(begin, end, "am") ||
            parse_token(begin, end, "A.M.") ||
            parse_token(begin, end, "a.m.") ||
            parse_token(begin, end, "a")) {
        // The hour should be between 1 and 12
        if (inout_hour < 1 || inout_hour > 12) {
            // Invalidate the hour for later checks to see
            inout_hour = -1;
            return sbs.fail();
        } else {
            // 12:00 AM is 00:00
            if (inout_hour == 12) {
                inout_hour = 0;
            }
            return sbs.succeed();
        }
    } else if (parse_token(begin, end, "PM") || parse_token(begin, end, "pm") ||
               parse_token(begin, end, "P.M.") ||
               parse_token(begin, end, "p.m.") ||
               parse_token(begin, end, "p")) {
        // The hour should be between 1 and 12
        if (inout_hour < 1 || inout_hour > 12) {
            // Invalidate the hour for later checks to see
            inout_hour = -1;
            return sbs.fail();
        } else {
            if (inout_hour < 12) {
                inout_hour += 12;
            }
            return sbs.succeed();
        }
    } else {
        return sbs.fail();
    }
}

bool parse::parse_time(const char *&begin, const char *end, time_hmst &out_hmst)
{
    if (parse_flex_time(begin, end, out_hmst)) {
        // Allows 1-digit hour, AM/PM
        // Almost all ISO 8601 times will match against this, except for when
        // only the hour is specified.
    } else {
        return false;
    }
    return true;
}


bool dynd::string_to_time(const char *begin, const char *end, time_hmst &out_hmst)
{
    time_hmst hmst;
    skip_whitespace(begin, end);
    if (!parse_time(begin, end, hmst)) {
        return false;
    }
    skip_whitespace(begin, end);
    if (begin == end) {
        out_hmst = hmst;
        return true;
    } else {
        return false;
    }
}
