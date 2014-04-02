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

// HH, HH:MM, HH:MM:SS, HH:MM:SS.SS*
static bool parse_iso8601_time(const char *&begin, const char *end, time_hmst &out_hmst)
{
    saved_begin_state sbs(begin);

    // HH
    int hour;
    if (!parse_2digit_int_no_ws(begin, end, hour)) {
        return sbs.fail();
    }
    // :MM
    if (!parse_token_no_ws(begin, end, ':')) {
        // If there's no ':', stop here and match just the HH
        if (time_hmst::is_valid(hour, 0, 0, 0)) {
            out_hmst.hour = hour;
            out_hmst.minute = 0;
            out_hmst.second = 0;
            out_hmst.tick = 0;
            return sbs.succeed();
        } else {
            return sbs.fail();
        }
    }
    int minute;
    if (!parse_2digit_int_no_ws(begin, end, minute)) {
        return sbs.fail();
    }
    // :SS
    if (!parse_token_no_ws(begin, end, ':')) {
        // If there's no ':', stop here and match just the HH:MM
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

bool dynd::parse_time(const char *&begin, const char *end, time_hmst &out_hmst)
{
    return parse_iso8601_time(begin, end, out_hmst);
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
