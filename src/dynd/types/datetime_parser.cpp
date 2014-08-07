//
// Copyright (C) 2011-14 Mark Wiebe, DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <string>

#include <dynd/parser_util.hpp>
#include <dynd/types/date_parser.hpp>
#include <dynd/types/time_parser.hpp>
#include <dynd/types/datetime_parser.hpp>
#include <dynd/types/datetime_util.hpp>

using namespace std;
using namespace dynd;
using namespace dynd::parse;

// Fri Dec 19 15:10:11 1997, Fri 19 Dec 15:10:11 1997
static bool parse_postgres_datetime(const char *&begin, const char *end,
                                    datetime_struct &out_dt)
{
    saved_begin_state sbs(begin);

    // "Fri "
    int weekday;
    if (!parse_str_weekday_no_ws(begin, end, weekday)) {
        return sbs.fail();
    }
    if (!skip_required_whitespace(begin, end)) {
        return sbs.fail();
    }
    // "Dec 19 " or "19 Dec "
    int month, day;
    if (parse_1or2digit_int_no_ws(begin, end, day)) {
        if (!skip_required_whitespace(begin, end)) {
            return sbs.fail();
        }
        if (!parse_str_month_no_ws(begin, end, month)) {
            return sbs.fail();
        }
    } else if (parse_str_month_no_ws(begin, end, month)) {
        if (!skip_required_whitespace(begin, end)) {
            return sbs.fail();
        }
        if (!parse_1or2digit_int_no_ws(begin, end, day)) {
            return sbs.fail();
        }
    } else {
        return sbs.fail();
    }
    if (!skip_required_whitespace(begin, end)) {
        return sbs.fail();
    }
    // "15:10:11 "
    if (!parse_time_no_tz(begin, end, out_dt.hmst)) {
        return sbs.fail();
    }
    if (!skip_required_whitespace(begin, end)) {
        return sbs.fail();
    }
    // "1997"
    int year;
    if (!parse_4digit_int_no_ws(begin, end, year)) {
        return sbs.fail();
    }
    if (date_ymd::is_valid(year, month, day)) {
        out_dt.ymd.year = year;
        out_dt.ymd.month = month;
        out_dt.ymd.day = day;
        if (out_dt.ymd.get_weekday() != weekday) {
            return sbs.fail();
        }
        return sbs.succeed();
    } else {
        return sbs.fail();
    }
}

// 19971219151011
static bool parse_iso8601_nodashes_datetime(const char *&begin, const char *end,
                                            datetime_struct &out_dt)
{
    saved_begin_state sbs(begin);
    // YYYY
    int year;
    if (!parse_4digit_int_no_ws(begin, end, year)) {
        return sbs.fail();
    }
    // MM
    int month;
    if (!parse_2digit_int_no_ws(begin, end, month)) {
        return sbs.fail();
    }
    // DD
    int day;
    if (!parse_2digit_int_no_ws(begin, end, day)) {
        return sbs.fail();
    }
    // Validate the date
    if (!date_ymd::is_valid(year, month, day)) {
        return sbs.fail();
    }

    // HH
    bool done = false;
    int hour;
    if (!parse_2digit_int_no_ws(begin, end, hour)) {
        hour = 0;
        done = true;
    }
    // MM
    int minute;
    if (!done && !parse_2digit_int_no_ws(begin, end, minute)) {
        minute = 0;
        done = true;
    }
    // SS
    int second;
    if (!done && !parse_2digit_int_no_ws(begin, end, second)) {
        second = 0;
        done = true;
    }
    // .SS*
    int tick = 0;
    if (!parse_token_no_ws(begin, end, '.')) {
        done = true;
    }
    if (!done && begin < end && isdigit(*begin)) {
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
    }

    // Validate the time
    if (!time_hmst::is_valid(hour, minute, second, tick)) {
        return sbs.fail();
    }

    out_dt.ymd.year = year;
    out_dt.ymd.month = month;
    out_dt.ymd.day = day;
    out_dt.hmst.hour = hour;
    out_dt.hmst.minute = minute;
    out_dt.hmst.second = second;
    out_dt.hmst.tick = tick;
    return sbs.succeed();
}

// <date> T <time>, <date>:<time>, <date> <time>
// optionally followed by a timezone specifier
static bool parse_date_time_datetime(const char *&begin, const char *end,
                                     datetime_struct &out_dt,
                                     date_parse_order_t ambig,
                                     int century_window,
                                     const char *&out_tz_begin,
                                     const char *&out_tz_end)
{
    saved_begin_state sbs(begin);

    if (!parse_date(begin, end, out_dt.ymd, ambig, century_window)) {
        return sbs.fail();
    }
    // "T", ":', or whitespace may separate a date and a time
    if (parse_token(begin, end, 'T')) {
      // Allow whitespace around the T
      skip_whitespace(begin, end);
    } else if (!parse_token_no_ws(begin, end, ':')) {
      if (!skip_required_whitespace(begin, end)) {
        out_dt.hmst.set_to_zero();
        return sbs.succeed();
      }
      if (begin == end) {
        // If there was just the date, set the rest to zero
        out_dt.hmst.set_to_zero();
        return sbs.succeed();
      }
    }
    if (!parse_time(begin, end, out_dt.hmst, out_tz_begin, out_tz_end)) {
        // In most cases we don't accept just an hour as a time, but for
        // ISO dash-formatted dates, we do, so try again as that.
        begin = sbs.saved_begin();
        if (!parse_iso8601_dashes_date(begin, end, out_dt.ymd)) {
            return sbs.fail();
        }
        // Either a "T" or whitespace may separate a date and a time
        if (!parse_token_no_ws(begin, end, 'T') &&
                !skip_required_whitespace(begin, end)) {
            return sbs.fail();
        }
        int hour;
        // Require a 2 digit hour followed by a non-digit
        if (!parse_2digit_int_no_ws(begin, end, hour) ||
                (begin < end && isdigit(*begin))) {
            return sbs.fail();
        }
        if (time_hmst::is_valid(hour, 0, 0, 0)) {
            out_dt.hmst.hour = hour;
            out_dt.hmst.minute = 0;
            out_dt.hmst.second = 0;
            out_dt.hmst.tick = 0;
            return sbs.succeed();
        } else {
            return sbs.fail();
        }
    }
    return sbs.succeed();
}

bool parse::parse_datetime(const char *&begin, const char *end,
                           date_parse_order_t ambig, int century_window,
                           datetime_struct &out_dt, const char *&out_tz_begin,
                           const char *&out_tz_end)
{
    if (parse_date_time_datetime(begin, end, out_dt, ambig, century_window,
                                 out_tz_begin, out_tz_end)) {
        // <date>T<time>, <date> <time>
    } else if (parse_postgres_datetime(begin, end, out_dt)) {
        // Fri Dec 19 15:10:11 1997, Fri 19 Dec 15:10:11 1997
    } else if (parse_iso8601_nodashes_datetime(begin, end, out_dt)) {
        // 19971219151011
    } else {
        return false;
    }
    return true;
}

bool dynd::string_to_datetime(const char *begin, const char *end,
                              date_parse_order_t ambig, int century_window,
                              assign_error_mode errmode,
                              datetime_struct &out_dt,
                              const char *&out_tz_begin,
                              const char *&out_tz_end)
{
    datetime_struct dt;
    skip_whitespace(begin, end);
    if (!parse_datetime(begin, end, ambig, century_window, dt, out_tz_begin,
                        out_tz_end)) {
        return false;
    }
    skip_whitespace(begin, end);
    if (begin == end || errmode == assign_error_nocheck) {
        out_dt = dt;
        return true;
    } else {
        return false;
    }
}
