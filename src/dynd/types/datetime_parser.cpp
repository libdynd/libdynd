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

bool parse::parse_datetime(const char *&begin, const char *end,
                           datetime_struct &out_dt,
                           date_parser_ambiguous_t ambig,
                           bool allow_2digit_year)
{
    saved_begin_state sbs(begin);

    if (!parse_date(begin, end, out_dt.ymd, ambig, allow_2digit_year)) {
        return sbs.fail();
    }
    // Either a "T" or whitespace may separate a date and a time
    if (!parse_token_no_ws(begin, end, 'T') &&
            !skip_required_whitespace(begin, end)) {
        return sbs.fail();
    }
    if (!parse_time(begin, end, out_dt.hmst)) {
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

bool dynd::string_to_datetime(const char *begin, const char *end,
                              datetime_struct &out_dt,
                              date_parser_ambiguous_t ambig,
                              bool allow_2digit_year)
{
    datetime_struct dt;
    skip_whitespace(begin, end);
    if (!parse_datetime(begin, end, dt, ambig, allow_2digit_year)) {
        return false;
    }
    skip_whitespace(begin, end);
    if (begin == end) {
        out_dt = dt;
        return true;
    } else {
        return false;
    }
}
