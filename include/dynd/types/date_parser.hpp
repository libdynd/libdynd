//
// Copyright (C) 2011-15 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#pragma once

#include <dynd/config.hpp>
#include <dynd/types/date_util.hpp>

namespace dynd {

/**
 * Parses a date. Accepts a wide variety of inputs, but rejects ambiguous
 * formats like MM/DD/YY vs DD/MM/YY. Skips whitespace at the beginning/end,
 * and fails if the full buffer is not a single date. If a time is after the
 * date, and is midnight, it will match successfully, ignoring any time zone
 * information.
 *
 * \param begin  The start of the UTF-8 buffer to parse.
 * \param end  One past the last character of the buffer to parse.
 * \param out_ymd  The date to fill.
 * \param ambig  Order to use for ambiguous cases like "01/02/03"
 *               or "01/02/1995".
 * \param century_window  Number describing how to handle dates with
 *                        two digit years. Values 1 to 99 mean to use
 *                        a sliding window starting that many years back.
 *                        Values 1000 and higher mean to use a fixed window
 *                        starting at the year given. The value 0 means to
 *                        disallow two digit years.
 *
 * \returns  True if the parse is successful, false otherwise.
 */
DYND_API bool string_to_date(const char *begin, const char *end, date_ymd &out_ymd,
                    date_parse_order_t ambig, int century_window,
                    assign_error_mode errmode);

namespace parse {

    /**
     * Parses a date. Accepts a wide variety of inputs, but rejects ambiguous
     * formats like MM/DD/YY vs DD/MM/YY. This function does not parse after
     * the date is matched, so can be used when parsing date and time together.
     *
     * \param begin  The start of a range of UTF-8 characters. This is modified
     *               to point immediately after the parsed date if true is returned.
     * \param end  The end of a range of UTF-8 characters.
     * \param out_ymd  If true is returned, this has been filled with the parsed
     *                 date.
     * \param ambig  Order to use for ambiguous cases like "01/02/03"
     *               or "01/02/1995".
     * \param century_window  Number describing how to handle dates with
     *                        two digit years. Values 1 to 99 mean to use
     *                        a sliding window starting that many years back.
     *                        Values 1000 and higher mean to use a fixed window
     *                        starting at the year given. The value 0 means to
     *                        disallow two digit years.
     *
     * \returns  True if a date was parsed successfully, false otherwise.
     */
    DYND_API bool parse_date(const char *&begin, const char *end, date_ymd &out_ymd,
                    date_parse_order_t ambig, int century_window);

    /**
     * Parses a date in ISO 8601 dashes form like YYYY-MM-DD, +YYYYYY-MM-DD, or
     * -YYYYYY-MM-DD.
     *
     * \param begin  The start of a range of UTF-8 characters. This is modified
     *               to point immediately after the parsed date if true is returned.
     * \param end  The end of a range of UTF-8 characters.
     * \param out_ymd  If true is returned, this has been filled with the parsed
     *                 date.
     */
    DYND_API bool parse_iso8601_dashes_date(const char *&begin, const char *end,
                                   date_ymd &out_ymd);

    /**
     * Parses a string month: Jan == 1, Dec == 12.
     *
     * \param begin  The start of a range of UTF-8 characters. This is modified
     *               to point immediately after the parsed value if true is returned.
     * \param end  The end of a range of UTF-8 characters.
     * \param out_month  If true is returned, this has been filled with the parsed
     *                   month.
     */
    DYND_API bool parse_str_month_no_ws(const char *&begin, const char *end, int &out_month);

    /**
     * Parses a string month: Jan == 1, Dec == 12. Accepts a period after month
     * abbreviations like "Jan.".
     *
     * \param begin  The start of a range of UTF-8 characters. This is modified
     *               to point immediately after the parsed value if true is returned.
     * \param end  The end of a range of UTF-8 characters.
     * \param out_month  If true is returned, this has been filled with the parsed
     *                   month.
     */
    DYND_API bool parse_str_month_punct_no_ws(const char *&begin, const char *end, int &out_month);

    /**
     * Parses a string weekday: Mon == 0, Sun == 6.
     *
     * \param begin  The start of a range of UTF-8 characters. This is modified
     *               to point immediately after the parsed value if true is returned.
     * \param end  The end of a range of UTF-8 characters.
     * \param out_weekday  If true is returned, this has been filled with the parsed
     *                   weekday.
     */
    DYND_API bool parse_str_weekday_no_ws(const char *&begin, const char *end,
                           int &out_weekday);

} // namespace parse

} // namespace dynd
