//
// Copyright (C) 2011-14 Mark Wiebe, DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#ifndef _DYND__DATE_PARSER_HPP_
#define _DYND__DATE_PARSER_HPP_

#include <dynd/config.hpp>

namespace dynd {

struct date_ymd;

enum date_parser_ambiguous_t {
    // Don't allow dates like 01/02/2003
    date_parser_ambiguous_disallow,
    // 01/02/2003 means February 1, 2003
    date_parser_ambiguous_dayfirst,
    // 01/02/2003 means January 2, 2003
    date_parser_ambiguous_monthfirst
};

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
 * \param ambig  How to handle the 01/02/2003 ambiguity. Defaults to disallow,
 *                   can also be dayfirst or monthfirst.
 * \param allow_2digit_year  If true, allows 2 digit years resolved with a sliding
 *                           window starting 70 years ago. Defaults to true.
 *
 * \returns  True if the parse is successful, false otherwise.
 */
bool
string_to_date(const char *begin, const char *end, date_ymd &out_ymd,
               date_parser_ambiguous_t ambig = date_parser_ambiguous_disallow,
               bool allow_2digit_year = true);

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
     * \param ambig  How to handle the 01/02/2003 ambiguity.
     * \param allow_2digit_year  If true, allows 2 digit years resolved with a sliding
     *                           window starting 70 years ago.
     *
     * \returns  True if a date was parsed successfully, false otherwise.
     */
    bool parse_date(const char *&begin, const char *end, date_ymd &out_ymd,
                    date_parser_ambiguous_t ambig, bool allow_2digit_year);

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
    bool parse_iso8601_dashes_date(const char *&begin, const char *end,
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
    bool parse_str_month_no_ws(const char *&begin, const char *end, int &out_month);

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
    bool parse_str_month_punct_no_ws(const char *&begin, const char *end, int &out_month);

    /**
     * Parses a string weekday: Mon == 0, Sun == 6.
     *
     * \param begin  The start of a range of UTF-8 characters. This is modified
     *               to point immediately after the parsed value if true is returned.
     * \param end  The end of a range of UTF-8 characters.
     * \param out_weekday  If true is returned, this has been filled with the parsed
     *                   weekday.
     */
    bool parse_str_weekday_no_ws(const char *&begin, const char *end,
                           int &out_weekday);

} // namespace parse

} // namespace dynd

#endif // _DYND__DATE_PARSER_HPP_
