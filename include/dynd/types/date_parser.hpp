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
 *
 * \returns  True if the parse is successful, false otherwise.
 */
bool string_to_date(const char *begin, const char *end, date_ymd &out_ymd,
                    date_parser_ambiguous_t ambig=date_parser_ambiguous_disallow);

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
     *
     * \returns  True if a date was parsed successfully, false otherwise.
     */
    bool parse_date(const char *&begin, const char *end, date_ymd &out_ymd,
                    date_parser_ambiguous_t ambig);

} // namespace parse

} // namespace dynd

#endif // _DYND__DATE_PARSER_HPP_
