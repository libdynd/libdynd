//
// Copyright (C) 2011-14 Mark Wiebe, DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#ifndef _DYND__DATETIME_PARSER_HPP_
#define _DYND__DATETIME_PARSER_HPP_

#include <dynd/config.hpp>
#include <dynd/types/date_parser.hpp>

namespace dynd {

struct datetime_struct;

/**
 * Parses a datetime.
 *
 * \param begin  The start of the UTF-8 buffer to parse.
 * \param end  One past the last character of the buffer to parse.
 * \param out_dt  The datetime to fill.
 * \param ambig  How to handle the 01/02/2003 ambiguity. Defaults to disallow,
 *                   can also be dayfirst or monthfirst.
 * \param allow_2digit_year  If true, allows 2 digit years resolved with a sliding
 *                           window starting 70 years ago. Defaults to true.
 *
 * \returns  True if the parse is successful, false otherwise.
 */
bool string_to_datetime(
    const char *begin, const char *end, datetime_struct &out_dt,
    date_parser_ambiguous_t ambig = date_parser_ambiguous_disallow,
    bool allow_2digit_year = true);

namespace parse {

    /**
     * Parses a datetime
     *
     * \param begin  The start of a range of UTF-8 characters. This is modified
     *               to point immediately after the parsed datetime if true is returned.
     * \param end  The end of a range of UTF-8 characters.
     * \param out_dt  The datetime to fill.
     * \param ambig  How to handle the 01/02/2003 ambiguity.
     * \param allow_2digit_year  If true, allows 2 digit years resolved with a sliding
     *                           window starting 70 years ago.
     *
     * \returns  True if a datetime was parsed successfully, false otherwise.
     */
    bool parse_datetime(const char *&begin, const char *end,
                        datetime_struct &out_dt, date_parser_ambiguous_t ambig,
                        bool allow_2digit_year);

} // namespace parse

} // namespace parse

#endif // _DYND__DATETIME_PARSER_HPP_
