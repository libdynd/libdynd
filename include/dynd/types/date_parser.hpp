//
// Copyright (C) 2011-14 Mark Wiebe, DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#ifndef _DYND__DATE_PARSER_HPP_
#define _DYND__DATE_PARSER_HPP_

#include <dynd/config.hpp>

namespace dynd {

struct date_ymd;

/**
 * Parses a date. Accepts a wide variety of inputs, but rejects ambiguous
 * formats like MM/DD/YY vs DD/MM/YY. Skips whitespace at the beginning/end,
 * and fails if the full buffer is not a single date. If a time is after the
 * date, and is midnight, it will match successfully, ignoring any time zone
 * information.
 *
 * \param begin  The start of the buffer to parse.
 * \param end  One past the last character of the buffer to parse.
 * \param out_ymd  The date to fill.
 *
 * \returns  True if the parse is successful, false otherwise.
 */
bool string_to_date(const char *begin, const char *end, date_ymd& out_ymd);

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
 *
 * \returns  True if a date was parsed successfully, false otherwise.
 */
bool parse_date(const char *&begin, const char *end, date_ymd& out_ymd);



} // namespace dynd

#endif // _DYND__DATE_PARSER_HPP_
