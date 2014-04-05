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
bool string_to_datetime(const char *begin, const char *end,
                        datetime_struct &out_dt, date_parse_order_t ambig,
                        int century_window);

namespace parse {

    /**
     * Parses a datetime
     *
     * \param begin  The start of a range of UTF-8 characters. This is modified
     *               to point immediately after the parsed datetime if true is returned.
     * \param end  The end of a range of UTF-8 characters.
     * \param out_dt  The datetime to fill.
     * \param ambig  Order to use for ambiguous cases like "01/02/03"
     *               or "01/02/1995".
     * \param century_window  Number describing how to handle dates with
     *                        two digit years. Values 1 to 99 mean to use
     *                        a sliding window starting that many years back.
     *                        Values 1000 and higher mean to use a fixed window
     *                        starting at the year given. The value 0 means to
     *                        disallow two digit years.
     *
     * \returns  True if a datetime was parsed successfully, false otherwise.
     */
    bool parse_datetime(const char *&begin, const char *end,
                        datetime_struct &out_dt, date_parse_order_t ambig,
                        int century_window);

} // namespace parse

} // namespace parse

#endif // _DYND__DATETIME_PARSER_HPP_
