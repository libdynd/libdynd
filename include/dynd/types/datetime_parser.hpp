//
// Copyright (C) 2011-15 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#pragma once

#include <dynd/config.hpp>
#include <dynd/types/date_parser.hpp>
#include <dynd/typed_data_assign.hpp>

namespace dynd {

struct datetime_struct;

/**
 * Parses a datetime.
 *
 * \param begin  The start of the UTF-8 buffer to parse.
 * \param end  One past the last character of the buffer to parse.
 * \param ambig  Order to use for ambiguous cases like "01/02/03"
 *               or "01/02/1995".
 * \param century_window  Number describing how to handle dates with
 *                        two digit years. Values 1 to 99 mean to use
 *                        a sliding window starting that many years back.
 *                        Values 1000 and higher mean to use a fixed window
 *                        starting at the year given. The value 0 means to
 *                        disallow two digit years.
 * \param out_dt  The datetime to fill.
 * \param out_tz_begin  If a timezone is parsed, this is set to the beginning of
 *                      matched timezone.
 * \param out_tz_end  If a timezone is parsed, this is set to the end of
 *                    matched timezone.
 *
 * \returns  True if the parse is successful, false otherwise.
 */
DYND_API bool string_to_datetime(const char *begin, const char *end,
                        date_parse_order_t ambig, int century_window,
                        assign_error_mode errmode, datetime_struct &out_dt,
                        const char *&out_tz_begin, const char *&out_tz_end);

namespace parse {

    /**
     * Parses a datetime
     *
     * \param begin  The start of a range of UTF-8 characters. This is modified
     *               to point immediately after the parsed datetime if true is returned.
     * \param end  The end of a range of UTF-8 characters.
     * \param ambig  Order to use for ambiguous cases like "01/02/03"
     *               or "01/02/1995".
     * \param century_window  Number describing how to handle dates with
     *                        two digit years. Values 1 to 99 mean to use
     *                        a sliding window starting that many years back.
     *                        Values 1000 and higher mean to use a fixed window
     *                        starting at the year given. The value 0 means to
     *                        disallow two digit years.
     * \param out_dt  The datetime to fill.
     * \param out_tz_begin  If a timezone is parsed, this is set to the beginning of
     *                      matched timezone.
     * \param out_tz_end  If a timezone is parsed, this is set to the end of
     *                    matched timezone.
     *
     * \returns  True if a datetime was parsed successfully, false otherwise.
     */
    DYND_API bool parse_datetime(const char *&begin, const char *end,
                        date_parse_order_t ambig, int century_window,
                        datetime_struct &out_dt, const char *&out_tz_begin,
                        const char *&out_tz_end);

} // namespace parse

} // namespace parse
