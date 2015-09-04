//
// Copyright (C) 2011-15 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#pragma once

#include <dynd/config.hpp>

namespace dynd {

struct time_hmst;

/**
 * Parses a time.
 *
 * \param begin  The start of the UTF-8 buffer to parse.
 * \param end  One past the last character of the buffer to parse.
 * \param out_hmst  The time to fill.
 * \param out_tz_begin  If a timezone is parsed, this is set to the beginning of
 *                      matched timezone.
 * \param out_tz_end  If a timezone is parsed, this is set to the end of
 *                    matched timezone.
 *
 * \returns  True if the parse is successful, false otherwise.
 */
bool DYND_API string_to_time(const char *begin, const char *end, time_hmst &out_hmst,
                    const char *&out_tz_begin, const char *&out_tz_end);

namespace parse {

    /**
     * Parses a time zone specifier.
     *
     * \param begin  The start of a range of UTF-8 characters. This is modified
     *               to point immediately after the parsed time if true is returned.
     * \param end  The end of a range of UTF-8 characters.
     *
     * \param out_tz_begin  If a timezone is parsed, this is set to the beginning of
     *                      matched timezone.
     * \param out_tz_end  If a timezone is parsed, this is set to the end of
     *                    matched timezone.
     *
     * \returns  True if a time zone was parsed successfully, false otherwise.
     */
    DYND_API bool parse_timezone(const char *&begin, const char *end,
                        const char *&out_tz_begin, const char *&out_tz_end);

    /**
     * Parses a time, possibly including a time zone
     *
     * \param begin  The start of a range of UTF-8 characters. This is modified
     *               to point immediately after the parsed time if true is returned.
     * \param end  The end of a range of UTF-8 characters.
     * \param out_hmst  The time to fill.
     * \param out_tz_begin  If a timezone is parsed, this is set to the beginning of
     *                      matched timezone.
     * \param out_tz_end  If a timezone is parsed, this is set to the end of
     *                    matched timezone.
     *
     * \returns  True if a time was parsed successfully, false otherwise.
     */
    DYND_API bool parse_time(const char *&begin, const char *end, time_hmst &out_hmst,
                    const char *&out_tz_begin, const char *&out_tz_end);

    /**
     * Parses a time without a time zone.
     *
     * \param begin  The start of a range of UTF-8 characters. This is modified
     *               to point immediately after the parsed time if true is returned.
     * \param end  The end of a range of UTF-8 characters.
     * \param out_hmst  The time to fill.
     *
     * \returns  True if a time was parsed successfully, false otherwise.
     */
    DYND_API bool parse_time_no_tz(const char *&begin, const char *end,
                          time_hmst &out_hmst);

    /**
     * Without skipping whitespace, parses an AM/PM indicator string and modifies
     * the provided hour appropriately. If the AM/PM is incompatible with the
     * hour value, sets the hour value to -1.
     */
    DYND_API bool parse_time_ampm(const char *&begin, const char *end, int& inout_hour);

} // namespace parse

} // namespace parse
