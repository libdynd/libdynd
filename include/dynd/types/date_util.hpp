//
// Copyright (C) 2011-15 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#pragma once

#include <algorithm>
#include <limits>
#include <iostream>

#include <dynd/config.hpp>
#include <dynd/typed_data_assign.hpp>

#define DYND_DATE_NA (std::numeric_limits<int32_t>::min())

#define DYND_TICKS_PER_DAY (24LL * 60LL * 60LL * 10000000LL)
#define DYND_SECONDS_PER_DAY (24LL * 60LL * 60LL)

namespace dynd {

namespace ndt {
    class type;
} // namespace ndt

/**
 * An enumeration for describing how to interpret ambiguous
 * dates such as "01/02/03" or "01/02/1995".
 */
enum date_parse_order_t {
    date_parse_no_ambig,
    date_parse_ymd,
    date_parse_mdy,
    date_parse_dmy
};

DYND_API std::ostream& operator<<(std::ostream& o, date_parse_order_t date_order);

struct DYND_API date_ymd {
    int16_t year;
    int8_t month;
    int8_t day;

    static const int month_lengths[2][12];
    static const int month_starts[2][13];

    static inline bool is_leap_year(int year) {
        return year % 4 == 0 && (year % 100 != 0 || year % 400 == 0);
    }

    static inline int get_month_length(int year, int month) {
        if (month >= 1 && month <= 12) {
            return month_lengths[is_leap_year(year)][month - 1];
        } else {
            return 0;
        }
    }

    static inline bool is_valid(int year, int month, int day) {
        if (month < 1 || month > 12) {
            return false;
        } else if (day < 1 ||
                   day > month_lengths[is_leap_year(year)][month - 1]) {
            return false;
        } else {
            return true;
        }
    }

    inline bool is_valid() const {
        return is_valid(year, month, day);
    }

    inline bool is_na() const {
        return month == -128;
    }

    /**
     * Gets the length of the month this date is
     * in. Returns 0 for an invalid date.
     */
    inline int get_month_length() const {
        return get_month_length(year, month);
    }

    /**
     * Gets the 0-based day of the week, where
     * 0 is Monday, 6 is Sunday.
     */
    inline int get_weekday() const {
        int days = to_days();
        // January 5, 1970 is Monday
        int weekday = (days - 4) % 7;
        if (weekday < 0) {
            weekday += 7;
        }
        return weekday;
    }

    /**
     * Gets the 0-based day of the year,
     * values 0-365, but returns -1 for an invalid
     * date.
     */
    inline int get_day_of_year() const {
        if (is_valid(year, month, day)) {
            return month_starts[is_leap_year(year)][month-1] + day - 1;
        } else {
            return -1;
        }
    }

    /**
     * Converts the ymd into a days offset from January 1, 1970.
     */
    static int32_t to_days(int year, int month, int day);

    /**
     * Converts the ymd into a days offset from January 1, 1970.
     */
    int32_t to_days() const {
        return to_days(year, month, day);
    }

    /**
     * Converts the ymd into an ISO 8601 string, or "" if it
     * is invalid. For years from 0001 to 9999, uses "####-##-##",
     * and for years outside that range uses "-######-##-##" or
     * "+######-##-##".
     */
    static std::string to_str(int year, int month, int day);

    /**
     * Converts the ymd into an ISO 8601 string, or "" if it
     * is invalid. For years from 0001 to 9999, uses "####-##-##",
     * and for years outside that range uses "-######-##-##" or
     * "+######-##-##".
     */
    inline std::string to_str() const {
        return to_str(year, month, day);
    }

    /**
     * Sets the year/month/day from a 1970 epoch days offset.
     *
     * \param days  A days offset from January 1, 1970.
     */
    void set_from_days(int32_t days);

    /**
     * Sets the year/month/day from the ticks-based date, throwing
     * away the time portion.
     */
    inline void set_from_ticks(int64_t ticks) {
        if (ticks >= 0) {
            set_from_days(static_cast<int32_t>(ticks / DYND_TICKS_PER_DAY));
        } else {
            set_from_days(static_cast<int32_t>(
                (ticks - (DYND_TICKS_PER_DAY - 1)) / DYND_TICKS_PER_DAY));
        }
    }

    /**
     * Sets the year/month/day to NA.
     */
    inline void set_to_na() {
        month = -128;
    }

    /**
     * Sets the year/month/day from a string. Accepts a wide variety of
     * inputs, but by default rejects ambiguous formats like MM/DD/YY vs
     * DD/MM/YY. Two digit years are handled with a sliding window
     * starting 70 years ago by default.
     *
     * \param s  Date string.
     * \param ambig  Order to use for ambiguous cases like "01/02/03"
     *               or "01/02/1995".
     * \param century_window  Number describing how to handle dates with
     *                        two digit years. Values 1 to 99 mean to use
     *                        a sliding window starting that many years back.
     *                        Values 1000 and higher mean to use a fixed window
     *                        starting at the year given. The value 0 means to
     *                        disallow two digit years.
     */
    inline void
    set_from_str(const std::string &s,
                 date_parse_order_t ambig = date_parse_no_ambig,
                 int century_window = 70,
                 assign_error_mode errmode = assign_error_fractional)
    {
        return set_from_str(s.data(), s.data() + s.size(), ambig, century_window, errmode);
    }

    void set_from_str(const char *begin, const char *end,
                      date_parse_order_t ambig,
                      int century_window,
                      assign_error_mode errmode);

    /**
     * When a year is a two digit year that should be resolved as a four
     * digit year, this function provides the resolution
     * using a fixed window strategy. The window is from `year_start` to
     * 100 years later.
     *
     * \param year  The year to apply the fixed window to.
     * \param year_start  The year at which the window starts.
     *
     * \returns  The adjusted year.
     */
    static int resolve_2digit_year_fixed_window(int year, int year_start);

    /**
     * When a year is a two digit year that should be resolved as a four
     * digit year, this function provides the resolution
     * using a sliding window strategy. The window is from
     * the current year with `years_ago` years subtracted from it, and is
     * one century long.
     *
     * This function simply retrieves the current year, and calls the variant
     * with selectable `year_start`.
     *
     * \param year  The year to apply the sliding window to.
     * \param years_ago  The number of years before the current date for the
     *                   start of the window.
     *
     * \returns  The adjusted year.
     */
    static int resolve_2digit_year_sliding_window(int year, int years_ago);

    /**
     * This function resolves a two digit year, choosing between a sliding
     * or fixed window approach based on the parameter value.
     *
     * \param year  The year to apply the century selection to.
     * \param century_window  Number describing how to handle dates with
     *                        two digit years. Values 1 to 99 mean to use
     *                        a sliding window starting that many years back.
     *                        Values 1000 and higher mean to use a fixed window
     *                        starting at the year given. The value 0 means to
     *                        disallow two digit years.
     *
     * \returns  The adjusted year.
     */
    static int resolve_2digit_year(int year, int century_window);

    /**
     * Returns the current date in the local time zone.
     */
    static date_ymd get_current_local_date();

    /**
     * Returns an ndt::type corresponding to the date_ymd structure.
     */
    static const ndt::type& type();
};

} // namespace dynd
