//
// Copyright (C) 2011-14 Mark Wiebe, DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#ifndef _DYND__DATE_UTIL_HPP_
#define _DYND__DATE_UTIL_HPP_

#include <algorithm>
#include <limits>
#include <iostream>

#include <dynd/config.hpp>

#define DYND_DATE_NA (std::numeric_limits<int32_t>::min())

#define DYND_TICKS_PER_DAY (24LL * 60LL * 60LL * 10000000LL)

namespace dynd {

namespace ndt {
    class type;
} // namespace ndt

struct date_ymd {
    int16_t year;
    int8_t month;
    int8_t day;

    static const int month_lengths[2][12];
    static const int month_starts[2][13];

    static inline bool is_leap_year(int year) {
        return year % 4 == 0 && (year % 100 != 0 || year % 400 == 0);
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
     * Converts the ymd into an ISO 8601 string, or "NA" if it
     * is invalid. For years from 0001 to 9999, uses "####-##-##",
     * and for years outside that range uses "-######-##-##" or
     * "+######-##-##".
     */
    static std::string to_str(int year, int month, int day);

    /**
     * Converts the ymd into an ISO 8601 string, or "NA" if it
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
     * inputs, but rejects ambiguous formats like MM/DD/YY vs DD/MM/YY.
     *
     * \param s  Date string.
     */
    void set_from_str(const std::string& s);

    /**
     * Sets the year/month/day from a string. Accepts a wide variety of
     * inputs, and interprets ambiguous formats like MM/DD/YY vs DD/MM/YY
     * using the monthfirst parameter.
     *
     * \param s  Date string.
     * \param monthfirst  If true, expect MM/DD/YY, otherwise expect DD/MM/YY.
     */
    void set_from_str(const std::string& s, bool monthfirst);

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
 
#endif // _DYND__DATE_UTIL_HPP_