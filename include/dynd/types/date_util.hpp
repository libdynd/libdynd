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

    /**
     * Converts the ymd into a days offset from January 1, 1970.
     */
    static int32_t to_days(int year, int month, int day) {
        if (is_valid(year, month, day)) {
            // Start with 365 days a year
            int result = (year - 1970) * 365;
            // Use the inclusion-exclusion principle to count leap years
            if (result >= 0) {
                result += ((year - (1968 + 1)) / 4) -
                          ((year - (1900 + 1)) / 100) +
                          ((year - (1600 + 1)) / 400);
            } else {
                result += ((year - 1972) / 4) -
                          ((year - 2000) / 100) +
                          ((year - 2000) / 400);
            }
            // Add in the months and days
            result += month_starts[is_leap_year(year)][month-1];
            result += day-1;
            return result;
        } else {
            return DYND_DATE_NA;
        }
    }

    /**
     * Converts the ymd into a days offset from January 1, 1970.
     */
    int32_t to_days() const {
        return to_days(year, month, day);
    }

    /**
     * Sets the year/month/day from a 1970 epoch days offset.
     *
     * \param days  A days offset from January 1, 1970.
     */
    void set_from_days(int days) {
        if (days != DYND_DATE_NA) {
            int yearcalc;
            // Make the days relative to year 0
            days += 719528;
            // To a 400 year cycle
            if (days >= 0) {
                yearcalc = 400 * (days / (400 * 365 + 100 - 4 + 1));
                days = days % (400 * 365 + 100 - 4 + 1);
            } else {
                yearcalc = 400 * ((days - (400 * 365 + 100 - 4)) / (400 * 365 + 100 - 4 + 1));
                days = days % (400 * 365 + 100 - 4 + 1);
                if (days < 0) {
                    days += (400 * 365 + 100 - 4 + 1);
                }
            }
            if (days >= 366) {
                // To a 100 year cycle
                yearcalc += 100 * ((days - 1) / (100 * 365 + 25 - 1));
                days = (days - 1) % (100 * 365 + 25 - 1);
                if (days >= 365) {
                    // To a 4 year cycle
                    yearcalc += 4 * ((days + 1) / (4 * 365 + 1));
                    days = (days + 1) % (4 * 365 + 1);
                    if (days >= 366) {
                        // To a 1 year cycle
                        yearcalc += (days - 1) / 365;
                        days = (days - 1) % 365;
                    }
                }
            }
            // Search for the month
            const int *monthstart = month_starts[is_leap_year(yearcalc)];
            const int *monthfound = std::upper_bound(monthstart + 1, monthstart + 13, days);
            // Set the ymd
            year = yearcalc;
            month = static_cast<int8_t>(monthfound - monthstart);
            day = days - *(monthfound - 1) + 1;
        } else {
            year = 0;
            month = -128;
            day = 0;
        }
    }

    static date_ymd get_current_local_date();

    /**
     * Returns an ndt::type corresponding to the date_ymd structure.
     */
    static const ndt::type& type();
};

} // namespace dynd
 
#endif // _DYND__DATE_UTIL_HPP_