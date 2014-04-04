//
// Copyright (C) 2011-14 Mark Wiebe, DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#ifndef _DYND__DATETIME_UTIL_HPP_
#define _DYND__DATETIME_UTIL_HPP_

#include <dynd/config.hpp>
#include <dynd/types/date_util.hpp>
#include <dynd/types/time_util.hpp>

#define DYND_DATETIME_NA (std::numeric_limits<int64_t>::min())

namespace dynd {

enum datetime_tz_t {
    // The abstract time zone is disconnected from a real physical
    // time. It is a time based on an abstract calendar.
    tz_abstract,
    // The UTC time zone. This cannot represent added leap seconds,
    // as it is based on the POSIX approach.
    tz_utc,
    // TODO: A "time zone" based on TAI atomic clock time. This can represent
    // the leap seconds UTC cannot, but converting to/from UTC is not
    // lossless.
    //tz_tai,
    // TODO: other time zones based on a time zone database
    // tz_other
};

struct datetime_struct {
    date_ymd ymd;
    time_hmst hmst;

    inline bool is_valid() const {
        return ymd.is_valid() && hmst.is_valid();
    }

    inline bool is_na() const {
        return ymd.is_na();
    }

    int64_t to_ticks() const {
        if (is_valid()) {
            return ymd.to_days() * DYND_TICKS_PER_DAY + hmst.to_ticks();
        } else {
            return DYND_DATETIME_NA;
        }
    }

    std::string to_str() const;

    void set_from_ticks(int64_t ticks) {
        if (ticks != DYND_DATETIME_NA) {
            int32_t days;
            if (ticks >= 0) {
                days = static_cast<int32_t>(ticks / DYND_TICKS_PER_DAY);
                ticks = ticks % DYND_TICKS_PER_DAY;
            } else {
                days = static_cast<int32_t>((ticks - (DYND_TICKS_PER_DAY - 1)) / DYND_TICKS_PER_DAY);
                ticks = ticks % DYND_TICKS_PER_DAY;
                if (ticks < 0) {
                    ticks += DYND_TICKS_PER_DAY;
                }
            }
            ymd.set_from_days(days);
            hmst.set_from_ticks(ticks);
        } else {
            set_to_na();
        }
    }

    void set_to_na() {
        ymd.set_to_na();
    }

    /**
     * Sets the datetime from a string, which is a date followed by a time.
     * Two digit years are handled with a sliding window starting 70 years
     * ago.
     *
     * \param s  Datetime string.
     */
    void set_from_str(const std::string& s);

    /**
     * Sets the datetime from a string. Accepts a wide variety of
     * inputs, and interprets ambiguous formats like MM/DD/YY vs DD/MM/YY
     * using the monthfirst parameter.
     *
     * \param s  Datetime string.
     * \param monthfirst  If true, expect MM/DD/YY, otherwise expect DD/MM/YY.
     * \param allow_2digit_year  If true, uses a sliding window starting 70 years ago
     *                           to resolve two digit years.  Defaults to true.
     */
    void set_from_str(const std::string &s, bool monthfirst,
                      bool allow_2digit_year = true);

    /**
     * Returns an ndt::type corresponding to the datetime_struct structure.
     */
    static const ndt::type& type();
};

} // namespace dynd

#endif // _DYND__DATETIME_UTIL_HPP_
