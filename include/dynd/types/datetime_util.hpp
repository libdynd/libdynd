//
// Copyright (C) 2011-15 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#pragma once

#include <dynd/config.hpp>
#include <dynd/types/date_util.hpp>
#include <dynd/types/time_util.hpp>
#include <dynd/typed_data_assign.hpp>

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

struct DYND_API datetime_struct {
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
     * Sets the datetime from a string. Accepts a wide variety of
     * inputs, and interprets ambiguous formats like MM/DD/YY vs DD/MM/YY
     * using the monthfirst parameter.
     *
     * \param s  Datetime string.
     * \param ambig  Order to use for ambiguous cases like "01/02/03"
     *               or "01/02/1995".
     * \param century_window  Number describing how to handle dates with
     *                        two digit years. Values 1 to 99 mean to use
     *                        a sliding window starting that many years back.
     *                        Values 1000 and higher mean to use a fixed window
     *                        starting at the year given. The value 0 means to
     *                        disallow two digit years.
     * \param errmode  The error mode to use for how strict to be when converting
     *                 values. In mode assign_error_nocheck, tries to do a "best interpretation"
     *                 conversion.
     *
     * \returns  The time zone, if any, found in the string
     */
    inline std::string
    set_from_str(const std::string &s,
                 date_parse_order_t ambig = date_parse_no_ambig,
                 int century_window = 70,
                 assign_error_mode errmode = assign_error_fractional)
    {
        const char *tz_begin = NULL, *tz_end = NULL;
        set_from_str(s.data(), s.data() + s.size(), ambig, century_window,
                     errmode, tz_begin, tz_end);
        return std::string(tz_begin, tz_end);
    }

    void set_from_str(const char *begin, const char *end,
                      date_parse_order_t ambig, int century_window,
                      assign_error_mode errmode, const char *&out_tz_begin,
                      const char *&out_tz_end);

    /**
     * Returns an ndt::type corresponding to the datetime_struct structure.
     */
    static const ndt::type& type();
};

} // namespace dynd
