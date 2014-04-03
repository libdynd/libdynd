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
        return ymd.to_days() * DYND_TICKS_PER_DAY + hmst.to_ticks();
    }

    void set_from_ticks(int64_t ticks) {
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
    }

    /**
     * Returns an ndt::type corresponding to the datetime_struct structure.
     */
    static const ndt::type& type();
};

} // namespace dynd

#endif // _DYND__DATETIME_UTIL_HPP_
