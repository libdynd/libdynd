//
// Copyright (C) 2011-14 Mark Wiebe, DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#ifndef _DYND__DATETIME_UTIL_HPP_
#define _DYND__DATETIME_UTIL_HPP_

#include <dynd/config.hpp>
#include <dynd/types/date_util.hpp>

#define DYND_DATETIME_NA (std::numeric_limits<int64_t>::min())

#define DYND_TICKS_PER_MICROSECOND (10LL)
#define DYND_TICKS_PER_MILLISECOND (10000LL)
#define DYND_TICKS_PER_SECOND (10000000LL)
#define DYND_TICKS_PER_MINUTE (60LL * 10000000LL)
#define DYND_TICKS_PER_HOUR (60LL * 60LL * 10000000LL)
// DYND_TICKS_PER_DAY is defined in date_util.hpp

namespace dynd {

struct datetime_struct {
    date_ymd ymd;
    int8_t hour;
    int8_t minute;
    int8_t second;
    int32_t tick;

    inline bool is_valid() const {
        return ymd.is_valid() &&
            hour >= 0 && hour < 24 &&
            minute >= 0 && minute < 60 &&
            second >= 0 && second <= 60 &&  // leap second can == 60
            tick >= 0 && tick < 10000000;
    }

    inline bool is_na() const {
        return ymd.is_na();
    }

    int64_t to_ticks() const {
        return (((((ymd.to_days() * 24LL) + hour) * 60LL + minute) * 60LL +
                 second) * 10000000LL) + tick;
    }

    void set_from_ticks(int64_t ticks) {
        tick = ticks % 10000000LL;
        ticks = ticks / 10000000LL;
        second = ticks % 60;
        ticks = ticks / 60;
        minute = ticks % 60;
        ticks = ticks / 60;
        hour = ticks % 24;
        ticks = ticks / 24;
        ymd.set_from_days(static_cast<int32_t>(ticks));
    }

    /**
     * Returns an ndt::type corresponding to the datetime_struct structure.
     */
    static const ndt::type& type();
};

} // namespace dynd

#endif // _DYND__DATETIME_UTIL_HPP_
