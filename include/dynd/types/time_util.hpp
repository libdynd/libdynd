//
// Copyright (C) 2011-15 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#pragma once

#include <algorithm>
#include <limits>
#include <iostream>

#include <dynd/config.hpp>

#define DYND_TIME_NA (std::numeric_limits<int64_t>::min())

#define DYND_NANOSECONDS_PER_TICK (100LL)

#define DYND_TICKS_PER_MICROSECOND (10LL)
#define DYND_TICKS_PER_MILLISECOND (10000LL)
#define DYND_TICKS_PER_SECOND (10000000LL)
#define DYND_TICKS_PER_MINUTE (60LL * 10000000LL)
#define DYND_TICKS_PER_HOUR (60LL * 60LL * 10000000LL)
// DYND_TICKS_PER_DAY is defined in date_util.hpp

namespace dynd {

namespace ndt {
    class type;
} // namespace ndt

struct DYND_API time_hmst {
    int8_t hour;
    int8_t minute;
    int8_t second;
    int32_t tick;

    static inline bool is_valid(int hour, int minute, int second, int tick) {
        return hour >= 0 && hour < 24 && minute >= 0 && minute < 60 &&
               second >= 0 && second <= 60 && // leap second can == 60
               tick >= 0 && tick < DYND_TICKS_PER_SECOND;
    }

    inline bool is_valid() const {
        return is_valid(hour, minute, second, tick);
    }

    inline bool is_na() const {
        return hour == -128;
    }

    /**
     * Converts the hmst to a ticks offset from midnight.
     */
    static int64_t to_ticks(int hour, int minute, int second, int tick);

    /**
     * Converts the hmst to a ticks offset from midnight.
     */
    int64_t to_ticks() const {
        return to_ticks(hour, minute, second, tick);
    }

    /**
     * Converts the hmst into an ISO 8601 string, or "" if it
     * is invalid. Uses the shortest representation, except for
     * having at least minutes. ##:##:##.#####.
     */
    static std::string to_str(int hour, int minute, int second, int tick);

    /**
     * Converts the hmst into an ISO 8601 string, or "" if it
     * is invalid. Uses the shortest representation, except for
     * having at least minutes. ##:##:##.#####.
     */
    inline std::string to_str() const {
        return to_str(hour, minute, second, tick);
    }

    /**
     * Sets the hmst from a ticks value.
     *
     * \param ticks  A ticks offset from midnight.
     */
    void set_from_ticks(int64_t ticks);

    /**
     * Sets the hmst to NA.
     */
    inline void set_to_zero() {
        memset(this, 0, sizeof(*this));
    }

    /**
     * Sets the hmst to NA.
     */
    inline void set_to_na() {
        hour = -128;
    }
    void set_from_str(const char *begin, const char *end,
                      const char *&out_tz_begin, const char *&out_tz_end);
    /**
     * Sets the hmst from a string.
     *
     * \param s  Date string.
     *
     * \returns  The time zone, if any, found in the string
     */
    inline std::string set_from_str(const std::string& s) {
        const char *tz_begin = NULL, *tz_end = NULL;
        set_from_str(s.data(), s.data() + s.size(), tz_begin, tz_end);
        return std::string(tz_begin, tz_end);
    }


    /**
     * Returns the current time in the local time zone.
     */
    static time_hmst get_current_local_time();

    /**
     * Returns an ndt::type corresponding to the time_hmst structure.
     */
    static const ndt::type& type();
};

} // namespace dynd
