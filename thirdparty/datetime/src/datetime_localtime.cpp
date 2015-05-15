/*
 * This file implements string parsing and creation for NumPy datetime.
 *
 * Written by Mark Wiebe (mwwiebe@gmail.com)
 * Copyright (c) 2011 by Enthought, Inc.
 *
 * See LICENSE.txt for the license.
 */

#include <stdexcept>
#include <time.h>

#include "datetime_localtime.h"

using namespace datetime;

/* Platform-specific time_t typedef */
#if defined(_MSC_VER)
typedef __time64_t os_time_t;
#else
typedef time_t os_time_t;
#endif

/*
 * Wraps `localtime` functionality for multiple platforms. This
 * converts a time value to a time structure in the local timezone.
 *
 * Throws an exception on failure.
 */
void get_localtime(os_time_t *ts, struct tm *tms)
{
#if defined(_WIN32)
    if (_localtime64_s(tms, ts) != 0) {
        throw std::runtime_error("Failed to use '_localtime64_s' to convert "
                                "to a local time");
    }
#else
    if (localtime_r(ts, tms) == NULL) {
        throw std::runtime_error("Failed to use 'localtime_r' to convert "
                                "to a local time");
    }
#endif
}

void datetime::fill_current_local_date(datetime_fields *out)
{
    os_time_t rawtime = 0;
    struct tm tm_;

#if defined(_MSC_VER)
    _time64(&rawtime);
#else
    time(&rawtime);
#endif
    get_localtime(&rawtime, &tm_);
    out->year = tm_.tm_year + 1900;
    out->month = tm_.tm_mon + 1;
    out->day = tm_.tm_mday;
}

void datetime::fill_current_local_date(date_ymd *out)
{
    os_time_t rawtime = 0;
    struct tm tm_;

#if defined(_MSC_VER)
    _time64(&rawtime);
#else
    time(&rawtime);
#endif
    get_localtime(&rawtime, &tm_);
    out->year = tm_.tm_year + 1900;
    out->month = tm_.tm_mon + 1;
    out->day = tm_.tm_mday;
}

datetime_val_t datetime::get_current_utc_datetime_seconds()
{
    os_time_t rawtime = 0;

#if defined(_MSC_VER)
    _time64(&rawtime);
#else
    time(&rawtime);
#endif

    return rawtime;
}

