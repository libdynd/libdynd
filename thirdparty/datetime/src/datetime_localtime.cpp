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
#if defined(_MSC_VER)
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

/*
 * Wraps `gmtime` functionality for multiple platforms. This
 * converts a time value to a time structure in UTC.
 *
 * Throws an exception on failure.
 */
void get_gmtime(os_time_t *ts, struct tm *tms)
{
#if defined(_MSC_VER)
    if (_gmtime64_s(tms, ts) != 0) {
        throw std::runtime_error("Failed to use '_gmtime64_s' to convert "
                                "to a UTC time");
    }
#else
    if (gmtime_r(ts, tms) == NULL) {
        throw std::runtime_error("Failed to use 'gmtime_r' to convert "
                                "to a UTC time");
    }
#endif
}

/*
 * Wraps `mktime` functionality for multiple platforms. This
 * converts a local time struct to an UTC value.
 *
 * Throws an exception on failure.
 */
static os_time_t
get_mktime(struct tm *tms)
{
    os_time_t ts;
#if defined(_MSC_VER)
    ts = _mktime64(tms);
    if (ts == -1) {
        throw std::runtime_error("Failed to use mktimes' to convert "
                                "local time to UTC");
    }
#else
    ts = mktime(tms);
    if (ts == -1) {
        throw std::runtime_error("Failed to use mktimes' to convert "
                                "local time to UTC");
    }
#endif
    return ts;
}


void datetime::convert_utc_to_local(datetime_fields *out_dtf_local,
                const datetime_fields *dtf_utc, int *out_timezone_offset)
{
    os_time_t rawtime = 0, localrawtime;
    struct tm tm_;
    int64_t year_correction = 0;

    /* Make a copy of the input 'dts' to modify */
    *out_dtf_local = *dtf_utc;

    /* HACK: Use a year < 2038 for later years for small time_t */
    if (sizeof(os_time_t) == 4 && out_dtf_local->year >= 2038) {
        if (is_leapyear(out_dtf_local->year)) {
            /* 2036 is a leap year */
            year_correction = out_dtf_local->year - 2036;
            out_dtf_local->year -= year_correction;
        }
        else {
            /* 2037 is not a leap year */
            year_correction = out_dtf_local->year - 2037;
            out_dtf_local->year -= year_correction;
        }
    }

    /*
     * Convert everything in 'dts' to a time_t, to minutes precision.
     * This is POSIX time, which skips leap-seconds, but because
     * we drop the seconds value from the datetime_fields, everything
     * is ok for this operation.
     */
    rawtime = (os_time_t)out_dtf_local->as_days() * 24 * 60 * 60;
    rawtime += dtf_utc->hour * 60 * 60;
    rawtime += dtf_utc->min * 60;

    /* localtime converts a 'time_t' into a local 'struct tm' */
    get_localtime(&rawtime, &tm_);

    /* Copy back all the values except seconds */
    out_dtf_local->min = tm_.tm_min;
    out_dtf_local->hour = tm_.tm_hour;
    out_dtf_local->day = tm_.tm_mday;
    out_dtf_local->month = tm_.tm_mon + 1;
    out_dtf_local->year = tm_.tm_year + 1900;

    /* Extract the timezone offset that was applied */
    rawtime /= 60;
    localrawtime = (os_time_t)out_dtf_local->as_days() * 24 * 60;
    localrawtime += out_dtf_local->hour * 60;
    localrawtime += out_dtf_local->min;

    *out_timezone_offset = int(localrawtime - rawtime);

    /* Reapply the year 2038 year correction HACK */
    out_dtf_local->year += year_correction;
}

void datetime::convert_local_to_utc(datetime_fields *out_dtf_utc,
                const datetime_fields *dtf_local)
{
    int64_t year_correction = 0;

    /* Make a copy of the input 'dts' to modify */
    *out_dtf_utc = *dtf_local;

    /* HACK: Use a year < 2038 for later years for small time_t */
    if (sizeof(os_time_t) == 4 && out_dtf_utc->year >= 2038) {
        if (is_leapyear(out_dtf_utc->year)) {
            /* 2036 is a leap year */
            year_correction = out_dtf_utc->year - 2036;
            out_dtf_utc->year -= year_correction;
        }
        else {
            /* 2037 is not a leap year */
            year_correction = out_dtf_utc->year - 2037;
            out_dtf_utc->year -= year_correction;
        }
    }

    /*
     * ISO 8601 states to treat date-times without a timezone offset
     * or 'Z' for UTC as local time. The C standard libary functions
     * mktime and gmtime allow us to do this conversion.
     *
     * Only do this timezone adjustment for recent and future years.
     * In this case, "recent" is defined to be 1970 and later, because
     * on MS Windows, mktime raises an error when given an earlier date.
     */
    if (out_dtf_utc->year >= 1970) {
        os_time_t rawtime = 0;
        struct tm tm_;

        tm_.tm_sec = out_dtf_utc->sec;
        tm_.tm_min = out_dtf_utc->min;
        tm_.tm_hour = out_dtf_utc->hour;
        tm_.tm_mday = out_dtf_utc->day;
        tm_.tm_mon = out_dtf_utc->month - 1;
        tm_.tm_year = int(out_dtf_utc->year - 1900);
        tm_.tm_isdst = -1;

        /* mktime converts a local 'struct tm' into a time_t */
        rawtime = get_mktime(&tm_);

        /* gmtime converts a 'time_t' into a UTC 'struct tm' */
        get_gmtime(&rawtime, &tm_);
        out_dtf_utc->sec = tm_.tm_sec;
        out_dtf_utc->min = tm_.tm_min;
        out_dtf_utc->hour = tm_.tm_hour;
        out_dtf_utc->day = tm_.tm_mday;
        out_dtf_utc->month = tm_.tm_mon + 1;
        out_dtf_utc->year = tm_.tm_year + 1900;
    }

    /* Reapply the year 2038 year correction HACK */
    out_dtf_utc->year += year_correction;
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