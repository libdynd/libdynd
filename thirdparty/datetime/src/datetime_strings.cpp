/*
 * This file implements string parsing and creation for NumPy datetime.
 *
 * Written by Mark Wiebe (mwwiebe@gmail.com)
 * Copyright (c) 2011 by Enthought, Inc.
 *
 * See LICENSE.txt for the license.
 */

#include <string.h>
#include <ctype.h>
#include <stdio.h>

#include <stdexcept>
#include <sstream>

#include "datetime_main.h"
#include "datetime_strings.h"
#include "datetime_localtime.h"

using namespace std;
using namespace datetime;

/*
 * Parses (almost) standard ISO 8601 date strings. The differences are:
 *
 * + The date "20100312" is parsed as the year 20100312, not as
 *   equivalent to "2010-03-12". The '-' in the dates are not optional.
 * + Only seconds may have a decimal point, with up to 18 digits after it
 *   (maximum attoseconds precision).
 * + Either a 'T' as in ISO 8601 or a ' ' may be used to separate
 *   the date and the time. Both are treated equivalently.
 * + Doesn't (yet) handle the "YYYY-DDD" or "YYYY-Www" formats.
 * + Doesn't handle leap seconds (seconds value has 60 in these cases).
 * + Doesn't handle 24:00:00 as synonym for midnight (00:00:00) tomorrow
 * + Accepts special values "", "NA", "NaT" (not a time), "null" as
 *   missing value tokens.
 *
 * 'str' must be a NULL-terminated string, and 'len' must be its length.
 * 'unit' should contain -1 if the unit is unknown, or the unit
 *      which will be used if it is.
 * 'casting' controls how the detected unit from the string is allowed
 *           to be cast to the 'unit' parameter.
 *
 * 'out' gets filled with the parsed date-time.
 * 'out_abstract' gets set to 1 if the parsed time had no timezone information.
 * 'out_bestunit' gives a suggested unit based on the amount of
 *      resolution provided in the string, or -1 for NA.
 * 'out_missing' gets set to 1 if the parsed value was "NA" or another
 *      missing value token. If the argument provided is NULL, raises
 *      an exception on a missing value.
 *
 * Returns 0 on success, -1 on failure.
 */
void datetime::parse_iso_8601_datetime(const char *str, size_t len,
                    datetime_unit_t unit,
                    bool is_abstract,
                    datetime_conversion_rule_t casting,
                    datetime_fields *out,
                    datetime_unit_t *out_bestunit,
                    bool *out_missing)
{
    int year_leap = 0;
    int i, numdigits;
    const char *substr;
    ptrdiff_t sublen;
    datetime_unit_t bestunit;

    /* Initialize the output to all zeros */
    memset(out, 0, sizeof(datetime_fields));
    out->month = 1;
    out->day = 1;
    
    /*
     * Convert the empty string, case-variants of "NaT",
     * case-variants of "NA", case-variants of "null",
     * and case-variants of "none" to not-a-time.
     */
    if (len <= 0 || (len == 3 &&
                        tolower(str[0]) == 'n' &&
                        tolower(str[1]) == 'a' &&
                        tolower(str[2]) == 't') ||
                    (len == 2 &&
                        tolower(str[0]) == 'n' &&
                        tolower(str[1]) == 'a') ||
                    (len == 4 && tolower(str[0]) == 'n' &&
                        tolower(str[1]) == 'u' &&
                        tolower(str[2]) == 'l' &&
                        tolower(str[3]) == 'l') ||
                    (len == 4 && tolower(str[0]) == 'n' &&
                        tolower(str[1]) == 'o' &&
                        tolower(str[2]) == 'n' &&
                        tolower(str[3]) == 'e')) {
        out->year = DATETIME_DATETIME_NAT;

        if (out_bestunit != NULL) {
            *out_bestunit = datetime_unit_unspecified;
        }
        if (out_missing != NULL) {
            *out_missing = true;
        } else {
            std::stringstream ss;
            ss << "cannot parse \"" << str << "\" as a datetime without missing value support";
            throw std::runtime_error(ss.str());
        }
        return;
    }

    /* Anything else isn't a missing value */
    if (out_missing != NULL) {
        *out_missing = false;
    }

    substr = str;
    sublen = len;

    /* Skip leading whitespace */
    while (sublen > 0 && isspace(*substr)) {
        ++substr;
        --sublen;
    }

    /* Leading '-' sign for negative year */
    if (*substr == '-') {
        ++substr;
        --sublen;
    }

    if (sublen == 0) {
        goto parse_error;
    }

    /* PARSE THE YEAR (digits until the '-' character) */
    out->year = 0;
    while (sublen > 0 && isdigit(*substr)) {
        out->year = 10 * out->year + (*substr - '0');
        ++substr;
        --sublen;
    }

    /* Negate the year if necessary */
    if (str[0] == '-') {
        out->year = -out->year;
    }
    /* Check whether it's a leap-year */
    year_leap = is_leapyear(out->year);

    /* Next character must be a '-' or the end of the string */
    if (sublen == 0) {
        bestunit = datetime_unit_year;
        goto finish;
    }
    else if (*substr == '-') {
        ++substr;
        --sublen;
    }
    else {
        goto parse_error;
    }

    /* Can't have a trailing '-' */
    if (sublen == 0) {
        goto parse_error;
    }

    /* PARSE THE MONTH (2 digits) */
    if (sublen >= 2 && isdigit(substr[0]) && isdigit(substr[1])) {
        out->month = 10 * (substr[0] - '0') + (substr[1] - '0');

        if (out->month < 1 || out->month > 12) {
            std::stringstream ss;
            ss << "month out of range in datetime string \"" << str << "\"";
            throw std::runtime_error(ss.str());
        }
        substr += 2;
        sublen -= 2;
    }
    else {
        goto parse_error;
    }

    /* Next character must be a '-' or the end of the string */
    if (sublen == 0) {
        bestunit = datetime_unit_month;
        goto finish;
    }
    else if (*substr == '-') {
        ++substr;
        --sublen;
    }
    else {
        goto parse_error;
    }

    /* Can't have a trailing '-' */
    if (sublen == 0) {
        goto parse_error;
    }

    /* PARSE THE DAY (2 digits) */
    if (sublen >= 2 && isdigit(substr[0]) && isdigit(substr[1])) {
        out->day = 10 * (substr[0] - '0') + (substr[1] - '0');

        if (out->day < 1 || out->day > days_per_month_table[year_leap][out->month-1]) {
            std::stringstream ss;
            ss << "day out of range in datetime string \"" << str << "\"";
            throw std::runtime_error(ss.str());
        }
        substr += 2;
        sublen -= 2;
    }
    else {
        goto parse_error;
    }

    /* Next character must be a 'T', ' ', or end of string */
    if (sublen == 0) {
        bestunit = datetime_unit_day;
        goto finish;
    }
    else if (*substr != 'T' && *substr != ' ') {
        goto parse_error;
    }
    else {
        ++substr;
        --sublen;
    }

    /* PARSE THE HOURS (2 digits) */
    if (sublen >= 2 && isdigit(substr[0]) && isdigit(substr[1])) {
        out->hour = 10 * (substr[0] - '0') + (substr[1] - '0');

        if (out->hour < 0 || out->hour >= 24) {
            std::stringstream ss;
            ss << "hour out of range in datetime string \"" << str << "\"";
            throw std::runtime_error(ss.str());
        }
        substr += 2;
        sublen -= 2;
    }
    else {
        goto parse_error;
    }

    /* Next character must be a ':' or the end of the string */
    if (sublen > 0 && *substr == ':') {
        ++substr;
        --sublen;
    }
    else {
        bestunit = datetime_unit_hour;
        goto parse_timezone;
    }

    /* Can't have a trailing ':' */
    if (sublen == 0) {
        goto parse_error;
    }

    /* PARSE THE MINUTES (2 digits) */
    if (sublen >= 2 && isdigit(substr[0]) && isdigit(substr[1])) {
        out->min = 10 * (substr[0] - '0') + (substr[1] - '0');

        if (out->min < 0 || out->min >= 60) {
            std::stringstream ss;
            ss << "minute out of range in datetime string \"" << str << "\"";
            throw std::runtime_error(ss.str());
        }
        substr += 2;
        sublen -= 2;
    }
    else {
        goto parse_error;
    }

    /* Next character must be a ':' or the end of the string */
    if (sublen > 0 && *substr == ':') {
        ++substr;
        --sublen;
    }
    else {
        bestunit = datetime_unit_minute;
        goto parse_timezone;
    }

    /* Can't have a trailing ':' */
    if (sublen == 0) {
        goto parse_error;
    }

    /* PARSE THE SECONDS (2 digits) */
    if (sublen >= 2 && isdigit(substr[0]) && isdigit(substr[1])) {
        out->sec = 10 * (substr[0] - '0') + (substr[1] - '0');

        if (out->sec < 0 || out->sec >= 60) {
            std::stringstream ss;
            ss << "second out of range in datetime string \"" << str << "\"";
            throw std::runtime_error(ss.str());
        }
        substr += 2;
        sublen -= 2;
    }
    else {
        goto parse_error;
    }

    /* Next character may be a '.' indicating fractional seconds */
    if (sublen > 0 && *substr == '.') {
        ++substr;
        --sublen;
    }
    else {
        bestunit = datetime_unit_second;
        goto parse_timezone;
    }

    /* PARSE THE MICROSECONDS (0 to 6 digits) */
    numdigits = 0;
    for (i = 0; i < 6; ++i) {
        out->us *= 10;
        if (sublen > 0  && isdigit(*substr)) {
            out->us += (*substr - '0');
            ++substr;
            --sublen;
            ++numdigits;
        }
    }

    if (sublen == 0 || !isdigit(*substr)) {
        if (numdigits > 3) {
            bestunit = datetime_unit_us;
        }
        else {
            bestunit = datetime_unit_ms;
        }
        goto parse_timezone;
    }

    /* PARSE THE TICKS (1 digits) */
    numdigits = 0;
    for (i = 0; i < 1; ++i) {
        out->tick *= 10;
        if (sublen > 0  && isdigit(*substr)) {
            out->tick += (*substr - '0');
            ++substr;
            --sublen;
            ++numdigits;
        }
    }

    bestunit = datetime_unit_tick;

parse_timezone:
    if (sublen == 0) {
        // If there's no timezone, make sure the abstract timezone
        // was requested
        if (!is_abstract && casting != datetime_conversion_relaxed) {
            std::stringstream ss;
            ss << "cannot parse \"" << str << "\" as a datetime with timezone using rule \"" << casting << "\"";
            ss << ", because no timezone was present in the string";
            throw std::runtime_error(ss.str());
        }
        goto finish;
    }

    /* UTC specifier */
    if (*substr == 'Z') {
        /* "Z" means not local */
        if (is_abstract && casting != datetime_conversion_relaxed) {
            std::stringstream ss;
            ss << "cannot parse \"" << str << "\" as an abstract datetime using rule \"" << casting << "\"";
            ss << ", because a timezone was present in the string";
            throw std::runtime_error(ss.str());
        }

        if (sublen == 1) {
            goto finish;
        }
        else {
            ++substr;
            --sublen;
        }
    }
    /* Time zone offset */
    else if (*substr == '-' || *substr == '+') {
        int offset_neg = 0, offset_hour = 0, offset_minute = 0;

        // A time zone offset means it isn't abstract
        if (is_abstract && casting != datetime_conversion_relaxed) {
            std::stringstream ss;
            ss << "cannot parse \"" << str << "\" as an abstract datetime using rule \"" << casting << "\"";
            ss << ", because a timezone was present in the string";
            throw std::runtime_error(ss.str());
        }

        if (*substr == '-') {
            offset_neg = 1;
        }
        ++substr;
        --sublen;

        /* The hours offset */
        if (sublen >= 2 && isdigit(substr[0]) && isdigit(substr[1])) {
            offset_hour = 10 * (substr[0] - '0') + (substr[1] - '0');
            substr += 2;
            sublen -= 2;
            if (offset_hour >= 24) {
                std::stringstream ss;
                ss << "timezone hours offset out of range in datetime string \"" << str << "\"";
                throw std::runtime_error(ss.str());
            }
        }
        else {
            goto parse_error;
        }

        /* The minutes offset is optional */
        if (sublen > 0) {
            /* Optional ':' */
            if (*substr == ':') {
                ++substr;
                --sublen;
            }

            /* The minutes offset (at the end of the string) */
            if (sublen >= 2 && isdigit(substr[0]) && isdigit(substr[1])) {
                offset_minute = 10 * (substr[0] - '0') + (substr[1] - '0');
                substr += 2;
                sublen -= 2;
                if (offset_minute >= 60) {
                    std::stringstream ss;
                    ss << "timezone minutes offset out of range in datetime string \"" << str << "\"";
                    throw std::runtime_error(ss.str());
                }
            }
            else {
                goto parse_error;
            }
        }

        // Apply the timezone offset unless we're parsing
        // for an abstract timezone, in which case we throw it away
        if (!is_abstract) {
            if (offset_neg) {
                offset_hour = -offset_hour;
                offset_minute = -offset_minute;
            }
            out->add_minutes(-60 * offset_hour - offset_minute);
        }
    }

    /* Skip trailing whitespace */
    while (sublen > 0 && isspace(*substr)) {
        ++substr;
        --sublen;
    }

    if (sublen != 0) {
        goto parse_error;
    }

finish:
    if (out_bestunit != NULL) {
        *out_bestunit = bestunit;
    }

    /* Check the casting rule */
    if (unit != datetime_unit_unspecified) {
        // In the abstract time zone case, allow extra zeros at the end
        if (!(is_abstract && out->divisible_by_unit(unit))
                && !satisfies_conversion_rule(unit, bestunit, casting)) {
            std::stringstream ss;
            ss << "cannot parse \"" << str << "\" as a datetime with unit ";
            ss << unit << " and " << casting << " casting";
            throw std::runtime_error(ss.str());
        }
    }

    return;

parse_error:
    std::stringstream ss;
    ss << "error parsing datetime string \"" << str << "\" at position ";
    ss << (int)(substr-str);
    throw std::runtime_error(ss.str());
}

/*
 * Provides a string length to use for converting datetime
 * objects with the given local and unit settings.
 */
int datetime::get_datetime_iso_8601_strlen(datetime_unit_t unit, bool is_abstract, int tzoffset)
{
    int len = 0;

    /* If no unit is provided, return the maximum length */
    if (unit == datetime_unit_unspecified) {
        return DATETIME_MAX_ISO8601_STRLEN;
    }

    switch (unit) {
        case datetime_unit_autodetect:
        case datetime_unit_tick:
            len += 1;  /* "#" */
        case datetime_unit_us:
            len += 3;  /* "###" */
        case datetime_unit_ms:
            len += 4;  /* ".###" */
        case datetime_unit_second:
            len += 3;  /* ":##" */
        case datetime_unit_minute:
            len += 3;  /* ":##" */
        case datetime_unit_hour:
            len += 3;  /* "T##" */
        case datetime_unit_day:
        case datetime_unit_week:
            len += 3;  /* "-##" */
        case datetime_unit_month:
            len += 3;  /* "-##" */
        case datetime_unit_year:
            len += 21; /* 64-bit year */
            break;
        default:
            throw runtime_error("Unrecognized datetime unit");
    }

    if (unit >= datetime_unit_hour && !is_abstract) {
        if (tzoffset != -1) {
            len += 5;  /* "+####" or "-####" */
        }
        else {
            len += 1;  /* "Z" */
        }
    }

    len += 1; /* NULL terminator */

    return len;
}

/*
 * Finds the largest unit whose value is nonzero, and for which
 * the remainder for the rest of the units is zero.
 */
static datetime_unit_t lossless_unit_from_datetime_fields(const datetime_fields *dts)
{
    if (dts->tick != 0) {
        return datetime_unit_tick;
    }
    else if (dts->us % 1000 != 0) {
        return datetime_unit_us;
    }
    else if (dts->us != 0) {
        return datetime_unit_ms;
    }
    else if (dts->sec != 0) {
        return datetime_unit_second;
    }
    else if (dts->min != 0) {
        return datetime_unit_minute;
    }
    else if (dts->hour != 0) {
        return datetime_unit_hour;
    }
    else if (dts->day != 1) {
        return datetime_unit_day;
    }
    else if (dts->month != 1) {
        return datetime_unit_month;
    }
    else {
        return datetime_unit_year;
    }
}

/*
 * Converts an datetime_fields to an (almost) ISO 8601
 * NULL-terminated string. If the string fits in the space exactly,
 * it leaves out the NULL terminator and returns success.
 *
 * The differences from ISO 8601 are the 'NA' missing value string, and
 * the number of year digits is >= 4 instead of strictly 4.
 *
 * 'unit' restricts the output to that unit. Set 'unit' to
 * datetime_unit_unspecified to auto-detect a unit after which
 * all the values are zero.
 *
 * If 'is_abstract' is true, produces a string with no 'Z' or
 * timezone offset at the end.
 *
 *  'tzoffset' is used if 'is_abstract' is false, and 'tzoffset' is
 *  set to a value other than -1. This is a manual override for
 *  the local time zone to use, as an offset in minutes.
 *
 *  'casting' controls whether data loss is allowed by truncating
 *  the data to a coarser unit.
 *
 *  Throws an exception on error.
 */
size_t datetime::make_iso_8601_datetime(const datetime_fields *dts, char *outstr, size_t outlen,
                    datetime_unit_t unit, bool is_abstract, int tzoffset,
                    datetime_conversion_rule_t casting)
{
    datetime_fields dtf_local;
    int timezone_offset = 0;

    char *substr = outstr;
    ptrdiff_t sublen = outlen;
    int tmplen;

    /* Handle NaT, and treat a datetime with generic units as NA */
    if (dts->year == DATETIME_DATETIME_NAT || unit == datetime_unit_unspecified) {
        if (outlen < 2) {
            goto string_too_short;
        }
        outstr[0] = 'N';
        outstr[1] = 'A';
        if (outlen > 2) {
            outstr[2] = '\0';
        }

        return 2;
    }

    /* Automatically detect a good unit */
    if (unit == datetime_unit_autodetect) {
        unit = lossless_unit_from_datetime_fields(dts);
        /*
         * Use at least minutes
         */
        if (unit < datetime_unit_minute) {
            unit = datetime_unit_minute;
        }
    }
    /*
     * Print weeks with the same precision as days.
     *
     * TODO: Could print weeks with YYYY-Www format if the week
     *       epoch is a Monday.
     */
    else if (unit == datetime_unit_week) {
        unit = datetime_unit_day;
    }

    // Use the manually provided tzoffset if applicable
    if (!is_abstract && tzoffset != -1) {
        /* Make a copy of the datetime_fields we can modify */
        dtf_local = *dts;

        /* Set and apply the required timezone offset */
        timezone_offset = tzoffset;
        dtf_local.add_minutes(timezone_offset);

        dts = &dtf_local;
    }

    /*
     * Now the datetimestruct data is in the final form for
     * the string representation, so ensure that the data
     * is being cast according to the casting rule.
     */
    if (casting != datetime_conversion_relaxed) {
        datetime_unit_t unitprec;

        unitprec = lossless_unit_from_datetime_fields(dts);
        if (unitprec > unit) {
            std::stringstream ss;
            ss << "cannot create a string with unit precision " << unit;
            ss << " which has data at precision " << unitprec;
            throw std::runtime_error(ss.str());
        }
    }

    /* YEAR */
#if defined(_MSC_VER) && (_MSC_VER >= 1400)
    tmplen = _snprintf_s(substr, sublen, sublen, "%04I64d", dts->year);
#elif defined(_MSC_VER)
    tmplen = _snprintf(substr, sublen, "%04I64d", dts->year);
#else
    tmplen = snprintf(substr, sublen, "%04lld", dts->year);
#endif
    /* If it ran out of space or there isn't space for the NULL terminator */
    if (tmplen < 0 || tmplen > sublen) {
        goto string_too_short;
    }
    substr += tmplen;
    sublen -= tmplen;

    /* Stop if the unit is years */
    if (unit == datetime_unit_year) {
        if (sublen > 0) {
            *substr = '\0';
        }
        return substr - outstr;
    }

    /* MONTH */
    if (sublen < 1 ) {
        goto string_too_short;
    }
    substr[0] = '-';
    if (sublen < 2 ) {
        goto string_too_short;
    }
    substr[1] = (char)((dts->month / 10) + '0');
    if (sublen < 3 ) {
        goto string_too_short;
    }
    substr[2] = (char)((dts->month % 10) + '0');
    substr += 3;
    sublen -= 3;

    /* Stop if the unit is months */
    if (unit == datetime_unit_month) {
        if (sublen > 0) {
            *substr = '\0';
        }
        return substr - outstr;
    }

    /* DAY */
    if (sublen < 1 ) {
        goto string_too_short;
    }
    substr[0] = '-';
    if (sublen < 2 ) {
        goto string_too_short;
    }
    substr[1] = (char)((dts->day / 10) + '0');
    if (sublen < 3 ) {
        goto string_too_short;
    }
    substr[2] = (char)((dts->day % 10) + '0');
    substr += 3;
    sublen -= 3;

    /* Stop if the unit is days */
    if (unit == datetime_unit_day) {
        if (sublen > 0) {
            *substr = '\0';
        }
        return substr - outstr;
    }

    /* HOUR */
    if (sublen < 1 ) {
        goto string_too_short;
    }
    substr[0] = 'T';
    if (sublen < 2 ) {
        goto string_too_short;
    }
    substr[1] = (char)((dts->hour / 10) + '0');
    if (sublen < 3 ) {
        goto string_too_short;
    }
    substr[2] = (char)((dts->hour % 10) + '0');
    substr += 3;
    sublen -= 3;

    /* Stop if the unit is hours */
    if (unit == datetime_unit_hour) {
        goto add_time_zone;
    }

    /* MINUTE */
    if (sublen < 1 ) {
        goto string_too_short;
    }
    substr[0] = ':';
    if (sublen < 2 ) {
        goto string_too_short;
    }
    substr[1] = (char)((dts->min / 10) + '0');
    if (sublen < 3 ) {
        goto string_too_short;
    }
    substr[2] = (char)((dts->min % 10) + '0');
    substr += 3;
    sublen -= 3;

    /* Stop if the unit is minutes */
    if (unit == datetime_unit_minute) {
        goto add_time_zone;
    }

    /* SECOND */
    if (sublen < 1 ) {
        goto string_too_short;
    }
    substr[0] = ':';
    if (sublen < 2 ) {
        goto string_too_short;
    }
    substr[1] = (char)((dts->sec / 10) + '0');
    if (sublen < 3 ) {
        goto string_too_short;
    }
    substr[2] = (char)((dts->sec % 10) + '0');
    substr += 3;
    sublen -= 3;

    /* Stop if the unit is seconds */
    if (unit == datetime_unit_second) {
        goto add_time_zone;
    }

    /* MILLISECOND */
    if (sublen < 1 ) {
        goto string_too_short;
    }
    substr[0] = '.';
    if (sublen < 2 ) {
        goto string_too_short;
    }
    substr[1] = (char)((dts->us / 100000) % 10 + '0');
    if (sublen < 3 ) {
        goto string_too_short;
    }
    substr[2] = (char)((dts->us / 10000) % 10 + '0');
    if (sublen < 4 ) {
        goto string_too_short;
    }
    substr[3] = (char)((dts->us / 1000) % 10 + '0');
    substr += 4;
    sublen -= 4;

    /* Stop if the unit is milliseconds */
    if (unit == datetime_unit_ms) {
        goto add_time_zone;
    }

    /* MICROSECOND */
    if (sublen < 1 ) {
        goto string_too_short;
    }
    substr[0] = (char)((dts->us / 100) % 10 + '0');
    if (sublen < 2 ) {
        goto string_too_short;
    }
    substr[1] = (char)((dts->us / 10) % 10 + '0');
    if (sublen < 3 ) {
        goto string_too_short;
    }
    substr[2] = (char)(dts->us % 10 + '0');
    substr += 3;
    sublen -= 3;

    /* Stop if the unit is microseconds */
    if (unit == datetime_unit_us) {
        goto add_time_zone;
    }

    /* TICK */
    if (sublen < 1 ) {
        goto string_too_short;
    }
    substr[0] = (char)(dts->tick % 10 + '0');
    substr += 1;
    sublen -= 1;

add_time_zone:
    if (!is_abstract) {
        if (tzoffset != -1) {
            /* Add the +/- sign */
            if (sublen < 1) {
                goto string_too_short;
            }
            if (timezone_offset < 0) {
                substr[0] = '-';
                timezone_offset = -timezone_offset;
            }
            else {
                substr[0] = '+';
            }
            substr += 1;
            sublen -= 1;

            /* Add the timezone offset */
            if (sublen < 1 ) {
                goto string_too_short;
            }
            substr[0] = (char)((timezone_offset / (10*60)) % 10 + '0');
            if (sublen < 2 ) {
                goto string_too_short;
            }
            substr[1] = (char)((timezone_offset / 60) % 10 + '0');
            if (sublen < 3 ) {
                goto string_too_short;
            }
            substr[2] = (char)(((timezone_offset % 60) / 10) % 10 + '0');
            if (sublen < 4 ) {
                goto string_too_short;
            }
            substr[3] = (char)((timezone_offset % 60) % 10 + '0');
            substr += 4;
            sublen -= 4;
        }
        /* UTC "Zulu" time */
        else {
            if (sublen < 1) {
                goto string_too_short;
            }
            substr[0] = 'Z';
            substr += 1;
            sublen -= 1;
        }
    }

    /* Add a NULL terminator, and return */
    if (sublen > 0) {
        substr[0] = '\0';
    }

    return substr - outstr;

string_too_short:
    std::stringstream ss;
    ss << "The string buffer provided for ISO datetime formatting ";
    ss << "was too short, require a length greater than" << outlen;
    throw std::runtime_error(ss.str());
}

std::string datetime::make_iso_8601_datetime(const datetime_fields *dtf,
                    datetime_unit_t unit, bool is_abstract,
                    int tzoffset, datetime_conversion_rule_t casting)
{
    size_t result_size = get_datetime_iso_8601_strlen(unit, is_abstract, tzoffset);
    std::string result(result_size, '\0');
    result.resize(make_iso_8601_datetime(dtf,
                &result[0], result_size, unit, is_abstract, tzoffset, casting));
    return result;
}
