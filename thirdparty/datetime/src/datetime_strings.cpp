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

#include <stdexcept>
#include <sstream>

#include "datetime_main.h"
#include "datetime_strings.h"
#include "datetime_localtime.h"

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
 * + Accepts special values "NaT" (not a time), "Today", (current
 *   day according to local time) and "Now" (current time in UTC).
 *
 * 'str' must be a NULL-terminated string, and 'len' must be its length.
 * 'unit' should contain -1 if the unit is unknown, or the unit
 *      which will be used if it is.
 * 'casting' controls how the detected unit from the string is allowed
 *           to be cast to the 'unit' parameter.
 *
 * 'out' gets filled with the parsed date-time.
 * 'out_local' gets set to 1 if the parsed time was in local time,
 *      to 0 otherwise. The values 'now' and 'today' don't get counted
 *      as local, and neither do UTC +/-#### timezone offsets, because
 *      they aren't using the computer's local timezone offset.
 * 'out_bestunit' gives a suggested unit based on the amount of
 *      resolution provided in the string, or -1 for NaT.
 * 'out_special' gets set to 1 if the parsed time was 'today',
 *      'now', or ''/'NaT'. For 'today', the unit recommended is
 *      'D', for 'now', the unit recommended is 's', and for 'NaT'
 *      the unit recommended is 'Y'.
 *
 * Returns 0 on success, -1 on failure.
 */
void datetime::parse_iso_8601_datetime(char *str, size_t len,
                    datetime_unit_t unit,
                    datetime_conversion_rule_t casting,
                    datetime_fields *out,
                    bool *out_local,
                    datetime_unit_t *out_bestunit,
                    bool *out_special)
{
    int year_leap = 0;
    int i, numdigits;
    char *substr;
    ptrdiff_t sublen;
    datetime_unit_t bestunit;

    /* Initialize the output to all zeros */
    memset(out, 0, sizeof(datetime_fields));
    out->month = 1;
    out->day = 1;

    /*
     * Convert the empty string and case-variants of "NaT" to not-a-time.
     */
    if (len <= 0 || (len == 3 &&
                        tolower(str[0]) == 'n' &&
                        tolower(str[1]) == 'a' &&
                        tolower(str[2]) == 't')) {
        out->year = DATETIME_DATETIME_NAT;

        /*
         * Indicate that this was a special value, and
         * that no unit can be recommended.
         */
        if (out_local != NULL) {
            *out_local = false;
        }
        if (out_bestunit != NULL) {
            *out_bestunit = datetime_unit_unspecified;
        }
        if (out_special != NULL) {
            *out_special = true;
        }
    }

    /*
     * The string "today" means take today's date in local time, and
     * convert it to a date representation. This date representation, if
     * forced into a time unit, will be at midnight UTC.
     * This is perhaps a little weird, but done so that the
     * 'datetime64[D]' type produces the date you expect, rather than
     * switching to an adjacent day depending on the current time and your
     * timezone.
     */
    if (len == 5 && tolower(str[0]) == 't' &&
                    tolower(str[1]) == 'o' &&
                    tolower(str[2]) == 'd' &&
                    tolower(str[3]) == 'a' &&
                    tolower(str[4]) == 'y') {
        fill_current_local_date(out);

        bestunit = datetime_unit_day;

        /*
         * Indicate that this was a special value, and
         * is a date (unit 'D').
         */
        if (out_local != NULL) {
            *out_local = false;
        }
        if (out_bestunit != NULL) {
            *out_bestunit = bestunit;
        }
        if (out_special != NULL) {
            *out_special = true;
        }

        /* Check the casting rule */
        if (unit != datetime_unit_unspecified &&
                    !satisfies_conversion_rule(unit, bestunit, casting)) {
            std::stringstream ss;
            ss << "cannot parse \"" << str << "\" as a datetime with unit ";
            ss << unit << " and " << casting << " casting";
            throw std::runtime_error(ss.str());
        }

        return;
    }

    /* The string "now" resolves to the current UTC time */
    if (len == 3 && tolower(str[0]) == 'n' &&
                    tolower(str[1]) == 'o' &&
                    tolower(str[2]) == 'w') {
        datetime_val_t rawtime = get_current_utc_datetime_seconds();

        bestunit = datetime_unit_second;

        /*
         * Indicate that this was a special value.
         */
        if (out_local != NULL) {
            *out_local = false;
        }
        if (out_bestunit != NULL) {
            *out_bestunit = bestunit;
        }
        if (out_special != NULL) {
            *out_special = true;
        }

        /* Check the casting rule */
        if (unit != datetime_unit_unspecified &&
                    !satisfies_conversion_rule(unit, bestunit, casting)) {
            std::stringstream ss;
            ss << "cannot parse \"" << str << "\" as a datetime with unit ";
            ss << unit << " and " << casting << " casting";
            throw std::runtime_error(ss.str());
        }

        out->set_from_datetime_val(rawtime, bestunit);
        return;
    }

    /* Anything else isn't a special value */
    if (out_special != NULL) {
        *out_special = false;
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
        if (out_local != NULL) {
            *out_local = false;
        }
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
        if (out_local != NULL) {
            *out_local = false;
        }
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
        if (out_local != NULL) {
            *out_local = false;
        }
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

    /* PARSE THE PICOSECONDS (0 to 6 digits) */
    numdigits = 0;
    for (i = 0; i < 6; ++i) {
        out->ps *= 10;
        if (sublen > 0 && isdigit(*substr)) {
            out->ps += (*substr - '0');
            ++substr;
            --sublen;
            ++numdigits;
        }
    }

    if (sublen == 0 || !isdigit(*substr)) {
        if (numdigits > 3) {
            bestunit = datetime_unit_ps;
        }
        else {
            bestunit = datetime_unit_ns;
        }
        goto parse_timezone;
    }

    /* PARSE THE ATTOSECONDS (0 to 6 digits) */
    numdigits = 0;
    for (i = 0; i < 6; ++i) {
        out->as *= 10;
        if (sublen > 0 && isdigit(*substr)) {
            out->as += (*substr - '0');
            ++substr;
            --sublen;
            ++numdigits;
        }
    }

    if (numdigits > 3) {
        bestunit = datetime_unit_as;
    }
    else {
        bestunit = datetime_unit_fs;
    }

parse_timezone:
    if (sublen == 0) {
        convert_local_to_utc(out, out);

        /* Since neither "Z" nor a time-zone was specified, it's local */
        if (out_local != NULL) {
            *out_local = true;
        }

        goto finish;
    }

    /* UTC specifier */
    if (*substr == 'Z') {
        /* "Z" means not local */
        if (out_local != NULL) {
            *out_local = false;
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

        /*
         * Since "local" means local with respect to the current
         * machine, we say this is non-local.
         */
        if (out_local != NULL) {
            *out_local = false;
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

        /* Apply the time zone offset */
        if (offset_neg) {
            offset_hour = -offset_hour;
            offset_minute = -offset_minute;
        }
        out->add_minutes(-60 * offset_hour - offset_minute);
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
    if (unit != datetime_unit_unspecified &&
                !satisfies_conversion_rule(unit, bestunit, casting)) {
        std::stringstream ss;
        ss << "cannot parse \"" << str << "\" as a datetime with unit ";
        ss << unit << " and " << casting << " casting";
        throw std::runtime_error(ss.str());
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
int datetime::get_datetime_iso_8601_strlen(bool local, datetime_unit_t unit)
{
    int len = 0;

    /* If no unit is provided, return the maximum length */
    if (unit == datetime_unit_unspecified) {
        return DATETIME_MAX_ISO8601_STRLEN;
    }

    switch (unit) {
        case datetime_unit_as:
            len += 3;  /* "###" */
        case datetime_unit_fs:
            len += 3;  /* "###" */
        case datetime_unit_ps:
            len += 3;  /* "###" */
        case datetime_unit_ns:
            len += 3;  /* "###" */
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
    }

    if (unit >= datetime_unit_hour) {
        if (local) {
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
static datetime_unit_t
lossless_unit_from_datetimestruct(datetime_fields *dts)
{
    if (dts->as % 1000 != 0) {
        return datetime_unit_as;
    }
    else if (dts->as != 0) {
        return datetime_unit_fs;
    }
    else if (dts->ps % 1000 != 0) {
        return datetime_unit_ps;
    }
    else if (dts->ps != 0) {
        return datetime_unit_ns;
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
 * The differences from ISO 8601 are the 'NaT' string, and
 * the number of year digits is >= 4 instead of strictly 4.
 *
 * If 'local' is non-zero, it produces a string in local time with
 * a +-#### timezone offset, otherwise it uses timezone Z (UTC).
 *
 * 'unit' restricts the output to that unit. Set 'unit' to
 * datetime_unit_unspecified to auto-detect a unit after which
 * all the values are zero.
 *
 *  'tzoffset' is used if 'local' is enabled, and 'tzoffset' is
 *  set to a value other than -1. This is a manual override for
 *  the local time zone to use, as an offset in minutes.
 *
 *  'casting' controls whether data loss is allowed by truncating
 *  the data to a coarser unit. This interacts with 'local', slightly,
 *  in order to form a date unit string as a local time, the casting
 *  must be unsafe.
 *
 *  Throws an exception on error.
 */
void datetime::make_iso_8601_datetime(datetime_fields *dts, char *outstr, int outlen,
                    bool local, datetime_unit_t unit, int tzoffset,
                    datetime_conversion_rule_t casting)
{
    datetime_fields dtf_local;
    int timezone_offset = 0;

    char *substr = outstr, sublen = outlen;
    int tmplen;

    /* Handle NaT, and treat a datetime with generic units as NaT */
    if (dts->year == DATETIME_DATETIME_NAT || unit == datetime_unit_unspecified) {
        if (outlen < 3) {
            goto string_too_short;
        }
        outstr[0] = 'N';
        outstr[1] = 'a';
        outstr[2] = 'T';
        if (outlen > 3) {
            outstr[3] = '\0';
        }

        return;
    }

    /*
     * Only do local time within a reasonable year range. The years
     * earlier than 1970 are not made local, because the Windows API
     * raises an error when they are attempted. For consistency, this
     * restriction is applied to all platforms.
     *
     * Note that this only affects how the datetime becomes a string.
     * The result is still completely unambiguous, it only means
     * that datetimes outside this range will not include a time zone
     * when they are printed.
     */
    if ((dts->year < 1970 || dts->year >= 10000) && tzoffset == -1) {
        local = 0;
    }

    /* Automatically detect a good unit */
    if (unit == -1) {
        unit = lossless_unit_from_datetimestruct(dts);
        /*
         * If there's a timezone, use at least minutes precision,
         * and never split up hours and minutes by default
         */
        if ((unit < datetime_unit_minute && local) || unit == datetime_unit_hour) {
            unit = datetime_unit_minute;
        }
        /* Don't split up dates by default */
        else if (unit < datetime_unit_day) {
            unit = datetime_unit_day;
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

    /* Use the C API to convert from UTC to local time */
    if (local && tzoffset == -1) {
        convert_utc_to_local(&dtf_local, dts, &timezone_offset);

        /* Set dts to point to our local time instead of the UTC time */
        dts = &dtf_local;
    }
    /* Use the manually provided tzoffset */
    else if (local) {
        /* Make a copy of the datetime_fields we can modify */
        dtf_local = *dts;
        dts = &dtf_local;

        /* Set and apply the required timezone offset */
        timezone_offset = tzoffset;
        dts->add_minutes(timezone_offset);
    }

    /*
     * Now the datetimestruct data is in the final form for
     * the string representation, so ensure that the data
     * is being cast according to the casting rule.
     */
    if (casting != datetime_conversion_relaxed) {
        /* Producing a date as a local time is always 'unsafe' */
        if (unit <= datetime_unit_day && local) {
            throw std::runtime_error("cannot create a local timezone-based date string in strict conversion mode");
        }
        /* Only 'relaxed' allows data loss */
        else {
            datetime_unit_t unitprec;

            unitprec = lossless_unit_from_datetimestruct(dts);
            if (unitprec > unit) {
                std::stringstream ss;
                ss << "cannot create a string with unit precision " << unit;
                ss << " which has data at precision " << unitprec;
                throw std::runtime_error(ss.str());
            }
        }
    }

    /* YEAR */
#if defined(_MSC_VER)
    tmplen = _snprintf(substr, sublen, "%04I64d", dts->year);
#elif defined(__APPLE__) || defined(__FreeBSD__)
    tmplen = snprintf(substr, sublen, "%04Ld", dts->year);
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
        return;
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
        return;
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
        return;
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

    /* NANOSECOND */
    if (sublen < 1 ) {
        goto string_too_short;
    }
    substr[0] = (char)((dts->ps / 100000) % 10 + '0');
    if (sublen < 2 ) {
        goto string_too_short;
    }
    substr[1] = (char)((dts->ps / 10000) % 10 + '0');
    if (sublen < 3 ) {
        goto string_too_short;
    }
    substr[2] = (char)((dts->ps / 1000) % 10 + '0');
    substr += 3;
    sublen -= 3;

    /* Stop if the unit is nanoseconds */
    if (unit == datetime_unit_ns) {
        goto add_time_zone;
    }

    /* PICOSECOND */
    if (sublen < 1 ) {
        goto string_too_short;
    }
    substr[0] = (char)((dts->ps / 100) % 10 + '0');
    if (sublen < 2 ) {
        goto string_too_short;
    }
    substr[1] = (char)((dts->ps / 10) % 10 + '0');
    if (sublen < 3 ) {
        goto string_too_short;
    }
    substr[2] = (char)(dts->ps % 10 + '0');
    substr += 3;
    sublen -= 3;

    /* Stop if the unit is picoseconds */
    if (unit == datetime_unit_ps) {
        goto add_time_zone;
    }

    /* FEMTOSECOND */
    if (sublen < 1 ) {
        goto string_too_short;
    }
    substr[0] = (char)((dts->as / 100000) % 10 + '0');
    if (sublen < 2 ) {
        goto string_too_short;
    }
    substr[1] = (char)((dts->as / 10000) % 10 + '0');
    if (sublen < 3 ) {
        goto string_too_short;
    }
    substr[2] = (char)((dts->as / 1000) % 10 + '0');
    substr += 3;
    sublen -= 3;

    /* Stop if the unit is femtoseconds */
    if (unit == datetime_unit_fs) {
        goto add_time_zone;
    }

    /* ATTOSECOND */
    if (sublen < 1 ) {
        goto string_too_short;
    }
    substr[0] = (char)((dts->as / 100) % 10 + '0');
    if (sublen < 2 ) {
        goto string_too_short;
    }
    substr[1] = (char)((dts->as / 10) % 10 + '0');
    if (sublen < 3 ) {
        goto string_too_short;
    }
    substr[2] = (char)(dts->as % 10 + '0');
    substr += 3;
    sublen -= 3;

add_time_zone:
    if (local) {
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

    /* Add a NULL terminator, and return */
    if (sublen > 0) {
        substr[0] = '\0';
    }

    return;

string_too_short:
    std::stringstream ss;
    ss << "The string buffer provided for ISO datetime formatting ";
    ss << "was too short, require a length greater than" << outlen;
    throw std::runtime_error(ss.str());
}
