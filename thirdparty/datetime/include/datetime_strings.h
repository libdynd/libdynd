#ifndef DATETIME_STRINGS_H
#define DATETIME_STRINGS_H

#include <string>

#include "datetime_main.h"

namespace datetime {

/*
 * Upper bound on the length of a DATETIME ISO 8601 string
 *   YEAR: 21 (64-bit year)
 *   MONTH: 3
 *   DAY: 3
 *   HOURS: 3
 *   MINUTES: 3
 *   SECONDS: 3
 *   ATTOSECONDS: 1 + 3*6
 *   TIMEZONE: 5
 *   NULL TERMINATOR: 1
 */
#define DATETIME_MAX_ISO8601_STRLEN (21+3*5+1+3*6+6+1)

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
 * 'is_abstract' should be true if no timezone specification is
 *      desired, false otherwise.
 * 'casting' controls how the detected unit from the string is allowed
 *           to be cast to the 'unit' parameter.
 *
 * 'out' gets filled with the parsed date-time.
 * 'out_bestunit' gives a suggested unit based on the amount of
 *      resolution provided in the string, or -1 for NA.
 * 'out_missing' gets set to 1 if the parsed value was "NA" or another
 *      missing value token. If the argument provided is NULL, raises
 *      an exception on a missing value.
 *
 * Returns 0 on success, -1 on failure.
 */
void parse_iso_8601_datetime(const char *str, size_t len,
                    datetime_unit_t unit,
                    bool is_abstract,
                    datetime_conversion_rule_t casting,
                    datetime_fields *out,
                    datetime_unit_t *out_bestunit,
                    bool *out_missing=NULL);

/*
 * Provides a string length to use for converting datetime
 * objects with the given local and unit settings.
 */
int get_datetime_iso_8601_strlen(datetime_unit_t unit, bool is_abstract, int tzoffset);

/**
 * Simplified interface to date parsing.
 *
 * \param str  The input date string.
 * \param unit  The unit to parse it as (must be a date unit).
 * \param casting  The casting rule to apply to the parsing.
 */
inline date_val_t parse_iso_8601_date(const std::string& str, datetime_unit_t unit,
                    datetime_conversion_rule_t casting)
{
    datetime_fields fld;
    bool out_missing = false;
    parse_iso_8601_datetime(str.c_str(), str.length(), unit, true, casting,
                    &fld, NULL, &out_missing);
    return fld.as_date_val(unit);
}

/**
 * Simplified interface to datetime parsing.
 *
 * \param str  The input datetime string.
 * \param unit  The unit to parse it as (must be a date unit).
 * \param casting  The casting rule to apply to the parsing.
 */
inline datetime_val_t parse_iso_8601_datetime(const std::string& str, datetime_unit_t unit,
                    bool is_abstract, datetime_conversion_rule_t casting)
{
    datetime_fields fld;
    bool out_missing = false;
    parse_iso_8601_datetime(str.c_str(), str.length(), unit, is_abstract, casting,
                    &fld, NULL, &out_missing);
    return fld.as_ticks();
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
 * datetime_unit_autodetect to auto-detect a unit after which
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
size_t make_iso_8601_datetime(const datetime_fields *dts, char *outstr, size_t outlen,
                    datetime_unit_t unit, bool is_abstract, int tzoffset,
                    datetime_conversion_rule_t casting);

std::string make_iso_8601_datetime(const datetime_fields *dtf,
                    datetime_unit_t unit = datetime_unit_unspecified, bool is_abstract = true,
                    int tzoffset = -1, datetime_conversion_rule_t casting = datetime_conversion_strict);

inline std::string make_iso_8601_datetime(datetime_val_t datetime, datetime_unit_t unit, bool is_abstract) {
    datetime_fields dtf;
    dtf.set_from_datetime_val(datetime, unit);
    return make_iso_8601_datetime(&dtf, datetime::datetime_unit_autodetect, is_abstract);
}

inline std::string make_iso_8601_date(date_val_t date, datetime_unit_t unit = datetime_unit_day) {
    datetime_fields dtf;
    dtf.set_from_date_val(date, unit);
    return make_iso_8601_datetime(&dtf, unit, true);
}

} // namespace datetime

#endif // DATETIME_STRINGS_H
