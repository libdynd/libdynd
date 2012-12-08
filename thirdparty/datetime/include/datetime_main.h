#ifndef DATETIME_MAIN_H
#define DATETIME_MAIN_H

#include <iostream>

#include "datetime_types.h"

namespace datetime {

enum datetime_unit_t {
    datetime_unit_unspecified,
    datetime_unit_year,
    datetime_unit_month,
    datetime_unit_week,
    datetime_unit_day,
    datetime_unit_hour,
    datetime_unit_minute,
    datetime_unit_second,
    datetime_unit_ms,
    datetime_unit_us,
    datetime_unit_ns,
    datetime_unit_ps,
    datetime_unit_fs,
    datetime_unit_as
};

enum datetime_conversion_rule_t {
    datetime_conversion_exact,
    datetime_conversion_strict,
    datetime_conversion_relaxed
};

std::ostream& operator<<(std::ostream& o, datetime_unit_t unit);

std::ostream& operator<<(std::ostream& o, datetime_conversion_rule_t rule);

/*
 * This structure contains an exploded view of a date-time value.
 * NaT is represented by year == DATETIME_DATE_NAT.
 */
struct datetime_fields {
    int64_t year;
    int32_t month, day, hour, min, sec, us, ps, as;

    datetime_fields()
        : year(0), month(0), day(0), hour(0),
          min(0), sec(0), us(0), ps(0), as(0)
    {
    }

    /*
     * Converts a datetime from a datetimestruct to a datetime based
     * on some metadata. The date is assumed to be valid.
     *
     * \param unit  The datetime unit to use.
     *
     * Throws an exception on error.
     */
    datetime_val_t as_datetime_val(datetime_unit_t unit) const;

    /*
     * Converts a datetime from a datetimestruct to a date based
     * on some metadata. The date is assumed to be valid.
     *
     * \param unit  The datetime unit to use, must be a date unit.
     *
     * Throws an exception on error.
     */
    date_val_t as_date_val(datetime_unit_t unit) const;
    
    /*
     * Converts a datetime based on the given metadata into a datetimestruct
     */
    void set_from_datetime_val(datetime_val_t val, datetime_unit_t unit);
    
    /**
     * Fills in the year, month, day in 'dts' based on the days
     * offset from 1970. Leaves the rest of the fields alone.
     */
    void fill_from_days(datetime_val_t days);

    /** The days offset from the 1970 epoch */
    datetime_val_t as_days() const;
    /** The minutes offset from the 1970 epoch */
    datetime_val_t as_minutes() const;
    
    /**
     * Adjusts based on a seconds offset.  Assumes
     * the current values are valid (i.e. each field is within its
     * valid range)
     */
    void add_seconds(int seconds);
    /**
     * Adjusts based on a minutes offset. Assumes
     * the current values are valid (i.e. each field is within its
     * valid range)
     */
    void add_minutes(int minutes);
};

extern int days_per_month_table[2][12];

/*
 * Returns true if the given year is a leap year, false otherwise.
 */
inline bool is_leapyear(int64_t year)
{
    return (year & 0x3) == 0 && /* year % 4 == 0 */
           ((year % 100) != 0 ||
            (year % 400) == 0);
}

/*
 * Modifies 'days' in place to be the day offset within the year,
 * and returns the year.
 */
datetime_val_t days_to_yearsdays(datetime_val_t* inout_days);

/* Extracts the month number from a 'datetime64[D]' value */
int days_to_month_number(datetime_val_t days);

bool satisfies_conversion_rule(datetime_unit_t dest, datetime_unit_t src, datetime_conversion_rule_t rule);

} // namespace datetime

#endif // DATETIME_MAIN_H