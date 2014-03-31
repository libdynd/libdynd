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
    datetime_unit_tick,
    // Not a normal unit, special flag for some functions
    datetime_unit_autodetect
};

enum datetime_conversion_rule_t {
    datetime_conversion_exact,
    datetime_conversion_strict,
    datetime_conversion_relaxed
};

std::ostream& operator<<(std::ostream& o, datetime_unit_t unit);

std::ostream& operator<<(std::ostream& o, datetime_conversion_rule_t rule);

/** A date as year, day offset within year */
struct date_yd {
    int32_t year, day;
};

/** A date as year, month, day */
struct date_ymd {
    int32_t year, month, day;
};

/*
 * This structure contains an exploded view of a date-time value.
 * NaT is represented by year == DATETIME_DATE_NAT.
 */
struct datetime_fields {
    int64_t year;
    int32_t month, day, hour, min, sec, us, tick;

    datetime_fields()
        : year(0), month(0), day(0), hour(0),
          min(0), sec(0), us(0), tick(0)
    {
    }

    /*
     * Converts a datetime from a datetimestruct to a ticks value.
     *
     * Throws an exception on error.
     */
    datetime_val_t as_ticks() const;

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
    
    /*
     * Converts a date based on the given metadata into a datetimestruct
     */
    void set_from_date_val(date_val_t val, datetime_unit_t unit) {
        set_from_datetime_val(val == DATETIME_DATE_NAT ? DATETIME_DATETIME_NAT : val, unit);
    }
    
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

    /**
     * Returns true if the datetime is divisible by the
     * specified unit, e.g. has no non-zero value with a smaller
     * unit.
     */
    bool divisible_by_unit(datetime_unit_t unit);
};

/**
 * Returns the number of days in the specified month.
 *
 * \param year  The year of the requested month.
 * \param month  The month whose length is requested.
 *
 * \returns  The number of days in the month specified.
 */
int get_month_size(int32_t year, int32_t month);

bool is_valid_ymd(int32_t year, int32_t month, int32_t day);

inline bool is_valid_ymd(const date_ymd& ymd) {
    return is_valid_ymd(ymd.year, ymd.month, ymd.day);
}

/** 
 * Converts any date value into a 'days' date and filled date_yd/date_ymd structures.
 */
void date_to_days_yd_and_ymd(date_val_t date, datetime_unit_t unit,
                int32_t& out_days, date_yd& out_yd, date_ymd& out_ymd);

/** 
 * Converts any date value into a filled date_ymd structure.
 */
void date_to_ymd(date_val_t date, datetime_unit_t unit, date_ymd& out_ymd);

/** 
 * Converts any date value into a days unit.
 */
void date_to_days(date_val_t date, datetime_unit_t unit, int32_t& out_days);

/**
 * Converts any date value into a standard C 'struct tm'.
 */
void date_to_struct_tm(date_val_t date, datetime_unit_t unit, struct tm& out_tm);

/*
 * Converts a 'days' date into a date_yd year + day offset structure.
 */
void days_to_yeardays(int32_t days, date_yd& out_yd);

/*
 * Converts a 'days' date into a date_ymd structure.
 */
void days_to_ymd(int32_t days, date_ymd& out_ymd);

/*
 * Modifies 'days' in place to be the day offset within the year,
 * and returns the year.
 */
int64_t days_to_yeardays(int64_t* inout_days);

/*
 * Converts a year + days offset to year/month/day struct.
 */
void yeardays_to_ymd(int32_t year, int32_t days, date_ymd& out_ymd);

/**
 * Converts a year/month/day date into a 32-bit days date.
 */
int32_t ymd_to_days(int32_t year, int32_t month, int32_t day);

/**
 * Converts a year/month/day date into a 32-bit days date.
 */
inline int32_t ymd_to_days(const date_ymd& ymd) {
    return ymd_to_days(ymd.year, ymd.month, ymd.day);
}

/**
 * Converts a year/month/day date into a 64-bit days date.
 */
int64_t ymd_to_days(int64_t year, int32_t month, int32_t day);

extern const int days_per_month_table[2][12];

/*
 * Returns true if the given year is a leap year, false otherwise.
 */
inline bool is_leapyear(int64_t year)
{
    return (year & 0x3) == 0 && /* year % 4 == 0 */
           ((year % 100) != 0 ||
            (year % 400) == 0);
}

/* Extracts the month number from a 'datetime64[D]' value */
int days_to_month_number(datetime_val_t days);

bool satisfies_conversion_rule(
    datetime_unit_t dest, datetime_unit_t src,
    datetime_conversion_rule_t rule);

} // namespace datetime

#endif // DATETIME_MAIN_H

