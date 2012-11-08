/*
 * This file implements core functionality for NumPy datetime.
 *
 * Written by Mark Wiebe (mwwiebe@gmail.com)
 * Copyright (c) 2011 by Enthought, Inc.
 *
 * See LICENSE.txt for the license.
 */

#include <stdexcept>

#include "datetime_main.h"
#include "datetime_strings.h"

using namespace datetime;

std::ostream& operator<<(std::ostream& o, datetime_unit_t unit)
{
    switch (unit) {
        case datetime_unit_unspecified:
            o << "unspecified";
            break;
        case datetime_unit_year:
            o << "year";
            break;
        case datetime_unit_month:
            o << "month";
            break;
        case datetime_unit_week:
            o << "week";
            break;
        case datetime_unit_day:
            o << "day";
            break;
        case datetime_unit_hour:
            o << "hour";
            break;
        case datetime_unit_minute:
            o << "minute";
            break;
        case datetime_unit_second:
            o << "second";
            break;
        case datetime_unit_ms:
            o << "ms";
            break;
        case datetime_unit_us:
            o << "us";
            break;
        case datetime_unit_ns:
            o << "ns";
            break;
        case datetime_unit_ps:
            o << "ps";
            break;
        case datetime_unit_fs:
            o << "fs";
            break;
        case datetime_unit_as:
            o << "as";
            break;
        default:
            o << "<invalid " << (int)unit << ">";
            break;        
    }
    return o;
}

std::ostream& operator<<(std::ostream& o, datetime_conversion_rule_t rule)
{
    switch (rule) {
        case datetime_conversion_exact:
            o << "exact";
            break;
        case datetime_conversion_strict:
            o << "strict";
            break;
        case datetime_conversion_relaxed:
            o << "relaxed";
            break;
        default:
            o << "<invalid " << (int)rule << ">";
            break;        
    }
    return o;
}

/* Days per month, regular year and leap year */
int datetime::days_per_month_table[2][12] = {
    { 31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31 },
    { 31, 29, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31 }
};

/*
 * Calculates the days offset from the 1970 epoch.
 */
datetime_val_t datetime::datetime_fields::as_days() const
{
    int i, month;
    int64_t year;
    datetime_val_t days = 0;
    int *month_lengths;

    year = this->year - 1970;
    days = year * 365;

    /* Adjust for leap years */
    if (days >= 0) {
        /*
         * 1968 is the closest leap year before 1970.
         * Exclude the current year, so add 1.
         */
        year += 1;
        /* Add one day for each 4 years */
        days += year / 4;
        /* 1900 is the closest previous year divisible by 100 */
        year += 68;
        /* Subtract one day for each 100 years */
        days -= year / 100;
        /* 1600 is the closest previous year divisible by 400 */
        year += 300;
        /* Add one day for each 400 years */
        days += year / 400;
    }
    else {
        /*
         * 1972 is the closest later year after 1970.
         * Include the current year, so subtract 2.
         */
        year -= 2;
        /* Subtract one day for each 4 years */
        days += year / 4;
        /* 2000 is the closest later year divisible by 100 */
        year -= 28;
        /* Add one day for each 100 years */
        days -= year / 100;
        /* 2000 is also the closest later year divisible by 400 */
        /* Subtract one day for each 400 years */
        days += year / 400;
    }

    month_lengths = days_per_month_table[is_leapyear(this->year)];
    month = this->month - 1;

    /* Add the months */
    for (i = 0; i < month; ++i) {
        days += month_lengths[i];
    }

    /* Add the days */
    days += this->day - 1;

    return days;
}

datetime_val_t datetime::datetime_fields::as_minutes() const
{
    return (as_days() * 24 + this->hour) * 60 + this->min;
}

/*
 * Modifies 'days' to be the day offset within the year,
 * and returns the year.
 */
datetime_val_t datetime::days_to_yearsdays(datetime_val_t* inout_days)
{
    const datetime_val_t days_per_400years = (400*365 + 100 - 4 + 1);
    /* Adjust so it's relative to the year 2000 (divisible by 400) */
    datetime_val_t days = *inout_days - (365*30 + 7);
    datetime_val_t year;

    /* Break down the 400 year cycle to get the year and day within the year */
    if (days >= 0) {
        year = 400 * (days / days_per_400years);
        days = days % days_per_400years;
    }
    else {
        year = 400 * ((days - (days_per_400years - 1)) / days_per_400years);
        days = days % days_per_400years;
        if (days < 0) {
            days += days_per_400years;
        }
    }

    /* Work out the year/day within the 400 year cycle */
    if (days >= 366) {
        year += 100 * ((days-1) / (100*365 + 25 - 1));
        days = (days-1) % (100*365 + 25 - 1);
        if (days >= 365) {
            year += 4 * ((days+1) / (4*365 + 1));
            days = (days+1) % (4*365 + 1);
            if (days >= 366) {
                year += (days-1) / 365;
                days = (days-1) % 365;
            }
        }
    }

    *inout_days = days;
    return year + 2000;
}

/* Extracts the month number from a 'datetime64[D]' value */
int datetime::days_to_month_number(datetime_val_t days)
{
    int64_t year;
    int *month_lengths, i;

    year = days_to_yearsdays(&days);
    month_lengths = days_per_month_table[is_leapyear(year)];

    for (i = 0; i < 12; ++i) {
        if (days < month_lengths[i]) {
            return i + 1;
        }
        else {
            days -= month_lengths[i];
        }
    }

    /* Should never get here */
    return 1;
}

/*
 * Fills in the year, month, day in 'dts' based on the days
 * offset from 1970.
 */
void datetime::datetime_fields::fill_from_days(datetime_val_t days)
{
    int *month_lengths, i;

    this->year = days_to_yearsdays(&days);
    month_lengths = days_per_month_table[is_leapyear(this->year)];

    for (i = 0; i < 12; ++i) {
        if (days < month_lengths[i]) {
            this->month = i + 1;
            this->day = (int)days + 1;
        } else {
            days -= month_lengths[i];
        }
    }
}

datetime_val_t datetime::datetime_fields::as_datetime_val(datetime_unit_t unit) const
{
    /* If the datetimestruct is NaT, return NaT */
    if (this->year == DATETIME_DATETIME_NAT) {
        return DATETIME_DATETIME_NAT;
    }

    if (unit == datetime_unit_year) {
        /* Truncate to the year */
        return this->year - 1970;
    }
    else if (unit == datetime_unit_month) {
        /* Truncate to the month */
        return 12 * (this->year - 1970) + (this->month - 1);
    }
    else {
        /* Otherwise calculate the number of days to start */
        datetime_val_t days = this->as_days();

        switch (unit) {
            case datetime_unit_week:
                /* Truncate to weeks */
                return (days >= 0) ? (days / 7) : ((days - 6) / 7);
            case datetime_unit_day:
                return days;
            case datetime_unit_hour:
                return days * 24 +
                      this->hour;
            case datetime_unit_minute:
                return (days * 24 +
                      this->hour) * 60 +
                      this->min;
            case datetime_unit_second:
                return ((days * 24 +
                      this->hour) * 60 +
                      this->min) * 60 +
                      this->sec;
            case datetime_unit_ms:
                return (((days * 24 +
                      this->hour) * 60 +
                      this->min) * 60 +
                      this->sec) * 1000 +
                      this->us / 1000;
            case datetime_unit_us:
                return (((days * 24 +
                      this->hour) * 60 +
                      this->min) * 60 +
                      this->sec) * 1000000 +
                      this->us;
            case datetime_unit_ns:
                return ((((days * 24 +
                      this->hour) * 60 +
                      this->min) * 60 +
                      this->sec) * 1000000 +
                      this->us) * 1000 +
                      this->ps / 1000;
            case datetime_unit_ps:
                return ((((days * 24 +
                      this->hour) * 60 +
                      this->min) * 60 +
                      this->sec) * 1000000 +
                      this->us) * 1000000 +
                      this->ps;
            case datetime_unit_fs:
                /* only 2.6 hours */
                return (((((days * 24 +
                      this->hour) * 60 +
                      this->min) * 60 +
                      this->sec) * 1000000 +
                      this->us) * 1000000 +
                      this->ps) * 1000 +
                      this->as / 1000;
            case datetime_unit_as:
                /* only 9.2 secs */
                return (((((days * 24 +
                      this->hour) * 60 +
                      this->min) * 60 +
                      this->sec) * 1000000 +
                      this->us) * 1000000 +
                      this->ps) * 1000000 +
                      this->as;
            default:
                throw std::runtime_error(
                        "datetime metadata with corrupt unit value");
        }
    }
}

void datetime::datetime_fields::set_from_datetime_val(datetime_val_t val, datetime_unit_t unit)
{
    int64_t perday;

    /* Initialize the output to all zeros */
    memset(this, 0, sizeof(datetime_fields));
    this->year = 1970;
    this->month = 1;
    this->day = 1;

    /* NaT is signaled in the year */
    if (val == DATETIME_DATETIME_NAT) {
        this->year = DATETIME_DATETIME_NAT;
        return;
    }

    /*
     * Note that care must be taken with the / and % operators
     * for negative values.
     */
    switch (unit) {
        case datetime_unit_year:
            this->year = 1970 + val;
            break;

        case datetime_unit_month:
            if (val >= 0) {
                this->year  = 1970 + val / 12;
                this->month = val % 12 + 1;
            }
            else {
                this->year  = 1969 + (val + 1) / 12;
                this->month = 12 + (val + 1)% 12;
            }
            break;

        case datetime_unit_week:
            /* A week is 7 days */
            this->fill_from_days(val * 7);
            break;

        case datetime_unit_day:
            this->fill_from_days(val);
            break;

        case datetime_unit_hour:
            perday = 24LL;

            if (val >= 0) {
                this->fill_from_days(val / perday);
                val = val % perday;
            }
            else {
                this->fill_from_days((val - (perday-1)) / perday);
                val = (perday-1) + (val + 1) % perday;
            }
            this->hour = (int)val;
            break;

        case datetime_unit_minute:
            perday = 24LL * 60;

            if (val >= 0) {
                this->fill_from_days(val / perday);
                val  = val % perday;
            }
            else {
                this->fill_from_days((val - (perday-1)) / perday);
                val = (perday-1) + (val + 1) % perday;
            }
            this->hour = (int)(val / 60);
            this->min = (int)(val % 60);
            break;

        case datetime_unit_second:
            perday = 24LL * 60 * 60;

            if (val >= 0) {
                this->fill_from_days(val / perday);
                val = val % perday;
            }
            else {
                this->fill_from_days((val - (perday-1)) / perday);
                val = (perday-1) + (val + 1) % perday;
            }
            this->hour = (int)(val / (60*60));
            this->min = (int)((val / 60) % 60);
            this->sec = (int)(val % 60);
            break;

        case datetime_unit_ms:
            perday = 24LL * 60 * 60 * 1000;

            if (val >= 0) {
                this->fill_from_days(val / perday);
                val = val % perday;
            }
            else {
                this->fill_from_days((val - (perday-1)) / perday);
                val = (perday-1) + (val + 1) % perday;
            }
            this->hour = (int)(val / (60*60*1000LL));
            this->min = (int)((val / (60*1000LL)) % 60);
            this->sec = (int)((val / 1000LL) % 60);
            this->us = (int)((val % 1000LL) * 1000);
            break;

        case datetime_unit_us:
            perday = 24LL * 60LL * 60LL * 1000LL * 1000LL;

            if (val >= 0) {
                this->fill_from_days(val / perday);
                val = val % perday;
            }
            else {
                this->fill_from_days((val - (perday-1)) / perday);
                val = (perday-1) + (val + 1) % perday;
            }
            this->hour = (int)(val / (60*60*1000000LL));
            this->min = (int)((val / (60*1000000LL)) % 60);
            this->sec = (int)((val / 1000000LL) % 60);
            this->us = (int)(val % 1000000LL);
            break;

        case datetime_unit_ns:
            perday = 24LL * 60LL * 60LL * 1000LL * 1000LL * 1000LL;

            if (val >= 0) {
                this->fill_from_days(val / perday);
                val = val % perday;
            }
            else {
                this->fill_from_days((val - (perday-1)) / perday);
                val = (perday-1) + (val + 1) % perday;
            }
            this->hour = (int)(val / (60*60*1000000000LL));
            this->min = (int)((val / (60*1000000000LL)) % 60);
            this->sec = (int)((val / 1000000000LL) % 60);
            this->us = (int)((val / 1000LL) % 1000000LL);
            this->ps = (int)((val % 1000LL) * 1000);
            break;

        case datetime_unit_ps:
            perday = 24LL * 60 * 60 * 1000 * 1000 * 1000 * 1000;

            if (val >= 0) {
                this->fill_from_days(val / perday);
                val = val % perday;
            }
            else {
                this->fill_from_days((val - (perday-1)) / perday);
                val = (perday-1) + (val + 1) % perday;
            }
            this->hour = (int)(val / (60*60*1000000000000LL));
            this->min = (int)((val / (60*1000000000000LL)) % 60);
            this->sec = (int)((val / 1000000000000LL) % 60);
            this->us = (int)((val / 1000000LL) % 1000000LL);
            this->ps = (int)(val % 1000000LL);
            break;

        case datetime_unit_fs:
            /* entire range is only +- 2.6 hours */
            if (val >= 0) {
                this->hour = (int)(val / (60*60*1000000000000000LL));
                this->min = (int)((val / (60*1000000000000000LL)) % 60);
                this->sec = (int)((val / 1000000000000000LL) % 60);
                this->us = (int)((val / 1000000000LL) % 1000000LL);
                this->ps = (int)((val / 1000LL) % 1000000LL);
                this->as = (int)((val % 1000LL) * 1000);
            }
            else {
                datetime_val_t minutes;

                minutes = val / (60*1000000000000000LL);
                val = val % (60*1000000000000000LL);
                if (val < 0) {
                    val += (60*1000000000000000LL);
                    --minutes;
                }
                /* Offset the negative minutes */
                this->add_minutes((int)minutes);
                this->sec = (int)((val / 1000000000000000LL) % 60);
                this->us = (int)((val / 1000000000LL) % 1000000LL);
                this->ps = (int)((val / 1000LL) % 1000000LL);
                this->as = (int)((val % 1000LL) * 1000);
            }
            break;

        case datetime_unit_as:
            /* entire range is only +- 9.2 seconds */
            if (val >= 0) {
                this->sec = (int)((val / 1000000000000000000LL) % 60);
                this->us = (int)((val / 1000000000000LL) % 1000000LL);
                this->ps = (int)((val / 1000000LL) % 1000000LL);
                this->as = (int)(val % 1000000LL);
            }
            else {
                datetime_val_t seconds;

                seconds = val / 1000000000000000000LL;
                val = val % 1000000000000000000LL;
                if (val < 0) {
                    val += 1000000000000000000LL;
                    --seconds;
                }
                /* Offset the negative seconds */
                this->add_seconds((int)seconds);
                this->us = (int)((val / 1000000000000LL) % 1000000LL);
                this->ps = (int)((val / 1000000LL) % 1000000LL);
                this->as = (int)(val % 1000000LL);
            }
            break;

        default:
            throw std::runtime_error(
                        "datetime metadata is corrupted with invalid "
                        "base unit");
    }
}

void datetime::datetime_fields::add_seconds(int seconds)
{
    int minutes;

    this->sec += seconds;
    if (this->sec < 0) {
        minutes = this->sec / 60;
        this->sec = this->sec % 60;
        if (this->sec < 0) {
            --minutes;
            this->sec += 60;
        }
        this->add_minutes(minutes);
    }
    else if (this->sec >= 60) {
        minutes = this->sec / 60;
        this->sec = this->sec % 60;
        this->add_minutes(minutes);
    }
}

void datetime::datetime_fields::add_minutes(int minutes)
{
    int isleap;

    /* MINUTES */
    this->min += minutes;
    while (this->min < 0) {
        this->min += 60;
        this->hour--;
    }
    while (this->min >= 60) {
        this->min -= 60;
        this->hour++;
    }

    /* HOURS */
    while (this->hour < 0) {
        this->hour += 24;
        this->day--;
    }
    while (this->hour >= 24) {
        this->hour -= 24;
        this->day++;
    }

    /* DAYS */
    if (this->day < 1) {
        this->month--;
        if (this->month < 1) {
            this->year--;
            this->month = 12;
        }
        isleap = is_leapyear(this->year);
        this->day += days_per_month_table[isleap][this->month-1];
    }
    else if (this->day > 28) {
        isleap = is_leapyear(this->year);
        if (this->day > days_per_month_table[isleap][this->month-1]) {
            this->day -= days_per_month_table[isleap][this->month-1];
            this->month++;
            if (this->month > 12) {
                this->year++;
                this->month = 1;
            }
        }
    }
}

bool datetime::satisfies_conversion_rule(datetime_unit_t dst, datetime_unit_t src, datetime_conversion_rule_t rule)
{
    if (src == datetime_unit_unspecified || dst == datetime_unit_unspecified)
        return false;

    switch (rule) {
        case datetime_conversion_strict:
            return src <= dst &&
                    !(src <= datetime_unit_day && datetime_unit_hour <= dst);
        case datetime_conversion_relaxed:
            return !(src <= datetime_unit_day && datetime_unit_hour <= dst) &&
                    !(dst <= datetime_unit_day && datetime_unit_hour <= src);
        default:
            return false;
    }
}