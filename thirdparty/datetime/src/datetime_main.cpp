/*
 * This file implements core functionality for NumPy datetime.
 *
 * Written by Mark Wiebe (mwwiebe@gmail.com)
 * Copyright (c) 2011 by Enthought, Inc.
 *
 * See LICENSE.txt for the license.
 */

#include <stdexcept>
#include <sstream>
#include <string.h>

#include "datetime_main.h"
#include "datetime_strings.h"

using namespace std;
using namespace datetime;

std::ostream& datetime::operator<<(std::ostream& o, datetime_unit_t unit)
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
        case datetime_unit_tick:
            o << "tick";
            break;
        case datetime_unit_autodetect:
            o << "<autodetect>";
            break;
        default:
            o << "<invalid " << (int)unit << ">";
            break;        
    }
    return o;
}

std::ostream& datetime::operator<<(std::ostream& o, datetime_conversion_rule_t rule)
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
const int datetime::days_per_month_table[2][12] = {
    { 31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31 },
    { 31, 29, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31 }
};

int datetime::get_month_size(int32_t year, int32_t month)
{
    const int *month_lengths = days_per_month_table[is_leapyear(year)];
    if (1 <= month && month <= 12) {
        return month_lengths[month-1];
    } else {
        stringstream ss;
        ss << "invalid month value " << month;
        throw runtime_error(ss.str());
    }
}

bool datetime::is_valid_ymd(int32_t year, int32_t month, int32_t day)
{
    if (year == DATETIME_DATE_NAT) {
        return false;
    }
    if (month < 1 || month > 12) {
        return false;
    }
    const int *month_lengths = days_per_month_table[is_leapyear(year)];
    if (day < 1 || day > month_lengths[month-1]) {
        return false;
    }
    return true;
}

void datetime::date_to_ymd(date_val_t date, datetime_unit_t unit, date_ymd& out_ymd)
{
    if (date == DATETIME_DATE_NAT) {
        out_ymd.year = DATETIME_DATE_NAT;
        out_ymd.month = 0;
        out_ymd.day = 0;
        return;
    }

    switch (unit) {
        case datetime_unit_year: {
            out_ymd.year = 1970 + date;
            out_ymd.month = 1;
            out_ymd.day = 1;
            break;
        }
        case datetime_unit_month: {
            if (date >= 0) {
                out_ymd.year  = 1970 + date / 12;
                out_ymd.month = date % 12 + 1;
                out_ymd.day = 1;
            }
            else {
                out_ymd.year  = 1969 + (date + 1) / 12;
                out_ymd.month = 12 + (date + 1)% 12;
                out_ymd.day = 1;
            }
            break;
        }
        case datetime_unit_day: {
            date_yd yd;
            days_to_yeardays(date, yd);
            yeardays_to_ymd(yd.year, yd.day, out_ymd);
            break;
        }
        default: {
            stringstream ss;
            ss << "datetime unit " << unit << " cannot be used as a date unit";
            throw runtime_error(ss.str());
        }
    }
}

void datetime::date_to_days(date_val_t date, datetime_unit_t unit, int32_t& out_days)
{
    if (date == DATETIME_DATE_NAT) {
        out_days = DATETIME_DATE_NAT;
        return;
    }

    switch (unit) {
        case datetime_unit_year: {
            date_ymd ymd;
            ymd.year = 1970 + date;
            ymd.month = 1;
            ymd.day = 1;
            out_days = ymd_to_days(ymd);
            break;
        }
        case datetime_unit_month: {
            date_ymd ymd;
            if (date >= 0) {
                ymd.year  = 1970 + date / 12;
                ymd.month = date % 12 + 1;
                ymd.day = 1;
            }
            else {
                ymd.year  = 1969 + (date + 1) / 12;
                ymd.month = 12 + (date + 1)% 12;
                ymd.day = 1;
            }
            out_days = ymd_to_days(ymd);
            break;
        }
        case datetime_unit_day: {
            out_days = date;
            break;
        }
        default: {
            stringstream ss;
            ss << "datetime unit " << unit << " cannot be used as a date unit";
            throw runtime_error(ss.str());
        }
    }
}

void datetime::date_to_days_yd_and_ymd(date_val_t date, datetime_unit_t unit,
                int32_t& out_days, date_yd& out_yd, date_ymd& out_ymd)
{
    if (date == DATETIME_DATE_NAT) {
        out_days = DATETIME_DATE_NAT;
        out_yd.year = DATETIME_DATE_NAT;
        out_yd.day = 0;
        out_ymd.year = DATETIME_DATE_NAT;
        out_ymd.month = 0;
        out_ymd.day = 0;
        return;
    }

    switch (unit) {
        case datetime_unit_year: {
            out_ymd.year = 1970 + date;
            out_ymd.month = 1;
            out_ymd.day = 1;
            out_yd.year = 1970;
            out_yd.day = 0;
            out_days = ymd_to_days(out_ymd);
            break;
        }
        case datetime_unit_month: {
            if (date >= 0) {
                out_ymd.year  = 1970 + date / 12;
                out_ymd.month = date % 12 + 1;
                out_ymd.day = 1;
            }
            else {
                out_ymd.year  = 1969 + (date + 1) / 12;
                out_ymd.month = 12 + (date + 1)% 12;
                out_ymd.day = 1;
            }
            out_days = ymd_to_days(out_ymd);
            days_to_yeardays(out_days, out_yd);
            break;
        }
        case datetime_unit_day: {
            out_days = date;
            days_to_yeardays(out_days, out_yd);
            yeardays_to_ymd(out_yd.year, out_yd.day, out_ymd);
            break;
        }
        default: {
            stringstream ss;
            ss << "datetime unit " << unit << " cannot be used as a date unit";
            throw runtime_error(ss.str());
        }
    }
}

/*
 * Modifies 'days' to be the day offset within the year,
 * and returns the year.
 */
template<class T>
inline T days_to_yeardays_templ(T* inout_days)
{
    const T days_per_400years = (400*365 + 100 - 4 + 1);
    /* Adjust so it's relative to the year 2000 (divisible by 400) */
    T days = *inout_days - (365*30 + 7);
    T year;

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

void datetime::days_to_yeardays(int32_t days, date_yd& out_yd)
{
    if (days == DATETIME_DATE_NAT) {
        out_yd.year = DATETIME_DATE_NAT;
        out_yd.day = 0;
        return;
    }

    out_yd.day = days;
    out_yd.year = days_to_yeardays_templ<int32_t>(&out_yd.day);
}

void datetime::days_to_ymd(int32_t days, date_ymd& out_ymd)
{
    if (days == DATETIME_DATE_NAT) {
        out_ymd.year = DATETIME_DATE_NAT;
        out_ymd.month = 0;
        out_ymd.day = 0;
        return;
    }

    date_yd yd;
    days_to_yeardays(days, yd);
    yeardays_to_ymd(yd.year, yd.day, out_ymd);
}

int64_t datetime::days_to_yeardays(int64_t* inout_days)
{
    return days_to_yeardays_templ<int64_t>(inout_days);
}

void datetime::yeardays_to_ymd(int32_t year, int32_t days, date_ymd& out_ymd)
{
    if (year == DATETIME_DATE_NAT) {
        out_ymd.year = DATETIME_DATE_NAT;
        out_ymd.month = 0;
        out_ymd.day = 0;
        return;
    }

    const int *month_lengths = days_per_month_table[is_leapyear(year)];

    out_ymd.year = year;
    for (int i = 0; i < 12; ++i) {
        if (days < month_lengths[i]) {
            out_ymd.month = i + 1;
            out_ymd.day = (int32_t)days + 1;
            break;
        } else {
            days -= month_lengths[i];
        }
    }

}

template<class T>
inline T ymd_to_days_templ(T year, int32_t month, int32_t day)
{
    T original_year = year;
    int i;
    T days = 0;

    year = year - 1970;
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

    const int *month_lengths = days_per_month_table[is_leapyear(original_year)];
    month = month - 1;

    /* Add the months */
    for (i = 0; i < month; ++i) {
        days += month_lengths[i];
    }

    /* Add the days */
    days += day - 1;

    return days;
}

int32_t datetime::ymd_to_days(int32_t year, int32_t month, int32_t day)
{
    return ymd_to_days_templ<int32_t>(year, month, day);
}

int64_t datetime::ymd_to_days(int64_t year, int32_t month, int32_t day)
{
    return ymd_to_days_templ<int64_t>(year, month, day);
}

/*
 * Calculates the days offset from the 1970 epoch.
 */
datetime_val_t datetime::datetime_fields::as_days() const
{
    if (this->year != DATETIME_DATETIME_NAT) {
        return ymd_to_days(this->year, this->month, this->day);
    } else {
        return DATETIME_DATETIME_NAT;
    }
}

datetime_val_t datetime::datetime_fields::as_minutes() const
{
    return (as_days() * 24 + this->hour) * 60 + this->min;
}

/* Extracts the month number from a 'datetime64[D]' value */
int datetime::days_to_month_number(datetime_val_t days)
{
    int64_t year;

    year = days_to_yeardays(&days);
    const int *month_lengths = days_per_month_table[is_leapyear(year)];

    for (int i = 0; i < 12; ++i) {
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
    this->year = days_to_yeardays(&days);
    const int *month_lengths = days_per_month_table[is_leapyear(this->year)];

    for (int i = 0; i < 12; ++i) {
        if (days < month_lengths[i]) {
            this->month = i + 1;
            this->day = (int)days + 1;
            break;
        } else {
            days -= month_lengths[i];
        }
    }
}

datetime_val_t datetime::datetime_fields::as_ticks() const
{
    /* If the datetimestruct is NaT, return NaT */
    if (this->year == DATETIME_DATETIME_NAT) {
        return DATETIME_DATETIME_NAT;
    }

    datetime_val_t days = this->as_days();

    return ((((days * 24LL +
            this->hour) * 60LL +
            this->min) * 60LL +
            this->sec) * 1000000LL +
            this->us) * 10LL +
            this->tick;
}

date_val_t datetime::datetime_fields::as_date_val(datetime_unit_t unit) const
{
    /* If the datetimestruct is NaT, return NaT */
    if (this->year == DATETIME_DATETIME_NAT) {
        return DATETIME_DATE_NAT;
    }

    if (unit == datetime_unit_year) {
        /* Truncate to the year */
        return static_cast<date_val_t>(this->year - 1970);
    }
    else if (unit == datetime_unit_month) {
        /* Truncate to the month */
        return static_cast<date_val_t>(12 * (this->year - 1970) + (this->month - 1));
    }
    else {
        /* Otherwise calculate the number of days to start */
        datetime_val_t days = this->as_days();

        switch (unit) {
            case datetime_unit_week:
                /* Truncate to weeks */
                return static_cast<date_val_t>((days >= 0) ? (days / 7) : ((days - 6) / 7));
            case datetime_unit_day:
                return static_cast<date_val_t>(days);
            default: {
                stringstream ss;
                ss << "as_date_val requires a date unit, got " << unit;
                throw std::runtime_error(ss.str());
            }
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

        case datetime_unit_tick:
            perday = 24LL * 60LL * 60LL * 1000LL * 1000LL * 10LL;

            if (val >= 0) {
                this->fill_from_days(val / perday);
                val = val % perday;
            }
            else {
                this->fill_from_days((val - (perday-1)) / perday);
                val = (perday-1) + (val + 1) % perday;
            }
            this->hour = (int)(val / (60*60*10000000LL));
            this->min = (int)((val / (60*10000000LL)) % 60);
            this->sec = (int)((val / 10000000LL) % 60);
            this->us = (int)((val / 10LL) % 1000000LL);
            this->tick = (int)(val % 10LL);
            break;

        default:
            cout << "corrupt unit " << unit << endl;
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

bool datetime::datetime_fields::divisible_by_unit(datetime_unit_t unit)
{
    switch (unit) {
        case datetime_unit_unspecified:
            return true;
        case datetime_unit_year:
            return month == 1 && day == 1 && hour == 0 && min == 0 &&
                sec == 0 && us == 0 && tick == 0;
        case datetime_unit_month:
            return day == 1 && hour == 0 && min == 0 &&
                sec == 0 && us == 0 && tick == 0;
        case datetime_unit_day:
            return hour == 0 && min == 0 &&
                sec == 0 && us == 0 && tick == 0;
        case datetime_unit_hour:
            return min == 0 && sec == 0 && us == 0 && tick == 0;
        case datetime_unit_minute:
            return sec == 0 && us == 0 && tick == 0;
        case datetime_unit_second:
            return us == 0 && tick == 0;
        case datetime_unit_ms:
            return (us % 1000) == 0 && tick == 0;
        case datetime_unit_us:
            return tick == 0;
        case datetime_unit_tick:
            return true;
        default:
            return false;
    }
}

void datetime::date_to_struct_tm(date_val_t date, datetime_unit_t unit, struct tm& out_tm)
{
    int32_t days;
    date_yd yd;
    date_ymd ymd;
    date_to_days_yd_and_ymd(date, unit, days, yd, ymd);

    memset(&out_tm, 0, sizeof(struct tm));
    out_tm.tm_year = ymd.year - 1900;
    out_tm.tm_yday = yd.day;
    out_tm.tm_mon = ymd.month - 1;
    out_tm.tm_mday = ymd.day;
    // 1970-01-04 is Sunday
    out_tm.tm_wday = (int)((days - 3) % 7);
    if (out_tm.tm_wday < 0) {
        out_tm.tm_wday += 7;
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
