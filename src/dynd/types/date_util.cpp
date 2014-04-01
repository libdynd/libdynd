//
// Copyright (C) 2011-14 Mark Wiebe, DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <time.h>

#include <dynd/types/date_util.hpp>
#include <dynd/types/date_parser.hpp>
#include <dynd/types/cstruct_type.hpp>

using namespace std;
using namespace dynd;

const int date_ymd::month_lengths[2][12] = {
    {31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31},
    {31, 29, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31},
};

const int date_ymd::month_starts[2][13] = {
    {0, 31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334, 365},
    {0, 31, 60, 91, 121, 152, 182, 213, 244, 274, 305, 335, 366}
};

int32_t date_ymd::to_days(int year, int month, int day)
{
    if (is_valid(year, month, day)) {
        // Start with 365 days a year
        int result = (year - 1970) * 365;
        // Use the inclusion-exclusion principle to count leap years
        if (result >= 0) {
            result += ((year - (1968 + 1)) / 4) -
                        ((year - (1900 + 1)) / 100) +
                        ((year - (1600 + 1)) / 400);
        } else {
            result += ((year - 1972) / 4) -
                        ((year - 2000) / 100) +
                        ((year - 2000) / 400);
        }
        // Add in the months and days
        result += month_starts[is_leap_year(year)][month-1];
        result += day-1;
        return result;
    } else {
        return DYND_DATE_NA;
    }
}

std::string date_ymd::to_str(int year, int month, int day)
{
    string s;
    if (!is_valid(year, month, day)) {
        s = "NA";
    } else if (year >= 1 && year <= 9999) {
        // ISO 8601 date
        s.resize(10);
        s[0] = '0' + (year / 1000);
        s[1] = '0' + ((year / 100) % 10);
        s[2] = '0' + ((year / 10) % 10);
        s[3] = '0' + (year % 10);
        s[4] = '-';
        s[5] = '0' + (month / 10);
        s[6] = '0' + (month % 10);
        s[7] = '-';
        s[8] = '0' + (day / 10);
        s[9] = '0' + (day % 10);
    } else {
        // Expanded ISO 8601 date, using +/- 6 digit year
        s.resize(13);
        if (year >= 0) {
            s[0] = '+';
        } else {
            s[0] = '-';
            year = -year;
        }
        s[1] = '0' + (year / 100000);
        s[2] = '0' + ((year / 10000) % 10);
        s[3] = '0' + ((year / 1000) % 10);
        s[4] = '0' + ((year / 100) % 10);
        s[5] = '0' + ((year / 10) % 10);
        s[6] = '0' + (year % 10);
        s[7] = '-';
        s[8] = '0' + (month / 10);
        s[9] = '0' + (month % 10);
        s[10] = '-';
        s[11] = '0' + (day / 10);
        s[12] = '0' + (day % 10);
    }
    return s;
}


void date_ymd::set_from_days(int32_t days)
{
    if (days != DYND_DATE_NA) {
        int yearcalc;
        // Make the days relative to year 0
        days += 719528;
        // To a 400 year cycle
        if (days >= 0) {
            yearcalc = 400 * (days / (400 * 365 + 100 - 4 + 1));
            days = days % (400 * 365 + 100 - 4 + 1);
        } else {
            yearcalc = 400 * ((days - (400 * 365 + 100 - 4)) / (400 * 365 + 100 - 4 + 1));
            days = days % (400 * 365 + 100 - 4 + 1);
            if (days < 0) {
                days += (400 * 365 + 100 - 4 + 1);
            }
        }
        if (days >= 366) {
            // To a 100 year cycle
            yearcalc += 100 * ((days - 1) / (100 * 365 + 25 - 1));
            days = (days - 1) % (100 * 365 + 25 - 1);
            if (days >= 365) {
                // To a 4 year cycle
                yearcalc += 4 * ((days + 1) / (4 * 365 + 1));
                days = (days + 1) % (4 * 365 + 1);
                if (days >= 366) {
                    // To a 1 year cycle
                    yearcalc += (days - 1) / 365;
                    days = (days - 1) % 365;
                }
            }
        }
        // Search for the month
        const int *monthstart = month_starts[is_leap_year(yearcalc)];
        const int *monthfound = std::upper_bound(monthstart + 1, monthstart + 13, days);
        // Set the ymd
        year = yearcalc;
        month = static_cast<int8_t>(monthfound - monthstart);
        day = days - *(monthfound - 1) + 1;
    } else {
        year = 0;
        month = -128;
        day = 0;
    }
}

void date_ymd::set_from_str(const std::string& s)
{
    if (!string_to_date(s.data(), s.data() + s.size(), *this)) {
        stringstream ss;
        ss << "Unable to parse ";
        print_escaped_utf8_string(ss, s);
        ss << " as a date";
        throw invalid_argument(ss.str());
    }
}

void date_ymd::set_from_str(const std::string& DYND_UNUSED(s), bool DYND_UNUSED(monthfirst))
{
    throw runtime_error("TODO: date_ymd::set_from_str");
}

date_ymd date_ymd::get_current_local_date()
{
    struct tm tm_;
#if defined(_MSC_VER)
    __time64_t rawtime;
    _time64(&rawtime);
    if (_localtime64_s(&tm_, &rawtime) != 0) {
        throw std::runtime_error("Failed to use '_localtime64_s' to convert "
                                "to a local time");
    }
#else
    time_t rawtime;
    time(&rawtime);
    if (localtime_r(&rawtime, &tm_) == NULL) {
        throw std::runtime_error("Failed to use 'localtime_r' to convert "
                                "to a local time");
    }
#endif
    date_ymd ymd;
    ymd.year = tm_.tm_year + 1900;
    ymd.month = tm_.tm_mon + 1;
    ymd.day = tm_.tm_mday;
    return ymd;
}
 
const ndt::type& date_ymd::type()
{
    static ndt::type tp = ndt::make_cstruct(
            ndt::make_type<int16_t>(), "year",
            ndt::make_type<int8_t>(), "month",
            ndt::make_type<int8_t>(), "day");
    return tp;
}

