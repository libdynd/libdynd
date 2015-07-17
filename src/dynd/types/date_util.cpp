//
// Copyright (C) 2011-15 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <time.h>

#include <dynd/types/date_util.hpp>
#include <dynd/types/date_parser.hpp>
#include <dynd/types/struct_type.hpp>

using namespace std;
using namespace dynd;

const int date_ymd::month_lengths[2][12] = {
    {31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31},
    {31, 29, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31},
};

const int date_ymd::month_starts[2][13] = {
    {0, 31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334, 365},
    {0, 31, 60, 91, 121, 152, 182, 213, 244, 274, 305, 335, 366}};

std::ostream &dynd::operator<<(std::ostream &o, date_parse_order_t date_order)
{
  switch (date_order) {
  case date_parse_no_ambig:
    return (o << "NoAmbig");
  case date_parse_ymd:
    return (o << "YMD");
  case date_parse_mdy:
    return (o << "MDY");
  case date_parse_dmy:
    return (o << "DMY");
  default:
    return (o << "<invalid dateorder " << (int)date_order << ">");
  }
}

int32_t date_ymd::to_days(int year, int month, int day)
{
  if (is_valid(year, month, day)) {
    // Start with 365 days a year
    int result = (year - 1970) * 365;
    // Use the inclusion-exclusion principle to count leap years
    if (result >= 0) {
      result += ((year - (1968 + 1)) / 4) - ((year - (1900 + 1)) / 100) +
                ((year - (1600 + 1)) / 400);
    } else {
      result +=
          ((year - 1972) / 4) - ((year - 2000) / 100) + ((year - 2000) / 400);
    }
    // Add in the months and days
    result += month_starts[is_leap_year(year)][month - 1];
    result += day - 1;
    return result;
  } else {
    return DYND_DATE_NA;
  }
}

std::string date_ymd::to_str(int year, int month, int day)
{
  string s;
  if (is_valid(year, month, day)) {
    if (year >= 1 && year <= 9999) {
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
      yearcalc =
          400 * ((days - (400 * 365 + 100 - 4)) / (400 * 365 + 100 - 4 + 1));
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
    const int *monthfound =
        std::upper_bound(monthstart + 1, monthstart + 13, days);
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

void date_ymd::set_from_str(const char *begin, const char *end,
                            date_parse_order_t ambig = date_parse_no_ambig,
                            int century_window = 70,
                            assign_error_mode errmode = assign_error_fractional)
{
  if (!string_to_date(begin, end, *this, ambig, century_window, errmode)) {
    stringstream ss;
    ss << "Unable to parse ";
    print_escaped_utf8_string(ss, begin, end);
    ss << " as a date";
    throw invalid_argument(ss.str());
  }
}

int date_ymd::resolve_2digit_year_fixed_window(int year, int year_start)
{
  int century_start = (year_start / 100) * 100;
  int year_start_in_century = year_start - century_start;

  if (year >= year_start_in_century) {
    return century_start + year;
  } else {
    return century_start + 100 + year;
  }
}

int date_ymd::resolve_2digit_year_sliding_window(int year, int years_ago)
{
  // Get the current date from the system
  int32_t current_date_days;
#if defined(_MSC_VER)
  __time64_t rawtime;
  _time64(&rawtime);
#else
  time_t rawtime;
  time(&rawtime);
#endif
  if (rawtime >= 0) {
    current_date_days = static_cast<int32_t>(rawtime / DYND_SECONDS_PER_DAY);
  } else {
    current_date_days = static_cast<int32_t>(
        (rawtime - (DYND_SECONDS_PER_DAY - 1)) / DYND_SECONDS_PER_DAY);
  }
  date_ymd current_date;
  current_date.set_from_days(current_date_days);

  // Use it to resolve the 2 digit year with the sliding window
  return resolve_2digit_year_fixed_window(year, current_date.year - years_ago);
}

int date_ymd::resolve_2digit_year(int year, int century_window)
{
  if (century_window >= 1 && century_window <= 99) {
    return date_ymd::resolve_2digit_year_sliding_window(year, century_window);
  } else if (century_window >= 1000) {
    return date_ymd::resolve_2digit_year_fixed_window(year, century_window);
  } else {
    stringstream ss;
    ss << "invalid century_window value " << century_window
       << ", must be 1-99 for a sliding window, or >= 1000 for a fixed "
          "window";
    throw invalid_argument(ss.str());
  }
}

date_ymd date_ymd::get_current_local_date()
{
  struct tm tm_;
#if defined(_WIN32)
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

const ndt::type &date_ymd::type()
{
  static ndt::type tp = ndt::struct_type::make({"year", "month", "day"},
                                               {ndt::type::make<int16_t>(),
                                                ndt::type::make<int8_t>(),
                                                ndt::type::make<int8_t>()});
  return tp;
}
