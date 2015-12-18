//
// Copyright (C) 2011-15 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <string>

#include <dynd/parse.hpp>
#include <dynd/types/date_parser.hpp>
#include <dynd/types/date_util.hpp>

using namespace std;
using namespace dynd;

// Needs to be alphabetically sorted
static named_value named_month_table[] = {
    named_value("apr", 4 + 12),  named_value("april", 4),     named_value("aug", 8 + 12),  named_value("august", 8),
    named_value("dec", 12 + 12), named_value("december", 12), named_value("feb", 2 + 12),  named_value("february", 2),
    named_value("jan", 1 + 12),  named_value("january", 1),   named_value("jul", 7 + 12),  named_value("july", 7),
    named_value("jun", 6 + 12),  named_value("june", 6),      named_value("mar", 3 + 12),  named_value("march", 3),
    named_value("may", 5 + 12),  named_value("nov", 11 + 12), named_value("november", 11), named_value("oct", 10 + 12),
    named_value("october", 10),  named_value("sep", 9 + 12),  named_value("sept", 9 + 12), named_value("september", 9),
};

bool dynd::parse_str_month_no_ws(const char *&begin, const char *end, int &out_month)
{
  if (parse_ci_alpha_str_named_value_no_ws(begin, end, named_month_table, out_month)) {
    if (out_month > 12) {
      out_month -= 12;
    }
    return true;
  }
  else {
    return false;
  }
}

bool dynd::parse_str_month_punct_no_ws(const char *&begin, const char *end, int &out_month)
{
  if (parse_ci_alpha_str_named_value_no_ws(begin, end, named_month_table, out_month)) {
    if (out_month > 12) {
      // If the token matched has 12 added to it, it's an abbreviation, so
      // accept a period after it.
      parse_token_no_ws(begin, end, '.');
      out_month -= 12;
      return true;
    }
    else {
      return true;
    }
  }
  else {
    return false;
  }
}

// Needs to be alphabetically sorted
static named_value named_weekday_table[] = {
    named_value("fri", 4), named_value("friday", 4),    named_value("mon", 0), named_value("monday", 0),
    named_value("sat", 5), named_value("saturday", 5),  named_value("sun", 6), named_value("sunday", 6),
    named_value("thu", 3), named_value("thursday", 3),  named_value("tue", 1), named_value("tuesday", 1),
    named_value("wed", 2), named_value("wednesday", 2),
};

bool dynd::parse_str_weekday_no_ws(const char *&begin, const char *end, int &out_weekday)
{
  return parse_ci_alpha_str_named_value_no_ws(begin, end, named_weekday_table, out_weekday);
}

// sMMsDD for separator character 's'
// Returns true on success
static bool parse_md(const char *&begin, const char *end, char sep, int &out_month, int &out_day)
{
  saved_begin_state sbs(begin);
  // sMM
  if (!parse_token_no_ws(begin, end, sep)) {
    return sbs.fail();
  }
  if (!parse_1or2digit_int_no_ws(begin, end, out_month)) {
    return sbs.fail();
  }
  // sDD
  if (!parse_token_no_ws(begin, end, sep)) {
    return sbs.fail();
  }
  if (!parse_1or2digit_int_no_ws(begin, end, out_day)) {
    return sbs.fail();
  }
  else if (begin < end && isdigit(begin[0])) {
    // Don't match if the next character is another digit
    return sbs.fail();
  }
  return sbs.succeed();
}

// sMMMsDD for separator character 's' and string-based month
// Returns true on success
static bool parse_md_str_month(const char *&begin, const char *end, char sep, int &out_month, int &out_day)
{
  saved_begin_state sbs(begin);
  // sMMM
  if (!parse_token_no_ws(begin, end, sep)) {
    return sbs.fail();
  }
  if (!parse_str_month_no_ws(begin, end, out_month)) {
    return sbs.fail();
  }
  // sDD
  if (!parse_token_no_ws(begin, end, sep)) {
    return sbs.fail();
  }
  if (!parse_1or2digit_int_no_ws(begin, end, out_day)) {
    return sbs.fail();
  }
  else if (begin < end && isdigit(begin[0])) {
    // Don't match if the next character is another digit
    return sbs.fail();
  }
  return sbs.succeed();
}

// 1979s03s22, 1979sMARs22
// and if ambig is YMD, also accept 79s03s22, 79sMARs22
// for separator character 's', where MM is numbers and MMM is a string.
static bool parse_ymd_sep_date(const char *&begin, const char *end, char sep, date_ymd &out_ymd,
                               date_parse_order_t ambig, int century_window)
{
  saved_begin_state sbs(begin);
  // YYYY
  int year;
  if (!parse_4digit_int_no_ws(begin, end, year)) {
    if (century_window != 0 && ambig == date_parse_ymd) {
      // Accept 79s03s22 as well in this case
      if (!parse_2digit_int_no_ws(begin, end, year)) {
        return sbs.fail();
      }
      year = date_ymd::resolve_2digit_year(year, century_window);
    }
    else {
      return sbs.fail();
    }
  }
  // sMMsDD
  int month, day;
  if (!parse_md(begin, end, sep, month, day)) {
    // sMMMsDD with a string month
    if (!parse_md_str_month(begin, end, sep, month, day)) {
      return sbs.fail();
    }
  }
  // Validate and return the date
  if (!date_ymd::is_valid(year, month, day)) {
    return sbs.fail();
  }
  out_ymd.year = year;
  out_ymd.month = month;
  out_ymd.day = day;
  return sbs.succeed();
}

// MMsDDsYYYY for separator character 's'
static bool parse_mdy_ambig_sep_date(const char *&begin, const char *end, char sep, date_ymd &out_ymd,
                                     int century_window)
{
  saved_begin_state sbs(begin);
  // MM
  int month;
  if (!parse_1or2digit_int_no_ws(begin, end, month)) {
    return sbs.fail();
  }
  // sDD
  if (!parse_token_no_ws(begin, end, sep)) {
    return sbs.fail();
  }
  int day;
  if (!parse_1or2digit_int_no_ws(begin, end, day)) {
    return sbs.fail();
  }
  // sYYYY
  if (!parse_token_no_ws(begin, end, sep)) {
    return sbs.fail();
  }
  int year;
  if (!parse_4digit_int_no_ws(begin, end, year)) {
    if (century_window != 0) {
      if (!parse_2digit_int_no_ws(begin, end, year)) {
        return sbs.fail();
      }
      else if (begin < end && isdigit(begin[0])) {
        // Don't match if the next character is another digit
        return sbs.fail();
      }
      year = date_ymd::resolve_2digit_year(year, century_window);
    }
    else {
      return sbs.fail();
    }
  }
  else if (begin < end && isdigit(begin[0])) {
    // Don't match if the next character is another digit
    return sbs.fail();
  }
  // Validate and return the date
  if (!date_ymd::is_valid(year, month, day)) {
    return sbs.fail();
  }
  out_ymd.year = year;
  out_ymd.month = month;
  out_ymd.day = day;
  return sbs.succeed();
}

// DDsMMsYYYY for separator character 's'
static bool parse_dmy_ambig_sep_date(const char *&begin, const char *end, char sep, date_ymd &out_ymd,
                                     int century_window)
{
  saved_begin_state sbs(begin);
  // DD
  int day;
  if (!parse_1or2digit_int_no_ws(begin, end, day)) {
    return sbs.fail();
  }
  // sMM
  if (!parse_token_no_ws(begin, end, sep)) {
    return sbs.fail();
  }
  int month;
  if (!parse_1or2digit_int_no_ws(begin, end, month)) {
    return sbs.fail();
  }
  // sYYYY
  if (!parse_token_no_ws(begin, end, sep)) {
    return sbs.fail();
  }
  int year;
  if (!parse_4digit_int_no_ws(begin, end, year)) {
    if (century_window != 0) {
      if (!parse_2digit_int_no_ws(begin, end, year)) {
        return sbs.fail();
      }
      else if (begin < end && isdigit(begin[0])) {
        // Don't match if the next character is another digit
        return sbs.fail();
      }
      year = date_ymd::resolve_2digit_year(year, century_window);
    }
    else {
      return sbs.fail();
    }
  }
  else if (begin < end && isdigit(begin[0])) {
    // Don't match if the next character is another digit
    return sbs.fail();
  }
  // Validate and return the date
  if (!date_ymd::is_valid(year, month, day)) {
    return sbs.fail();
  }
  out_ymd.year = year;
  out_ymd.month = month;
  out_ymd.day = day;
  return sbs.succeed();
}

// DDsMMMsYYYY
// if ambig is DMY or MDY, also accept DDsMMMsYY (because the month
// is a string we can unambiguously interpret both to mean "year at the end")
// for separator character 's', where MM is numbers and MMM is a string.

static bool parse_dmy_str_month_sep_date(const char *&begin, const char *end, char sep, date_ymd &out_ymd,
                                         date_parse_order_t ambig, int century_window)
{
  saved_begin_state sbs(begin);
  // DD
  int day;
  if (!parse_1or2digit_int_no_ws(begin, end, day)) {
    return sbs.fail();
  }
  // sMMM string month
  if (!parse_token_no_ws(begin, end, sep)) {
    return sbs.fail();
  }
  int month;
  if (!parse_str_month_no_ws(begin, end, month)) {
    return sbs.fail();
  }
  // sYYYY
  if (!parse_token_no_ws(begin, end, sep)) {
    return sbs.fail();
  }
  int year;
  if (!parse_4digit_int_no_ws(begin, end, year)) {
    if (century_window && (ambig == date_parse_dmy || ambig == date_parse_mdy)) {
      if (!parse_2digit_int_no_ws(begin, end, year)) {
        return sbs.fail();
      }
      else if (begin < end && isdigit(begin[0])) {
        // Don't match if the next character is another digit
        return sbs.fail();
      }
      year = date_ymd::resolve_2digit_year(year, century_window);
    }
    else {
      return sbs.fail();
    }
  }
  else if (begin < end && isdigit(begin[0])) {
    // Don't match if the next character is another digit
    return sbs.fail();
  }
  // Validate and return the date
  if (!date_ymd::is_valid(year, month, day)) {
    return sbs.fail();
  }
  out_ymd.year = year;
  out_ymd.month = month;
  out_ymd.day = day;
  return sbs.succeed();
}

// DDsMMMsYYYY
// if ambig is DMY or MDY, also accept DDsMMMsYY (because the month
// is a string we can unambiguously interpret both to mean "year at the end")
// for arbitrary amount of whitespace 's', where MMM is a string month.
static bool parse_dmy_str_month_ws_date(const char *&begin, const char *end, date_ymd &out_ymd,
                                        date_parse_order_t ambig, int century_window)
{
  saved_begin_state sbs(begin);
  // DD
  int day;
  if (!parse_1or2digit_int_no_ws(begin, end, day)) {
    return sbs.fail();
  }
  // sMMM string month
  skip_whitespace(begin, end);
  int month;
  if (!parse_str_month_no_ws(begin, end, month)) {
    return sbs.fail();
  }
  // sYYYY
  skip_whitespace(begin, end);
  int year;
  if (!parse_4digit_int_no_ws(begin, end, year)) {
    if (century_window && (ambig == date_parse_dmy || ambig == date_parse_mdy)) {
      if (!parse_2digit_int_no_ws(begin, end, year)) {
        return sbs.fail();
      }
      else if (begin < end && isdigit(begin[0])) {
        // Don't match if the next character is another digit
        return sbs.fail();
      }
      year = date_ymd::resolve_2digit_year(year, century_window);
    }
    else {
      return sbs.fail();
    }
  }
  else if (begin < end && isdigit(begin[0])) {
    // Don't match if the next character is another digit
    return sbs.fail();
  }
  // Validate and return the date
  if (!date_ymd::is_valid(year, month, day)) {
    return sbs.fail();
  }
  out_ymd.year = year;
  out_ymd.month = month;
  out_ymd.day = day;
  return sbs.succeed();
}

// YYYY-MM-DD, +YYYYYY-MM-DD, or -YYYYYY-MM-DD
bool dynd::parse_iso8601_dashes_date(const char *&begin, const char *end, date_ymd &out_ymd)
{
  saved_begin_state sbs(begin);
  // YYYY, +YYYYYY, or -YYYYYY
  int year;
  if (parse_token_no_ws(begin, end, '-')) {
    if (!parse_6digit_int_no_ws(begin, end, year)) {
      return sbs.fail();
    }
    year = -year;
  }
  else if (parse_token_no_ws(begin, end, '+')) {
    if (!parse_6digit_int_no_ws(begin, end, year)) {
      return sbs.fail();
    }
  }
  else {
    if (!parse_4digit_int_no_ws(begin, end, year)) {
      return sbs.fail();
    }
  }
  // -MM-DD
  int month, day;
  if (!parse_md(begin, end, '-', month, day)) {
    return sbs.fail();
  }
  // Validate and return the date
  if (!date_ymd::is_valid(year, month, day)) {
    return sbs.fail();
  }
  out_ymd.year = year;
  out_ymd.month = month;
  out_ymd.day = day;
  return sbs.succeed();
}

// YYYYMMDD
static bool parse_iso8601_nodashes_date(const char *&begin, const char *end, date_ymd &out_ymd)
{
  saved_begin_state sbs(begin);
  // YYYY
  int year;
  if (!parse_4digit_int_no_ws(begin, end, year)) {
    return sbs.fail();
  }
  // MM
  int month;
  if (!parse_2digit_int_no_ws(begin, end, month)) {
    return sbs.fail();
  }
  // DD
  int day;
  if (!parse_2digit_int_no_ws(begin, end, day)) {
    return sbs.fail();
  }
  // Disallow another digit immediately following
  if (begin < end && isdigit(*begin)) {
    return sbs.fail();
  }
  // Validate and return the date
  if (!date_ymd::is_valid(year, month, day)) {
    return sbs.fail();
  }
  out_ymd.year = year;
  out_ymd.month = month;
  out_ymd.day = day;
  return sbs.succeed();
}

static bool parse_mdy_long_format_date(const char *&begin, const char *end, date_ymd &out_ymd, int century_window)
{
  saved_begin_state sbs(begin);
  int month;
  if (!parse_str_month_punct_no_ws(begin, end, month)) {
    return sbs.fail();
  }
  if (!skip_required_whitespace(begin, end)) {
    return sbs.fail();
  }
  int day;
  if (!parse_1or2digit_int_no_ws(begin, end, day)) {
    return sbs.fail();
  }
  // Comma allowed, but not required
  parse_token(begin, end, ',');
  skip_whitespace(begin, end);
  int year;
  if (!parse_4digit_int_no_ws(begin, end, year)) {
    if (century_window != 0) {
      if (!parse_2digit_int_no_ws(begin, end, year)) {
        return sbs.fail();
      }
      else if (begin < end && isdigit(begin[0])) {
        // Don't match if the next character is another digit
        return sbs.fail();
      }
      year = date_ymd::resolve_2digit_year(year, century_window);
    }
    else {
      return sbs.fail();
    }
  }
  else if (begin < end && isdigit(begin[0])) {
    // Don't match if the next character is another digit
    return sbs.fail();
  }
  // Validate and return the date
  if (!date_ymd::is_valid(year, month, day)) {
    return sbs.fail();
  }
  out_ymd.year = year;
  out_ymd.month = month;
  out_ymd.day = day;
  return sbs.succeed();
}

// Skips Z, +####, -####, +##, -##, +##:##, -##:##
static void skip_timezone(const char *&begin, const char *end)
{
  skip_whitespace(begin, end);
  if (parse_token_no_ws(begin, end, 'Z')) {
    return;
  }
  else if (parse_token_no_ws(begin, end, "GMT")) {
    return;
  }
  else {
    if (!parse_token_no_ws(begin, end, '+')) {
      if (!parse_token_no_ws(begin, end, '-')) {
        return;
      }
    }
    int tzoffset;
    if (parse_4digit_int_no_ws(begin, end, tzoffset)) {
      return;
    }
    else if (parse_2digit_int_no_ws(begin, end, tzoffset)) {
      const char *saved_begin = begin;
      if (parse_token_no_ws(begin, end, ':')) {
        if (parse_2digit_int_no_ws(begin, end, tzoffset)) {
          return;
        }
        else {
          begin = saved_begin;
          return;
        }
      }
    }
  }
}

// Skips 00:00:00.0000, 00:00:00.0000Z, 00:00:00.0000-06:00, 00:00:00.0000-0600, 00:00:00.0000-06
static void skip_midnight_time(const char *&begin, const char *end)
{
  // Hour
  if (!parse_token_no_ws(begin, end, "00")) {
    return;
  }
  // Minute
  if (!parse_token_no_ws(begin, end, ":00")) {
    skip_timezone(begin, end);
    return;
  }
  // Second
  if (!parse_token_no_ws(begin, end, ":00")) {
    skip_timezone(begin, end);
    return;
  }
  // Second decimal
  if (!parse_token_no_ws(begin, end, ".0")) {
    skip_timezone(begin, end);
    return;
  }
  while (begin < end && *begin == '0') {
    ++begin;
  }
  skip_timezone(begin, end);
  return;
}

bool dynd::parse_date(const char *&begin, const char *end, date_ymd &out_ymd, date_parse_order_t ambig,
                      int century_window)
{
  int weekday;
  if (parse_str_weekday_no_ws(begin, end, weekday)) {
    // Optional comma and whitespace after the weekday
    parse_token(begin, end, ',');
    skip_whitespace(begin, end);
  }
  else {
    weekday = -1;
  }

  if (parse_iso8601_dashes_date(begin, end, out_ymd) || parse_iso8601_nodashes_date(begin, end, out_ymd)) {
    // 1979-03-22, +001979-03-22, -005999-01-01
    // 19790322
  }
  else if (parse_ymd_sep_date(begin, end, '/', out_ymd, ambig, century_window) ||
           parse_ymd_sep_date(begin, end, '-', out_ymd, ambig, century_window) ||
           parse_ymd_sep_date(begin, end, '.', out_ymd, ambig, century_window)) {
    // 1979/03/22 or 1979/Mar/22, 1979-03-22 or 1979-Mar-22
    // 1979.03.22 or 1979.Mar.22
  }
  else if (parse_dmy_str_month_sep_date(begin, end, '/', out_ymd, ambig, century_window) ||
           parse_dmy_str_month_sep_date(begin, end, '-', out_ymd, ambig, century_window) ||
           parse_dmy_str_month_sep_date(begin, end, '.', out_ymd, ambig, century_window)) {
    // 22/Mar/1979, 22-Mar-1979, 22.Mar.1979
  }
  else if (parse_dmy_str_month_ws_date(begin, end, out_ymd, ambig, century_window)) {
    // 22 Mar 1979
  }
  else if (parse_mdy_long_format_date(begin, end, out_ymd, century_window)) {
    // March 22, 1979
  }
  else if (ambig == date_parse_mdy && (parse_mdy_ambig_sep_date(begin, end, '/', out_ymd, century_window) ||
                                       parse_mdy_ambig_sep_date(begin, end, '-', out_ymd, century_window) ||
                                       parse_mdy_ambig_sep_date(begin, end, '.', out_ymd, century_window))) {
    // 03/22/1979, 03-22-1979, 03.22.1979
  }
  else if (ambig == date_parse_dmy && (parse_dmy_ambig_sep_date(begin, end, '/', out_ymd, century_window) ||
                                       parse_dmy_ambig_sep_date(begin, end, '-', out_ymd, century_window) ||
                                       parse_dmy_ambig_sep_date(begin, end, '.', out_ymd, century_window))) {
    // 22/03/1979, 22-03-1979, 22.03.1979
  }
  else {
    return false;
  }

  // If there was a weekday, validate it
  if (weekday >= 0) {
    if (out_ymd.get_weekday() != weekday) {
      return false;
    }
  }

  return true;
}

bool dynd::string_to_date(const char *begin, const char *end, date_ymd &out_ymd, date_parse_order_t ambig,
                          int century_window, assign_error_mode errmode)
{
  date_ymd ymd;
  skip_whitespace(begin, end);
  if (!parse_date(begin, end, ymd, ambig, century_window)) {
    return false;
  }
  if (errmode == assign_error_nocheck) {
    // If the assignment error mode is "nocheck", just take
    // what we got no matter what comes next.
    out_ymd = ymd;
    return true;
  }
  // Either a "T" or whitespace may separate a date and a time
  if (parse_token_no_ws(begin, end, 'T')) {
    skip_midnight_time(begin, end);
  }
  else if (skip_required_whitespace(begin, end)) {
    skip_midnight_time(begin, end);
  }
  skip_whitespace(begin, end);
  if (begin == end) {
    out_ymd = ymd;
    return true;
  }
  else {
    return false;
  }
}
