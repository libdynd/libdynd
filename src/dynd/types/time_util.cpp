//
// Copyright (C) 2011-15 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <time.h>

#include <dynd/types/time_util.hpp>
#include <dynd/types/time_parser.hpp>
#include <dynd/types/date_util.hpp>
#include <dynd/types/struct_type.hpp>

using namespace std;
using namespace dynd;

int64_t time_hmst::to_ticks(int hour, int minute, int second, int tick)
{
  if (is_valid(hour, minute, second, tick)) {
    return static_cast<int64_t>(tick) + second * DYND_TICKS_PER_SECOND +
           minute * DYND_TICKS_PER_MINUTE + hour * DYND_TICKS_PER_HOUR;
  } else {
    return DYND_TIME_NA;
  }
}

std::string time_hmst::to_str(int hour, int minute, int second, int tick)
{
  std::string s;
  if (is_valid(hour, minute, second, tick)) {
    s.resize(2 + 1 + 2 + 1 + 2 + 1 + 7);
    s[0] = '0' + (hour / 10);
    s[1] = '0' + (hour % 10);
    s[2] = ':';
    s[3] = '0' + (minute / 10);
    s[4] = '0' + (minute % 10);
    if (second != 0 || tick != 0) {
      s[5] = ':';
      s[6] = '0' + (second / 10);
      s[7] = '0' + (second % 10);
      if (tick != 0) {
        s[8] = '.';
        int i = 9, divisor = 1000000;
        while (tick != 0) {
          s[i] = '0' + (tick / divisor);
          tick = tick % divisor;
          divisor = divisor / 10;
          ++i;
        }
        s.resize(i);
      } else {
        s.resize(8);
      }
    } else {
      s.resize(5);
    }
  }
  return s;
}

void time_hmst::set_from_ticks(int64_t ticks)
{
  if (ticks >= 0 && ticks < DYND_TICKS_PER_DAY) {
    tick = static_cast<int32_t>(ticks % DYND_TICKS_PER_SECOND);
    ticks = ticks / DYND_TICKS_PER_SECOND;
    second = static_cast<int8_t>(ticks % 60);
    ticks = ticks / 60;
    minute = static_cast<int8_t>(ticks % 60);
    hour = static_cast<int8_t>(ticks / 60);
  } else {
    set_to_na();
  }
}

void time_hmst::set_from_str(const char *begin, const char *end,
                             const char *&out_tz_begin, const char *&out_tz_end)
{
  if (!string_to_time(begin, end, *this, out_tz_begin, out_tz_end)) {
    stringstream ss;
    ss << "Unable to parse ";
    print_escaped_utf8_string(ss, begin, end);
    ss << " as a time";
    throw invalid_argument(ss.str());
  }
}

time_hmst time_hmst::get_current_local_time()
{
  // TODO: Could use C++11 chrono library
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
  time_hmst hmst;
  hmst.hour = tm_.tm_hour;
  hmst.minute = tm_.tm_min;
  hmst.second = tm_.tm_sec;
  hmst.tick = 0;
  return hmst;
}

const ndt::type &time_hmst::type()
{
  static ndt::type tp = ndt::struct_type::make(
      {"hour", "minute", "second", "tick"},
      {ndt::type::make<int8_t>(), ndt::type::make<int8_t>(),
       ndt::type::make<int8_t>(), ndt::type::make<int32_t>()});
  return tp;
}
