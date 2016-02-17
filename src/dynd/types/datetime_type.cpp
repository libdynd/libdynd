//
// Copyright (C) 2011-15 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <time.h>

#include <cerrno>
#include <algorithm>

#include <dynd/callable.hpp>
#include <dynd/types/datetime_type.hpp>
#include <dynd/types/date_util.hpp>
#include <dynd/types/string_type.hpp>
#include <dynd/types/typevar_type.hpp>
#include <dynd/exceptions.hpp>
#include <dynd/array_iter.hpp>
#include <dynd/parse.hpp>
#include <dynd/functional.hpp>

#include <datetime_strings.h>
#include <datetime_localtime.h>

using namespace std;
using namespace dynd;

namespace {

struct datetime_get_date_kernel : nd::base_kernel<datetime_get_date_kernel, 1> {
  ndt::type datetime_tp;

  datetime_get_date_kernel(const ndt::type &datetime_tp) : datetime_tp(datetime_tp) {}

  void single(char *dst, char *const *src)
  {
    const ndt::datetime_type *dd = reinterpret_cast<const ndt::datetime_type *>(datetime_tp.extended());
    datetime_tz_t tz = dd->get_timezone();
    if (tz == tz_utc || tz == tz_abstract) {
      int64_t days = **reinterpret_cast<int64_t *const *>(src);
      if (days < 0) {
        days -= (DYND_TICKS_PER_DAY - 1);
      }
      days /= DYND_TICKS_PER_DAY;

      *reinterpret_cast<int32_t *>(dst) = static_cast<int32_t>(days);
    }
    else {
      throw runtime_error("datetime date property only implemented for "
                          "UTC and abstract timezones");
    }
  }

  static void instantiate(char *DYND_UNUSED(static_data), char *DYND_UNUSED(data), nd::kernel_builder *ckb,
                          const ndt::type &DYND_UNUSED(dst_tp), const char *DYND_UNUSED(dst_arrmeta),
                          intptr_t DYND_UNUSED(nsrc), const ndt::type *src_tp,
                          const char *const *DYND_UNUSED(src_arrmeta), kernel_request_t kernreq,
                          intptr_t DYND_UNUSED(nkwd), const nd::array *DYND_UNUSED(kwds),
                          const std::map<std::string, ndt::type> &DYND_UNUSED(tp_vars))
  {
    ckb->emplace_back<datetime_get_date_kernel>(kernreq, src_tp[0]);
  }
};

struct datetime_get_time_kernel : nd::base_kernel<datetime_get_time_kernel, 1> {
  ndt::type datetime_tp;

  datetime_get_time_kernel(const ndt::type &datetime_tp) : datetime_tp(datetime_tp) {}

  void single(char *dst, char *const *src)
  {
    const ndt::datetime_type *dd = reinterpret_cast<const ndt::datetime_type *>(datetime_tp.extended());
    datetime_tz_t tz = dd->get_timezone();
    if (tz == tz_utc || tz == tz_abstract) {
      int64_t ticks = **reinterpret_cast<int64_t *const *>(src);
      ticks %= DYND_TICKS_PER_DAY;
      if (ticks < 0) {
        ticks += DYND_TICKS_PER_DAY;
      }
      *reinterpret_cast<int64_t *>(dst) = ticks;
    }
    else {
      throw runtime_error("datetime time property only implemented for "
                          "UTC and abstract timezones");
    }
  }

  static void instantiate(char *DYND_UNUSED(static_data), char *DYND_UNUSED(data), nd::kernel_builder *ckb,
                          const ndt::type &DYND_UNUSED(dst_tp), const char *DYND_UNUSED(dst_arrmeta),
                          intptr_t DYND_UNUSED(nsrc), const ndt::type *src_tp,
                          const char *const *DYND_UNUSED(src_arrmeta), kernel_request_t kernreq,
                          intptr_t DYND_UNUSED(nkwd), const nd::array *DYND_UNUSED(kwds),
                          const std::map<std::string, ndt::type> &DYND_UNUSED(tp_vars))
  {
    ckb->emplace_back<datetime_get_time_kernel>(kernreq, src_tp[0]);
  }
};

struct datetime_get_year_kernel : nd::base_kernel<datetime_get_year_kernel, 1> {
  ndt::type datetime_tp;

  datetime_get_year_kernel(const ndt::type &datetime_tp) : datetime_tp(datetime_tp) {}

  void single(char *dst, char *const *src)
  {
    const ndt::datetime_type *dd = reinterpret_cast<const ndt::datetime_type *>(datetime_tp.extended());
    datetime_tz_t tz = dd->get_timezone();
    if (tz == tz_utc || tz == tz_abstract) {
      date_ymd ymd;
      ymd.set_from_ticks(**reinterpret_cast<int64_t *const *>(src));
      *reinterpret_cast<int32_t *>(dst) = ymd.year;
    }
    else {
      throw runtime_error("datetime property access only implemented for "
                          "UTC and abstract timezones");
    }
  }

  static void instantiate(char *DYND_UNUSED(static_data), char *DYND_UNUSED(data), nd::kernel_builder *ckb,
                          const ndt::type &DYND_UNUSED(dst_tp), const char *DYND_UNUSED(dst_arrmeta),
                          intptr_t DYND_UNUSED(nsrc), const ndt::type *src_tp,
                          const char *const *DYND_UNUSED(src_arrmeta), kernel_request_t kernreq,
                          intptr_t DYND_UNUSED(nkwd), const nd::array *DYND_UNUSED(kwds),
                          const std::map<std::string, ndt::type> &DYND_UNUSED(tp_vars))
  {
    ckb->emplace_back<datetime_get_year_kernel>(kernreq, src_tp[0]);
  }
};

struct datetime_get_month_kernel : nd::base_kernel<datetime_get_month_kernel, 1> {
  ndt::type datetime_tp;

  datetime_get_month_kernel(const ndt::type &datetime_tp) : datetime_tp(datetime_tp) {}

  void single(char *dst, char *const *src)
  {
    const ndt::datetime_type *dd = reinterpret_cast<const ndt::datetime_type *>(datetime_tp.extended());
    datetime_tz_t tz = dd->get_timezone();
    if (tz == tz_utc || tz == tz_abstract) {
      date_ymd ymd;
      ymd.set_from_ticks(**reinterpret_cast<int64_t *const *>(src));
      *reinterpret_cast<int32_t *>(dst) = ymd.month;
    }
    else {
      throw runtime_error("datetime property access only implemented for "
                          "UTC and abstract timezones");
    }
  }

  static void instantiate(char *DYND_UNUSED(static_data), char *DYND_UNUSED(data), nd::kernel_builder *ckb,
                          const ndt::type &DYND_UNUSED(dst_tp), const char *DYND_UNUSED(dst_arrmeta),
                          intptr_t DYND_UNUSED(nsrc), const ndt::type *src_tp,
                          const char *const *DYND_UNUSED(src_arrmeta), kernel_request_t kernreq,
                          intptr_t DYND_UNUSED(nkwd), const nd::array *DYND_UNUSED(kwds),
                          const std::map<std::string, ndt::type> &DYND_UNUSED(tp_vars))
  {
    ckb->emplace_back<datetime_get_month_kernel>(kernreq, src_tp[0]);
  }
};

struct datetime_get_day_kernel : nd::base_kernel<datetime_get_day_kernel, 1> {
  ndt::type datetime_tp;

  datetime_get_day_kernel(const ndt::type &datetime_tp) : datetime_tp(datetime_tp) {}

  void single(char *dst, char *const *src)
  {
    const ndt::datetime_type *dd = reinterpret_cast<const ndt::datetime_type *>(datetime_tp.extended());
    datetime_tz_t tz = dd->get_timezone();
    if (tz == tz_utc || tz == tz_abstract) {
      date_ymd ymd;
      ymd.set_from_ticks(**reinterpret_cast<int64_t *const *>(src));
      *reinterpret_cast<int32_t *>(dst) = ymd.day;
    }
    else {
      throw runtime_error("datetime property access only implemented for "
                          "UTC and abstract timezones");
    }
  }

  static void instantiate(char *DYND_UNUSED(static_data), char *DYND_UNUSED(data), nd::kernel_builder *ckb,
                          const ndt::type &DYND_UNUSED(dst_tp), const char *DYND_UNUSED(dst_arrmeta),
                          intptr_t DYND_UNUSED(nsrc), const ndt::type *src_tp,
                          const char *const *DYND_UNUSED(src_arrmeta), kernel_request_t kernreq,
                          intptr_t DYND_UNUSED(nkwd), const nd::array *DYND_UNUSED(kwds),
                          const std::map<std::string, ndt::type> &DYND_UNUSED(tp_vars))
  {
    ckb->emplace_back<datetime_get_day_kernel>(kernreq, src_tp[0]);
  }
};

struct datetime_get_hour_kernel : nd::base_kernel<datetime_get_hour_kernel, 1> {
  ndt::type datetime_tp;

  datetime_get_hour_kernel(const ndt::type &datetime_tp) : datetime_tp(datetime_tp) {}

  void single(char *dst, char *const *src)
  {
    const ndt::datetime_type *dd = reinterpret_cast<const ndt::datetime_type *>(datetime_tp.extended());
    datetime_tz_t tz = dd->get_timezone();
    if (tz == tz_utc || tz == tz_abstract) {
      int64_t hour = **reinterpret_cast<int64_t *const *>(src) % DYND_TICKS_PER_DAY;
      if (hour < 0) {
        hour += DYND_TICKS_PER_DAY;
      }
      hour /= DYND_TICKS_PER_HOUR;
      *reinterpret_cast<int32_t *>(dst) = static_cast<int32_t>(hour);
    }
    else {
      throw runtime_error("datetime property access only implemented for "
                          "UTC and abstract timezones");
    }
  }

  static void instantiate(char *DYND_UNUSED(static_data), char *DYND_UNUSED(data), nd::kernel_builder *ckb,
                          const ndt::type &DYND_UNUSED(dst_tp), const char *DYND_UNUSED(dst_arrmeta),
                          intptr_t DYND_UNUSED(nsrc), const ndt::type *src_tp,
                          const char *const *DYND_UNUSED(src_arrmeta), kernel_request_t kernreq,
                          intptr_t DYND_UNUSED(nkwd), const nd::array *DYND_UNUSED(kwds),
                          const std::map<std::string, ndt::type> &DYND_UNUSED(tp_vars))
  {
    ckb->emplace_back<datetime_get_hour_kernel>(kernreq, src_tp[0]);
  }
};

struct datetime_get_minute_kernel : nd::base_kernel<datetime_get_minute_kernel, 1> {
  ndt::type datetime_tp;

  datetime_get_minute_kernel(const ndt::type &datetime_tp) : datetime_tp(datetime_tp) {}

  void single(char *dst, char *const *src)
  {
    const ndt::datetime_type *dd = reinterpret_cast<const ndt::datetime_type *>(datetime_tp.extended());
    datetime_tz_t tz = dd->get_timezone();
    if (tz == tz_utc || tz == tz_abstract) {
      int64_t minute = **reinterpret_cast<int64_t *const *>(src) % DYND_TICKS_PER_HOUR;
      if (minute < 0) {
        minute += DYND_TICKS_PER_HOUR;
      }
      minute /= DYND_TICKS_PER_MINUTE;
      *reinterpret_cast<int32_t *>(dst) = static_cast<int32_t>(minute);
    }
    else {
      throw runtime_error("datetime property access only implemented for UTC and "
                          "abstract timezones");
    }
  }

  static void instantiate(char *DYND_UNUSED(static_data), char *DYND_UNUSED(data), nd::kernel_builder *ckb,
                          const ndt::type &DYND_UNUSED(dst_tp), const char *DYND_UNUSED(dst_arrmeta),
                          intptr_t DYND_UNUSED(nsrc), const ndt::type *src_tp,
                          const char *const *DYND_UNUSED(src_arrmeta), kernel_request_t kernreq,
                          intptr_t DYND_UNUSED(nkwd), const nd::array *DYND_UNUSED(kwds),
                          const std::map<std::string, ndt::type> &DYND_UNUSED(tp_vars))
  {
    ckb->emplace_back<datetime_get_minute_kernel>(kernreq, src_tp[0]);
  }
};

struct datetime_get_second_kernel : nd::base_kernel<datetime_get_second_kernel, 1> {
  ndt::type datetime_tp;

  datetime_get_second_kernel(const ndt::type &datetime_tp) : datetime_tp(datetime_tp) {}

  void single(char *dst, char *const *src)
  {
    const ndt::datetime_type *dd = reinterpret_cast<const ndt::datetime_type *>(datetime_tp.extended());
    datetime_tz_t tz = dd->get_timezone();
    if (tz == tz_utc || tz == tz_abstract) {
      int64_t second = **reinterpret_cast<int64_t *const *>(src) % DYND_TICKS_PER_MINUTE;
      if (second < 0) {
        second += DYND_TICKS_PER_MINUTE;
      }
      second /= DYND_TICKS_PER_SECOND;
      *reinterpret_cast<int32_t *>(dst) = static_cast<int32_t>(second);
    }
    else {
      throw runtime_error("datetime property access only implemented for "
                          "UTC and abstract timezones");
    }
  }

  static void instantiate(char *DYND_UNUSED(static_data), char *DYND_UNUSED(data), nd::kernel_builder *ckb,
                          const ndt::type &DYND_UNUSED(dst_tp), const char *DYND_UNUSED(dst_arrmeta),
                          intptr_t DYND_UNUSED(nsrc), const ndt::type *src_tp,
                          const char *const *DYND_UNUSED(src_arrmeta), kernel_request_t kernreq,
                          intptr_t DYND_UNUSED(nkwd), const nd::array *DYND_UNUSED(kwds),
                          const std::map<std::string, ndt::type> &DYND_UNUSED(tp_vars))
  {
    ckb->emplace_back<datetime_get_second_kernel>(kernreq, src_tp[0]);
  }
};

struct datetime_get_microsecond_kernel : nd::base_kernel<datetime_get_microsecond_kernel, 1> {
  ndt::type datetime_tp;

  datetime_get_microsecond_kernel(const ndt::type &datetime_tp) : datetime_tp(datetime_tp) {}

  void single(char *dst, char *const *src)
  {
    const ndt::datetime_type *dd = reinterpret_cast<const ndt::datetime_type *>(datetime_tp.extended());
    datetime_tz_t tz = dd->get_timezone();
    if (tz == tz_utc || tz == tz_abstract) {
      int64_t microsecond = **reinterpret_cast<int64_t *const *>(src) % DYND_TICKS_PER_SECOND;
      if (microsecond < 0) {
        microsecond += DYND_TICKS_PER_SECOND;
      }
      microsecond /= DYND_TICKS_PER_MICROSECOND;
      *reinterpret_cast<int32_t *>(dst) = static_cast<int32_t>(microsecond);
    }
    else {
      throw runtime_error("datetime property access only implemented for "
                          "UTC and abstract timezones");
    }
  }

  static void instantiate(char *DYND_UNUSED(static_data), char *DYND_UNUSED(data), nd::kernel_builder *ckb,
                          const ndt::type &DYND_UNUSED(dst_tp), const char *DYND_UNUSED(dst_arrmeta),
                          intptr_t DYND_UNUSED(nsrc), const ndt::type *src_tp,
                          const char *const *DYND_UNUSED(src_arrmeta), kernel_request_t kernreq,
                          intptr_t DYND_UNUSED(nkwd), const nd::array *DYND_UNUSED(kwds),
                          const std::map<std::string, ndt::type> &DYND_UNUSED(tp_vars))
  {
    ckb->emplace_back<datetime_get_microsecond_kernel>(kernreq, src_tp[0]);
  }
};

struct datetime_get_tick_kernel : nd::base_kernel<datetime_get_tick_kernel, 1> {
  ndt::type datetime_tp;

  datetime_get_tick_kernel(const ndt::type &datetime_tp) : datetime_tp(datetime_tp) {}

  void single(char *dst, char *const *src)
  {
    const ndt::datetime_type *dd = reinterpret_cast<const ndt::datetime_type *>(datetime_tp.extended());
    datetime_tz_t tz = dd->get_timezone();
    if (tz == tz_utc || tz == tz_abstract) {
      int64_t tick = **reinterpret_cast<int64_t *const *>(src) % 10000000LL;
      if (tick < 0) {
        tick += 10000000LL;
      }
      *reinterpret_cast<int32_t *>(dst) = static_cast<int32_t>(tick);
    }
    else {
      throw runtime_error("datetime property access only implemented for "
                          "UTC and abstract timezones");
    }
  }

  static void instantiate(char *DYND_UNUSED(static_data), char *DYND_UNUSED(data), nd::kernel_builder *ckb,
                          const ndt::type &DYND_UNUSED(dst_tp), const char *DYND_UNUSED(dst_arrmeta),
                          intptr_t DYND_UNUSED(nsrc), const ndt::type *src_tp,
                          const char *const *DYND_UNUSED(src_arrmeta), kernel_request_t kernreq,
                          intptr_t DYND_UNUSED(nkwd), const nd::array *DYND_UNUSED(kwds),
                          const std::map<std::string, ndt::type> &DYND_UNUSED(tp_vars))
  {
    ckb->emplace_back<datetime_get_tick_kernel>(kernreq, src_tp[0]);
  }
};

} // anonymous namespace

ndt::datetime_type::datetime_type(datetime_tz_t timezone)
    : base_type(datetime_id, datetime_kind, 8, alignof(int64_t), type_flag_none, 0, 0, 0), m_timezone(timezone)
{
}

ndt::datetime_type::~datetime_type() {}

void ndt::datetime_type::set_cal(const char *DYND_UNUSED(arrmeta), char *data, assign_error_mode errmode, int32_t year,
                                 int32_t month, int32_t day, int32_t hour, int32_t minute, int32_t second,
                                 int32_t tick) const
{
  if (errmode != assign_error_nocheck) {
    if (!date_ymd::is_valid(year, month, day)) {
      stringstream ss;
      ss << "invalid input year/month/day " << year << "/" << month << "/" << day;
      throw runtime_error(ss.str());
    }
    if (hour < 0 || hour >= 24) {
      stringstream ss;
      ss << "invalid input hour " << hour << " for " << type(this, true);
      throw runtime_error(ss.str());
    }
    if (minute < 0 || minute >= 60) {
      stringstream ss;
      ss << "invalid input minute " << minute << " for " << type(this, true);
      throw runtime_error(ss.str());
    }
    if (second < 0 || second >= 60) {
      stringstream ss;
      ss << "invalid input second " << second << " for " << type(this, true);
      throw runtime_error(ss.str());
    }
    if (tick < 0 || tick >= 1000000000) {
      stringstream ss;
      ss << "invalid input tick (100*nanosecond) " << tick << " for " << type(this, true);
      throw runtime_error(ss.str());
    }
  }

  datetime_struct dts;
  dts.ymd.year = year;
  dts.ymd.month = month;
  dts.ymd.day = day;
  dts.hmst.hour = hour;
  dts.hmst.minute = minute;
  dts.hmst.second = second;
  dts.hmst.tick = tick;

  *reinterpret_cast<int64_t *>(data) = dts.to_ticks();
}

void ndt::datetime_type::set_from_utf8_string(const char *DYND_UNUSED(arrmeta), char *data, const char *utf8_begin,
                                              const char *utf8_end, const eval::eval_context *ectx) const
{
  datetime_struct dts;
  const char *tz_begin = NULL, *tz_end = NULL;
  dts.set_from_str(utf8_begin, utf8_end, date_parse_no_ambig, 70, ectx->errmode, tz_begin, tz_end);
  if (m_timezone != tz_abstract && tz_begin != tz_end) {
    if (m_timezone == tz_utc &&
        (compare_range_to_literal(tz_begin, tz_end, "Z") || compare_range_to_literal(tz_begin, tz_end, "UTC"))) {
      // It's a UTC time to a UTC time zone
    }
    else {
      stringstream ss;
      ss << "DyND time zone support is partial, cannot handle ";
      ss.write(tz_begin, tz_end - tz_begin);
      throw runtime_error(ss.str());
    }
  }
  *reinterpret_cast<int64_t *>(data) = dts.to_ticks();
}

void ndt::datetime_type::get_cal(const char *DYND_UNUSED(arrmeta), const char *data, int32_t &out_year,
                                 int32_t &out_month, int32_t &out_day, int32_t &out_hour, int32_t &out_min,
                                 int32_t &out_sec, int32_t &out_tick) const
{
  datetime_struct dts;
  dts.set_from_ticks(*reinterpret_cast<const int64_t *>(data));
  out_year = dts.ymd.year;
  out_month = dts.ymd.month;
  out_day = dts.ymd.day;
  out_hour = dts.hmst.hour;
  out_min = dts.hmst.minute;
  out_sec = dts.hmst.second;
  out_tick = dts.hmst.tick;
}

void ndt::datetime_type::print_data(std::ostream &o, const char *DYND_UNUSED(arrmeta), const char *data) const
{
  datetime_struct dts;
  dts.set_from_ticks(*reinterpret_cast<const int64_t *>(data));
  o << dts.to_str();
  if (m_timezone == tz_utc) {
    o << "Z";
  }
}

void ndt::datetime_type::print_type(std::ostream &o) const
{
  if (m_timezone == tz_abstract) {
    o << "datetime";
  }
  else {
    o << "datetime[tz='";
    switch (m_timezone) {
    case tz_utc:
      o << "UTC";
      break;
    default:
      o << "(invalid " << (int32_t)m_timezone << ")";
      break;
    }
    o << "']";
  }
}

bool ndt::datetime_type::is_lossless_assignment(const type &dst_tp, const type &src_tp) const
{
  if (dst_tp.extended() == this) {
    if (src_tp.extended() == this) {
      return true;
    }
    else if (src_tp.get_id() == date_id) {
      // There is only one possibility for the datetime type (TODO: timezones!)
      return true;
    }
    else {
      return false;
    }
  }
  else {
    return false;
  }
}

bool ndt::datetime_type::operator==(const base_type &rhs) const
{
  if (this == &rhs) {
    return true;
  }
  else if (rhs.get_id() != datetime_id) {
    return false;
  }
  else {
    const datetime_type &r = static_cast<const datetime_type &>(rhs);
    // TODO: When "other" timezone data is supported, need to compare them too
    return m_timezone == r.m_timezone;
  }
}

std::map<std::string, nd::callable> ndt::datetime_type::get_dynamic_type_functions() const
{
  return std::map<std::string, nd::callable>();
}

///////// properties on the nd::array

std::map<std::string, nd::callable> ndt::datetime_type::get_dynamic_array_properties() const
{
  std::map<std::string, nd::callable> properties;
  properties["year"] = nd::functional::adapt(ndt::make_type<int32_t>(),
                                             nd::callable::make<datetime_get_year_kernel>(ndt::type("(Any) -> Any")));
  properties["month"] = nd::functional::adapt(ndt::make_type<int32_t>(),
                                              nd::callable::make<datetime_get_month_kernel>(ndt::type("(Any) -> Any")));
  properties["day"] = nd::functional::adapt(ndt::make_type<int32_t>(),
                                            nd::callable::make<datetime_get_day_kernel>(ndt::type("(Any) -> Any")));

  return properties;
}
