//
// Copyright (C) 2011-15 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <dynd/array.hpp>
#include <dynd/func/callable.hpp>
#include <dynd/types/time_type.hpp>
#include <dynd/types/typevar_type.hpp>
#include <dynd/parser_util.hpp>
#include <dynd/functional.hpp>

using namespace std;
using namespace dynd;

namespace {
struct time_get_tick_kernel : nd::base_kernel<time_get_tick_kernel, 1> {
  void single(char *dst, char *const *src)
  {
    int64_t ticks = **reinterpret_cast<int64_t *const *>(src);
    *reinterpret_cast<int32_t *>(dst) = static_cast<int32_t>(ticks % 10000000);
  }
};

struct time_get_struct_kernel : nd::base_kernel<time_get_struct_kernel, 1> {
  void single(char *dst, char *const *src)
  {
    time_hmst *dst_struct = reinterpret_cast<time_hmst *>(dst);
    dst_struct->set_from_ticks(**reinterpret_cast<int64_t *const *>(src));
  }
};

struct time_set_struct_kernel : nd::base_kernel<time_set_struct_kernel, 1> {
  void single(char *dst, char *const *src)
  {
    time_hmst *src_struct = *reinterpret_cast<time_hmst *const *>(src);
    *reinterpret_cast<int64_t *>(dst) = src_struct->to_ticks();
  }
};

struct time_get_hour_kernel : nd::base_kernel<time_get_hour_kernel, 1> {
  void single(char *dst, char *const *src)
  {
    int64_t ticks = **reinterpret_cast<int64_t *const *>(src);
    *reinterpret_cast<int32_t *>(dst) = static_cast<int32_t>(ticks / DYND_TICKS_PER_HOUR);
  }
};

struct time_get_minute_kernel : nd::base_kernel<time_get_minute_kernel, 1> {
  void single(char *dst, char *const *src)
  {
    int64_t ticks = **reinterpret_cast<int64_t *const *>(src);
    *reinterpret_cast<int32_t *>(dst) = static_cast<int32_t>((ticks / DYND_TICKS_PER_MINUTE) % 60);
  }
};

struct time_get_second_kernel : nd::base_kernel<time_get_second_kernel, 1> {
  void single(char *dst, char *const *src)
  {
    int64_t ticks = **reinterpret_cast<int64_t *const *>(src);
    *reinterpret_cast<int32_t *>(dst) = static_cast<int32_t>((ticks / DYND_TICKS_PER_SECOND) % 60);
  }
};

struct time_get_microsecond_kernel : nd::base_kernel<time_get_microsecond_kernel, 1> {
  void single(char *dst, char *const *src)
  {
    int64_t ticks = **reinterpret_cast<int64_t *const *>(src);
    *reinterpret_cast<int32_t *>(dst) = static_cast<int32_t>((ticks / DYND_TICKS_PER_MICROSECOND) % 1000000);
  }
};
}

ndt::time_type::time_type(datetime_tz_t timezone)
    : base_type(time_type_id, datetime_kind, 8, alignof(int64_t), type_flag_none, 0, 0, 0), m_timezone(timezone)
{
}

ndt::time_type::~time_type() {}

void ndt::time_type::set_time(const char *DYND_UNUSED(arrmeta), char *data, assign_error_mode errmode, int32_t hour,
                              int32_t minute, int32_t second, int32_t tick) const
{
  if (errmode != assign_error_nocheck && !time_hmst::is_valid(hour, minute, second, tick)) {
    stringstream ss;
    ss << "invalid input time " << hour << ":" << minute << ":" << second << ", ticks: " << tick;
    throw runtime_error(ss.str());
  }

  *reinterpret_cast<int64_t *>(data) = time_hmst::to_ticks(hour, minute, second, tick);
}

void ndt::time_type::set_from_utf8_string(const char *DYND_UNUSED(arrmeta), char *data, const char *utf8_begin,
                                          const char *utf8_end, const eval::eval_context *DYND_UNUSED(ectx)) const
{
  time_hmst hmst;
  const char *tz_begin = NULL, *tz_end = NULL;
  // TODO: Use errmode to adjust strictness of the parsing
  hmst.set_from_str(utf8_begin, utf8_end, tz_begin, tz_end);
  if (m_timezone != tz_abstract && tz_begin != tz_end) {
    if (m_timezone == tz_utc && (parse::compare_range_to_literal(tz_begin, tz_end, "Z") ||
                                 parse::compare_range_to_literal(tz_begin, tz_end, "UTC"))) {
      // It's a UTC time to a UTC time zone
    }
    else {
      stringstream ss;
      ss << "DyND time zone support is partial, cannot handle ";
      ss.write(tz_begin, tz_end - tz_begin);
      throw runtime_error(ss.str());
    }
  }
  *reinterpret_cast<int64_t *>(data) = hmst.to_ticks();
}

time_hmst ndt::time_type::get_time(const char *DYND_UNUSED(arrmeta), const char *data) const
{
  time_hmst hmst;
  hmst.set_from_ticks(*reinterpret_cast<const int64_t *>(data));
  return hmst;
}

void ndt::time_type::print_data(std::ostream &o, const char *DYND_UNUSED(arrmeta), const char *data) const
{
  time_hmst hmst;
  hmst.set_from_ticks(*reinterpret_cast<const int64_t *>(data));
  o << hmst.to_str();
  if (m_timezone == tz_utc) {
    o << "Z";
  }
}

void ndt::time_type::print_type(std::ostream &o) const
{
  if (m_timezone == tz_abstract) {
    o << "time";
  }
  else {
    o << "time[tz='";
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

bool ndt::time_type::is_lossless_assignment(const type &dst_tp, const type &src_tp) const
{
  if (dst_tp.extended() == this) {
    if (src_tp.extended() == this) {
      return true;
    }
    else if (src_tp.get_type_id() == time_type_id) {
      // There is only one possibility for the time type (TODO: timezones!)
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

bool ndt::time_type::operator==(const base_type &rhs) const
{
  if (this == &rhs) {
    return true;
  }
  else if (rhs.get_type_id() != time_type_id) {
    return false;
  }
  else {
    const time_type &r = static_cast<const time_type &>(rhs);
    // TODO: When "other" timezone data is supported, need to compare them too
    return m_timezone == r.m_timezone;
  }
}

size_t ndt::time_type::make_comparison_kernel(void *ckb, intptr_t ckb_offset, const type &src0_tp,
                                              const char *src0_arrmeta, const type &src1_tp, const char *src1_arrmeta,
                                              comparison_type_t comptype, const eval::eval_context *ectx) const
{
  if (this == src0_tp.extended()) {
    if (*this == *src1_tp.extended()) {
      return make_builtin_type_comparison_kernel(ckb, ckb_offset, int64_type_id, int64_type_id, comptype);
    }
    else if (!src1_tp.is_builtin()) {
      return src1_tp.extended()->make_comparison_kernel(ckb, ckb_offset, src0_tp, src0_arrmeta, src1_tp, src1_arrmeta,
                                                        comptype, ectx);
    }
  }

  throw not_comparable_error(src0_tp, src1_tp, comptype);
}

///////// properties on the nd::array

void ndt::time_type::get_dynamic_array_properties(std::map<std::string, nd::callable> &properties) const
{

  properties["hour"] = nd::functional::adapt(ndt::type::make<int32_t>(),
                                             nd::callable::make<time_get_hour_kernel>(ndt::type("(Any) -> Any")));
  properties["minute"] = nd::functional::adapt(ndt::type::make<int32_t>(),
                                               nd::callable::make<time_get_minute_kernel>(ndt::type("(Any) -> Any")));
  properties["second"] = nd::functional::adapt(ndt::type::make<int32_t>(),
                                               nd::callable::make<time_get_second_kernel>(ndt::type("(Any) -> Any")));
  properties["microsecond"] = nd::functional::adapt(
      ndt::type::make<int32_t>(), nd::callable::make<time_get_microsecond_kernel>(ndt::type("(Any) -> Any")));
  properties["tick"] = nd::functional::adapt(ndt::type::make<int32_t>(),
                                             nd::callable::make<time_get_tick_kernel>(ndt::type("(Any) -> Any")));
}

void ndt::time_type::get_dynamic_array_functions(std::map<std::string, nd::callable> &functions) const
{
  functions["to_struct"] = nd::functional::adapt(ndt::type("{hour: int8, minute: int8, second: int8, tick: int32}"),
                                                 nd::callable::make<time_get_struct_kernel>(ndt::type("(Any) -> Any")));
}
