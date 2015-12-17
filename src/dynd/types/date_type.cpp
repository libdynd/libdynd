//
// Copyright (C) 2011-15 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <time.h>

#include <cerrno>
#include <algorithm>

#include <dynd/callable.hpp>
#include <dynd/types/date_type.hpp>
#include <dynd/types/string_type.hpp>
#include <dynd/types/adapt_type.hpp>
#include <dynd/types/option_type.hpp>
#include <dynd/types/typevar_type.hpp>
#include <dynd/functional.hpp>
#include <dynd/func/elwise.hpp>
#include <dynd/exceptions.hpp>
#include <dynd/array_iter.hpp>
#include <dynd/parse.hpp>

#include <datetime_strings.h>
#include <datetime_localtime.h>

using namespace std;
using namespace dynd;

namespace {

struct date_get_year_kernel : nd::base_kernel<date_get_year_kernel, 1> {
  void single(char *dst, char *const *src)
  {
    date_ymd ymd;
    ymd.set_from_days(**reinterpret_cast<int32_t *const *>(src));
    *reinterpret_cast<int32_t *>(dst) = ymd.year;
  }
};

struct date_get_month_kernel : nd::base_kernel<date_get_month_kernel, 1> {
  void single(char *dst, char *const *src)
  {
    date_ymd ymd;
    ymd.set_from_days(**reinterpret_cast<int32_t *const *>(src));
    *reinterpret_cast<int32_t *>(dst) = ymd.month;
  }
};

struct date_get_day_kernel : nd::base_kernel<date_get_day_kernel, 1> {
  void single(char *dst, char *const *src)
  {
    date_ymd ymd;
    ymd.set_from_days(**reinterpret_cast<int32_t *const *>(src));
    *reinterpret_cast<int32_t *>(dst) = ymd.day;
  }
};

} // anonymous namespace

struct date_set_struct_kernel : nd::base_kernel<date_set_struct_kernel, 1> {
  void single(char *dst, char *const *src)
  {
    const date_ymd *src_struct = *reinterpret_cast<date_ymd *const *>(src);
    *reinterpret_cast<int32_t *>(dst) = src_struct->to_days();
  }
};

struct date_get_struct_kernel : nd::base_kernel<date_get_struct_kernel, 1> {
  void single(char *dst, char *const *src)
  {
    date_ymd *dst_struct = reinterpret_cast<date_ymd *>(dst);
    dst_struct->set_from_days(**reinterpret_cast<int32_t *const *>(src));
  }
};

struct date_get_weekday_kernel : nd::base_kernel<date_get_weekday_kernel, 1> {
  void single(char *dst, char *const *src)
  {
    int32_t days = **reinterpret_cast<int32_t *const *>(src);
    // 1970-01-05 is Monday
    int weekday = (int)((days - 4) % 7);
    if (weekday < 0) {
      weekday += 7;
    }
    *reinterpret_cast<int32_t *>(dst) = weekday;
  }
};

ndt::date_type::date_type() : base_type(date_type_id, datetime_kind, 4, alignof(int32_t), type_flag_none, 0, 0, 0) {}

ndt::date_type::~date_type() {}

void ndt::date_type::set_ymd(const char *DYND_UNUSED(arrmeta), char *data, assign_error_mode errmode, int32_t year,
                             int32_t month, int32_t day) const
{
  if (errmode != assign_error_nocheck && !date_ymd::is_valid(year, month, day)) {
    stringstream ss;
    ss << "invalid input year/month/day " << year << "/" << month << "/" << day;
    throw runtime_error(ss.str());
  }

  *reinterpret_cast<int32_t *>(data) = date_ymd::to_days(year, month, day);
}

void ndt::date_type::set_from_utf8_string(const char *DYND_UNUSED(arrmeta), char *data, const char *utf8_begin,
                                          const char *utf8_end, const eval::eval_context *ectx) const
{
  date_ymd ymd;
  ymd.set_from_str(utf8_begin, utf8_end, ectx->date_parse_order, ectx->century_window, ectx->errmode);
  *reinterpret_cast<int32_t *>(data) = ymd.to_days();
}

date_ymd ndt::date_type::get_ymd(const char *DYND_UNUSED(arrmeta), const char *data) const
{
  date_ymd ymd;
  ymd.set_from_days(*reinterpret_cast<const int32_t *>(data));
  return ymd;
}

void ndt::date_type::print_data(std::ostream &o, const char *DYND_UNUSED(arrmeta), const char *data) const
{
  date_ymd ymd;
  ymd.set_from_days(*reinterpret_cast<const int32_t *>(data));
  std::string s = ymd.to_str();
  if (s.empty()) {
    o << "NA";
  }
  else {
    o << s;
  }
}

void ndt::date_type::print_type(std::ostream &o) const { o << "date"; }

bool ndt::date_type::is_lossless_assignment(const type &dst_tp, const type &src_tp) const
{
  if (dst_tp.extended() == this) {
    if (src_tp.extended() == this) {
      return true;
    }
    else if (src_tp.get_type_id() == date_type_id) {
      // There is only one possibility for the date type (TODO: timezones!)
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

bool ndt::date_type::operator==(const base_type &rhs) const
{
  if (this == &rhs) {
    return true;
  }
  else if (rhs.get_type_id() != date_type_id) {
    return false;
  }
  else {
    // There is only one possibility for the date type (TODO: timezones!)
    return true;
  }
}

size_t ndt::date_type::make_comparison_kernel(void *ckb, intptr_t ckb_offset, const type &src0_tp,
                                              const char *src0_arrmeta, const type &src1_tp, const char *src1_arrmeta,
                                              comparison_type_t comptype, const eval::eval_context *ectx) const
{
  if (this == src0_tp.extended()) {
    if (*this == *src1_tp.extended()) {
      return make_builtin_type_comparison_kernel(ckb, ckb_offset, int32_type_id, int32_type_id, comptype);
    }
    else if (!src1_tp.is_builtin()) {
      return src1_tp.extended()->make_comparison_kernel(ckb, ckb_offset, src0_tp, src0_arrmeta, src1_tp, src1_arrmeta,
                                                        comptype, ectx);
    }
  }

  throw not_comparable_error(src0_tp, src1_tp, comptype);
}

///////// functions on the type

static int32_t fn_type_today(ndt::type DYND_UNUSED(dt)) { return date_ymd::get_current_local_date().to_days(); }

static int32_t date_from_ymd(int year, int month, int day)
{
  date_ymd ymd;
  ymd.year = year;
  ymd.month = month;
  ymd.day = day;
  if (!ymd.is_valid()) {
    stringstream ss;
    ss << "invalid year/month/day " << ymd.year << "/" << ymd.month << "/" << ymd.day;
    throw runtime_error(ss.str());
  }
  return ymd.to_days();
}

static int32_t fn_type_construct(ndt::type DYND_UNUSED(dt), int year, int month, int day)
{
  return date_from_ymd(year, month, day);
}

/*
static nd::array fn_type_construct(ndt::type DYND_UNUSED(dt), int32_t year,
                                   int32_t month, int32_t day)
{
  return

  nd::callable af =
nd::functional::elwise(nd::functional::apply(date_from_ymd));


  return af(year_as_int, month_as_int, day_as_int)
      .view_scalars(ndt::date_type::make());
}
*/

void ndt::date_type::get_dynamic_type_functions(std::map<std::string, nd::callable> &functions) const
{
  functions["today"] = nd::functional::apply(&fn_type_today, "self");
  functions["__construct__"] = nd::functional::apply(&fn_type_construct, "self", "year", "month", "day");
}

///////// properties on the nd::array

void ndt::date_type::get_dynamic_array_properties(std::map<std::string, nd::callable> &properties) const
{
  properties["year"] = nd::functional::adapt(ndt::make_type<int32_t>(),
                                             nd::callable::make<date_get_year_kernel>(ndt::type("(Any) -> Any")));
  properties["month"] = nd::functional::adapt(ndt::make_type<int32_t>(),
                                              nd::callable::make<date_get_month_kernel>(ndt::type("(Any) -> Any")));
  properties["day"] = nd::functional::adapt(ndt::make_type<int32_t>(),
                                            nd::callable::make<date_get_day_kernel>(ndt::type("(Any) -> Any")));
}

///////// functions on the nd::array

/*
struct strftime_kernel : nd::base_kernel<strftime_kernel> {
  nd::array self;
  std::string format;

  strftime_kernel(const nd::array &self, std::string format) : self(self), format(format) {}

  void single(nd::array *dst, nd::array *const *DYND_UNUSED(src)) { *dst = helper(self, format); }

  static void resolve_dst_type(char *DYND_UNUSED(static_data), char *DYND_UNUSED(data), ndt::type &dst_tp,
                               intptr_t DYND_UNUSED(nsrc), const ndt::type *DYND_UNUSED(src_tp),
                               intptr_t DYND_UNUSED(nkwd), const nd::array *kwds,
                               const std::map<std::string, ndt::type> &DYND_UNUSED(tp_vars))
  {
    dst_tp = helper(kwds[0], kwds[1].as<std::string>()).get_type();
  }

  static intptr_t instantiate(char *DYND_UNUSED(static_data), char *DYND_UNUSED(data), void *ckb, intptr_t ckb_offset,
                              const ndt::type &DYND_UNUSED(dst_tp), const char *DYND_UNUSED(dst_arrmeta),
                              intptr_t DYND_UNUSED(nsrc), const ndt::type *DYND_UNUSED(src_tp),
                              const char *const *DYND_UNUSED(src_arrmeta), kernel_request_t kernreq,
                              const eval::eval_context *DYND_UNUSED(ectx), intptr_t DYND_UNUSED(nkwd),
                              const nd::array *kwds, const std::map<std::string, ndt::type> &DYND_UNUSED(tp_vars))
  {
    make(ckb, kernreq, ckb_offset, kwds[0], kwds[1].as<std::string>());
    return ckb_offset;
  }

  static nd::array helper(const nd::array &n, const std::string &format)
  {
    // TODO: Allow 'format' itself to be an array, with broadcasting, etc.
    if (format.empty()) {
      throw runtime_error("format string for strftime should not be empty");
    }
    return n.replace_dtype(
        ndt::unary_expr_type::make(ndt::make_type<ndt::string_type>(), n.get_dtype(), make_strftime_kernelgen(format)));
  }
};
*/

/*
struct replace_kernel : nd::base_kernel<replace_kernel> {
  nd::array self;
  int32_t year;
  int32_t month;
  int32_t day;

  replace_kernel(const nd::array &self, int32_t year, int32_t month, int32_t day)
      : self(self), year(year), month(month), day(day)
  {
  }

  void single(nd::array *dst, nd::array *const *DYND_UNUSED(src)) { *dst = helper(self, year, month, day); }

  static void resolve_dst_type(char *DYND_UNUSED(static_data), char *DYND_UNUSED(data), ndt::type &dst_tp,
                               intptr_t DYND_UNUSED(nsrc), const ndt::type *DYND_UNUSED(src_tp),
                               intptr_t DYND_UNUSED(nkwd), const nd::array *kwds,
                               const std::map<std::string, ndt::type> &DYND_UNUSED(tp_vars))
  {
    dst_tp = helper(kwds[0], kwds[1].is_missing() ? numeric_limits<int32_t>::max() : kwds[1].as<int32_t>(),
                    kwds[2].is_missing() ? numeric_limits<int32_t>::max() : kwds[2].as<int32_t>(),
                    kwds[3].is_missing() ? numeric_limits<int32_t>::max() : kwds[3].as<int32_t>())
                 .get_type();
  }

  static intptr_t instantiate(char *DYND_UNUSED(static_data), char *DYND_UNUSED(data), void *ckb, intptr_t ckb_offset,
                              const ndt::type &DYND_UNUSED(dst_tp), const char *DYND_UNUSED(dst_arrmeta),
                              intptr_t DYND_UNUSED(nsrc), const ndt::type *DYND_UNUSED(src_tp),
                              const char *const *DYND_UNUSED(src_arrmeta), kernel_request_t kernreq,
                              const eval::eval_context *DYND_UNUSED(ectx), intptr_t DYND_UNUSED(nkwd),
                              const nd::array *kwds, const std::map<std::string, ndt::type> &DYND_UNUSED(tp_vars))
  {
    make(ckb, kernreq, ckb_offset, kwds[0],
         kwds[1].is_missing() ? numeric_limits<int32_t>::max() : kwds[1].as<int32_t>(),
         kwds[2].is_missing() ? numeric_limits<int32_t>::max() : kwds[2].as<int32_t>(),
         kwds[3].is_missing() ? numeric_limits<int32_t>::max() : kwds[3].as<int32_t>());
    return ckb_offset;
  }

  static nd::array helper(const nd::array &n, int32_t year, int32_t month, int32_t day)
  {
    // TODO: Allow 'year', 'month', and 'day' to be arrays, with broadcasting,
    // etc.
    if (year == numeric_limits<int32_t>::max() && month == numeric_limits<int32_t>::max() &&
        day == numeric_limits<int32_t>::max()) {
      throw std::runtime_error("no parameters provided to date.replace, should provide at least one");
    }
    return n.replace_dtype(
        ndt::unary_expr_type::make(ndt::date_type::make(), n.get_dtype(), make_replace_kernelgen(year, month, day)));
  }
};

static nd::array function_ndo_replace(const nd::array &n, int32_t year, int32_t month, int32_t day)
{
  nd::callable f =
      nd::callable::make<weekday_kernel>(ndt::type("(self: Any, year: int32, month: int32, day: int32) -> Any"));
  return f(kwds("self", n, "year", year, "month", month, "day", day));
}
*/

void ndt::date_type::get_dynamic_array_functions(std::map<std::string, nd::callable> &functions) const
{
  functions["to_struct"] = nd::functional::adapt(ndt::type("{year: int16, month: int8, day: int8}"),
                                                 nd::callable::make<date_get_struct_kernel>(ndt::type("(Any) -> Any")));
  functions["weekday"] = nd::functional::adapt(ndt::make_type<int32_t>(),
                                               nd::callable::make<date_get_weekday_kernel>(ndt::type("(Any) -> Any")));

  //          "strftime", nd::callable::make<strftime_kernel>(ndt::type("(self: Any, format: string) -> Any"))),
  //        "replace", gfunc::make_callable_with_default(&function_ndo_replace, "self", "year", "month", "day",
  //                                                   numeric_limits<int32_t>::max(), numeric_limits<int32_t>::max(),
  //                                                 numeric_limits<int32_t>::max()))};
}
