//
// Copyright (C) 2011-15 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <dynd/kernels/base_kernel.hpp>
#include <dynd/kernels/datetime_adapter_kernels.hpp>
#include <dynd/parser_util.hpp>
#include <dynd/types/datetime_parser.hpp>
#include <dynd/types/datetime_type.hpp>
#include <dynd/types/callable_type.hpp>
#include <dynd/func/callable.hpp>
#include <dynd/string.hpp>

using namespace std;
using namespace dynd;

namespace {
template <class T>
inline T floordiv(T a, T b)
{
  return (a - (a < 0 ? b - 1 : 0)) / b;
}
} // anonymous namespace

/**
 * Matches netcdf date metadata like "hours since 2001-1-1T03:00".
 */
static bool parse_datetime_since(const char *begin, const char *end, int64_t &out_epoch_datetime,
                                 int64_t &out_unit_factor, int64_t &out_unit_divisor)
{
  if (parse::parse_token(begin, end, "hours")) {
    out_unit_factor = DYND_TICKS_PER_HOUR;
  } else if (parse::parse_token(begin, end, "minutes")) {
    out_unit_factor = DYND_TICKS_PER_MINUTE;
  } else if (parse::parse_token(begin, end, "seconds")) {
    out_unit_factor = DYND_TICKS_PER_SECOND;
  } else if (parse::parse_token(begin, end, "milliseconds")) {
    out_unit_factor = DYND_TICKS_PER_MILLISECOND;
  } else if (parse::parse_token(begin, end, "microseconds")) {
    out_unit_factor = DYND_TICKS_PER_MICROSECOND;
  } else if (parse::parse_token(begin, end, "nanoseconds")) {
    out_unit_divisor = DYND_NANOSECONDS_PER_TICK;
  } else {
    return false;
  }
  if (!parse::skip_required_whitespace(begin, end)) {
    return false;
  }
  // The tokens supported by netcdf, as from the udunits libarary
  if (!parse::parse_token(begin, end, "since") && !parse::parse_token(begin, end, "after") &&
      !parse::parse_token(begin, end, "from") && !parse::parse_token(begin, end, "ref") &&
      !parse::parse_token(begin, end, '@')) {
    return false;
  }
  if (!parse::skip_required_whitespace(begin, end)) {
    return false;
  }
  datetime_struct epoch;
  const char *tz_begin = NULL, *tz_end = NULL;
  if (!parse::parse_datetime(begin, end, date_parse_no_ambig, 0, epoch, tz_begin, tz_end)) {
    int year;
    if (parse::parse_date(begin, end, epoch.ymd, date_parse_no_ambig, 0)) {
      epoch.hmst.set_to_zero();
    } else if (parse::parse_4digit_int_no_ws(begin, end, year)) {
      epoch.ymd.year = static_cast<int16_t>(year);
      epoch.ymd.month = 1;
      epoch.ymd.day = 1;
      epoch.hmst.set_to_zero();
    } else {
      return false;
    }
  }
  // TODO: Apply TZ to make the epoch UTC
  if (tz_begin != tz_end && !parse::compare_range_to_literal(tz_begin, tz_end, "UTC") &&
      !parse::compare_range_to_literal(tz_begin, tz_end, "GMT")) {
    return false;
  }
  parse::skip_whitespace(begin, end);
  out_epoch_datetime = epoch.to_ticks();
  return begin == end;
}

namespace {
template <class Tsrc, class Tdst>
struct int_multiply_and_offset_ck : nd::base_kernel<int_multiply_and_offset_ck<Tsrc, Tdst>, 1> {
  pair<Tdst, Tdst> m_factor_offset;

  void single(char *dst, char *const *src)
  {
    Tsrc value = *reinterpret_cast<Tsrc *>(src[0]);
    *reinterpret_cast<Tdst *>(dst) = value != std::numeric_limits<Tsrc>::min()
                                         ? m_factor_offset.first * static_cast<Tdst>(value) + m_factor_offset.second
                                         : std::numeric_limits<Tdst>::min();
  }
};

template <class Tsrc, class Tdst>
static intptr_t instantiate_int_multiply_and_offset_callable(
    char *static_data, size_t DYND_UNUSED(data_size), char *DYND_UNUSED(data), void *ckb, intptr_t ckb_offset,
    const ndt::type &DYND_UNUSED(dst_tp), const char *DYND_UNUSED(dst_arrmeta), intptr_t DYND_UNUSED(nsrc),
    const ndt::type *DYND_UNUSED(src_tp), const char *const *DYND_UNUSED(src_arrmeta), kernel_request_t kernreq,
    const eval::eval_context *DYND_UNUSED(ectx), intptr_t DYND_UNUSED(nkwd), const nd::array *DYND_UNUSED(kwds),
    const std::map<std::string, ndt::type> &DYND_UNUSED(tp_vars))
{
  typedef int_multiply_and_offset_ck<Tsrc, Tdst> self_type;
  self_type *self = self_type::make(ckb, kernreq, ckb_offset);
  self->m_factor_offset = *reinterpret_cast<pair<Tdst, Tdst> *>(static_data);
  return ckb_offset;
}

template <class Tsrc, class Tdst>
nd::callable make_int_multiply_and_offset_callable(Tdst factor, Tdst offset, const ndt::type &func_proto)
{
  return nd::callable(func_proto, kernel_request_single, single_t(), make_pair(factor, offset), 0, NULL, NULL,
                      &instantiate_int_multiply_and_offset_callable<Tsrc, Tdst>);
}

template <class Tsrc, class Tdst>
struct int_offset_and_divide_ck : nd::base_kernel<int_offset_and_divide_ck<Tsrc, Tdst>, 1> {
  pair<Tdst, Tdst> m_offset_divisor;

  void single(char *dst, char *const *src)
  {
    Tsrc value = *reinterpret_cast<Tsrc *>(src[0]);
    if (value != std::numeric_limits<Tsrc>::min()) {
      value += m_offset_divisor.first;
      *reinterpret_cast<Tdst *>(dst) = floordiv(value, m_offset_divisor.second);
    } else {
      *reinterpret_cast<Tdst *>(dst) = std::numeric_limits<Tdst>::min();
    }
  }
};

template <class Tsrc, class Tdst>
static intptr_t instantiate_int_offset_and_divide_callable(
    char *static_data, size_t DYND_UNUSED(data_size), char *DYND_UNUSED(data), void *ckb, intptr_t ckb_offset,
    const ndt::type &DYND_UNUSED(dst_tp), const char *DYND_UNUSED(dst_arrmeta), intptr_t DYND_UNUSED(nsrc),
    const ndt::type *DYND_UNUSED(src_tp), const char *const *DYND_UNUSED(src_arrmeta), kernel_request_t kernreq,
    const eval::eval_context *DYND_UNUSED(ectx), intptr_t DYND_UNUSED(nkwd), const nd::array *DYND_UNUSED(kwds),
    const std::map<std::string, ndt::type> &DYND_UNUSED(tp_vars))
{
  typedef int_offset_and_divide_ck<Tsrc, Tdst> self_type;
  self_type *self = self_type::make(ckb, kernreq, ckb_offset);
  self->m_offset_divisor = *reinterpret_cast<pair<Tdst, Tdst> *>(static_data);
  return ckb_offset;
}

template <class Tsrc, class Tdst>
nd::callable make_int_offset_and_divide_callable(Tdst offset, Tdst divisor, const ndt::type &func_proto)
{
  return nd::callable(func_proto, kernel_request_single, single_t(), make_pair(offset, divisor), 0, NULL, NULL,
                      &instantiate_int_offset_and_divide_callable<Tsrc, Tdst>);
}

} // anonymous namespace

bool dynd::make_datetime_adapter_callable(const ndt::type &value_tp, const ndt::type &operand_tp, const nd::string &op,
                                          nd::callable &out_forward, nd::callable &out_reverse)
{
  int64_t epoch_datetime, unit_factor = 1, unit_divisor = 1;
  if (value_tp.get_type_id() != datetime_type_id) {
    return false;
  }
  if (parse_datetime_since(op.begin(), op.end(), epoch_datetime, unit_factor, unit_divisor)) {
    switch (operand_tp.get_type_id()) {
    case int64_type_id:
      if (unit_divisor > 1) {
        // TODO: This is a bad implementation, should do divide_and_offset to
        //       avoid overflow issues.
        out_forward = make_int_offset_and_divide_callable<int64_t, int64_t>(
            epoch_datetime * unit_divisor, unit_divisor,
            ndt::callable_type::make(value_tp, ndt::type::make<int64_t>()));
        out_reverse = make_int_multiply_and_offset_callable<int64_t, int64_t>(
            unit_divisor, -epoch_datetime * unit_divisor,
            ndt::callable_type::make(ndt::type::make<int64_t>(), value_tp));
      } else {
        out_forward = make_int_multiply_and_offset_callable<int64_t, int64_t>(
            unit_factor, epoch_datetime, ndt::callable_type::make(value_tp, ndt::type::make<int64_t>()));
        out_reverse = make_int_offset_and_divide_callable<int64_t, int64_t>(
            -epoch_datetime, unit_factor, ndt::callable_type::make(ndt::type::make<int64_t>(), value_tp));
      }
      return true;
    default:
      return false;
    }
  } else {
    return false;
  }
}
