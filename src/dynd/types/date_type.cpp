//
// Copyright (C) 2011-14 Mark Wiebe, DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <time.h>

#include <cerrno>
#include <algorithm>

#include <dynd/types/date_type.hpp>
#include <dynd/types/property_type.hpp>
#include <dynd/types/cstruct_type.hpp>
#include <dynd/types/string_type.hpp>
#include <dynd/types/unary_expr_type.hpp>
#include <dynd/types/option_type.hpp>
#include <dynd/types/typevar_type.hpp>
#include <dynd/kernels/date_assignment_kernels.hpp>
#include <dynd/kernels/date_expr_kernels.hpp>
#include <dynd/kernels/string_assignment_kernels.hpp>
#include <dynd/kernels/assignment_kernels.hpp>
#include <dynd/kernels/date_adapter_kernels.hpp>
#include <dynd/exceptions.hpp>
#include <dynd/func/make_callable.hpp>
#include <dynd/array_iter.hpp>
#include <dynd/parser_util.hpp>

#include <datetime_strings.h>
#include <datetime_localtime.h>

using namespace std;
using namespace dynd;

date_type::date_type()
    : base_type(date_type_id, datetime_kind, 4, scalar_align_of<int32_t>::value,
                type_flag_scalar, 0, 0, 0)
{
}

date_type::~date_type()
{
}

void date_type::set_ymd(const char *DYND_UNUSED(arrmeta), char *data,
                assign_error_mode errmode, int32_t year, int32_t month, int32_t day) const
{
    if (errmode != assign_error_nocheck && !date_ymd::is_valid(year, month, day)) {
        stringstream ss;
        ss << "invalid input year/month/day " << year << "/" << month << "/" << day;
        throw runtime_error(ss.str());
    }

    *reinterpret_cast<int32_t *>(data) = date_ymd::to_days(year, month, day);
}

void date_type::set_from_utf8_string(const char *DYND_UNUSED(arrmeta), char *data,
                                     const char *utf8_begin,
                                     const char *utf8_end,
                                     const eval::eval_context *ectx) const
{
    date_ymd ymd;
    ymd.set_from_str(utf8_begin, utf8_end, ectx->date_parse_order,
                     ectx->century_window, ectx->errmode);
    *reinterpret_cast<int32_t *>(data) = ymd.to_days();
}

date_ymd date_type::get_ymd(const char *DYND_UNUSED(arrmeta), const char *data) const
{
    date_ymd ymd;
    ymd.set_from_days(*reinterpret_cast<const int32_t *>(data));
    return ymd;
}

void date_type::print_data(std::ostream& o, const char *DYND_UNUSED(arrmeta), const char *data) const
{
    date_ymd ymd;
    ymd.set_from_days(*reinterpret_cast<const int32_t *>(data));
    string s = ymd.to_str();
    if (s.empty()) {
        o << "NA";
    } else {
        o << s;
    }
}

void date_type::print_type(std::ostream& o) const
{
    o << "date";
}

bool date_type::is_lossless_assignment(const ndt::type& dst_tp, const ndt::type& src_tp) const
{
    if (dst_tp.extended() == this) {
        if (src_tp.extended() == this) {
            return true;
        } else if (src_tp.get_type_id() == date_type_id) {
            // There is only one possibility for the date type (TODO: timezones!)
            return true;
        } else {
            return false;
        }
    } else {
        return false;
    }
}

bool date_type::operator==(const base_type& rhs) const
{
    if (this == &rhs) {
        return true;
    } else if (rhs.get_type_id() != date_type_id) {
        return false;
    } else {
        // There is only one possibility for the date type (TODO: timezones!)
        return true;
    }
}

size_t date_type::make_assignment_kernel(
                ckernel_builder *ckb, intptr_t ckb_offset,
                const ndt::type& dst_tp, const char *dst_arrmeta,
                const ndt::type& src_tp, const char *src_arrmeta,
                kernel_request_t kernreq,
                const eval::eval_context *ectx) const
{
    if (this == dst_tp.extended()) {
        if (src_tp.get_type_id() == date_type_id) {
            return make_pod_typed_data_assignment_kernel(ckb, ckb_offset,
                            get_data_size(), get_data_alignment(), kernreq);
        } else if (src_tp.get_kind() == string_kind) {
            // Assignment from strings
            return make_string_to_date_assignment_kernel(
                ckb, ckb_offset, src_tp, src_arrmeta, kernreq, ectx);
        } else if (src_tp.get_kind() == struct_kind) {
            // Convert to struct using the "struct" property
            return ::make_assignment_kernel(
                ckb, ckb_offset, ndt::make_property(dst_tp, "struct"),
                dst_arrmeta, src_tp, src_arrmeta, kernreq, ectx);
        } else if (!src_tp.is_builtin()) {
            return src_tp.extended()->make_assignment_kernel(
                ckb, ckb_offset, dst_tp, dst_arrmeta, src_tp, src_arrmeta,
                kernreq, ectx);
        }
    } else {
        if (dst_tp.get_kind() == string_kind) {
            // Assignment to strings
            return make_date_to_string_assignment_kernel(
                ckb, ckb_offset, dst_tp, dst_arrmeta, kernreq, ectx);
        } else if (dst_tp.get_kind() == struct_kind) {
            // Convert to struct using the "struct" property
            return ::make_assignment_kernel(
                ckb, ckb_offset, dst_tp, dst_arrmeta,
                ndt::make_property(src_tp, "struct"), src_arrmeta, kernreq,
                ectx);
        }
        // TODO
    }

    stringstream ss;
    ss << "Cannot assign from " << src_tp << " to " << dst_tp;
    throw dynd::type_error(ss.str());
}

size_t date_type::make_comparison_kernel(
                ckernel_builder *ckb, intptr_t ckb_offset,
                const ndt::type& src0_tp, const char *src0_arrmeta,
                const ndt::type& src1_tp, const char *src1_arrmeta,
                comparison_type_t comptype,
                const eval::eval_context *ectx) const
{
    if (this == src0_tp.extended()) {
        if (*this == *src1_tp.extended()) {
            return make_builtin_type_comparison_kernel(ckb, ckb_offset,
                            int32_type_id, int32_type_id, comptype);
        } else if (!src1_tp.is_builtin()) {
            return src1_tp.extended()->make_comparison_kernel(ckb, ckb_offset,
                            src0_tp, src0_arrmeta,
                            src1_tp, src1_arrmeta,
                            comptype, ectx);
        }
    }

    throw not_comparable_error(src0_tp, src1_tp, comptype);
}

///////// properties on the type

//static pair<string, gfunc::callable> date_type_properties[] = {
//};

void date_type::get_dynamic_type_properties(const std::pair<std::string, gfunc::callable> **out_properties, size_t *out_count) const
{
    *out_properties = NULL; //date_type_properties;
    *out_count = 0; //sizeof(date_type_properties) / sizeof(date_type_properties[0]);
}

///////// functions on the type

static nd::array function_type_today(const ndt::type& dt) {
    date_ymd ymd = date_ymd::get_current_local_date();
    nd::array result = nd::empty(dt);
    *reinterpret_cast<int32_t *>(result.get_readwrite_originptr()) = ymd.to_days();
    // Make the result immutable (we own the only reference to the data at this point)
    result.flag_as_immutable();
    return result;
}

static nd::array function_type_construct(const ndt::type& DYND_UNUSED(dt),
                const nd::array& year, const nd::array& month, const nd::array& day)
{
    // TODO proper buffering
    nd::array year_as_int = year.ucast(ndt::make_type<int32_t>()).eval();
    nd::array month_as_int = month.ucast(ndt::make_type<int32_t>()).eval();
    nd::array day_as_int = day.ucast(ndt::make_type<int32_t>()).eval();
    nd::array result;

    array_iter<1,3> iter(ndt::make_date(), result, year_as_int, month_as_int, day_as_int);
    if (!iter.empty()) {
        date_ymd ymd;
        do {
            ymd.year = *reinterpret_cast<const int32_t *>(iter.data<1>());
            ymd.month = *reinterpret_cast<const int32_t *>(iter.data<2>());
            ymd.day = *reinterpret_cast<const int32_t *>(iter.data<3>());
            if (!ymd.is_valid()) {
                stringstream ss;
                ss << "invalid year/month/day " << ymd.year << "/" << ymd.month << "/" << ymd.day;
                throw runtime_error(ss.str());
            }
            *reinterpret_cast<int32_t *>(iter.data<0>()) = ymd.to_days();
        } while (iter.next());
    }

    return result;
}

void date_type::get_dynamic_type_functions(const std::pair<std::string, gfunc::callable> **out_functions, size_t *out_count) const
{
    static pair<string, gfunc::callable> date_type_functions[] = {
        pair<string, gfunc::callable>(
            "today", gfunc::make_callable(&function_type_today, "self")),
        pair<string, gfunc::callable>(
            "__construct__",
            gfunc::make_callable(&function_type_construct, "self", "year",
                                 "month", "day"))};

    *out_functions = date_type_functions;
    *out_count = sizeof(date_type_functions) / sizeof(date_type_functions[0]);
}

///////// properties on the nd::array

static nd::array property_ndo_get_year(const nd::array& n) {
    return n.replace_dtype(ndt::make_property(n.get_dtype(), "year"));
}

static nd::array property_ndo_get_month(const nd::array& n) {
    return n.replace_dtype(ndt::make_property(n.get_dtype(), "month"));
}

static nd::array property_ndo_get_day(const nd::array& n) {
    return n.replace_dtype(ndt::make_property(n.get_dtype(), "day"));
}

void date_type::get_dynamic_array_properties(const std::pair<std::string, gfunc::callable> **out_properties, size_t *out_count) const
{
    static pair<string, gfunc::callable> date_array_properties[] = {
        pair<string, gfunc::callable>(
            "year", gfunc::make_callable(&property_ndo_get_year, "self")),
        pair<string, gfunc::callable>(
            "month", gfunc::make_callable(&property_ndo_get_month, "self")),
        pair<string, gfunc::callable>(
            "day", gfunc::make_callable(&property_ndo_get_day, "self"))};

    *out_properties = date_array_properties;
    *out_count = sizeof(date_array_properties) / sizeof(date_array_properties[0]);
}

///////// functions on the nd::array

static nd::array function_ndo_to_struct(const nd::array& n) {
    return n.replace_dtype(ndt::make_property(n.get_dtype(), "struct"));
}

static nd::array function_ndo_strftime(const nd::array& n, const std::string& format) {
    // TODO: Allow 'format' itself to be an array, with broadcasting, etc.
    if (format.empty()) {
        throw runtime_error("format string for strftime should not be empty");
    }
    return n.replace_dtype(ndt::make_unary_expr(ndt::make_string(), n.get_dtype(),
                    make_strftime_kernelgen(format)));
}

static nd::array function_ndo_weekday(const nd::array& n) {
    return n.replace_dtype(ndt::make_property(n.get_dtype(), "weekday"));
}

static nd::array function_ndo_replace(const nd::array& n, int32_t year, int32_t month, int32_t day) {
    // TODO: Allow 'year', 'month', and 'day' to be arrays, with broadcasting, etc.
    if (year == numeric_limits<int32_t>::max() && month == numeric_limits<int32_t>::max() &&
                    day == numeric_limits<int32_t>::max()) {
        throw std::runtime_error("no parameters provided to date.replace, should provide at least one");
    }
    return n.replace_dtype(ndt::make_unary_expr(ndt::make_date(), n.get_dtype(),
                    make_replace_kernelgen(year, month, day)));
}

void date_type::get_dynamic_array_functions(
                const std::pair<std::string, gfunc::callable> **out_functions,
                size_t *out_count) const
{
    static pair<string, gfunc::callable> date_array_functions[] = {
        pair<string, gfunc::callable>(
            "to_struct", gfunc::make_callable(&function_ndo_to_struct, "self")),
        pair<string, gfunc::callable>(
            "strftime",
            gfunc::make_callable(&function_ndo_strftime, "self", "format")),
        pair<string, gfunc::callable>(
            "weekday", gfunc::make_callable(&function_ndo_weekday, "self")),
        pair<string, gfunc::callable>(
            "replace",
            gfunc::make_callable_with_default(
                &function_ndo_replace, "self", "year", "month", "day",
                numeric_limits<int32_t>::max(), numeric_limits<int32_t>::max(),
                numeric_limits<int32_t>::max()))};

    *out_functions = date_array_functions;
    *out_count = sizeof(date_array_functions) / sizeof(date_array_functions[0]);
}

///////// property accessor kernels (used by property_type)

namespace {
void get_property_kernel_year_single(char *dst, const char *const *src,
                                     ckernel_prefix *DYND_UNUSED(self))
{
    date_ymd ymd;
    ymd.set_from_days(**reinterpret_cast<const int32_t *const *>(src));
    *reinterpret_cast<int32_t *>(dst) = ymd.year;
}

void get_property_kernel_month_single(char *dst, const char *const *src,
                                      ckernel_prefix *DYND_UNUSED(self))
{
    date_ymd ymd;
    ymd.set_from_days(**reinterpret_cast<const int32_t *const *>(src));
    *reinterpret_cast<int32_t *>(dst) = ymd.month;
}

void get_property_kernel_day_single(char *dst, const char *const *src,
                                    ckernel_prefix *DYND_UNUSED(self))
{
    date_ymd ymd;
    ymd.set_from_days(**reinterpret_cast<const int32_t *const *>(src));
    *reinterpret_cast<int32_t *>(dst) = ymd.day;
}

void get_property_kernel_weekday_single(char *dst, const char *const *src,
                                        ckernel_prefix *DYND_UNUSED(self))
{
    int32_t days = **reinterpret_cast<const int32_t *const *>(src);
    // 1970-01-05 is Monday
    int weekday = (int)((days - 4) % 7);
    if (weekday < 0) {
        weekday += 7;
    }
    *reinterpret_cast<int32_t *>(dst) = weekday;
}

void get_property_kernel_struct_single(char *dst, const char *const *src,
                                       ckernel_prefix *DYND_UNUSED(self))
{
    date_ymd *dst_struct = reinterpret_cast<date_ymd *>(dst);
    dst_struct->set_from_days(**reinterpret_cast<const int32_t *const *>(src));
}

void set_property_kernel_struct_single(char *dst, const char *const*src,
                ckernel_prefix *DYND_UNUSED(self))
{
    const date_ymd *src_struct = *reinterpret_cast<const date_ymd *const *>(src);
    *reinterpret_cast<int32_t *>(dst) = src_struct->to_days();
}
} // anonymous namespace

namespace {
enum date_properties_t {
  dateprop_year,
  dateprop_month,
  dateprop_day,
  dateprop_weekday,
  dateprop_struct
};
}

size_t date_type::get_elwise_property_index(const std::string& property_name) const
{
  if (property_name == "year") {
    return dateprop_year;
  } else if (property_name == "month") {
    return dateprop_month;
  } else if (property_name == "day") {
    return dateprop_day;
  } else if (property_name == "weekday") {
    return dateprop_weekday;
  } else if (property_name == "struct") {
    // A read/write property for accessing a date as a struct
    return dateprop_struct;
  } else {
    stringstream ss;
    ss << "dynd date type does not have a kernel for property "
       << property_name;
    throw runtime_error(ss.str());
  }
}

ndt::type date_type::get_elwise_property_type(size_t property_index,
                                              bool &out_readable,
                                              bool &out_writable) const
{
  switch (property_index) {
  case dateprop_year:
  case dateprop_month:
  case dateprop_day:
  case dateprop_weekday:
    out_readable = true;
    out_writable = false;
    return ndt::make_type<int32_t>();
  case dateprop_struct:
    out_readable = true;
    out_writable = true;
    return date_ymd::type();
  default:
    out_readable = false;
    out_writable = false;
    return ndt::make_type<void>();
  }
}

size_t date_type::make_elwise_property_getter_kernel(
    ckernel_builder *ckb, intptr_t ckb_offset,
    const char *DYND_UNUSED(dst_arrmeta), const char *DYND_UNUSED(src_arrmeta),
    size_t src_property_index, kernel_request_t kernreq,
    const eval::eval_context *DYND_UNUSED(ectx)) const
{
  ckb_offset =
      make_kernreq_to_single_kernel_adapter(ckb, ckb_offset, 1, kernreq);
  ckernel_prefix *e = ckb->alloc_ck_leaf<ckernel_prefix>(ckb_offset);
  switch (src_property_index) {
  case dateprop_year:
    e->set_function<expr_single_t>(&get_property_kernel_year_single);
    return ckb_offset;
  case dateprop_month:
    e->set_function<expr_single_t>(&get_property_kernel_month_single);
    return ckb_offset;
  case dateprop_day:
    e->set_function<expr_single_t>(&get_property_kernel_day_single);
    return ckb_offset;
  case dateprop_weekday:
    e->set_function<expr_single_t>(&get_property_kernel_weekday_single);
    return ckb_offset;
  case dateprop_struct:
    e->set_function<expr_single_t>(&get_property_kernel_struct_single);
    return ckb_offset;
  default:
    stringstream ss;
    ss << "dynd date type given an invalid property index"
       << src_property_index;
    throw runtime_error(ss.str());
  }
}

size_t date_type::make_elwise_property_setter_kernel(
    ckernel_builder *ckb, intptr_t ckb_offset,
    const char *DYND_UNUSED(dst_arrmeta), size_t dst_property_index,
    const char *DYND_UNUSED(src_arrmeta), kernel_request_t kernreq,
    const eval::eval_context *DYND_UNUSED(ectx)) const
{
    ckb_offset =
        make_kernreq_to_single_kernel_adapter(ckb, ckb_offset, 1, kernreq);
    ckernel_prefix *e = ckb->alloc_ck_leaf<ckernel_prefix>(ckb_offset);
    switch (dst_property_index) {
        case dateprop_struct:
            e->set_function<expr_single_t>(&set_property_kernel_struct_single);
            return ckb_offset;
        default:
            stringstream ss;
            ss << "dynd date type given an invalid property index" << dst_property_index;
            throw runtime_error(ss.str());
    }
}

namespace {
struct date_is_avail_ck {
  static void single(char *dst, const char *const *src,
                     ckernel_prefix *DYND_UNUSED(self))
  {
    int32_t date = **reinterpret_cast<const int32_t *const *>(src);
    *dst = date != DYND_DATE_NA;
  }

  static void strided(char *dst, intptr_t dst_stride, const char *const *src,
                      const intptr_t *src_stride, size_t count,
                      ckernel_prefix *DYND_UNUSED(self))
  {
    const char *src0 = src[0];
    intptr_t src0_stride = src_stride[0];
    for (size_t i = 0; i != count; ++i) {
      int32_t date = *reinterpret_cast<const int32_t *>(src);
      *dst = date != DYND_DATE_NA;
      dst += dst_stride;
      src0 += src0_stride;
    }
  }

  static intptr_t instantiate(const arrfunc_type_data *DYND_UNUSED(self),
                              dynd::ckernel_builder *ckb, intptr_t ckb_offset,
                              const ndt::type &dst_tp,
                              const char *DYND_UNUSED(dst_arrmeta),
                              const ndt::type *src_tp,
                              const char *const *DYND_UNUSED(src_arrmeta),
                              kernel_request_t kernreq,
                              const eval::eval_context *DYND_UNUSED(ectx))
  {
    if (src_tp[0].get_type_id() != option_type_id ||
        src_tp[0].tcast<option_type>()->get_value_type().get_type_id() !=
            date_type_id) {
      stringstream ss;
      ss << "Expected source type ?date, got " << src_tp[0];
      throw type_error(ss.str());
    }
    if (dst_tp.get_type_id() != bool_type_id) {
      stringstream ss;
      ss << "Expected destination type bool, got " << dst_tp;
      throw type_error(ss.str());
    }
    ckernel_prefix *ckp = ckb->alloc_ck_leaf<ckernel_prefix>(ckb_offset);
    ckp->set_expr_function<date_is_avail_ck>(kernreq);
    return ckb_offset;
  }
};

struct date_assign_na_ck {
  static void single(char *dst, const char *const *DYND_UNUSED(src),
                     ckernel_prefix *DYND_UNUSED(self))
  {
    *reinterpret_cast<int32_t *>(dst) = DYND_DATE_NA;
  }

  static void strided(char *dst, intptr_t dst_stride,
                      const char *const *DYND_UNUSED(src),
                      const intptr_t *DYND_UNUSED(src_stride), size_t count,
                      ckernel_prefix *DYND_UNUSED(self))
  {
    for (size_t i = 0; i != count; ++i, dst += dst_stride) {
      *reinterpret_cast<int32_t *>(dst) = DYND_DATE_NA;
    }
  }

  static intptr_t instantiate(const arrfunc_type_data *DYND_UNUSED(self),
                              dynd::ckernel_builder *ckb, intptr_t ckb_offset,
                              const ndt::type &dst_tp,
                              const char *DYND_UNUSED(dst_arrmeta),
                              const ndt::type *DYND_UNUSED(src_tp),
                              const char *const *DYND_UNUSED(src_arrmeta),
                              kernel_request_t kernreq,
                              const eval::eval_context *DYND_UNUSED(ectx))
  {
    if (dst_tp.get_type_id() != option_type_id ||
        dst_tp.tcast<option_type>()->get_value_type().get_type_id() !=
            date_type_id) {
      stringstream ss;
      ss << "Expected destination type ?date, got " << dst_tp;
      throw type_error(ss.str());
    }
    ckernel_prefix *ckp = ckb->alloc_ck_leaf<ckernel_prefix>(ckb_offset);
    ckp->set_expr_function<date_assign_na_ck>(kernreq);
    return ckb_offset;
  }
};
} // anonymous namespace

nd::array date_type::get_option_nafunc() const
{
  nd::array naf = nd::empty(option_type::make_nafunc_type());
  arrfunc_type_data *is_avail =
      reinterpret_cast<arrfunc_type_data *>(naf.get_ndo()->m_data_pointer);
  arrfunc_type_data *assign_na = is_avail + 1;

  // Use a typevar instead of option[T] to avoid a circular dependency
  is_avail->func_proto =
      ndt::make_funcproto(ndt::make_typevar("T"), ndt::make_type<dynd_bool>());
  is_avail->instantiate = &date_is_avail_ck::instantiate;
  assign_na->func_proto = ndt::make_funcproto(0, NULL, ndt::make_typevar("T"));
  assign_na->instantiate = &date_assign_na_ck::instantiate;
  naf.flag_as_immutable();
  return naf;
}

bool date_type::adapt_type(const ndt::type &operand_tp, const nd::string &op,
                           nd::arrfunc &out_forward, nd::arrfunc &out_reverse)
    const
{
  return make_date_adapter_arrfunc(operand_tp, op, out_forward, out_reverse);
}

bool date_type::reverse_adapt_type(const ndt::type &value_tp,
                                   const nd::string &op,
                                   nd::arrfunc &out_forward,
                                   nd::arrfunc &out_reverse) const
{
  // Note that out_reverse and out_forward are swapped compared with
  // adapt_type
  return make_date_adapter_arrfunc(value_tp, op, out_reverse, out_forward);
}

const ndt::type& ndt::make_date()
{
  // Static instance of the type, which has a reference count > 0 for the
  // lifetime of the program. This static construction is inside a
  // function to ensure correct creation order during startup.
  static date_type dt;
  static const ndt::type static_instance(&dt, true);
  return static_instance;
}
