//
// Copyright (C) 2011-15 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <time.h>
#include <cerrno>

#include <dynd/kernels/date_expr_kernels.hpp>
#include <dynd/kernels/elwise_expr_kernels.hpp>
#include <dynd/types/string_type.hpp>
#include <datetime_strings.h>

using namespace std;
using namespace dynd;

static void ymd_to_struct_tm(struct tm &stm, int days, const date_ymd &ymd)
{
  memset(&stm, 0, sizeof(struct tm));
  stm.tm_year = ymd.year - 1900;
  stm.tm_yday = ymd.get_day_of_year();
  stm.tm_mon = ymd.month - 1;
  stm.tm_mday = ymd.day;
  // 1970-01-04 is Sunday
  stm.tm_wday = (int)((days - 3) % 7);
  if (stm.tm_wday < 0) {
    stm.tm_wday += 7;
  }
}

/////////////////////////////////////////
// strftime kernel

namespace {
struct date_strftime_kernel_extra {
  typedef date_strftime_kernel_extra extra_type;

  ckernel_prefix base;
  size_t format_size;
  const char *format;
  const string_type_arrmeta *dst_arrmeta;

  static void single_unary(ckernel_prefix *extra, char *dst, char *const *src)
  {
    extra_type *e = reinterpret_cast<extra_type *>(extra);
    const string_type_arrmeta *dst_md = e->dst_arrmeta;

    struct tm tm_val;
    int32_t date = **reinterpret_cast<int32_t *const *>(src);
    // Convert the date to a 'struct tm'
    date_ymd ymd;
    ymd.set_from_days(date);
    ymd_to_struct_tm(tm_val, date, ymd);
#ifdef _MSC_VER
    // Given an invalid format string strftime will abort unless an invalid
    // parameter handler is installed.
    disable_invalid_parameter_handler raii;
#endif
    dynd::string *dst_d = reinterpret_cast<dynd::string *>(dst);
    memory_block_pod_allocator_api *allocator = get_memory_block_pod_allocator_api(dst_md->blockref);

    // Call strftime, growing the string buffer if needed so it fits
    size_t str_size = e->format_size + 16;
    char *begin, *end;
    allocator->allocate(dst_md->blockref, str_size, 1, &begin, &end);
    dst_d->assign(begin, end - begin);
    for (int attempt = 0; attempt < 3; ++attempt) {
      // Force errno to zero
      errno = 0;
      size_t len = strftime(dst_d->begin(), str_size, e->format, &tm_val);
      if (len > 0) {
        allocator->resize(dst_md->blockref, len, &begin, &end);
        dst_d->assign(begin, end - begin);
        break;
      } else {
        if (errno != 0) {
          stringstream ss;
          ss << "error in strftime with format string \"" << e->format << "\" to strftime";
          throw runtime_error(ss.str());
        }
        str_size *= 2;
        allocator->resize(dst_md->blockref, str_size, &begin, &end);
        dst_d->assign(begin, end - begin);
      }
    }
  }

  static void strided_unary(ckernel_prefix *extra, char *dst, intptr_t dst_stride, char *const *src,
                            const intptr_t *src_stride, size_t count)
  {
    extra_type *e = reinterpret_cast<extra_type *>(extra);
    size_t format_size = e->format_size;
    const char *format = e->format;
    const string_type_arrmeta *dst_md = e->dst_arrmeta;

    struct tm tm_val;
#ifdef _MSC_VER
    // Given an invalid format string strftime will abort unless an invalid
    // parameter handler is installed.
    disable_invalid_parameter_handler raii;
#endif
    memory_block_pod_allocator_api *allocator = get_memory_block_pod_allocator_api(dst_md->blockref);
    char *src0 = src[0];
    intptr_t src0_stride = src_stride[0];
    for (size_t i = 0; i != count; ++i) {
      dynd::string *dst_d = reinterpret_cast<dynd::string *>(dst);
      int32_t date = *reinterpret_cast<int32_t *>(src0);
      // Convert the date to a 'struct tm'
      date_ymd ymd;
      ymd.set_from_days(date);
      ymd_to_struct_tm(tm_val, date, ymd);

      // Call strftime, growing the string buffer if needed so it fits
      size_t str_size = format_size + 16;
      char *begin, *end;
      allocator->allocate(dst_md->blockref, str_size, 1, &begin, &end);
      dst_d->assign(begin, end - begin);
      for (int attempt = 0; attempt < 3; ++attempt) {
        // Force errno to zero
        errno = 0;
        size_t len = strftime(dst_d->begin(), str_size, format, &tm_val);
        if (len > 0) {
          allocator->resize(dst_md->blockref, len, &begin, &end);
          dst_d->assign(begin, end - begin);
          break;
        } else {
          if (errno != 0) {
            stringstream ss;
            ss << "error in strftime with format string \"" << e->format << "\" to strftime";
            throw runtime_error(ss.str());
          }
          str_size *= 2;
          allocator->resize(dst_md->blockref, str_size, &begin, &end);
          dst_d->assign(begin, end - begin);
        }
      }
      dst += dst_stride;
      src0 += src0_stride;
    }
  }
};
} // anonymous namespace

class date_strftime_kernel_generator : public expr_kernel_generator {
  std::string m_format;

public:
  date_strftime_kernel_generator(const std::string &format) : expr_kernel_generator(true), m_format(format)
  {
  }

  virtual ~date_strftime_kernel_generator()
  {
  }

  size_t make_expr_kernel(void *ckb, intptr_t ckb_offset, const ndt::type &dst_tp, const char *dst_arrmeta,
                          size_t src_count, const ndt::type *src_tp, const char *const *src_arrmeta,
                          kernel_request_t kernreq, const eval::eval_context *ectx) const
  {
    if (src_count != 1) {
      stringstream ss;
      ss << "date_strftime_kernel_generator requires 1 src operand, ";
      ss << "received " << src_count;
      throw runtime_error(ss.str());
    }
    bool require_elwise = dst_tp.get_type_id() != string_type_id || src_tp[0].get_type_id() != date_type_id;
    // If the types don't match the ones for this generator,
    // call the elementwise dimension handler to handle one dimension,
    // giving 'this' as the next kernel generator to call
    if (require_elwise) {
      return make_elwise_dimension_expr_kernel(ckb, ckb_offset, dst_tp, dst_arrmeta, src_count, src_tp, src_arrmeta,
                                               kernreq, ectx, this);
    }

    date_strftime_kernel_extra *e =
        reinterpret_cast<ckernel_builder<kernel_request_host> *>(ckb)->alloc_ck<date_strftime_kernel_extra>(ckb_offset);
    switch (kernreq) {
    case kernel_request_single:
      e->base.function = reinterpret_cast<void *>(&date_strftime_kernel_extra::single_unary);
      break;
    case kernel_request_strided:
      e->base.function = reinterpret_cast<void *>(&date_strftime_kernel_extra::strided_unary);
      break;
    default: {
      stringstream ss;
      ss << "date_strftime_kernel_generator: unrecognized request " << (int)kernreq;
      throw runtime_error(ss.str());
    }
    }
    // The lifetime of kernels must be shorter than that of the kernel
    // generator,
    // so we can point at data in the kernel generator
    e->format_size = m_format.size();
    e->format = m_format.c_str();
    e->dst_arrmeta = reinterpret_cast<const string_type_arrmeta *>(dst_arrmeta);
    return ckb_offset;
  }

  void print_type(std::ostream &o) const
  {
    o << "strftime(op0, ";
    print_escaped_utf8_string(o, m_format);
    o << ")";
  }
};

expr_kernel_generator *dynd::make_strftime_kernelgen(const std::string &format)
{
  return new date_strftime_kernel_generator(format);
}

/////////////////////////////////////////
// replace kernel

namespace {
struct date_replace_kernel_extra {
  typedef date_replace_kernel_extra extra_type;

  ckernel_prefix base;
  int32_t year, month, day;

  static void single_unary(ckernel_prefix *extra, char *dst, char *const *src)
  {
    extra_type *e = reinterpret_cast<extra_type *>(extra);
    int32_t year = e->year, month = e->month, day = e->day;

    int32_t date = **reinterpret_cast<int32_t *const *>(src);
    // Convert the date to YMD form
    date_ymd ymd;
    ymd.set_from_days(date);

    // Replace the values as requested
    if (year != numeric_limits<int32_t>::max()) {
      ymd.year = year;
    }
    if (month != numeric_limits<int32_t>::max()) {
      ymd.month = month;
      if (-12 <= month && month <= -1) {
        // Use negative months to count from the end (like Python slicing,
        // though
        // the standard Python datetime.date doesn't support this)
        ymd.month = month + 13;
      } else if (1 <= month && month <= 12) {
        ymd.month = month;
      } else {
        stringstream ss;
        ss << "invalid month value " << month;
        throw runtime_error(ss.str());
      }
      // If the day isn't also being replaced, make sure the resulting date is
      // valid
      if (day == numeric_limits<int32_t>::max()) {
        if (!ymd.is_valid()) {
          stringstream ss;
          ss << "invalid replace resulting year/month/day " << year << "/" << month << "/" << day;
          throw runtime_error(ss.str());
        }
      }
    }
    if (day != numeric_limits<int32_t>::max()) {
      int month_size = ymd.get_month_length();
      if (1 <= day && day <= month_size) {
        ymd.day = day;
      } else if (-month_size <= day && day <= -1) {
        // Use negative days to count from the end (like Python slicing, though
        // the standard Python datetime.date doesn't support this)
        ymd.day = day + month_size + 1;
      } else {
        stringstream ss;
        ss << "invalid day value " << day << " for year/month " << year << "/" << month;
        throw runtime_error(ss.str());
      }
    }

    *reinterpret_cast<int32_t *>(dst) = ymd.to_days();
  }
  static void strided_unary(ckernel_prefix *extra, char *dst, intptr_t dst_stride, char *const *src,
                            const intptr_t *src_stride, size_t count)
  {
    const char *src0 = src[0];
    intptr_t src0_stride = src_stride[0];
    for (size_t i = 0; i != count; ++i) {
      single_unary(extra, dst, src);
      dst += dst_stride;
      src0 += src0_stride;
    }
  }
};
} // anonymous namespace

class date_replace_kernel_generator : public expr_kernel_generator {
  int32_t m_year, m_month, m_day;

public:
  date_replace_kernel_generator(int32_t year, int32_t month, int32_t day)
      : expr_kernel_generator(true), m_year(year), m_month(month), m_day(day)
  {
  }

  virtual ~date_replace_kernel_generator()
  {
  }

  size_t make_expr_kernel(void *ckb, intptr_t ckb_offset, const ndt::type &dst_tp, const char *dst_arrmeta,
                          size_t src_count, const ndt::type *src_tp, const char *const *src_arrmeta,
                          kernel_request_t kernreq, const eval::eval_context *ectx) const
  {
    if (src_count != 1) {
      stringstream ss;
      ss << "date_replace_kernel_generator requires 1 src operand, ";
      ss << "received " << src_count;
      throw runtime_error(ss.str());
    }
    bool require_elwise = dst_tp.get_type_id() != date_type_id || src_tp[0].get_type_id() != date_type_id;
    // If the types don't match the ones for this generator,
    // call the elementwise dimension handler to handle one dimension,
    // giving 'this' as the next kernel generator to call
    if (require_elwise) {
      return make_elwise_dimension_expr_kernel(ckb, ckb_offset, dst_tp, dst_arrmeta, src_count, src_tp, src_arrmeta,
                                               kernreq, ectx, this);
    }

    date_replace_kernel_extra *e =
        reinterpret_cast<ckernel_builder<kernel_request_host> *>(ckb)->alloc_ck<date_replace_kernel_extra>(ckb_offset);
    switch (kernreq) {
    case kernel_request_single:
      e->base.function = reinterpret_cast<void *>(&date_replace_kernel_extra::single_unary);
      break;
    case kernel_request_strided:
      e->base.function = reinterpret_cast<void *>(&date_replace_kernel_extra::strided_unary);
      break;
    default: {
      stringstream ss;
      ss << "date_replace_kernel_generator: unrecognized request " << (int)kernreq;
      throw runtime_error(ss.str());
    }
    }
    e->year = m_year;
    e->month = m_month;
    e->day = m_day;
    return ckb_offset;
  }

  void print_type(std::ostream &o) const
  {
    o << "replace(op0";
    if (m_year != numeric_limits<int32_t>::max()) {
      o << ", year=" << m_year;
    }
    if (m_month != numeric_limits<int32_t>::max()) {
      o << ", month=" << m_month;
    }
    if (m_day != numeric_limits<int32_t>::max()) {
      o << ", day=" << m_day;
    }
    o << ")";
  }
};

expr_kernel_generator *dynd::make_replace_kernelgen(int32_t year, int32_t month, int32_t day)
{
  return new date_replace_kernel_generator(year, month, day);
}
