//
// Copyright (C) 2011-15 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <dynd/kernels/base_kernel.hpp>
#include <dynd/kernels/datetime_assignment_kernels.hpp>
#include <dynd/types/datetime_type.hpp>
#include <datetime_strings.h>

using namespace std;
using namespace dynd;

/////////////////////////////////////////
// string to datetime assignment

namespace {
struct string_to_datetime_ck
    : nd::base_kernel<string_to_datetime_ck, kernel_request_host, 1> {
  ndt::type m_dst_datetime_tp;
  ndt::type m_src_string_tp;
  const char *m_src_arrmeta;
  assign_error_mode m_errmode;
  date_parse_order_t m_date_parse_order;
  int m_century_window;

  void single(char *dst, char *const *src)
  {
    const ndt::base_string_type *bst =
        static_cast<const ndt::base_string_type *>(m_src_string_tp.extended());
    const string &s = bst->get_utf8_string(m_src_arrmeta, src[0], m_errmode);
    datetime_struct dts;
    // TODO: properly distinguish "date" and "option[date]" with respect to NA
    // support
    if (s == "NA") {
      dts.set_to_na();
    } else {
      dts.set_from_str(s, m_date_parse_order, m_century_window);
    }
    *reinterpret_cast<int64_t *>(dst) = dts.to_ticks();
  }
};
} // anonymous namespace

size_t dynd::make_string_to_datetime_assignment_kernel(
    void *ckb, intptr_t ckb_offset, const ndt::type &dst_datetime_tp,
    const char *DYND_UNUSED(dst_arrmeta), const ndt::type &src_string_tp,
    const char *src_arrmeta, kernel_request_t kernreq,
    const eval::eval_context *ectx)
{
  typedef string_to_datetime_ck self_type;
  if (src_string_tp.get_kind() != string_kind) {
    stringstream ss;
    ss << "make_string_to_datetime_assignment_kernel: source type "
       << src_string_tp << " is not a string type";
    throw runtime_error(ss.str());
  }

  self_type *self = self_type::make(ckb, kernreq, ckb_offset);
  self->m_dst_datetime_tp = dst_datetime_tp;
  self->m_src_string_tp = src_string_tp;
  self->m_src_arrmeta = src_arrmeta;
  self->m_errmode = ectx->errmode;
  self->m_date_parse_order = ectx->date_parse_order;
  self->m_century_window = ectx->century_window;
  return ckb_offset;
}

/////////////////////////////////////////
// datetime to string assignment

namespace {
struct datetime_to_string_ck : nd::base_kernel<datetime_to_string_ck, kernel_request_host, 1> {
  ndt::type m_dst_string_tp;
  ndt::type m_src_datetime_tp;
  const char *m_dst_arrmeta;
  eval::eval_context m_ectx;

  void single(char *dst, char *const *src)
  {
    datetime_struct dts;
    dts.set_from_ticks(*reinterpret_cast<const int64_t *>(src[0]));
    string s = dts.to_str();
    if (s.empty()) {
      s = "NA";
    } else if (m_src_datetime_tp.extended<ndt::datetime_type>()->get_timezone() ==
               tz_utc) {
      s += "Z";
    }
    const ndt::base_string_type *bst =
        static_cast<const ndt::base_string_type *>(m_dst_string_tp.extended());
    bst->set_from_utf8_string(m_dst_arrmeta, dst, s, &m_ectx);
  }
};
} // anonymous namespace

size_t dynd::make_datetime_to_string_assignment_kernel(
    void *ckb, intptr_t ckb_offset, const ndt::type &dst_string_tp,
    const char *dst_arrmeta, const ndt::type &src_datetime_tp,
    const char *DYND_UNUSED(src_arrmeta), kernel_request_t kernreq,
    const eval::eval_context *ectx)
{
  typedef datetime_to_string_ck self_type;
  if (dst_string_tp.get_kind() != string_kind) {
    stringstream ss;
    ss << "get_datetime_to_string_assignment_kernel: dest type "
       << dst_string_tp << " is not a string type";
    throw runtime_error(ss.str());
  }

  self_type *self = self_type::make(ckb, kernreq, ckb_offset);
  self->m_dst_string_tp = dst_string_tp;
  self->m_src_datetime_tp = src_datetime_tp;
  self->m_dst_arrmeta = dst_arrmeta;
  self->m_ectx = *ectx;
  return ckb_offset;
}
