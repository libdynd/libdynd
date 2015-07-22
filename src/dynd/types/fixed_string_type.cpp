//
// Copyright (C) 2011-15 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <algorithm>

#include <dynd/types/fixed_string_type.hpp>
#include <dynd/types/string_type.hpp>
#include <dynd/kernels/string_assignment_kernels.hpp>
#include <dynd/kernels/string_comparison_kernels.hpp>
#include <dynd/kernels/string_numeric_assignment_kernels.hpp>
#include <dynd/exceptions.hpp>
#include <dynd/iter/string_iter.hpp>

using namespace std;
using namespace dynd;

ndt::fixed_string_type::fixed_string_type(intptr_t stringsize,
                                          string_encoding_t encoding)
    : base_string_type(fixed_string_type_id, 0, 1, type_flag_scalar, 0),
      m_stringsize(stringsize), m_encoding(encoding)
{
  switch (encoding) {
  case string_encoding_ascii:
  case string_encoding_utf_8:
    m_members.data_size = m_stringsize;
    m_members.data_alignment = 1;
    break;
  case string_encoding_ucs_2:
  case string_encoding_utf_16:
    m_members.data_size = m_stringsize * 2;
    m_members.data_alignment = 2;
    break;
  case string_encoding_utf_32:
    m_members.data_size = m_stringsize * 4;
    m_members.data_alignment = 4;
    break;
  default:
    throw runtime_error(
        "Unrecognized string encoding in dynd fixed_string type constructor");
  }
}

ndt::fixed_string_type::~fixed_string_type() {}

void ndt::fixed_string_type::get_string_range(const char **out_begin,
                                              const char **out_end,
                                              const char *DYND_UNUSED(arrmeta),
                                              const char *data) const
{
  // Beginning of the string
  *out_begin = data;

  switch (string_encoding_char_size_table[m_encoding]) {
  case 1: {
    const char *end =
        reinterpret_cast<const char *>(memchr(data, 0, get_data_size()));
    if (end != NULL) {
      *out_end = end;
    } else {
      *out_end = data + get_data_size();
    }
    break;
  }
  case 2: {
    const uint16_t *ptr = reinterpret_cast<const uint16_t *>(data);
    const uint16_t *ptr_max = ptr + get_data_size() / sizeof(uint16_t);
    while (ptr < ptr_max && *ptr != 0) {
      ++ptr;
    }
    *out_end = reinterpret_cast<const char *>(ptr);
    break;
  }
  case 4: {
    const uint32_t *ptr = reinterpret_cast<const uint32_t *>(data);
    const uint32_t *ptr_max = ptr + get_data_size() / sizeof(uint32_t);
    while (ptr < ptr_max && *ptr != 0) {
      ++ptr;
    }
    *out_end = reinterpret_cast<const char *>(ptr);
    break;
  }
  }
}

void ndt::fixed_string_type::set_from_utf8_string(
    const char *DYND_UNUSED(arrmeta), char *dst, const char *utf8_begin,
    const char *utf8_end, const eval::eval_context *ectx) const
{
  assign_error_mode errmode = ectx->errmode;
  char *dst_end = dst + get_data_size();
  next_unicode_codepoint_t next_fn =
      get_next_unicode_codepoint_function(string_encoding_utf_8, errmode);
  append_unicode_codepoint_t append_fn =
      get_append_unicode_codepoint_function(m_encoding, errmode);
  uint32_t cp;

  while (utf8_begin < utf8_end && dst < dst_end) {
    cp = next_fn(utf8_begin, utf8_end);
    append_fn(cp, dst, dst_end);
  }
  if (utf8_begin < utf8_end) {
    if (errmode != assign_error_nocheck) {
      throw std::runtime_error("Input is too large to convert to "
                               "destination fixed-size string");
    }
  } else if (dst < dst_end) {
    memset(dst, 0, dst_end - dst);
  }
}

void ndt::fixed_string_type::print_data(std::ostream &o,
                                        const char *DYND_UNUSED(arrmeta),
                                        const char *data) const
{
  uint32_t cp;
  next_unicode_codepoint_t next_fn;
  next_fn =
      get_next_unicode_codepoint_function(m_encoding, assign_error_nocheck);
  const char *data_end = data + get_data_size();

  // Print as an escaped string
  o << "\"";
  while (data < data_end) {
    cp = next_fn(data, data_end);
    if (cp != 0) {
      print_escaped_unicode_codepoint(o, cp, false);
    } else {
      break;
    }
  }
  o << "\"";
}

void ndt::fixed_string_type::print_type(std::ostream &o) const
{
  o << "fixed_string[" << m_stringsize;
  if (m_encoding != string_encoding_utf_8) {
    o << ",'" << m_encoding << "'";
  }
  o << "]";
}

ndt::type ndt::fixed_string_type::get_canonical_type() const
{
  return type(this, true);
}

bool ndt::fixed_string_type::is_lossless_assignment(
    const type &DYND_UNUSED(dst_tp), const type &DYND_UNUSED(src_tp)) const
{
  // Don't shortcut anything to 'nocheck' error checking, so that
  // decoding errors get caught appropriately.
  return false;
}

bool ndt::fixed_string_type::operator==(const base_type &rhs) const
{
  if (this == &rhs) {
    return true;
  } else if (rhs.get_type_id() != fixed_string_type_id) {
    return false;
  } else {
    const fixed_string_type *dt = static_cast<const fixed_string_type *>(&rhs);
    return m_encoding == dt->m_encoding && m_stringsize == dt->m_stringsize;
  }
}

intptr_t ndt::fixed_string_type::make_assignment_kernel(
    void *ckb, intptr_t ckb_offset, const type &dst_tp, const char *dst_arrmeta,
    const type &src_tp, const char *src_arrmeta, kernel_request_t kernreq,
    const eval::eval_context *ectx) const
{
  if (this == dst_tp.extended()) {
    switch (src_tp.get_type_id()) {
    case fixed_string_type_id: {
      const fixed_string_type *src_fs = src_tp.extended<fixed_string_type>();
      return make_fixed_string_assignment_kernel(
          ckb, ckb_offset, get_data_size(), m_encoding, src_fs->get_data_size(),
          src_fs->m_encoding, kernreq, ectx);
    }
    case string_type_id: {
      const base_string_type *src_fs = src_tp.extended<base_string_type>();
      return make_blockref_string_to_fixed_string_assignment_kernel(
          ckb, ckb_offset, get_data_size(), m_encoding, src_fs->get_encoding(),
          kernreq, ectx);
    }
    default: {
      if (!src_tp.is_builtin()) {
        return src_tp.extended()->make_assignment_kernel(
            ckb, ckb_offset, dst_tp, dst_arrmeta, src_tp, src_arrmeta, kernreq,
            ectx);
      } else {
        return make_builtin_to_string_assignment_kernel(
            ckb, ckb_offset, dst_tp, dst_arrmeta, src_tp.get_type_id(), kernreq,
            ectx);
      }
    }
    }
  } else {
    if (dst_tp.is_builtin()) {
      return make_string_to_builtin_assignment_kernel(
          ckb, ckb_offset, dst_tp.get_type_id(), src_tp, src_arrmeta, kernreq,
          ectx);
    } else {
      stringstream ss;
      ss << "Cannot assign from " << src_tp << " to " << dst_tp;
      throw dynd::type_error(ss.str());
    }
  }
}

size_t ndt::fixed_string_type::make_comparison_kernel(
    void *ckb, intptr_t ckb_offset, const type &src0_dt,
    const char *src0_arrmeta, const type &src1_dt, const char *src1_arrmeta,
    comparison_type_t comptype, const eval::eval_context *ectx) const
{
  if (this == src0_dt.extended()) {
    if (*this == *src1_dt.extended()) {
      return make_fixed_string_comparison_kernel(ckb, ckb_offset, m_stringsize,
                                                 m_encoding, comptype, ectx);
    } else if (src1_dt.get_kind() == string_kind) {
      return make_general_string_comparison_kernel(
          ckb, ckb_offset, src0_dt, src0_arrmeta, src1_dt, src1_arrmeta,
          comptype, ectx);
    } else if (!src1_dt.is_builtin()) {
      return src1_dt.extended()->make_comparison_kernel(
          ckb, ckb_offset, src0_dt, src0_arrmeta, src1_dt, src1_arrmeta,
          comptype, ectx);
    }
  }

  throw not_comparable_error(src0_dt, src1_dt, comptype);
}

void ndt::fixed_string_type::make_string_iter(
    dim_iter *out_di, string_encoding_t encoding, const char *arrmeta,
    const char *data, const memory_block_ptr &ref, intptr_t buffer_max_mem,
    const eval::eval_context *ectx) const
{
  const char *data_begin;
  const char *data_end;
  get_string_range(&data_begin, &data_end, arrmeta, data);
  iter::make_string_iter(out_di, encoding, m_encoding, data_begin, data_end,
                         ref, buffer_max_mem, ectx);
}
