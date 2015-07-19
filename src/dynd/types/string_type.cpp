//
// Copyright (C) 2011-15 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <dynd/func/arrfunc.hpp>
#include <dynd/kernels/base_kernel.hpp>
#include <dynd/types/string_type.hpp>
#include <dynd/memblock/pod_memory_block.hpp>
#include <dynd/kernels/string_assignment_kernels.hpp>
#include <dynd/kernels/string_comparison_kernels.hpp>
#include <dynd/kernels/string_numeric_assignment_kernels.hpp>
#include <dynd/types/fixed_string_type.hpp>
#include <dynd/types/option_type.hpp>
#include <dynd/types/typevar_type.hpp>
#include <dynd/iter/string_iter.hpp>
#include <dynd/exceptions.hpp>

#include <algorithm>

using namespace std;
using namespace dynd;

ndt::string_type::string_type(string_encoding_t encoding)
    : base_string_type(
          string_type_id, sizeof(string_type_data), sizeof(const char *),
          type_flag_scalar | type_flag_zeroinit | type_flag_blockref,
          sizeof(string_type_arrmeta)),
      m_encoding(encoding)
{
  switch (encoding) {
  case string_encoding_ascii:
  case string_encoding_ucs_2:
  case string_encoding_utf_8:
  case string_encoding_utf_16:
  case string_encoding_utf_32:
    break;
  default:
    throw runtime_error(
        "Unrecognized string encoding in string type constructor");
  }
}

ndt::string_type::~string_type() {}

void ndt::string_type::get_string_range(const char **out_begin,
                                        const char **out_end,
                                        const char *DYND_UNUSED(arrmeta),
                                        const char *data) const
{
  *out_begin = reinterpret_cast<const string_type_data *>(data)->begin;
  *out_end = reinterpret_cast<const string_type_data *>(data)->end;
}

void ndt::string_type::set_from_utf8_string(
    const char *arrmeta, char *dst, const char *utf8_begin,
    const char *utf8_end, const eval::eval_context *ectx) const
{
  const string_type_arrmeta *data_md =
      reinterpret_cast<const string_type_arrmeta *>(arrmeta);
  assign_error_mode errmode = ectx->errmode;
  const intptr_t src_charsize = 1;
  intptr_t dst_charsize = string_encoding_char_size_table[m_encoding];
  char *dst_begin = NULL, *dst_current, *dst_end = NULL;
  next_unicode_codepoint_t next_fn =
      get_next_unicode_codepoint_function(string_encoding_utf_8, errmode);
  append_unicode_codepoint_t append_fn =
      get_append_unicode_codepoint_function(m_encoding, errmode);
  uint32_t cp;

  memory_block_pod_allocator_api *allocator =
      get_memory_block_pod_allocator_api(data_md->blockref);

  // Allocate the initial output as the src number of characters + some padding
  // TODO: Don't add padding if the output is not a multi-character encoding
  allocator->allocate(data_md->blockref,
                      ((utf8_end - utf8_begin) / src_charsize + 16) *
                          dst_charsize * 1124 / 1024,
                      dst_charsize, &dst_begin, &dst_end);

  dst_current = dst_begin;
  while (utf8_begin < utf8_end) {
    cp = next_fn(utf8_begin, utf8_end);
    // Append the codepoint, or increase the allocated memory as necessary
    if (dst_end - dst_current >= 8) {
      append_fn(cp, dst_current, dst_end);
    } else {
      char *dst_begin_saved = dst_begin;
      allocator->resize(data_md->blockref, 2 * (dst_end - dst_begin),
                        &dst_begin, &dst_end);
      dst_current = dst_begin + (dst_current - dst_begin_saved);

      append_fn(cp, dst_current, dst_end);
    }
  }

  // Shrink-wrap the memory to just fit the string
  allocator->resize(data_md->blockref, dst_current - dst_begin, &dst_begin,
                    &dst_end);

  // Set the output
  reinterpret_cast<string_type_data *>(dst)->begin = dst_begin;
  reinterpret_cast<string_type_data *>(dst)->end = dst_end;
}

void ndt::string_type::print_data(std::ostream &o,
                                  const char *DYND_UNUSED(arrmeta),
                                  const char *data) const
{
  uint32_t cp;
  next_unicode_codepoint_t next_fn;
  next_fn =
      get_next_unicode_codepoint_function(m_encoding, assign_error_nocheck);
  const char *begin = reinterpret_cast<const string_type_data *>(data)->begin;
  const char *end = reinterpret_cast<const string_type_data *>(data)->end;

  // Print as an escaped string
  o << "\"";
  while (begin < end) {
    cp = next_fn(begin, end);
    print_escaped_unicode_codepoint(o, cp, false);
  }
  o << "\"";
}

void ndt::string_type::print_type(std::ostream &o) const
{

  o << "string";
  if (m_encoding != string_encoding_utf_8) {
    o << "['" << m_encoding << "']";
  }
}

bool ndt::string_type::is_unique_data_owner(const char *arrmeta) const
{
  const string_type_arrmeta *md =
      reinterpret_cast<const string_type_arrmeta *>(arrmeta);
  if (md->blockref != NULL && (md->blockref->m_use_count != 1 ||
                               md->blockref->m_type != pod_memory_block_type)) {
    return false;
  }
  return true;
}

ndt::type ndt::string_type::get_canonical_type() const
{
  return type(this, true);
}

void ndt::string_type::get_shape(intptr_t ndim, intptr_t i, intptr_t *out_shape,
                                 const char *DYND_UNUSED(arrmeta),
                                 const char *DYND_UNUSED(data)) const
{
  out_shape[i] = -1;
  if (i + 1 < ndim) {
    stringstream ss;
    ss << "requested too many dimensions from type " << type(this, true);
    throw runtime_error(ss.str());
  }
}

bool
ndt::string_type::is_lossless_assignment(const type &DYND_UNUSED(dst_tp),
                                         const type &DYND_UNUSED(src_tp)) const
{
  // Don't shortcut anything to 'nocheck' error checking, so that
  // decoding errors get caught appropriately.
  return false;
}

bool ndt::string_type::operator==(const base_type &rhs) const
{
  if (this == &rhs) {
    return true;
  } else if (rhs.get_type_id() != string_type_id) {
    return false;
  } else {
    const string_type *dt = static_cast<const string_type *>(&rhs);
    return m_encoding == dt->m_encoding;
  }
}

void ndt::string_type::arrmeta_default_construct(char *arrmeta,
                                                 bool blockref_alloc) const
{
  // Simply allocate a POD memory block
  if (blockref_alloc) {
    string_type_arrmeta *md = reinterpret_cast<string_type_arrmeta *>(arrmeta);
    md->blockref = make_pod_memory_block().release();
  }
}

void ndt::string_type::arrmeta_copy_construct(
    char *dst_arrmeta, const char *src_arrmeta,
    memory_block_data *embedded_reference) const
{
  // Copy the blockref, switching it to the embedded_reference if necessary
  const string_type_arrmeta *src_md =
      reinterpret_cast<const string_type_arrmeta *>(src_arrmeta);
  string_type_arrmeta *dst_md =
      reinterpret_cast<string_type_arrmeta *>(dst_arrmeta);
  dst_md->blockref = src_md->blockref ? src_md->blockref : embedded_reference;
  if (dst_md->blockref) {
    memory_block_incref(dst_md->blockref);
  }
}

void ndt::string_type::arrmeta_reset_buffers(char *arrmeta) const
{
  const string_type_arrmeta *md =
      reinterpret_cast<const string_type_arrmeta *>(arrmeta);
  if (md->blockref != NULL && md->blockref->m_type == pod_memory_block_type) {
    memory_block_pod_allocator_api *allocator =
        get_memory_block_pod_allocator_api(md->blockref);
    allocator->reset(md->blockref);
  } else {
    throw runtime_error(
        "can only reset the buffers of a dynd string "
        "type if the memory block reference was constructed by default");
  }
}

void ndt::string_type::arrmeta_finalize_buffers(char *arrmeta) const
{
  string_type_arrmeta *md = reinterpret_cast<string_type_arrmeta *>(arrmeta);
  if (md->blockref != NULL) {
    // Finalize the memory block
    memory_block_pod_allocator_api *allocator =
        get_memory_block_pod_allocator_api(md->blockref);
    if (allocator != NULL) {
      allocator->finalize(md->blockref);
    }
  }
}

void ndt::string_type::arrmeta_destruct(char *arrmeta) const
{
  string_type_arrmeta *md = reinterpret_cast<string_type_arrmeta *>(arrmeta);
  if (md->blockref) {
    memory_block_decref(md->blockref);
  }
}

void ndt::string_type::arrmeta_debug_print(const char *arrmeta, std::ostream &o,
                                           const std::string &indent) const
{
  const string_type_arrmeta *md =
      reinterpret_cast<const string_type_arrmeta *>(arrmeta);
  o << indent << "string arrmeta\n";
  memory_block_debug_print(md->blockref, o, indent + " ");
}

intptr_t ndt::string_type::make_assignment_kernel(
    void *ckb, intptr_t ckb_offset, const type &dst_tp, const char *dst_arrmeta,
    const type &src_tp, const char *src_arrmeta, kernel_request_t kernreq,
    const eval::eval_context *ectx) const
{
  if (this == dst_tp.extended()) {
    switch (src_tp.get_type_id()) {
    case string_type_id: {
      return make_blockref_string_assignment_kernel(
          ckb, ckb_offset, dst_arrmeta, get_encoding(), src_arrmeta,
          src_tp.extended<base_string_type>()->get_encoding(), kernreq, ectx);
    }
    case fixed_string_type_id: {
      return make_fixed_string_to_blockref_string_assignment_kernel(
          ckb, ckb_offset, dst_arrmeta, get_encoding(), src_tp.get_data_size(),
          src_tp.extended<base_string_type>()->get_encoding(), kernreq, ectx);
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

size_t ndt::string_type::make_comparison_kernel(
    void *ckb, intptr_t ckb_offset, const type &src0_dt,
    const char *src0_arrmeta, const type &src1_dt, const char *src1_arrmeta,
    comparison_type_t comptype, const eval::eval_context *ectx) const
{
  if (this == src0_dt.extended()) {
    if (*this == *src1_dt.extended()) {
      return make_string_comparison_kernel(ckb, ckb_offset, m_encoding,
                                           comptype, ectx);
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

void ndt::string_type::make_string_iter(dim_iter *out_di,
                                        string_encoding_t encoding,
                                        const char *arrmeta, const char *data,
                                        const memory_block_ptr &ref,
                                        intptr_t buffer_max_mem,
                                        const eval::eval_context *ectx) const
{
  const string_type_data *d = reinterpret_cast<const string_type_data *>(data);
  memory_block_ptr dataref = ref;
  const string_type_arrmeta *md =
      reinterpret_cast<const string_type_arrmeta *>(arrmeta);
  if (md->blockref != NULL) {
    dataref = memory_block_ptr(md->blockref);
  }
  iter::make_string_iter(out_di, encoding, m_encoding, d->begin, d->end,
                         dataref, buffer_max_mem, ectx);
}