//
// Copyright (C) 2011-15 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <stdexcept>
#include <sstream>

#include <dynd/shortvector.hpp>
#include <dynd/type.hpp>
#include <dynd/diagnostics.hpp>
#include <dynd/kernels/string_algorithm_kernels.hpp>
#include <dynd/types/string_type.hpp>

using namespace std;
using namespace dynd;

/////////////////////////////////////////////
// String concatenation kernel

void kernels::string_concatenation_kernel::init(size_t nop, const char *dst_arrmeta,
                                                const char **DYND_UNUSED(src_arrmeta))
{
  const string_type_arrmeta *sdm = reinterpret_cast<const string_type_arrmeta *>(dst_arrmeta);
  m_nop = nop;
  // This is a borrowed reference
  m_dst_blockref = sdm->blockref;
}

inline void concat_one_string(size_t nop, dynd::string *d, const dynd::string *const *s,
                              memory_block_pod_allocator_api *allocator, memory_block_data *dst_blockref)
{
  // Get the size of the concatenated string
  size_t size = 0;
  for (size_t i = 0; i != nop; ++i) {
    size += (s[i]->end() - s[i]->begin());
  }
  // Allocate the output
  size_t alignment = 1; // NOTE: This kernel is hardcoded for UTF-8, alignment 1
  char *begin, *end;
  allocator->allocate(dst_blockref, size, alignment, &begin, &end);
  d->assign(begin, end - begin);
  // Copy the string data
  char *dst = d->begin();
  for (size_t i = 0; i != nop; ++i) {
    size_t op_size = (s[i]->end() - s[i]->begin());
    DYND_MEMCPY(dst, s[i]->begin(), op_size);
    dst += op_size;
  }
}

void kernels::string_concatenation_kernel::single(char *dst, char *const *src, ckernel_prefix *extra)
{
  const extra_type *e = reinterpret_cast<const extra_type *>(extra);
  dynd::string *d = reinterpret_cast<dynd::string *>(dst);
  const dynd::string *const *s = reinterpret_cast<const dynd::string *const *>(src);
  memory_block_pod_allocator_api *allocator = get_memory_block_pod_allocator_api(e->m_dst_blockref);

  concat_one_string(e->m_nop, d, s, allocator, e->m_dst_blockref);
}

void kernels::string_concatenation_kernel::strided(char *dst, intptr_t dst_stride, char *const *src,
                                                   const intptr_t *src_stride, size_t count, ckernel_prefix *extra)
{
  const extra_type *e = reinterpret_cast<const extra_type *>(extra);
  size_t nop = e->m_nop;
  memory_block_pod_allocator_api *allocator = get_memory_block_pod_allocator_api(e->m_dst_blockref);

  // Loop to concatenate all the strings3
  shortvector<const char *> src_vec(nop, src);
  for (size_t i = 0; i != count; ++i) {
    dynd::string *d = reinterpret_cast<dynd::string *>(dst);
    const dynd::string *const *s = reinterpret_cast<const dynd::string *const *>(src_vec.get());
    concat_one_string(nop, d, s, allocator, e->m_dst_blockref);
    dst += dst_stride;
    for (size_t op = 0; op < nop; ++op) {
      src_vec[op] += src_stride[op];
    }
  }
}

/////////////////////////////////////////////
// String find kernel

void kernels::string_find_kernel::init(const ndt::type *src_tp, const char *const *src_arrmeta)
{
  if (src_tp[0].get_kind() != string_kind) {
    stringstream ss;
    ss << "Expected a string type for the string find kernel, not " << src_tp[0];
    throw runtime_error(ss.str());
  }
  if (src_tp[1].get_kind() != string_kind) {
    stringstream ss;
    ss << "Expected a string type for the string find kernel, not " << src_tp[1];
    throw runtime_error(ss.str());
  }
  m_base.destructor = &kernels::string_find_kernel::destruct;
  m_str_type = static_cast<const ndt::base_string_type *>(ndt::type(src_tp[0]).release());
  m_str_arrmeta = src_arrmeta[0];
  m_sub_type = static_cast<const ndt::base_string_type *>(ndt::type(src_tp[1]).release());
  m_sub_arrmeta = src_arrmeta[1];
}

void kernels::string_find_kernel::destruct(ckernel_prefix *extra)
{
  extra_type *e = reinterpret_cast<extra_type *>(extra);
  base_type_xdecref(e->m_str_type);
  base_type_xdecref(e->m_sub_type);
}

inline void find_one_string(intptr_t *d, const char *str_begin, const char *str_end, const char *sub_begin,
                            const char *sub_end, next_unicode_codepoint_t str_next_fn,
                            next_unicode_codepoint_t sub_next_fn)
{
  int32_t sub_first = sub_next_fn(sub_begin, sub_end);
  // TODO: This algorithm is slow and naive, should use fast algorithms...
  intptr_t pos = 0;
  while (str_begin < str_end) {
    int32_t str_cp = str_next_fn(str_begin, str_end);
    if (str_cp == sub_first) {
      // If the first character matched, try the rest
      const char *sub_match_begin = sub_begin, *str_match_begin = str_begin;
      bool matched = true;
      while (sub_match_begin < sub_end) {
        if (str_match_begin == str_end) {
          // End of the string
          matched = false;
          break;
        }
        int32_t sub_cp = str_next_fn(sub_match_begin, sub_end);
        str_cp = str_next_fn(str_match_begin, str_end);
        if (sub_cp != str_cp) {
          // Mismatched character
          matched = false;
          break;
        }
      }
      if (matched) {
        *d = pos;
        return;
      }
    }
    ++pos;
  }

  *d = -1;
}

void kernels::string_find_kernel::single(char *dst, char *const *src, ckernel_prefix *extra)
{
  const extra_type *e = reinterpret_cast<const extra_type *>(extra);
  string_encoding_t str_encoding = e->m_str_type->get_encoding();
  string_encoding_t sub_encoding = e->m_sub_type->get_encoding();
  // TODO: Get the error mode from the evaluation context
  next_unicode_codepoint_t str_next_fn = get_next_unicode_codepoint_function(str_encoding, assign_error_nocheck);
  next_unicode_codepoint_t sub_next_fn = get_next_unicode_codepoint_function(sub_encoding, assign_error_nocheck);

  intptr_t *d = reinterpret_cast<intptr_t *>(dst);
  // Get the extents of the string and substring
  const char *str_begin, *str_end;
  e->m_str_type->get_string_range(&str_begin, &str_end, e->m_str_arrmeta, src[0]);
  const char *sub_begin, *sub_end;
  e->m_sub_type->get_string_range(&sub_begin, &sub_end, e->m_sub_arrmeta, src[1]);
  find_one_string(d, str_begin, str_end, sub_begin, sub_end, str_next_fn, sub_next_fn);
}

void kernels::string_find_kernel::strided(char *dst, intptr_t dst_stride, char *const *src, const intptr_t *src_stride,
                                          size_t count, ckernel_prefix *extra)
{
  const extra_type *e = reinterpret_cast<const extra_type *>(extra);
  string_encoding_t str_encoding = e->m_str_type->get_encoding();
  string_encoding_t sub_encoding = e->m_sub_type->get_encoding();
  // TODO: Get the error mode from the evaluation context
  next_unicode_codepoint_t str_next_fn = get_next_unicode_codepoint_function(str_encoding, assign_error_nocheck);
  next_unicode_codepoint_t sub_next_fn = get_next_unicode_codepoint_function(sub_encoding, assign_error_nocheck);

  const char *src_str = src[0], *src_sub = src[1];
  for (size_t i = 0; i != count; ++i) {
    intptr_t *d = reinterpret_cast<intptr_t *>(dst);
    // Get the extents of the string and substring
    const char *str_begin, *str_end;
    e->m_str_type->get_string_range(&str_begin, &str_end, e->m_str_arrmeta, src_str);
    const char *sub_begin, *sub_end;
    e->m_sub_type->get_string_range(&sub_begin, &sub_end, e->m_sub_arrmeta, src_sub);
    find_one_string(d, str_begin, str_end, sub_begin, sub_end, str_next_fn, sub_next_fn);

    dst += dst_stride;
    src_str += src_stride[0];
    src_sub += src_stride[1];
  }
}
