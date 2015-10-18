//
// Copyright (C) 2011-15 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <stdexcept>
#include <sstream>

#include <dynd/type.hpp>
#include <dynd/diagnostics.hpp>
#include <dynd/kernels/string_comparison_kernels.hpp>
#include <dynd/types/fixed_string_type.hpp>
#include <dynd/types/string_type.hpp>
#include <dynd/types/convert_type.hpp>

using namespace std;
using namespace dynd;

/////////////////////////////////////////
// fixed_string comparison

namespace {
struct fixed_string_compare_kernel_extra {
  ckernel_prefix base;
  size_t string_size;
};

struct ascii_utf8_fixed_string_compare_kernel {
  static void less(ckernel_prefix *extra, char *dst, char *const *src)
  {
    size_t stringsize = reinterpret_cast<fixed_string_compare_kernel_extra *>(extra)->string_size;
    *reinterpret_cast<int *>(dst) = strncmp(src[0], src[1], stringsize) < 0;
  }

  static void less_equal(ckernel_prefix *extra, char *dst, char *const *src)
  {
    size_t stringsize = reinterpret_cast<fixed_string_compare_kernel_extra *>(extra)->string_size;
    *reinterpret_cast<int *>(dst) = strncmp(src[0], src[1], stringsize) <= 0;
  }

  static void equal(ckernel_prefix *extra, char *dst, char *const *src)
  {
    size_t stringsize = reinterpret_cast<fixed_string_compare_kernel_extra *>(extra)->string_size;
    *reinterpret_cast<int *>(dst) = strncmp(src[0], src[1], stringsize) == 0;
  }

  static void not_equal(ckernel_prefix *extra, char *dst, char *const *src)
  {
    size_t stringsize = reinterpret_cast<fixed_string_compare_kernel_extra *>(extra)->string_size;
    *reinterpret_cast<int *>(dst) = strncmp(src[0], src[1], stringsize) != 0;
  }

  static void greater_equal(ckernel_prefix *extra, char *dst, char *const *src)
  {
    size_t stringsize = reinterpret_cast<fixed_string_compare_kernel_extra *>(extra)->string_size;
    *reinterpret_cast<int *>(dst) = strncmp(src[0], src[1], stringsize) >= 0;
  }

  static void greater(ckernel_prefix *extra, char *dst, char *const *src)
  {
    size_t stringsize = reinterpret_cast<fixed_string_compare_kernel_extra *>(extra)->string_size;
    *reinterpret_cast<int *>(dst) = strncmp(src[0], src[1], stringsize) > 0;
  }
};

struct utf16_fixed_string_compare_kernel {
  static void less(ckernel_prefix *extra, char *dst, char *const *src)
  {
    size_t stringsize = reinterpret_cast<fixed_string_compare_kernel_extra *>(extra)->string_size;
    *reinterpret_cast<int *>(dst) = lexicographical_compare(
        reinterpret_cast<const uint16_t *>(src[0]), reinterpret_cast<const uint16_t *>(src[0]) + stringsize,
        reinterpret_cast<const uint16_t *>(src[1]), reinterpret_cast<const uint16_t *>(src[1]) + stringsize);
  }

  static void less_equal(ckernel_prefix *extra, char *dst, char *const *src)
  {
    size_t stringsize = reinterpret_cast<fixed_string_compare_kernel_extra *>(extra)->string_size;
    *reinterpret_cast<int *>(dst) = !lexicographical_compare(reinterpret_cast<const uint16_t *>(src[1]),
                                                             reinterpret_cast<const uint16_t *>(src[1]) + stringsize,
                                                             reinterpret_cast<const uint16_t *>(src[0]),
                                                             reinterpret_cast<const uint16_t *>(src[0]) + stringsize);
  }

  static void equal(ckernel_prefix *extra, char *dst, char *const *src)
  {
    size_t string_size = reinterpret_cast<fixed_string_compare_kernel_extra *>(extra)->string_size;
    const uint16_t *lhs = reinterpret_cast<const uint16_t *>(src[0]);
    const uint16_t *rhs = reinterpret_cast<const uint16_t *>(src[1]);
    for (size_t i = 0; i != string_size; ++i, ++lhs, ++rhs) {
      if (*lhs != *rhs) {
        *reinterpret_cast<int *>(dst) = false;
        return;
      }
    }
    *reinterpret_cast<int *>(dst) = true;
  }

  static void not_equal(ckernel_prefix *extra, char *dst, char *const *src)
  {
    size_t string_size = reinterpret_cast<fixed_string_compare_kernel_extra *>(extra)->string_size;
    const uint16_t *lhs = reinterpret_cast<const uint16_t *>(src[0]);
    const uint16_t *rhs = reinterpret_cast<const uint16_t *>(src[1]);
    for (size_t i = 0; i != string_size; ++i, ++lhs, ++rhs) {
      if (*lhs != *rhs) {
        *reinterpret_cast<int *>(dst) = true;
        return;
      }
    }
    *reinterpret_cast<int *>(dst) = false;
  }

  static void greater_equal(ckernel_prefix *extra, char *dst, char *const *src)
  {
    size_t stringsize = reinterpret_cast<fixed_string_compare_kernel_extra *>(extra)->string_size;
    *reinterpret_cast<int *>(dst) = !lexicographical_compare(reinterpret_cast<const uint16_t *>(src[0]),
                                                             reinterpret_cast<const uint16_t *>(src[0]) + stringsize,
                                                             reinterpret_cast<const uint16_t *>(src[1]),
                                                             reinterpret_cast<const uint16_t *>(src[1]) + stringsize);
  }

  static void greater(ckernel_prefix *extra, char *dst, char *const *src)
  {
    size_t stringsize = reinterpret_cast<fixed_string_compare_kernel_extra *>(extra)->string_size;
    *reinterpret_cast<int *>(dst) = lexicographical_compare(
        reinterpret_cast<const uint16_t *>(src[1]), reinterpret_cast<const uint16_t *>(src[1]) + stringsize,
        reinterpret_cast<const uint16_t *>(src[0]), reinterpret_cast<const uint16_t *>(src[0]) + stringsize);
  }
};

struct utf32_fixed_string_compare_kernel {
  static void less(ckernel_prefix *extra, char *dst, char *const *src)
  {
    size_t stringsize = reinterpret_cast<fixed_string_compare_kernel_extra *>(extra)->string_size;
    *reinterpret_cast<int *>(dst) = lexicographical_compare(
        reinterpret_cast<const uint32_t *>(src[0]), reinterpret_cast<const uint32_t *>(src[0]) + stringsize,
        reinterpret_cast<const uint32_t *>(src[1]), reinterpret_cast<const uint32_t *>(src[1]) + stringsize);
  }

  static void less_equal(ckernel_prefix *extra, char *dst, char *const *src)
  {
    size_t stringsize = reinterpret_cast<fixed_string_compare_kernel_extra *>(extra)->string_size;
    *reinterpret_cast<int *>(dst) = !lexicographical_compare(reinterpret_cast<const uint32_t *>(src[1]),
                                                             reinterpret_cast<const uint32_t *>(src[1]) + stringsize,
                                                             reinterpret_cast<const uint32_t *>(src[0]),
                                                             reinterpret_cast<const uint32_t *>(src[0]) + stringsize);
  }

  static void equal(ckernel_prefix *extra, char *dst, char *const *src)
  {
    size_t string_size = reinterpret_cast<fixed_string_compare_kernel_extra *>(extra)->string_size;
    const uint32_t *lhs = reinterpret_cast<const uint32_t *>(src[0]);
    const uint32_t *rhs = reinterpret_cast<const uint32_t *>(src[1]);
    for (size_t i = 0; i != string_size; ++i, ++lhs, ++rhs) {
      if (*lhs != *rhs) {
        *reinterpret_cast<int *>(dst) = false;
        return;
      }
    }
    *reinterpret_cast<int *>(dst) = true;
  }

  static void not_equal(ckernel_prefix *extra, char *dst, char *const *src)
  {
    size_t string_size = reinterpret_cast<fixed_string_compare_kernel_extra *>(extra)->string_size;
    const uint32_t *lhs = reinterpret_cast<const uint32_t *>(src[0]);
    const uint32_t *rhs = reinterpret_cast<const uint32_t *>(src[1]);
    for (size_t i = 0; i != string_size; ++i, ++lhs, ++rhs) {
      if (*lhs != *rhs) {
        *reinterpret_cast<int *>(dst) = true;
        return;
      }
    }
    *reinterpret_cast<int *>(dst) = false;
  }

  static void greater_equal(ckernel_prefix *extra, char *dst, char *const *src)
  {
    size_t stringsize = reinterpret_cast<fixed_string_compare_kernel_extra *>(extra)->string_size;
    *reinterpret_cast<int *>(dst) = !lexicographical_compare(reinterpret_cast<const uint32_t *>(src[0]),
                                                             reinterpret_cast<const uint32_t *>(src[0]) + stringsize,
                                                             reinterpret_cast<const uint32_t *>(src[1]),
                                                             reinterpret_cast<const uint32_t *>(src[1]) + stringsize);
  }

  static void greater(ckernel_prefix *extra, char *dst, char *const *src)
  {
    size_t stringsize = reinterpret_cast<fixed_string_compare_kernel_extra *>(extra)->string_size;
    *reinterpret_cast<int *>(dst) = lexicographical_compare(
        reinterpret_cast<const uint32_t *>(src[1]), reinterpret_cast<const uint32_t *>(src[1]) + stringsize,
        reinterpret_cast<const uint32_t *>(src[0]), reinterpret_cast<const uint32_t *>(src[0]) + stringsize);
  }
};
} // anonymous namespace

#define DYND_FIXEDSTRING_COMPARISON_TABLE_TYPE_LEVEL(type)                                                             \
  {                                                                                                                    \
    type##_fixed_string_compare_kernel::less, type##_fixed_string_compare_kernel::less,                                \
        type##_fixed_string_compare_kernel::less_equal, type##_fixed_string_compare_kernel::equal,                     \
        type##_fixed_string_compare_kernel::not_equal, type##_fixed_string_compare_kernel::greater_equal,              \
        type##_fixed_string_compare_kernel::greater                                                                    \
  }

size_t dynd::make_fixed_string_comparison_kernel(void *ckb, intptr_t ckb_offset, size_t string_size,
                                                 string_encoding_t encoding, comparison_type_t comptype,
                                                 const eval::eval_context *DYND_UNUSED(ectx))
{
  static int lookup[5] = {0, 1, 0, 1, 2};
  static expr_single_t fixed_string_comparisons_table[3][7] = {DYND_FIXEDSTRING_COMPARISON_TABLE_TYPE_LEVEL(ascii_utf8),
                                                               DYND_FIXEDSTRING_COMPARISON_TABLE_TYPE_LEVEL(utf16),
                                                               DYND_FIXEDSTRING_COMPARISON_TABLE_TYPE_LEVEL(utf32)};
  if (0 <= encoding && encoding < 5 && 0 <= comptype && comptype < 7) {
    fixed_string_compare_kernel_extra *e = reinterpret_cast<ckernel_builder<kernel_request_host> *>(ckb)
                                               ->alloc_ck<fixed_string_compare_kernel_extra>(ckb_offset);
    e->base.function = reinterpret_cast<void *>(fixed_string_comparisons_table[lookup[encoding]][comptype]);
    e->string_size = string_size;
    return ckb_offset;
  } else {
    stringstream ss;
    ss << "make_fixed_string_comparison_kernel: Unexpected encoding (" << encoding;
    ss << ") or comparison type (" << comptype << ")";
    throw runtime_error(ss.str());
  }
}

#undef DYND_FIXEDSTRING_COMPARISON_TABLE_TYPE_LEVEL

/////////////////////////////////////////
// blockref string comparison

namespace {
template <typename T>
struct string_compare_kernel {
  static void less(ckernel_prefix *DYND_UNUSED(self), char *dst, char *const *src)
  {
    const dynd::string *da = reinterpret_cast<const dynd::string *>(src[0]);
    const dynd::string *db = reinterpret_cast<const dynd::string *>(src[1]);
    *reinterpret_cast<int *>(dst) =
        lexicographical_compare(reinterpret_cast<const T *>(da->begin()), reinterpret_cast<const T *>(da->end()),
                                reinterpret_cast<const T *>(db->begin()), reinterpret_cast<const T *>(db->end()));
  }

  static void less_equal(ckernel_prefix *DYND_UNUSED(self), char *dst, char *const *src)
  {
    const dynd::string *da = reinterpret_cast<const dynd::string *>(src[0]);
    const dynd::string *db = reinterpret_cast<const dynd::string *>(src[1]);
    *reinterpret_cast<int *>(dst) =
        !lexicographical_compare(reinterpret_cast<const T *>(db->begin()), reinterpret_cast<const T *>(db->end()),
                                 reinterpret_cast<const T *>(da->begin()), reinterpret_cast<const T *>(da->end()));
  }

  static void equal(ckernel_prefix *DYND_UNUSED(self), char *dst, char *const *src)
  {
    const dynd::string *da = reinterpret_cast<const dynd::string *>(src[0]);
    const dynd::string *db = reinterpret_cast<const dynd::string *>(src[1]);
    *reinterpret_cast<int *>(dst) = (da->end() - da->begin() == db->end() - db->begin()) &&
                                    memcmp(da->begin(), db->begin(), da->end() - da->begin()) == 0;
  }

  static void not_equal(ckernel_prefix *DYND_UNUSED(self), char *dst, char *const *src)
  {
    const dynd::string *da = reinterpret_cast<const dynd::string *>(src[0]);
    const dynd::string *db = reinterpret_cast<const dynd::string *>(src[1]);
    *reinterpret_cast<int *>(dst) = (da->end() - da->begin() != db->end() - db->begin()) ||
                                    memcmp(da->begin(), db->begin(), da->end() - da->begin()) != 0;
  }

  static void greater_equal(ckernel_prefix *DYND_UNUSED(self), char *dst, char *const *src)
  {
    const dynd::string *da = reinterpret_cast<const dynd::string *>(src[0]);
    const dynd::string *db = reinterpret_cast<const dynd::string *>(src[1]);
    *reinterpret_cast<int *>(dst) =
        !lexicographical_compare(reinterpret_cast<const T *>(da->begin()), reinterpret_cast<const T *>(da->end()),
                                 reinterpret_cast<const T *>(db->begin()), reinterpret_cast<const T *>(db->end()));
  }

  static void greater(ckernel_prefix *DYND_UNUSED(self), char *dst, char *const *src)
  {
    const dynd::string *da = reinterpret_cast<const dynd::string *>(src[0]);
    const dynd::string *db = reinterpret_cast<const dynd::string *>(src[1]);
    *reinterpret_cast<int *>(dst) =
        lexicographical_compare(reinterpret_cast<const T *>(db->begin()), reinterpret_cast<const T *>(db->end()),
                                reinterpret_cast<const T *>(da->begin()), reinterpret_cast<const T *>(da->end()));
  }
};
} // anonymous namespace

#define DYND_STRING_COMPARISON_TABLE_TYPE_LEVEL(type)                                                                  \
  {                                                                                                                    \
    string_compare_kernel<type>::less, string_compare_kernel<type>::less, string_compare_kernel<type>::less_equal,     \
        string_compare_kernel<type>::equal, string_compare_kernel<type>::not_equal,                                    \
        string_compare_kernel<type>::greater_equal, string_compare_kernel<type>::greater                               \
  }

size_t dynd::make_string_comparison_kernel(void *ckb, intptr_t ckb_offset, string_encoding_t encoding,
                                           comparison_type_t comptype, const eval::eval_context *DYND_UNUSED(ectx))
{
  static int lookup[5] = {0, 1, 0, 1, 2};
  static expr_single_t string_comparisons_table[3][7] = {DYND_STRING_COMPARISON_TABLE_TYPE_LEVEL(uint8_t),
                                                         DYND_STRING_COMPARISON_TABLE_TYPE_LEVEL(uint16_t),
                                                         DYND_STRING_COMPARISON_TABLE_TYPE_LEVEL(uint32_t)};
  if (0 <= encoding && encoding < 5 && 0 <= comptype && comptype < 7) {
    ckernel_prefix *e =
        reinterpret_cast<ckernel_builder<kernel_request_host> *>(ckb)->alloc_ck<ckernel_prefix>(ckb_offset);
    e->function = reinterpret_cast<void *>(string_comparisons_table[lookup[encoding]][comptype]);
    return ckb_offset;
  } else {
    stringstream ss;
    ss << "make_string_comparison_kernel: Unexpected encoding (" << encoding;
    ss << ") or comparison type (" << comptype << ")";
    throw runtime_error(ss.str());
  }
}

#undef DYND_STRING_COMPARISON_TABLE_TYPE_LEVEL

size_t dynd::make_general_string_comparison_kernel(void *ckb, intptr_t ckb_offset, const ndt::type &src0_dt,
                                                   const char *src0_arrmeta, const ndt::type &src1_dt,
                                                   const char *src1_arrmeta, comparison_type_t comptype,
                                                   const eval::eval_context *ectx)
{
  // TODO: Make more efficient, direct comparison kernels
  ndt::type sdt = ndt::string_type::make();
  return make_comparison_kernel(ckb, ckb_offset, ndt::convert_type::make(sdt, src0_dt), src0_arrmeta,
                                ndt::convert_type::make(sdt, src1_dt), src1_arrmeta, comptype, ectx);
}
