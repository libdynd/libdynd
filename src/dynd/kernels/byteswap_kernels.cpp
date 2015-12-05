//
// Copyright (C) 2011-15 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <stdexcept>

#include <dynd/diagnostics.hpp>
#include <dynd/func/callable.hpp>
#include <dynd/kernels/base_kernel.hpp>
#include <dynd/kernels/byteswap_kernels.hpp>

using namespace std;
using namespace dynd;

namespace {

template <typename T>
struct aligned_fixed_size_byteswap
    : nd::base_kernel<aligned_fixed_size_byteswap<T>, 1> {
  void single(char *dst, char *const *src)
  {
    DYND_ASSERT_ALIGNED(dst, 0, sizeof(T),
                        "type: " << ndt::type(dynd::type_id_of<T>::value));
    DYND_ASSERT_ALIGNED(src, 0, sizeof(T),
                        "type: " << ndt::type(dynd::type_id_of<T>::value));
    *reinterpret_cast<T *>(dst) =
        byteswap_value(**reinterpret_cast<T *const *>(src));
  }

  void strided(char *dst, intptr_t dst_stride, char *const *src,
               const intptr_t *src_stride, size_t count)
  {
    DYND_ASSERT_ALIGNED(dst, dst_stride, sizeof(T),
                        "type: " << ndt::type(dynd::type_id_of<T>::value));
    DYND_ASSERT_ALIGNED(src, src_stride, sizeof(T),
                        "type: " << ndt::type(dynd::type_id_of<T>::value));
    char *src0 = src[0];
    intptr_t src0_stride = src_stride[0];
    for (size_t i = 0; i != count; ++i) {
      *reinterpret_cast<T *>(dst) =
          byteswap_value(*reinterpret_cast<T *>(src0));
      dst += dst_stride;
      src0 += src0_stride;
    }
  }
};

template <typename T>
struct aligned_fixed_size_pairwise_byteswap_kernel
    : nd::base_kernel<aligned_fixed_size_pairwise_byteswap_kernel<T>, 1> {
  void single(char *dst, char *const *src)
  {
    DYND_ASSERT_ALIGNED(dst, 0, sizeof(T),
                        "type: " << ndt::type(dynd::type_id_of<T>::value));
    DYND_ASSERT_ALIGNED(src, 0, sizeof(T),
                        "type: " << ndt::type(dynd::type_id_of<T>::value));
    *reinterpret_cast<T *>(dst) =
        byteswap_value(**reinterpret_cast<T *const *>(src));
    *(reinterpret_cast<T *>(dst) + 1) =
        byteswap_value(*(*reinterpret_cast<T *const *>(src) + 1));
  }

  void strided(char *dst, intptr_t dst_stride, char *const *src,
               const intptr_t *src_stride, size_t count)
  {
    DYND_ASSERT_ALIGNED(dst, dst_stride, sizeof(T),
                        "type: " << ndt::type(dynd::type_id_of<T>::value));
    DYND_ASSERT_ALIGNED(src, src_stride, sizeof(T),
                        "type: " << ndt::type(dynd::type_id_of<T>::value));
    char *src0 = src[0];
    intptr_t src0_stride = src_stride[0];
    for (size_t i = 0; i != count; ++i) {
      *reinterpret_cast<T *>(dst) =
          byteswap_value(**reinterpret_cast<T *const *>(src0));
      *(reinterpret_cast<T *>(dst) + 1) =
          byteswap_value(*(*reinterpret_cast<T *const *>(src0) + 1));
      dst += dst_stride;
      src0 += src0_stride;
    }
  }
};
} // anonymous namespace

namespace {
struct byteswap_ck : nd::base_kernel<byteswap_ck, 1> {
  size_t data_size;

  byteswap_ck(size_t data_size) : data_size(data_size)
  {
  }

  void single(char *dst, char *const *src)
  {
    // Do a different loop for in-place swap versus copying swap,
    // so this one kernel function works correctly for both cases.
    if (src[0] == dst) {
      // In-place swap
      for (size_t j = 0; j < data_size / 2; ++j) {
        std::swap(dst[j], dst[data_size - j - 1]);
      }
    } else {
      for (size_t j = 0; j < data_size; ++j) {
        dst[j] = src[0][data_size - j - 1];
      }
    }
  }
};

struct pairwise_byteswap_ck : nd::base_kernel<pairwise_byteswap_ck, 1> {
  size_t data_size;

  pairwise_byteswap_ck(size_t data_size) : data_size(data_size)
  {
  }

  void single(char *dst, char *const *src)
  {
    // Do a different loop for in-place swap versus copying swap,
    // so this one kernel function works correctly for both cases.
    if (src[0] == dst) {
      // In-place swap
      for (size_t j = 0; j < data_size / 4; ++j) {
        std::swap(dst[j], dst[data_size / 2 - j - 1]);
      }
      for (size_t j = 0; j < data_size / 4; ++j) {
        std::swap(dst[data_size / 2 + j], dst[data_size - j - 1]);
      }
    } else {
      for (size_t j = 0; j < data_size / 2; ++j) {
        dst[j] = src[0][data_size / 2 - j - 1];
      }
      for (size_t j = 0; j < data_size / 2; ++j) {
        dst[data_size / 2 + j] = src[0][data_size - j - 1];
      }
    }
  }
};
} // anonymous namespace

size_t dynd::make_byteswap_assignment_function(void *ckb, intptr_t ckb_offset,
                                               intptr_t data_size,
                                               intptr_t data_alignment,
                                               kernel_request_t kernreq)
{
  if (data_size == data_alignment) {
    switch (data_size) {
    case 2:
      aligned_fixed_size_byteswap<uint16_t>::make(ckb, kernreq, ckb_offset);
      return ckb_offset;
    case 4:
      aligned_fixed_size_byteswap<uint32_t>::make(ckb, kernreq, ckb_offset);
      return ckb_offset;
    case 8:
      aligned_fixed_size_byteswap<uint64_t>::make(ckb, kernreq, ckb_offset);
      return ckb_offset;
    default:
      break;
    }
  }

  // Otherwise use the general case ckernel
  byteswap_ck::make(ckb, kernreq, ckb_offset, data_size);
  return ckb_offset;
}

size_t dynd::make_pairwise_byteswap_assignment_function(
    void *ckb, intptr_t ckb_offset, intptr_t data_size, intptr_t data_alignment,
    kernel_request_t kernreq)
{
  if (data_size == data_alignment) {
    switch (data_size) {
    case 4:
      aligned_fixed_size_pairwise_byteswap_kernel<uint16_t>::make(ckb, kernreq,
                                                                  ckb_offset);
      return ckb_offset;
    case 8:
      aligned_fixed_size_pairwise_byteswap_kernel<uint32_t>::make(ckb, kernreq,
                                                                  ckb_offset);
      return ckb_offset;
    case 16:
      aligned_fixed_size_pairwise_byteswap_kernel<uint64_t>::make(ckb, kernreq,
                                                                  ckb_offset);
      return ckb_offset;
    default:
      break;
    }
  }

  // Otherwise use the general case ckernel
  pairwise_byteswap_ck::make(ckb, kernreq, ckb_offset, data_size);
  return ckb_offset;
}
