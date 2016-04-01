//
// Copyright (C) 2011-16 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#pragma once

#include <dynd/parse.hpp>
#include <dynd/kernels/base_kernel.hpp>
#include <dynd/types/option_type.hpp>
#include <dynd/math.hpp>

namespace dynd {
namespace nd {
  namespace detail {

    template <type_id_t Src0TypeID, type_id_t Src0BaseID>
    struct is_na_kernel;

    template <>
    struct is_na_kernel<bool_id, bool_kind_id> : base_strided_kernel<is_na_kernel<bool_id, bool_kind_id>, 1> {
      void single(char *dst, char *const *src) { *dst = **reinterpret_cast<unsigned char *const *>(src) > 1; }

      void strided(char *dst, intptr_t dst_stride, char *const *src, const intptr_t *src_stride, size_t count)
      {
        // Available if the value is 0 or 1
        char *src0 = src[0];
        intptr_t src0_stride = src_stride[0];
        for (size_t i = 0; i != count; ++i) {
          *dst = *reinterpret_cast<unsigned char *>(src0) > 1;
          dst += dst_stride;
          src0 += src0_stride;
        }
      }
    };

    // option[T] for signed integer T
    // NA is the smallest negative value
    template <type_id_t Src0TypeID>
    struct is_na_kernel<Src0TypeID, int_kind_id> : base_strided_kernel<is_na_kernel<Src0TypeID, int_kind_id>, 1> {
      typedef typename type_of<Src0TypeID>::type A0;

      void single(char *dst, char *const *src)
      {
        *dst = **reinterpret_cast<A0 *const *>(src) == std::numeric_limits<A0>::min();
      }

      void strided(char *dst, intptr_t dst_stride, char *const *src, const intptr_t *src_stride, size_t count)
      {
        char *src0 = src[0];
        intptr_t src0_stride = src_stride[0];
        for (size_t i = 0; i != count; ++i) {
          *dst = *reinterpret_cast<A0 *>(src0) == std::numeric_limits<A0>::min();
          dst += dst_stride;
          src0 += src0_stride;
        }
      }
    };

    template <type_id_t Src0TypeID>
    struct is_na_kernel<Src0TypeID, uint_kind_id> : base_strided_kernel<is_na_kernel<Src0TypeID, uint_kind_id>, 1> {
      typedef typename type_of<Src0TypeID>::type src0_type;

      void single(char *dst, char *const *src)
      {
        *dst = **reinterpret_cast<src0_type *const *>(src) == std::numeric_limits<src0_type>::max();
      }

      void strided(char *dst, intptr_t dst_stride, char *const *src, const intptr_t *src_stride, size_t count)
      {
        char *src0 = src[0];
        intptr_t src0_stride = src_stride[0];
        for (size_t i = 0; i != count; ++i) {
          *dst = *reinterpret_cast<src0_type *>(src0) == std::numeric_limits<src0_type>::max();
          dst += dst_stride;
          src0 += src0_stride;
        }
      }
    };

    // option[float]
    // NA is 0x7f8007a2
    // Special rule adopted from R: Any NaN is NA
    template <>
    struct is_na_kernel<float32_id, float_kind_id> : base_strided_kernel<is_na_kernel<float32_id, float_kind_id>, 1> {
      void single(char *dst, char *const *src) { *dst = dynd::isnan(**reinterpret_cast<float *const *>(src)) != 0; }

      void strided(char *dst, intptr_t dst_stride, char *const *src, const intptr_t *src_stride, size_t count)
      {
        char *src0 = src[0];
        intptr_t src0_stride = src_stride[0];
        for (size_t i = 0; i != count; ++i) {
          *dst = dynd::isnan(*reinterpret_cast<float *>(src0)) != 0;
          dst += dst_stride;
          src0 += src0_stride;
        }
      }
    };

    // option[double]
    // NA is 0x7ff00000000007a2ULL
    // Special rule adopted from R: Any NaN is NA
    template <>
    struct is_na_kernel<float64_id, float_kind_id> : base_strided_kernel<is_na_kernel<float64_id, float_kind_id>, 1> {
      void single(char *dst, char *const *src) { *dst = dynd::isnan(**reinterpret_cast<double *const *>(src)) != 0; }

      void strided(char *dst, intptr_t dst_stride, char *const *src, const intptr_t *src_stride, size_t count)
      {
        char *src0 = src[0];
        intptr_t src0_stride = src_stride[0];
        for (size_t i = 0; i != count; ++i) {
          *dst = dynd::isnan(*reinterpret_cast<double *>(src0)) != 0;
          dst += dst_stride;
          src0 += src0_stride;
        }
      }
    };

    // option[complex[float]]
    // NA is two float NAs
    template <>
    struct is_na_kernel<complex_float32_id, complex_kind_id>
        : base_strided_kernel<is_na_kernel<complex_float32_id, complex_kind_id>, 1> {
      void single(char *dst, char *const *src)
      {
        *dst = (*reinterpret_cast<uint32_t *const *>(src))[0] == DYND_FLOAT32_NA_AS_UINT &&
               (*reinterpret_cast<uint32_t *const *>(src))[1] == DYND_FLOAT32_NA_AS_UINT;
      }

      void strided(char *dst, intptr_t dst_stride, char *const *src, const intptr_t *src_stride, size_t count)
      {
        char *src0 = src[0];
        intptr_t src0_stride = src_stride[0];
        for (size_t i = 0; i != count; ++i) {
          *dst = reinterpret_cast<uint32_t *>(src0)[0] == DYND_FLOAT32_NA_AS_UINT &&
                 reinterpret_cast<uint32_t *>(src0)[1] == DYND_FLOAT32_NA_AS_UINT;
          dst += dst_stride;
          src0 += src0_stride;
        }
      }
    };

    // option[complex[double]]
    // NA is two double NAs
    template <>
    struct is_na_kernel<complex_float64_id, complex_kind_id>
        : base_strided_kernel<is_na_kernel<complex_float64_id, complex_kind_id>, 1> {
      void single(char *dst, char *const *src)
      {
        *dst = (*reinterpret_cast<uint64_t *const *>(src))[0] == DYND_FLOAT64_NA_AS_UINT &&
               (*reinterpret_cast<uint64_t *const *>(src))[1] == DYND_FLOAT64_NA_AS_UINT;
      }

      void strided(char *dst, intptr_t dst_stride, char *const *src, const intptr_t *src_stride, size_t count)
      {
        // Available if the value is 0 or 1
        char *src0 = src[0];
        intptr_t src0_stride = src_stride[0];
        for (size_t i = 0; i != count; ++i) {
          *dst = reinterpret_cast<uint64_t *>(src0)[0] == DYND_FLOAT64_NA_AS_UINT &&
                 reinterpret_cast<uint64_t *>(src0)[1] == DYND_FLOAT64_NA_AS_UINT;
          dst += dst_stride;
          src0 += src0_stride;
        }
      }
    };

    template <>
    struct is_na_kernel<void_id, any_kind_id> : base_strided_kernel<is_na_kernel<void_id, any_kind_id>, 1> {
      void single(char *dst, char *const *DYND_UNUSED(src)) { *dst = 1; }

      void strided(char *dst, intptr_t dst_stride, char *const *DYND_UNUSED(src),
                   const intptr_t *DYND_UNUSED(src_stride), size_t count)
      {
        for (size_t i = 0; i != count; ++i) {
          *dst = 1;
          dst += dst_stride;
        }
      }
    };

    template <>
    struct is_na_kernel<bytes_id, bytes_kind_id> : base_strided_kernel<is_na_kernel<bytes_id, bytes_kind_id>, 1> {
      void single(char *dst, char *const *src) { *dst = reinterpret_cast<bytes *>(src[0])->begin() == NULL; }
    };

    template <>
    struct is_na_kernel<string_id, string_kind_id> : base_strided_kernel<is_na_kernel<string_id, string_kind_id>, 1> {
      void single(char *dst, char *const *src) { *dst = reinterpret_cast<string *>(src[0])->begin() == NULL; }
    };

    template <>
    struct is_na_kernel<pointer_id, any_kind_id> : base_strided_kernel<is_na_kernel<pointer_id, any_kind_id>, 1> {
      void single(char *DYND_UNUSED(dst), char *const *DYND_UNUSED(src))
      {
        throw std::runtime_error("is_missing for pointers is not yet implemented");
      }
    };

  } // namespace dynd::nd::detail

  template <type_id_t Src0ValueTypeID>
  struct is_na_kernel : detail::is_na_kernel<Src0ValueTypeID, base_id_of<Src0ValueTypeID>::value> {
  };

} // namespace dynd::nd
} // namespace dynd
