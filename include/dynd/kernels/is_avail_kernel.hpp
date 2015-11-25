//
// Copyright (C) 2011-15 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#pragma once

#include <dynd/kernels/base_kernel.hpp>
#include <dynd/kernels/base_virtual_kernel.hpp>
#include <dynd/types/option_type.hpp>
#include <dynd/types/time_type.hpp>
#include <dynd/math.hpp>

namespace dynd {
namespace nd {
  namespace detail {

    template <type_id_t Src0TypeID, type_kind_t Src0TypeKind>
    struct is_avail_kernel;

    template <>
    struct is_avail_kernel<bool_type_id, bool_kind> : base_kernel<is_avail_kernel<bool_type_id, bool_kind>, 1> {
      void single(char *dst, char *const *src)
      {
        *dst = **reinterpret_cast<unsigned char *const *>(src) <= 1;
      }

      void strided(char *dst, intptr_t dst_stride, char *const *src, const intptr_t *src_stride, size_t count)
      {
        // Available if the value is 0 or 1
        char *src0 = src[0];
        intptr_t src0_stride = src_stride[0];
        for (size_t i = 0; i != count; ++i) {
          *dst = *reinterpret_cast<unsigned char *>(src0) <= 1;
          dst += dst_stride;
          src0 += src0_stride;
        }
      }
    };

    // option[T] for signed integer T
    // NA is the smallest negative value
    template <type_id_t Src0TypeID>
    struct is_avail_kernel<Src0TypeID, sint_kind> : base_kernel<is_avail_kernel<Src0TypeID, sint_kind>, 1> {
      typedef typename type_of<Src0TypeID>::type A0;

      void single(char *dst, char *const *src)
      {
        *dst = **reinterpret_cast<A0 *const *>(src) != std::numeric_limits<A0>::min();
      }

      void strided(char *dst, intptr_t dst_stride, char *const *src, const intptr_t *src_stride, size_t count)
      {
        char *src0 = src[0];
        intptr_t src0_stride = src_stride[0];
        for (size_t i = 0; i != count; ++i) {
          *dst = *reinterpret_cast<A0 *>(src0) != std::numeric_limits<A0>::min();
          dst += dst_stride;
          src0 += src0_stride;
        }
      }
    };

    // option[float]
    // NA is 0x7f8007a2
    // Special rule adopted from R: Any NaN is NA
    template <>
    struct is_avail_kernel<float32_type_id, real_kind> : base_kernel<is_avail_kernel<float32_type_id, real_kind>, 1> {
      void single(char *dst, char *const *src)
      {
        *dst = dynd::isnan(**reinterpret_cast<float *const *>(src)) == 0;
      }

      void strided(char *dst, intptr_t dst_stride, char *const *src, const intptr_t *src_stride, size_t count)
      {
        char *src0 = src[0];
        intptr_t src0_stride = src_stride[0];
        for (size_t i = 0; i != count; ++i) {
          *dst = dynd::isnan(*reinterpret_cast<float *>(src0)) == 0;
          dst += dst_stride;
          src0 += src0_stride;
        }
      }
    };

    // option[double]
    // NA is 0x7ff00000000007a2ULL
    // Special rule adopted from R: Any NaN is NA
    template <>
    struct is_avail_kernel<float64_type_id, real_kind> : base_kernel<is_avail_kernel<float64_type_id, real_kind>, 1> {
      void single(char *dst, char *const *src)
      {
        *dst = dynd::isnan(**reinterpret_cast<double *const *>(src)) == 0;
      }

      void strided(char *dst, intptr_t dst_stride, char *const *src, const intptr_t *src_stride, size_t count)
      {
        char *src0 = src[0];
        intptr_t src0_stride = src_stride[0];
        for (size_t i = 0; i != count; ++i) {
          *dst = dynd::isnan(*reinterpret_cast<double *>(src0)) == 0;
          dst += dst_stride;
          src0 += src0_stride;
        }
      }
    };

    // option[complex[float]]
    // NA is two float NAs
    template <>
    struct is_avail_kernel<complex_float32_type_id,
                           complex_kind> : base_kernel<is_avail_kernel<complex_float32_type_id, complex_kind>, 1> {
      void single(char *dst, char *const *src)
      {
        *dst = (*reinterpret_cast<uint32_t *const *>(src))[0] != DYND_FLOAT32_NA_AS_UINT &&
               (*reinterpret_cast<uint32_t *const *>(src))[1] != DYND_FLOAT32_NA_AS_UINT;
      }

      void strided(char *dst, intptr_t dst_stride, char *const *src, const intptr_t *src_stride, size_t count)
      {
        char *src0 = src[0];
        intptr_t src0_stride = src_stride[0];
        for (size_t i = 0; i != count; ++i) {
          *dst = reinterpret_cast<uint32_t *>(src0)[0] != DYND_FLOAT32_NA_AS_UINT &&
                 reinterpret_cast<uint32_t *>(src0)[1] != DYND_FLOAT32_NA_AS_UINT;
          dst += dst_stride;
          src0 += src0_stride;
        }
      }
    };

    // option[complex[double]]
    // NA is two double NAs
    template <>
    struct is_avail_kernel<complex_float64_type_id,
                           complex_kind> : base_kernel<is_avail_kernel<complex_float64_type_id, complex_kind>, 1> {
      void single(char *dst, char *const *src)
      {
        *dst = (*reinterpret_cast<uint64_t *const *>(src))[0] != DYND_FLOAT64_NA_AS_UINT &&
               (*reinterpret_cast<uint64_t *const *>(src))[1] != DYND_FLOAT64_NA_AS_UINT;
      }

      void strided(char *dst, intptr_t dst_stride, char *const *src, const intptr_t *src_stride, size_t count)
      {
        // Available if the value is 0 or 1
        char *src0 = src[0];
        intptr_t src0_stride = src_stride[0];
        for (size_t i = 0; i != count; ++i) {
          *dst = reinterpret_cast<uint64_t *>(src0)[0] != DYND_FLOAT64_NA_AS_UINT &&
                 reinterpret_cast<uint64_t *>(src0)[1] != DYND_FLOAT64_NA_AS_UINT;
          dst += dst_stride;
          src0 += src0_stride;
        }
      }
    };

    template <>
    struct is_avail_kernel<void_type_id, void_kind> : base_kernel<is_avail_kernel<void_type_id, void_kind>, 1> {
      void single(char *dst, char *const *DYND_UNUSED(src))
      {
        *dst = 0;
      }

      void strided(char *dst, intptr_t dst_stride, char *const *DYND_UNUSED(src),
                   const intptr_t *DYND_UNUSED(src_stride), size_t count)
      {
        // Available if the value is 0 or 1
        for (size_t i = 0; i != count; ++i) {
          *dst = 0;
          dst += dst_stride;
        }
      }
    };

    template <>
    struct is_avail_kernel<string_type_id, string_kind> : base_kernel<is_avail_kernel<string_type_id, string_kind>, 1> {
      void single(char *dst, char *const *src)
      {
        string *std = *reinterpret_cast<string *const *>(src);
        *dst = std->begin() != NULL;
      }

      void strided(char *dst, intptr_t dst_stride, char *const *src, const intptr_t *src_stride, size_t count)
      {
        char *src0 = src[0];
        intptr_t src0_stride = src_stride[0];
        for (size_t i = 0; i != count; ++i) {
          string *std = reinterpret_cast<string *>(src0);
          *dst = std->begin() != NULL;
          dst += dst_stride;
          src0 += src0_stride;
        }
      }
    };

    template <>
    struct is_avail_kernel<date_type_id, datetime_kind> : base_kernel<is_avail_kernel<date_type_id, datetime_kind>, 1> {
      void single(char *dst, char *const *src)
      {
        int32_t date = **reinterpret_cast<int32_t *const *>(src);
        *dst = date != DYND_DATE_NA;
      }

      void strided(char *dst, intptr_t dst_stride, char *const *src, const intptr_t *src_stride, size_t count)
      {
        const char *src0 = src[0];
        intptr_t src0_stride = src_stride[0];
        for (size_t i = 0; i != count; ++i) {
          int32_t date = *reinterpret_cast<const int32_t *>(src0);
          *dst = date != DYND_DATE_NA;
          dst += dst_stride;
          src0 += src0_stride;
        }
      }
    };

    template <>
    struct is_avail_kernel<time_type_id, datetime_kind> : base_kernel<is_avail_kernel<time_type_id, datetime_kind>, 1> {
      void single(char *dst, char *const *src)
      {
        int64_t v = **reinterpret_cast<int64_t *const *>(src);
        *dst = v != DYND_TIME_NA;
      }

      void strided(char *dst, intptr_t dst_stride, char *const *src, const intptr_t *src_stride, size_t count)
      {
        char *src0 = src[0];
        intptr_t src0_stride = src_stride[0];
        for (size_t i = 0; i != count; ++i) {
          int64_t v = *reinterpret_cast<int64_t *>(src0);
          *dst = v != DYND_TIME_NA;
          dst += dst_stride;
          src0 += src0_stride;
        }
      }
    };

    template <>
    struct is_avail_kernel<datetime_type_id,
                           datetime_kind> : base_kernel<is_avail_kernel<datetime_type_id, datetime_kind>, 1> {
      void single(char *dst, char *const *src)
      {
        int64_t v = **reinterpret_cast<int64_t *const *>(src);
        *dst = v != DYND_DATETIME_NA;
      }

      void strided(char *dst, intptr_t dst_stride, char *const *src, const intptr_t *src_stride, size_t count)
      {
        char *src0 = src[0];
        intptr_t src0_stride = src_stride[0];
        for (size_t i = 0; i != count; ++i) {
          int64_t v = *reinterpret_cast<int64_t *>(src0);
          *dst = v != DYND_DATETIME_NA;
          dst += dst_stride;
          src0 += src0_stride;
        }
      }
    };

    template <>
    struct is_avail_kernel<fixed_dim_type_id,
                           dim_kind> : base_virtual_kernel<is_avail_kernel<fixed_dim_type_id, dim_kind>> {
      static intptr_t
      instantiate(char *DYND_UNUSED(static_data), char *DYND_UNUSED(data), void *ckb,
                  intptr_t ckb_offset, const ndt::type &DYND_UNUSED(dst_tp), const char *DYND_UNUSED(dst_arrmeta),
                  intptr_t DYND_UNUSED(nsrc), const ndt::type *src_tp, const char *const *DYND_UNUSED(src_arrmeta),
                  kernel_request_t kernreq, const eval::eval_context *DYND_UNUSED(ectx), intptr_t DYND_UNUSED(nkwd),
                  const nd::array *DYND_UNUSED(kwds), const std::map<std::string, ndt::type> &DYND_UNUSED(tp_vars))
      {
        switch (src_tp->get_dtype().get_type_id()) {
        case bool_type_id:
          is_avail_kernel<bool_type_id, bool_kind>::make(ckb, kernreq, ckb_offset);
          return ckb_offset;
        case int8_type_id:
          is_avail_kernel<int8_type_id, sint_kind>::make(ckb, kernreq, ckb_offset);
          return ckb_offset;
        case int16_type_id:
          is_avail_kernel<int16_type_id, sint_kind>::make(ckb, kernreq, ckb_offset);
          return ckb_offset;
        case int32_type_id:
          is_avail_kernel<int32_type_id, sint_kind>::make(ckb, kernreq, ckb_offset);
          return ckb_offset;
        case int64_type_id:
          is_avail_kernel<int64_type_id, sint_kind>::make(ckb, kernreq, ckb_offset);
          return ckb_offset;
        case int128_type_id:
          is_avail_kernel<int128_type_id, sint_kind>::make(ckb, kernreq, ckb_offset);
          return ckb_offset;
        case float32_type_id:
          is_avail_kernel<float32_type_id, real_kind>::make(ckb, kernreq, ckb_offset);
          return ckb_offset;
        case float64_type_id:
          is_avail_kernel<float64_type_id, real_kind>::make(ckb, kernreq, ckb_offset);
          return ckb_offset;
        case complex_float32_type_id:
          is_avail_kernel<complex_float32_type_id, complex_kind>::make(ckb, kernreq, ckb_offset);
          return ckb_offset;
        case complex_float64_type_id:
          is_avail_kernel<complex_float64_type_id, complex_kind>::make(ckb, kernreq, ckb_offset);
          return ckb_offset;
        default:
          throw type_error("fixed_dim_is_avail: expected built-in type");
        }
      }
    };

    template <>
    struct is_avail_kernel<pointer_type_id, expr_kind> : base_kernel<is_avail_kernel<pointer_type_id, expr_kind>, 1> {
      void single(char *DYND_UNUSED(dst), char *const *DYND_UNUSED(src))
      {
        throw std::runtime_error("is_avail for pointers is not yet implemented");
      }

      void strided(char *DYND_UNUSED(dst), intptr_t DYND_UNUSED(dst_stride), char *const *DYND_UNUSED(src),
                   const intptr_t *DYND_UNUSED(src_stride), size_t DYND_UNUSED(count))
      {
        throw std::runtime_error("is_avail for pointers is not yet implemented");
      }
    };

  } // namespace dynd::nd::detail

  template <type_id_t Src0ValueTypeID>
  struct is_avail_kernel : detail::is_avail_kernel<Src0ValueTypeID, type_kind_of<Src0ValueTypeID>::value> {
  };

} // namespace dynd::nd

namespace ndt {

  template <type_id_t Src0ValueTypeID>
  struct type::equivalent<nd::is_avail_kernel<Src0ValueTypeID>> {
    static type make()
    {
      return callable_type::make(type::make<bool1>(), option_type::make(Src0ValueTypeID));
    }
  };

} // namespace dynd::ndt
} // namespace dynd
