//
// Copyright (C) 2011-16 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#pragma once

#include <dynd/kernels/base_kernel.hpp>
#include <dynd/parse.hpp>
#include <dynd/types/option_type.hpp>

namespace dynd {
namespace nd {
  namespace detail {

    template <typename ReturnValueType, typename ReturnValueBaseType>
    struct assign_na_kernel;

    template <>
    struct assign_na_kernel<bool, ndt::bool_kind_type>
        : base_strided_kernel<assign_na_kernel<bool, ndt::bool_kind_type>, 0> {
      void single(char *dst, char *const *DYND_UNUSED(src)) { *dst = 2; }

      void strided(char *dst, intptr_t dst_stride, char *const *DYND_UNUSED(src),
                   const intptr_t *DYND_UNUSED(src_stride), size_t count) {
        if (dst_stride == 1) {
          memset(dst, 2, count);
        } else {
          for (size_t i = 0; i != count; ++i, dst += dst_stride) {
            *dst = 2;
          }
        }
      }
    };

    template <typename ReturnValueType>
    struct assign_na_kernel<ReturnValueType, ndt::int_kind_type>
        : base_strided_kernel<assign_na_kernel<ReturnValueType, ndt::int_kind_type>, 0> {
      void single(char *dst, char *const *DYND_UNUSED(src)) {
        *reinterpret_cast<ReturnValueType *>(dst) = std::numeric_limits<ReturnValueType>::min();
      }

      void strided(char *dst, intptr_t dst_stride, char *const *DYND_UNUSED(src),
                   const intptr_t *DYND_UNUSED(src_stride), size_t count) {
        for (size_t i = 0; i != count; ++i, dst += dst_stride) {
          *reinterpret_cast<ReturnValueType *>(dst) = std::numeric_limits<ReturnValueType>::min();
        }
      }
    };

    template <typename ReturnValueType>
    struct assign_na_kernel<ReturnValueType, ndt::uint_kind_type>
        : base_strided_kernel<assign_na_kernel<ReturnValueType, ndt::uint_kind_type>, 0> {
      void single(char *dst, char *const *DYND_UNUSED(src)) {
        *reinterpret_cast<ReturnValueType *>(dst) = std::numeric_limits<ReturnValueType>::max();
      }

      void strided(char *dst, intptr_t dst_stride, char *const *DYND_UNUSED(src),
                   const intptr_t *DYND_UNUSED(src_stride), size_t count) {
        for (size_t i = 0; i != count; ++i, dst += dst_stride) {
          *reinterpret_cast<ReturnValueType *>(dst) = std::numeric_limits<ReturnValueType>::max();
        }
      }
    };

    template <>
    struct assign_na_kernel<float, ndt::float_kind_type>
        : base_strided_kernel<assign_na_kernel<float, ndt::float_kind_type>, 0> {
      void single(char *dst, char *const *DYND_UNUSED(src)) {
        *reinterpret_cast<uint32_t *>(dst) = DYND_FLOAT32_NA_AS_UINT;
      }

      void strided(char *dst, intptr_t dst_stride, char *const *DYND_UNUSED(src),
                   const intptr_t *DYND_UNUSED(src_stride), size_t count) {
        for (size_t i = 0; i != count; ++i, dst += dst_stride) {
          *reinterpret_cast<uint32_t *>(dst) = DYND_FLOAT32_NA_AS_UINT;
        }
      }
    };

    template <>
    struct assign_na_kernel<double, ndt::float_kind_type>
        : base_strided_kernel<assign_na_kernel<double, ndt::float_kind_type>, 0> {
      void single(char *dst, char *const *DYND_UNUSED(src)) {
        *reinterpret_cast<uint64_t *>(dst) = DYND_FLOAT64_NA_AS_UINT;
      }

      void strided(char *dst, intptr_t dst_stride, char *const *DYND_UNUSED(src),
                   const intptr_t *DYND_UNUSED(src_stride), size_t count) {
        for (size_t i = 0; i != count; ++i, dst += dst_stride) {
          *reinterpret_cast<uint64_t *>(dst) = DYND_FLOAT64_NA_AS_UINT;
        }
      }
    };

    template <>
    struct assign_na_kernel<complex<float>, ndt::complex_kind_type>
        : base_strided_kernel<assign_na_kernel<complex<float>, ndt::complex_kind_type>, 0> {
      void single(char *dst, char *const *DYND_UNUSED(src)) {
        reinterpret_cast<uint32_t *>(dst)[0] = DYND_FLOAT32_NA_AS_UINT;
        reinterpret_cast<uint32_t *>(dst)[1] = DYND_FLOAT32_NA_AS_UINT;
      }

      void strided(char *dst, intptr_t dst_stride, char *const *DYND_UNUSED(src),
                   const intptr_t *DYND_UNUSED(src_stride), size_t count) {
        for (size_t i = 0; i != count; ++i, dst += dst_stride) {
          reinterpret_cast<uint32_t *>(dst)[0] = DYND_FLOAT32_NA_AS_UINT;
          reinterpret_cast<uint32_t *>(dst)[1] = DYND_FLOAT32_NA_AS_UINT;
        }
      }
    };

    template <>
    struct assign_na_kernel<complex<double>, ndt::complex_kind_type>
        : base_strided_kernel<assign_na_kernel<complex<double>, ndt::complex_kind_type>, 0> {
      void single(char *dst, char *const *DYND_UNUSED(src)) {
        reinterpret_cast<uint64_t *>(dst)[0] = DYND_FLOAT64_NA_AS_UINT;
        reinterpret_cast<uint64_t *>(dst)[1] = DYND_FLOAT64_NA_AS_UINT;
      }

      void strided(char *dst, intptr_t dst_stride, char *const *DYND_UNUSED(src),
                   const intptr_t *DYND_UNUSED(src_stride), size_t count) {
        for (size_t i = 0; i != count; ++i, dst += dst_stride) {
          reinterpret_cast<uint64_t *>(dst)[0] = DYND_FLOAT64_NA_AS_UINT;
          reinterpret_cast<uint64_t *>(dst)[1] = DYND_FLOAT64_NA_AS_UINT;
        }
      }
    };

    template <>
    struct assign_na_kernel<void, ndt::any_kind_type>
        : base_strided_kernel<assign_na_kernel<void, ndt::any_kind_type>, 0> {
      void single(char *DYND_UNUSED(dst), char *const *DYND_UNUSED(src)) {}
    };

    template <>
    struct assign_na_kernel<ndt::pointer_type, ndt::any_kind_type>
        : base_strided_kernel<assign_na_kernel<ndt::pointer_type, ndt::any_kind_type>, 0> {
      void single(char *DYND_UNUSED(dst), char *const *DYND_UNUSED(src)) {
        throw std::runtime_error("assign_na for pointers is not yet implemented");
      }
    };

    template <>
    struct assign_na_kernel<bytes, ndt::bytes_kind_type>
        : base_strided_kernel<assign_na_kernel<bytes, ndt::bytes_kind_type>, 1> {
      void single(char *res, char *const *DYND_UNUSED(args)) { reinterpret_cast<bytes *>(res)->clear(); }
    };

    template <>
    struct assign_na_kernel<string, ndt::string_kind_type>
        : base_strided_kernel<assign_na_kernel<string, ndt::string_kind_type>, 1> {
      void single(char *res, char *const *DYND_UNUSED(args)) { reinterpret_cast<bytes *>(res)->clear(); }
    };

  } // namespace dynd::nd::detail

  template <typename ReturnValueType>
  using assign_na_kernel = detail::assign_na_kernel<ReturnValueType, ndt::base_of_t<ReturnValueType>>;

} // namespace dynd::nd
} // namespace dynd
