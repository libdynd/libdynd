//
// Copyright (C) 2011-15 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#pragma once

#include <dynd/kernels/base_kernel.hpp>
#include <dynd/kernels/base_virtual_kernel.hpp>
#include <dynd/types/option_type.hpp>
#include <dynd/types/time_type.hpp>

namespace dynd {
namespace nd {
  namespace detail {

    template <type_id_t DstTypeID, type_kind_t DstTypeKind>
    struct assign_na_kernel;

    template <>
    struct assign_na_kernel<bool_type_id, bool_kind> : base_kernel<assign_na_kernel<bool_type_id, bool_kind>, 0> {
      void single(char *dst, char *const *DYND_UNUSED(src))
      {
        *dst = 2;
      }

      void strided(char *dst, intptr_t dst_stride, char *const *DYND_UNUSED(src),
                   const intptr_t *DYND_UNUSED(src_stride), size_t count)
      {
        if (dst_stride == 1) {
          memset(dst, 2, count);
        } else {
          for (size_t i = 0; i != count; ++i, dst += dst_stride) {
            *dst = 2;
          }
        }
      }
    };

    template <type_id_t DstTypeID>
    struct assign_na_kernel<DstTypeID, sint_kind> : base_kernel<assign_na_kernel<DstTypeID, sint_kind>, 0> {
      typedef typename type_of<DstTypeID>::type dst_type;

      void single(char *dst, char *const *DYND_UNUSED(src))
      {
        *reinterpret_cast<dst_type *>(dst) = std::numeric_limits<dst_type>::min();
      }

      void strided(char *dst, intptr_t dst_stride, char *const *DYND_UNUSED(src),
                   const intptr_t *DYND_UNUSED(src_stride), size_t count)
      {
        for (size_t i = 0; i != count; ++i, dst += dst_stride) {
          *reinterpret_cast<dst_type *>(dst) = std::numeric_limits<dst_type>::min();
        }
      }
    };

    template <>
    struct assign_na_kernel<float32_type_id, real_kind> : base_kernel<assign_na_kernel<float32_type_id, real_kind>, 0> {
      void single(char *dst, char *const *DYND_UNUSED(src))
      {
        *reinterpret_cast<uint32_t *>(dst) = DYND_FLOAT32_NA_AS_UINT;
      }

      void strided(char *dst, intptr_t dst_stride, char *const *DYND_UNUSED(src),
                   const intptr_t *DYND_UNUSED(src_stride), size_t count)
      {
        for (size_t i = 0; i != count; ++i, dst += dst_stride) {
          *reinterpret_cast<uint32_t *>(dst) = DYND_FLOAT32_NA_AS_UINT;
        }
      }
    };

    template <>
    struct assign_na_kernel<float64_type_id, real_kind> : base_kernel<assign_na_kernel<float64_type_id, real_kind>, 0> {
      void single(char *dst, char *const *DYND_UNUSED(src))
      {
        *reinterpret_cast<uint64_t *>(dst) = DYND_FLOAT64_NA_AS_UINT;
      }

      void strided(char *dst, intptr_t dst_stride, char *const *DYND_UNUSED(src),
                   const intptr_t *DYND_UNUSED(src_stride), size_t count)
      {
        for (size_t i = 0; i != count; ++i, dst += dst_stride) {
          *reinterpret_cast<uint64_t *>(dst) = DYND_FLOAT64_NA_AS_UINT;
        }
      }
    };

    template <>
    struct assign_na_kernel<complex_float32_type_id,
                            complex_kind> : base_kernel<assign_na_kernel<complex_float32_type_id, complex_kind>, 0> {
      void single(char *dst, char *const *DYND_UNUSED(src))
      {
        reinterpret_cast<uint32_t *>(dst)[0] = DYND_FLOAT32_NA_AS_UINT;
        reinterpret_cast<uint32_t *>(dst)[1] = DYND_FLOAT32_NA_AS_UINT;
      }

      void strided(char *dst, intptr_t dst_stride, char *const *DYND_UNUSED(src),
                   const intptr_t *DYND_UNUSED(src_stride), size_t count)
      {
        for (size_t i = 0; i != count; ++i, dst += dst_stride) {
          reinterpret_cast<uint32_t *>(dst)[0] = DYND_FLOAT32_NA_AS_UINT;
          reinterpret_cast<uint32_t *>(dst)[1] = DYND_FLOAT32_NA_AS_UINT;
        }
      }
    };

    template <>
    struct assign_na_kernel<complex_float64_type_id,
                            complex_kind> : base_kernel<assign_na_kernel<complex_float64_type_id, complex_kind>, 0> {
      void single(char *dst, char *const *DYND_UNUSED(src))
      {
        reinterpret_cast<uint64_t *>(dst)[0] = DYND_FLOAT64_NA_AS_UINT;
        reinterpret_cast<uint64_t *>(dst)[1] = DYND_FLOAT64_NA_AS_UINT;
      }

      void strided(char *dst, intptr_t dst_stride, char *const *DYND_UNUSED(src),
                   const intptr_t *DYND_UNUSED(src_stride), size_t count)
      {
        for (size_t i = 0; i != count; ++i, dst += dst_stride) {
          reinterpret_cast<uint64_t *>(dst)[0] = DYND_FLOAT64_NA_AS_UINT;
          reinterpret_cast<uint64_t *>(dst)[1] = DYND_FLOAT64_NA_AS_UINT;
        }
      }
    };

    template <>
    struct assign_na_kernel<void_type_id, void_kind> : base_kernel<assign_na_kernel<void_type_id, void_kind>, 0> {
      void single(char *DYND_UNUSED(dst), char *const *DYND_UNUSED(src))
      {
      }

      void strided(char *DYND_UNUSED(dst), intptr_t DYND_UNUSED(dst_stride), char *const *DYND_UNUSED(src),
                   const intptr_t *DYND_UNUSED(src_stride), size_t DYND_UNUSED(count))
      {
      }
    };

    template <>
    struct assign_na_kernel<fixed_dim_type_id,
                            dim_kind> : base_virtual_kernel<assign_na_kernel<fixed_dim_type_id, dim_kind>> {
      static intptr_t instantiate(char *DYND_UNUSED(static_data), char *DYND_UNUSED(data), void *ckb,
                                  intptr_t ckb_offset, const ndt::type &dst_tp, const char *DYND_UNUSED(dst_arrmeta),
                                  intptr_t DYND_UNUSED(nsrc), const ndt::type *DYND_UNUSED(src_tp),
                                  const char *const *DYND_UNUSED(src_arrmeta), kernel_request_t kernreq,
                                  const eval::eval_context *DYND_UNUSED(ectx), intptr_t DYND_UNUSED(nkwd),
                                  const nd::array *DYND_UNUSED(kwds),
                                  const std::map<std::string, ndt::type> &DYND_UNUSED(tp_vars))
      {
        switch (dst_tp.get_dtype().get_type_id()) {
        case bool_type_id:
          assign_na_kernel<bool_type_id, bool_kind>::make(ckb, kernreq, ckb_offset);
          return ckb_offset;
        case int8_type_id:
          assign_na_kernel<int8_type_id, sint_kind>::make(ckb, kernreq, ckb_offset);
          return ckb_offset;
        case int16_type_id:
          assign_na_kernel<int16_type_id, sint_kind>::make(ckb, kernreq, ckb_offset);
          return ckb_offset;
        case int32_type_id:
          assign_na_kernel<int32_type_id, sint_kind>::make(ckb, kernreq, ckb_offset);
          return ckb_offset;
        case int64_type_id:
          assign_na_kernel<int64_type_id, sint_kind>::make(ckb, kernreq, ckb_offset);
          return ckb_offset;
        case int128_type_id:
          assign_na_kernel<int128_type_id, sint_kind>::make(ckb, kernreq, ckb_offset);
          return ckb_offset;
        case float32_type_id:
          assign_na_kernel<float32_type_id, real_kind>::make(ckb, kernreq, ckb_offset);
          return ckb_offset;
        case float64_type_id:
          assign_na_kernel<float64_type_id, real_kind>::make(ckb, kernreq, ckb_offset);
          return ckb_offset;
        case complex_float32_type_id:
          assign_na_kernel<complex_float32_type_id, complex_kind>::make(ckb, kernreq, ckb_offset);
          return ckb_offset;
        case complex_float64_type_id:
          assign_na_kernel<complex_float64_type_id, complex_kind>::make(ckb, kernreq, ckb_offset);
          return ckb_offset;
        default:
          throw type_error("fixed_dim_assign_na: expected built-in type");
        }
      }
    };

    template <>
    struct assign_na_kernel<pointer_type_id, expr_kind> : base_kernel<assign_na_kernel<pointer_type_id, expr_kind>, 0> {
      void single(char *DYND_UNUSED(dst), char *const *DYND_UNUSED(src))
      {
        throw std::runtime_error("assign_na for pointers is not yet implemented");
      }

      void strided(char *DYND_UNUSED(dst), intptr_t DYND_UNUSED(dst_stride), char *const *DYND_UNUSED(src),
                   const intptr_t *DYND_UNUSED(src_stride), size_t DYND_UNUSED(count))
      {
        throw std::runtime_error("assign_na for pointers is not yet implemented");
      }
    };

    template <>
    struct assign_na_kernel<string_type_id, string_kind> : base_kernel<assign_na_kernel<string_type_id, string_kind>,
                                                                       1> {
      void single(char *dst, char *const *DYND_UNUSED(src))
      {
        string *std = reinterpret_cast<string *>(dst);
        if (std->begin() != NULL) {
          throw std::invalid_argument("Cannot assign an NA to a dynd string after "
                                      "it has been allocated");
        }
      }

      void strided(char *dst, intptr_t dst_stride, char *const *DYND_UNUSED(src),
                   const intptr_t *DYND_UNUSED(src_stride), size_t count)
      {
        for (size_t i = 0; i != count; ++i, dst += dst_stride) {
          string *std = reinterpret_cast<string *>(dst);
          if (std->begin() != NULL) {
            throw std::invalid_argument("Cannot assign an NA to a dynd string after "
                                        "it has been allocated");
          }
        }
      }
    };

    template <>
    struct assign_na_kernel<date_type_id, datetime_kind> : base_kernel<assign_na_kernel<date_type_id, datetime_kind>,
                                                                       1> {
      void single(char *dst, char *const *DYND_UNUSED(src))
      {
        *reinterpret_cast<int32_t *>(dst) = DYND_DATE_NA;
      }

      void strided(char *dst, intptr_t dst_stride, char *const *DYND_UNUSED(src),
                   const intptr_t *DYND_UNUSED(src_stride), size_t count)
      {
        for (size_t i = 0; i != count; ++i, dst += dst_stride) {
          *reinterpret_cast<int32_t *>(dst) = DYND_DATE_NA;
        }
      }
    };

    template <>
    struct assign_na_kernel<time_type_id, datetime_kind> : base_kernel<assign_na_kernel<time_type_id, datetime_kind>,
                                                                       1> {
      void single(char *dst, char *const *DYND_UNUSED(src))
      {
        *reinterpret_cast<int64_t *>(dst) = DYND_TIME_NA;
      }

      void strided(char *dst, intptr_t dst_stride, char *const *DYND_UNUSED(src),
                   const intptr_t *DYND_UNUSED(src_stride), size_t count)
      {
        for (size_t i = 0; i != count; ++i, dst += dst_stride) {
          *reinterpret_cast<int64_t *>(dst) = DYND_TIME_NA;
        }
      }
    };

    template <>
    struct assign_na_kernel<datetime_type_id,
                            datetime_kind> : base_kernel<assign_na_kernel<datetime_type_id, datetime_kind>, 1> {
      void single(char *dst, char *const *DYND_UNUSED(src))
      {
        *reinterpret_cast<int64_t *>(dst) = DYND_DATETIME_NA;
      }

      void strided(char *dst, intptr_t dst_stride, char *const *DYND_UNUSED(src),
                   const intptr_t *DYND_UNUSED(src_stride), size_t count)
      {
        for (size_t i = 0; i != count; ++i, dst += dst_stride) {
          *reinterpret_cast<int64_t *>(dst) = DYND_DATETIME_NA;
        }
      }
    };

  } // namespace dynd::nd::detail

  template <type_id_t DstTypeID>
  struct assign_na_kernel : detail::assign_na_kernel<DstTypeID, type_kind_of<DstTypeID>::value> {
  };

} // namespace dynd::nd

namespace ndt {

  template <type_id_t Src0ValueTypeID>
  struct type::equivalent<nd::assign_na_kernel<Src0ValueTypeID>> {
    static type make()
    {
      return callable_type::make(option_type::make(Src0ValueTypeID));
    }
  };

} // namespace dynd::ndt
} // namespace dynd
