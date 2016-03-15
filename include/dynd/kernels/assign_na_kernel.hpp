//
// Copyright (C) 2011-15 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#pragma once

#include <dynd/parse.hpp>
#include <dynd/kernels/base_kernel.hpp>
#include <dynd/types/option_type.hpp>

namespace dynd {
namespace nd {
  namespace detail {

    template <type_id_t DstTypeID, type_id_t DstBaseTypeID>
    struct assign_na_kernel;

    template <>
    struct assign_na_kernel<bool_id, bool_kind_id> : base_strided_kernel<assign_na_kernel<bool_id, bool_kind_id>, 0> {
      void single(char *dst, char *const *DYND_UNUSED(src)) { *dst = 2; }

      void strided(char *dst, intptr_t dst_stride, char *const *DYND_UNUSED(src),
                   const intptr_t *DYND_UNUSED(src_stride), size_t count)
      {
        if (dst_stride == 1) {
          memset(dst, 2, count);
        }
        else {
          for (size_t i = 0; i != count; ++i, dst += dst_stride) {
            *dst = 2;
          }
        }
      }
    };

    template <type_id_t RetTypeID>
    struct assign_na_kernel<RetTypeID, int_kind_id> : base_strided_kernel<assign_na_kernel<RetTypeID, int_kind_id>, 0> {
      typedef typename type_of<RetTypeID>::type ret_type;

      void single(char *dst, char *const *DYND_UNUSED(src))
      {
        *reinterpret_cast<ret_type *>(dst) = std::numeric_limits<ret_type>::min();
      }

      void strided(char *dst, intptr_t dst_stride, char *const *DYND_UNUSED(src),
                   const intptr_t *DYND_UNUSED(src_stride), size_t count)
      {
        for (size_t i = 0; i != count; ++i, dst += dst_stride) {
          *reinterpret_cast<ret_type *>(dst) = std::numeric_limits<ret_type>::min();
        }
      }
    };

    template <type_id_t DstTypeID>
    struct assign_na_kernel<DstTypeID, uint_kind_id>
        : base_strided_kernel<assign_na_kernel<DstTypeID, uint_kind_id>, 0> {
      typedef typename type_of<DstTypeID>::type dst_type;

      void single(char *dst, char *const *DYND_UNUSED(src))
      {
        *reinterpret_cast<dst_type *>(dst) = std::numeric_limits<dst_type>::max();
      }

      void strided(char *dst, intptr_t dst_stride, char *const *DYND_UNUSED(src),
                   const intptr_t *DYND_UNUSED(src_stride), size_t count)
      {
        for (size_t i = 0; i != count; ++i, dst += dst_stride) {
          *reinterpret_cast<dst_type *>(dst) = std::numeric_limits<dst_type>::max();
        }
      }
    };

    template <>
    struct assign_na_kernel<float32_id, float_kind_id>
        : base_strided_kernel<assign_na_kernel<float32_id, float_kind_id>, 0> {
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
    struct assign_na_kernel<float64_id, float_kind_id>
        : base_strided_kernel<assign_na_kernel<float64_id, float_kind_id>, 0> {
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
    struct assign_na_kernel<complex_float32_id, complex_kind_id>
        : base_strided_kernel<assign_na_kernel<complex_float32_id, complex_kind_id>, 0> {
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
    struct assign_na_kernel<complex_float64_id, complex_kind_id>
        : base_strided_kernel<assign_na_kernel<complex_float64_id, complex_kind_id>, 0> {
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
    struct assign_na_kernel<void_id, any_kind_id> : base_strided_kernel<assign_na_kernel<void_id, any_kind_id>, 0> {
      void single(char *DYND_UNUSED(dst), char *const *DYND_UNUSED(src)) {}

      void strided(char *DYND_UNUSED(dst), intptr_t DYND_UNUSED(dst_stride), char *const *DYND_UNUSED(src),
                   const intptr_t *DYND_UNUSED(src_stride), size_t DYND_UNUSED(count))
      {
      }
    };

    template <>
    struct assign_na_kernel<fixed_dim_id, fixed_dim_kind_id>
        : base_kernel<assign_na_kernel<fixed_dim_id, fixed_dim_kind_id>> {
      static void instantiate(char *DYND_UNUSED(static_data), char *DYND_UNUSED(data), kernel_builder *ckb,
                              const ndt::type &dst_tp, const char *DYND_UNUSED(dst_arrmeta), intptr_t DYND_UNUSED(nsrc),
                              const ndt::type *DYND_UNUSED(src_tp), const char *const *DYND_UNUSED(src_arrmeta),
                              kernel_request_t kernreq, intptr_t DYND_UNUSED(nkwd), const nd::array *DYND_UNUSED(kwds),
                              const std::map<std::string, ndt::type> &DYND_UNUSED(tp_vars))
      {
        switch (dst_tp.get_dtype().get_id()) {
        case bool_id:
          ckb->emplace_back<assign_na_kernel<bool_id, bool_kind_id>>(kernreq);
          break;
        case int8_id:
          ckb->emplace_back<assign_na_kernel<int8_id, int_kind_id>>(kernreq);
          break;
        case int16_id:
          ckb->emplace_back<assign_na_kernel<int16_id, int_kind_id>>(kernreq);
          break;
        case int32_id:
          ckb->emplace_back<assign_na_kernel<int32_id, int_kind_id>>(kernreq);
          break;
        case int64_id:
          ckb->emplace_back<assign_na_kernel<int64_id, int_kind_id>>(kernreq);
          break;
        case int128_id:
          ckb->emplace_back<assign_na_kernel<int128_id, int_kind_id>>(kernreq);
          break;
        case float32_id:
          ckb->emplace_back<assign_na_kernel<float32_id, float_kind_id>>(kernreq);
          break;
        case float64_id:
          ckb->emplace_back<assign_na_kernel<float64_id, float_kind_id>>(kernreq);
          break;
        case complex_float32_id:
          ckb->emplace_back<assign_na_kernel<complex_float32_id, complex_kind_id>>(kernreq);
          break;
        case complex_float64_id:
          ckb->emplace_back<assign_na_kernel<complex_float64_id, complex_kind_id>>(kernreq);
          break;
        default:
          throw type_error("fixed_dim_assign_na: expected built-in type");
        }
      }
    };

    template <>
    struct assign_na_kernel<pointer_id, any_kind_id>
        : base_strided_kernel<assign_na_kernel<pointer_id, any_kind_id>, 0> {
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
    struct assign_na_kernel<bytes_id, bytes_kind_id>
        : base_strided_kernel<assign_na_kernel<bytes_id, bytes_kind_id>, 1> {
      void single(char *res, char *const *DYND_UNUSED(args)) { reinterpret_cast<bytes *>(res)->clear(); }
    };

    template <>
    struct assign_na_kernel<string_id, string_kind_id>
        : base_strided_kernel<assign_na_kernel<string_id, string_kind_id>, 1> {
      void single(char *res, char *const *DYND_UNUSED(args)) { reinterpret_cast<bytes *>(res)->clear(); }
    };

  } // namespace dynd::nd::detail

  template <type_id_t DstTypeID>
  struct assign_na_kernel : detail::assign_na_kernel<DstTypeID, base_id_of<DstTypeID>::value> {
  };

} // namespace dynd::nd

namespace ndt {

  template <type_id_t Src0ValueTypeID>
  struct traits<nd::assign_na_kernel<Src0ValueTypeID>> {
    static type equivalent() { return callable_type::make(make_type<option_type>(Src0ValueTypeID)); }
  };

} // namespace dynd::ndt
} // namespace dynd
