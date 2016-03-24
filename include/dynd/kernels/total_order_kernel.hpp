//
// Copyright (C) 2011-15 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#pragma once

#include <dynd/kernels/base_kernel.hpp>
#include <dynd/types/fixed_string_type.hpp>

namespace dynd {
namespace nd {
  namespace detail {

    template <type_id_t Src0TypeID, type_id_t Src0BaseID, type_id_t Src1TypeID, type_id_t Src1BaseID>
    struct total_order_kernel;

    template <>
    struct total_order_kernel<bool_id, bool_kind_id, bool_id, bool_kind_id>
        : base_strided_kernel<total_order_kernel<bool_id, bool_kind_id, bool_id, bool_kind_id>, 2> {
      void single(char *dst, char *const *src)
      {
        *reinterpret_cast<int *>(dst) =
            static_cast<int>(*reinterpret_cast<bool1 *>(src[0])) < static_cast<int>(*reinterpret_cast<bool1 *>(src[1]));
      }
    };

    template <>
    struct total_order_kernel<int32_id, int_kind_id, int32_id, int_kind_id>
        : base_strided_kernel<total_order_kernel<int32_id, int_kind_id, int32_id, int_kind_id>, 2> {
      void single(char *dst, char *const *src)
      {
        *reinterpret_cast<int *>(dst) = *reinterpret_cast<int *>(src[0]) < *reinterpret_cast<int *>(src[1]);
      }
    };

    template <>
    struct total_order_kernel<fixed_string_id, string_kind_id, fixed_string_id, string_kind_id>
        : base_strided_kernel<total_order_kernel<fixed_string_id, string_kind_id, fixed_string_id, string_kind_id>, 2> {
      size_t size;

      total_order_kernel(size_t size) : size(size) {}

      void single(char *dst, char *const *src) { *reinterpret_cast<int *>(dst) = strncmp(src[0], src[1], size) < 0; }

    };

    template <>
    struct total_order_kernel<string_id, string_kind_id, string_id, string_kind_id>
        : base_strided_kernel<total_order_kernel<string_id, string_kind_id, string_id, string_kind_id>, 2> {
      void single(char *dst, char *const *src)
      {
        *reinterpret_cast<int *>(dst) = std::lexicographical_compare(
            reinterpret_cast<string *>(src[0])->begin(), reinterpret_cast<string *>(src[0])->end(),
            reinterpret_cast<string *>(src[1])->begin(), reinterpret_cast<string *>(src[1])->end());
      }
    };

  } // namespace dynd::nd::detail

  template <type_id_t Src0TypeID, type_id_t Src1TypeID>
  struct total_order_kernel : detail::total_order_kernel<Src0TypeID, base_id_of<Src0TypeID>::value, Src1TypeID,
                                                         base_id_of<Src1TypeID>::value> {
  };

} // namespace dynd::nd
} // namespace dynd
