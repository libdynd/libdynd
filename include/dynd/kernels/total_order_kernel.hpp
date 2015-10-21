//
// Copyright (C) 2011-15 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#pragma once

#include <dynd/kernels/base_kernel.hpp>

namespace dynd {
namespace nd {
  namespace detail {

    template <type_id_t Src0TypeID, type_kind_t Src0TypeKind, type_id_t Src1TypeID, type_kind_t Src1TypeKind>
    struct total_order_kernel : base_kernel<total_order_kernel<Src0TypeID, Src0TypeKind, Src1TypeID, Src1TypeKind>, 2> {
      static const size_t data_size = 0;

      void single(char *DYND_UNUSED(dst), char *const *DYND_UNUSED(src))
      {
      }
    };

    template <type_id_t Src0TypeID, type_id_t Src1TypeID>
    struct total_order_kernel<Src0TypeID, bool_kind, Src1TypeID,
                              bool_kind> : base_kernel<total_order_kernel<Src0TypeID, bool_kind, Src1TypeID, bool_kind>,
                                                       2> {
      static const size_t data_size = 0;

      void single(char *dst, char *const *src)
      {
        *reinterpret_cast<bool1 *>(dst) =
            static_cast<int>(*reinterpret_cast<bool1 *>(src[0])) < static_cast<int>(*reinterpret_cast<bool1 *>(src[1]));
      }
    };

  } // namespace dynd::nd::detail

  template <type_id_t Src0TypeID, type_id_t Src1TypeID>
  struct total_order_kernel : detail::total_order_kernel<Src0TypeID, type_kind_of<Src0TypeID>::value, Src1TypeID,
                                                         type_kind_of<Src1TypeID>::value> {
  };

} // namespace dynd::nd

namespace ndt {

  template <type_id_t Src0TypeID, type_id_t Src1TypeID>
  struct type::equivalent<nd::total_order_kernel<Src0TypeID, Src1TypeID>> {
    static type make()
    {
      return callable_type::make(type::make<bool>(), {type(Src0TypeID), type(Src1TypeID)});
    }
  };

} // namespace dynd::ndt
} // namespace dynd
