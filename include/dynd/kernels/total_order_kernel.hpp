//
// Copyright (C) 2011-15 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#pragma once

#include <dynd/kernels/base_kernel.hpp>
#include <dynd/kernels/base_virtual_kernel.hpp>
#include <dynd/types/fixed_string_type.hpp>

namespace dynd {
namespace nd {
  namespace detail {

    template <type_id_t Src0TypeID, type_kind_t Src0TypeKind, type_id_t Src1TypeID, type_kind_t Src1TypeKind>
    struct total_order_kernel;

    template <>
    struct total_order_kernel<
        bool_type_id, bool_kind, bool_type_id,
        bool_kind> : base_kernel<total_order_kernel<bool_type_id, bool_kind, bool_type_id, bool_kind>, 2> {
      static const size_t data_size = 0;

      void single(char *dst, char *const *src)
      {
        *reinterpret_cast<int *>(dst) =
            static_cast<int>(*reinterpret_cast<bool1 *>(src[0])) < static_cast<int>(*reinterpret_cast<bool1 *>(src[1]));
      }
    };

    template <>
    struct total_order_kernel<
        int32_type_id, sint_kind, int32_type_id,
        sint_kind> : base_kernel<total_order_kernel<int32_type_id, sint_kind, int32_type_id, sint_kind>, 2> {
      static const size_t data_size = 0;

      void single(char *dst, char *const *src)
      {
        *reinterpret_cast<int *>(dst) = *reinterpret_cast<int *>(src[0]) < *reinterpret_cast<int *>(src[1]);
      }
    };

    template <>
    struct total_order_kernel<fixed_string_type_id, string_kind, fixed_string_type_id,
                              string_kind> : base_kernel<total_order_kernel<fixed_string_type_id, string_kind,
                                                                            fixed_string_type_id, string_kind>,
                                                         2> {
      static const size_t data_size = 0;

      size_t size;

      total_order_kernel(size_t size) : size(size)
      {
      }

      void single(char *dst, char *const *src)
      {
        *reinterpret_cast<int *>(dst) = strncmp(src[0], src[1], size) < 0;
      }

      static intptr_t instantiate(char *DYND_UNUSED(static_data), char *DYND_UNUSED(data), void *ckb,
                                  intptr_t ckb_offset, const ndt::type &DYND_UNUSED(dst_tp),
                                  const char *DYND_UNUSED(dst_arrmeta), intptr_t DYND_UNUSED(nsrc),
                                  const ndt::type *src_tp, const char *const *DYND_UNUSED(src_arrmeta),
                                  kernel_request_t kernreq, const eval::eval_context *DYND_UNUSED(ectx),
                                  intptr_t DYND_UNUSED(nkwd), const nd::array *DYND_UNUSED(kwds),
                                  const std::map<std::string, ndt::type> &DYND_UNUSED(tp_vars))
      {
        total_order_kernel::make(ckb, kernreq, ckb_offset, src_tp[0].extended<ndt::fixed_string_type>()->get_size());
        return ckb_offset;
      }
    };

    template <>
    struct total_order_kernel<
        string_type_id, string_kind, string_type_id,
        string_kind> : base_kernel<total_order_kernel<string_type_id, string_kind, string_type_id, string_kind>, 2> {
      static const size_t data_size = 0;

      void single(char *dst, char *const *src)
      {
        *reinterpret_cast<int *>(dst) = std::lexicographical_compare(
            reinterpret_cast<string *>(src[0])->begin(), reinterpret_cast<string *>(src[0])->end(),
            reinterpret_cast<string *>(src[1])->begin(), reinterpret_cast<string *>(src[1])->end());
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
      return callable_type::make(type::make<int>(), {type(Src0TypeID), type(Src1TypeID)});
    }
  };

} // namespace dynd::ndt
} // namespace dynd
