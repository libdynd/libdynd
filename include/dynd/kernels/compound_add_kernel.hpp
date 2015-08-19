#pragma once

#include <dynd/kernels/base_kernel.hpp>

namespace dynd {
namespace nd {
  namespace detail {

    template <type_id_t DstTypeID, type_id_t Src0TypeID,
              bool WithBinaryOperator>
    struct compound_add_kernel;

    template <type_id_t DstTypeID, type_id_t Src0TypeID>
    struct compound_add_kernel<
        DstTypeID, Src0TypeID,
        false> : base_kernel<compound_add_kernel<DstTypeID, Src0TypeID, false>,
                             kernel_request_host, 1> {
      typedef typename type_of<DstTypeID>::type dst_type;
      typedef typename type_of<Src0TypeID>::type src0_type;

      static const std::size_t data_size = 0;

      void single(char *dst, char *const *src)
      {
        *reinterpret_cast<dst_type *>(dst) +=
            *reinterpret_cast<src0_type *>(src[0]);
      }

      void strided(char *dst, std::intptr_t dst_stride, char *const *src,
                   const std::intptr_t *src_stride, std::size_t count)
      {
        char *src0 = src[0];
        std::intptr_t src0_stride = src_stride[0];
        for (std::size_t i = 0; i < count; ++i) {
          *reinterpret_cast<dst_type *>(dst) +=
              *reinterpret_cast<src0_type *>(src0);
          dst += dst_stride;
          src0 += src0_stride;
        }
      }
    };

    template <type_id_t DstTypeID, type_id_t Src0TypeID>
    struct compound_add_kernel<
        DstTypeID, Src0TypeID,
        true> : base_kernel<compound_add_kernel<DstTypeID, Src0TypeID, true>,
                            kernel_request_host, 1> {
      typedef typename type_of<DstTypeID>::type dst_type;
      typedef typename type_of<Src0TypeID>::type src0_type;

      static const std::size_t data_size = 0;

      void single(char *dst, char *const *src)
      {
        *reinterpret_cast<dst_type *>(dst) =
            static_cast<dst_type>(*reinterpret_cast<dst_type *>(dst) +
                                  *reinterpret_cast<src0_type *>(src[0]));
      }
    };

  } // namespace dynd::nd::detail

  template <type_id_t DstTypeID, type_id_t Src0TypeID>
  struct is_precise_assignable {
    static const bool value =
        (type_kind_of<DstTypeID>::value == sint_kind &&
         type_kind_of<Src0TypeID>::value == real_kind) ||
        (DstTypeID == float32_type_id && Src0TypeID == float64_type_id);
  };

  template <type_id_t DstTypeID, type_id_t Src0TypeID>
  struct compound_add_kernel
      : detail::compound_add_kernel<
            DstTypeID, Src0TypeID,
            is_precise_assignable<DstTypeID, Src0TypeID>::value> {
  };

} // namespace dynd::nd

namespace ndt {

  template <type_id_t DstTypeID, type_id_t Src0TypeID>
  struct type::equivalent<nd::compound_add_kernel<DstTypeID, Src0TypeID>> {
    static type make()
    {
      return callable_type::make(type(DstTypeID), type(Src0TypeID));
    }
  };

} // namespace dynd::ndt
} // namespace dynd
