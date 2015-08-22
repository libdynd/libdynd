//
// Copyright (C) 2011-15 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#pragma once

#include <dynd/kernels/base_kernel.hpp>

namespace dynd {
namespace nd {

  template <typename DstType, typename Src0Type, bool UseBinaryOperator>
  struct compound_div_kernel;

  template <typename DstType, typename Src0Type>
  struct compound_div_kernel<
      DstType, Src0Type,
      false> : base_kernel<compound_div_kernel<DstType, Src0Type, false>, 1> {
    typedef DstType dst_type;
    typedef Src0Type src0_type;

    static const std::size_t data_size = 0;

    void single(char *dst, char *const *src)
    {
      *reinterpret_cast<dst_type *>(dst) /=
          *reinterpret_cast<src0_type *>(src[0]);
    }

    void strided(char *dst, std::intptr_t dst_stride, char *const *src,
                 const std::intptr_t *src_stride, std::size_t count)
    {
      char *src0 = src[0];
      std::intptr_t src0_stride = src_stride[0];
      for (std::size_t i = 0; i < count; ++i) {
        *reinterpret_cast<dst_type *>(dst) /=
            *reinterpret_cast<src0_type *>(src0);
        dst += dst_stride;
        src0 += src0_stride;
      }
    }
  };

  template <typename DstType, typename Src0Type>
  struct compound_div_kernel<
      DstType, Src0Type,
      true> : base_kernel<compound_div_kernel<DstType, Src0Type, true>, 1> {
    typedef DstType dst_type;
    typedef Src0Type src0_type;

    static const std::size_t data_size = 0;

    void single(char *dst, char *const *src)
    {
      *reinterpret_cast<dst_type *>(dst) =
          static_cast<dst_type>(*reinterpret_cast<dst_type *>(dst) /
                                *reinterpret_cast<src0_type *>(src[0]));
    }

    void strided(char *dst, std::intptr_t dst_stride, char *const *src,
                 const std::intptr_t *src_stride, std::size_t count)
    {
      char *src0 = src[0];
      std::intptr_t src0_stride = src_stride[0];
      for (std::size_t i = 0; i < count; ++i) {
        *reinterpret_cast<dst_type *>(dst) =
            static_cast<dst_type>(*reinterpret_cast<dst_type *>(dst) /
                                  *reinterpret_cast<src0_type *>(src0));
        dst += dst_stride;
        src0 += src0_stride;
      }
    }
  };

  template <typename Src0Type>
  struct compound_div_kernel<
      bool1, Src0Type,
      true> : base_kernel<compound_div_kernel<bool1, Src0Type, true>, 1> {
    typedef bool1 dst_type;
    typedef Src0Type src0_type;

    static const std::size_t data_size = 0;

    void single(char *dst, char *const *src)
    {
      *reinterpret_cast<dst_type *>(dst) =
          *reinterpret_cast<dst_type *>(dst) /
                  *reinterpret_cast<src0_type *>(src[0])
              ? true
              : false;
    }

    void strided(char *dst, std::intptr_t dst_stride, char *const *src,
                 const std::intptr_t *src_stride, std::size_t count)
    {
      char *src0 = src[0];
      std::intptr_t src0_stride = src_stride[0];
      for (std::size_t i = 0; i < count; ++i) {
        *reinterpret_cast<dst_type *>(dst) =
            *reinterpret_cast<dst_type *>(dst) /
                    *reinterpret_cast<src0_type *>(src0)
                ? true
                : false;
        dst += dst_stride;
        src0 += src0_stride;
      }
    }
  };

  template <type_id_t DstTypeID, type_id_t Src0TypeID>
  struct compound_div_kernel_t
      : compound_div_kernel<
            typename type_of<DstTypeID>::type,
            typename type_of<Src0TypeID>::type,
            !is_lossless_assignable<DstTypeID, Src0TypeID>::value> {
  };

} // namespace dynd::nd

namespace ndt {

  template <type_id_t DstTypeID, type_id_t Src0TypeID>
  struct type::equivalent<nd::compound_div_kernel_t<DstTypeID, Src0TypeID>> {
    static type make()
    {
      return callable_type::make(type(DstTypeID), type(Src0TypeID));
    }
  };

} // namespace dynd::ndt
} // namespace dynd
