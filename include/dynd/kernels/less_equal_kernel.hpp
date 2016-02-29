//
// Copyright (C) 2011-15 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#pragma once

#include <dynd/kernels/base_kernel.hpp>
#include <dynd/types/callable_type.hpp>

namespace dynd {
namespace nd {

  template <type_id_t I0, type_id_t I1>
  struct less_equal_kernel : base_strided_kernel<less_equal_kernel<I0, I1>, 2> {
    typedef typename type_of<I0>::type A0;
    typedef typename type_of<I1>::type A1;
    typedef typename std::common_type<A0, A1>::type T;

    void single(char *dst, char *const *src)
    {
      *reinterpret_cast<bool1 *>(dst) =
          static_cast<T>(*reinterpret_cast<A0 *>(src[0])) <= static_cast<T>(*reinterpret_cast<A1 *>(src[1]));
    }
  };

  template <type_id_t I0>
  struct less_equal_kernel<I0, I0> : base_strided_kernel<less_equal_kernel<I0, I0>, 2> {
    typedef typename type_of<I0>::type A0;

    void single(char *dst, char *const *src)
    {
      *reinterpret_cast<bool1 *>(dst) = *reinterpret_cast<A0 *>(src[0]) <= *reinterpret_cast<A0 *>(src[1]);
    }
  };

} // namespace dynd::nd

namespace ndt {

  template <type_id_t Src0TypeID, type_id_t Src1TypeID>
  struct traits<nd::less_equal_kernel<Src0TypeID, Src1TypeID>> {
    static type equivalent() { return callable_type::make(make_type<bool1>(), {type(Src0TypeID), type(Src1TypeID)}); }
  };

} // namespace dynd::ndt
} // namespace dynd
