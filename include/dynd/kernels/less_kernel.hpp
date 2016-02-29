//
// Copyright (C) 2011-15 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#pragma once

#include <dynd/kernels/base_kernel.hpp>
#include <dynd/types/callable_type.hpp>

namespace dynd {
namespace nd {

  template <type_id_t Arg0ID, type_id_t Arg1ID>
  struct less_kernel : base_strided_kernel<less_kernel<Arg0ID, Arg1ID>, 2> {
    typedef typename type_of<Arg0ID>::type arg0_type;
    typedef typename type_of<Arg1ID>::type arg1_type;
    typedef typename std::common_type<arg0_type, arg1_type>::type common_type;

    void single(char *dst, char *const *src)
    {
      *reinterpret_cast<bool1 *>(dst) = static_cast<common_type>(*reinterpret_cast<arg0_type *>(src[0])) <
                                        static_cast<common_type>(*reinterpret_cast<arg1_type *>(src[1]));
    }
  };

  template <type_id_t ArgID>
  struct less_kernel<ArgID, ArgID> : base_strided_kernel<less_kernel<ArgID, ArgID>, 2> {
    typedef typename type_of<ArgID>::type arg_type;

    void single(char *dst, char *const *src)
    {
      *reinterpret_cast<bool1 *>(dst) = *reinterpret_cast<arg_type *>(src[0]) < *reinterpret_cast<arg_type *>(src[1]);
    }
  };

} // namespace dynd::nd

namespace ndt {

  template <type_id_t Src0TypeID, type_id_t Src1TypeID>
  struct traits<nd::less_kernel<Src0TypeID, Src1TypeID>> {
    static type equivalent() { return callable_type::make(make_type<bool1>(), {type(Src0TypeID), type(Src1TypeID)}); }
  };

} // namespace dynd::ndt
} // namespace dynd
