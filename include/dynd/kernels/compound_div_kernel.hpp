#pragma once

#include <dynd/kernels/base_kernel.hpp>

namespace dynd {
namespace nd {

  template <type_id_t DstTypeID, type_id_t Src0TypeID>
  struct compound_div_kernel
      : base_kernel<compound_div_kernel<DstTypeID, Src0TypeID>,
                    kernel_request_host, 1> {
    typedef typename type_of<DstTypeID>::type dst_type;
    typedef typename type_of<Src0TypeID>::type src0_type;

    static const std::size_t data_size = 0;

    void single(char *dst, char *const *src)
    {
      *reinterpret_cast<dst_type *>(dst) /=
          static_cast<dst_type>(*reinterpret_cast<src0_type *>(src[0]));
    }
  };

} // namespace dynd::nd

namespace ndt {

  template <type_id_t DstTypeID, type_id_t Src0TypeID>
  struct type::equivalent<nd::compound_div_kernel<DstTypeID, Src0TypeID>> {
    static type make()
    {
      return callable_type::make(type(DstTypeID), type(Src0TypeID));
    }
  };

} // namespace dynd::ndt
} // namespace dynd
