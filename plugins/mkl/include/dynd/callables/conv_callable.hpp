//
// Copyright (C) 2011-16 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#pragma once

#include <dynd/callables/base_callable.hpp>
#include <dynd/kernels/conv_kernel.hpp>

namespace dynd {
namespace nd {
  namespace mkl {

    template <typename DataType>
    class conv_callable : public base_callable {
    public:
      conv_callable() : base_callable(ndt::type("(Fixed**N * Scalar, Fixed**N * Scalar) -> Fixed**N * Scalar")) {}

      ndt::type resolve_ret_type(size_t ndim, const ndt::type &src0_tp, const ndt::type &src1_tp) {
        size_t dst_size = src0_tp.extended<ndt::fixed_dim_type>()->get_fixed_dim_size() +
                          src1_tp.extended<ndt::fixed_dim_type>()->get_fixed_dim_size() - 1;

        if (ndim == 1) {
          return ndt::make_type<ndt::fixed_dim_type>(dst_size,
                                                     src0_tp.extended<ndt::fixed_dim_type>()->get_element_type());
        }

        return ndt::make_type<ndt::fixed_dim_type>(
            dst_size, resolve_ret_type(ndim - 1, src0_tp.extended<ndt::fixed_dim_type>()->get_element_type(),
                                       src1_tp.extended<ndt::fixed_dim_type>()->get_element_type()));
      }

      ndt::type resolve(base_callable *DYND_UNUSED(caller), char *DYND_UNUSED(data), call_graph &cg,
                        const ndt::type &dst_tp, size_t DYND_UNUSED(nsrc), const ndt::type *src_tp,
                        size_t DYND_UNUSED(nkwd), const array *DYND_UNUSED(kwds),
                        const std::map<std::string, ndt::type> &DYND_UNUSED(tp_vars)) {
        size_t ndim = src_tp[0].get_ndim();
        MKL_INT mode = VSL_CONV_MODE_DIRECT;

        switch (ndim) {
        case 1:
          cg.emplace_back([mode](kernel_builder &kb, kernel_request_t kernreq, char *DYND_UNUSED(data),
                                 const char *dst_arrmeta, size_t DYND_UNUSED(narg), const char *const *src_arrmeta) {
            kb.emplace_back<conv_kernel<1, DataType>>(kernreq, mode,
                                                      reinterpret_cast<const size_stride_t *>(dst_arrmeta),
                                                      reinterpret_cast<const size_stride_t *>(src_arrmeta[0]),
                                                      reinterpret_cast<const size_stride_t *>(src_arrmeta[1]));
          });
          break;
        case 2:
          cg.emplace_back([mode](kernel_builder &kb, kernel_request_t kernreq, char *DYND_UNUSED(data),
                                 const char *dst_arrmeta, size_t DYND_UNUSED(narg), const char *const *src_arrmeta) {
            kb.emplace_back<conv_kernel<2, DataType>>(kernreq, mode,
                                                      reinterpret_cast<const size_stride_t *>(dst_arrmeta),
                                                      reinterpret_cast<const size_stride_t *>(src_arrmeta[0]),
                                                      reinterpret_cast<const size_stride_t *>(src_arrmeta[1]));
          });
          break;
        default:
          throw std::runtime_error("conv is not supported for this data type");
        }

        if (dst_tp.is_symbolic()) {
          return resolve_ret_type(ndim, src_tp[0], src_tp[1]);
        }

        return dst_tp;
      }
    };

  } // namespace dynd::nd::mkl
} // namespace dynd::nd
} // namespace dynd
