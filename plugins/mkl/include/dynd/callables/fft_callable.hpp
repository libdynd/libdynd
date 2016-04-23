//
// Copyright (C) 2011-16 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#pragma once

#include <dynd/callables/base_callable.hpp>
#include <dynd/kernels/fft_kernel.hpp>

namespace dynd {
namespace nd {
  namespace mkl {

    class fft_callable : public base_callable {
    public:
      fft_callable() : base_callable(ndt::type("(Fixed**N * complex[float64]) -> Fixed**N * complex[float64]")) {}

      ndt::type resolve(base_callable *DYND_UNUSED(caller), char *DYND_UNUSED(data), call_graph &cg,
                        const ndt::type &DYND_UNUSED(dst_tp), size_t DYND_UNUSED(nsrc), const ndt::type *src_tp,
                        size_t DYND_UNUSED(nkwd), const array *DYND_UNUSED(kwds),
                        const std::map<std::string, ndt::type> &DYND_UNUSED(tp_vars)) {
        size_t ndim = src_tp[0].get_ndim();

        cg.emplace_back([ndim](kernel_builder &kb, kernel_request_t kernreq, char *DYND_UNUSED(data),
                               const char *DYND_UNUSED(dst_arrmeta), size_t DYND_UNUSED(narg),
                               const char *const *src_arrmeta) {
          kb.emplace_back<fft_kernel<dynd::complex<double>>>(kernreq, ndim,
                                                             reinterpret_cast<const size_stride_t *>(src_arrmeta[0]));
        });

        return src_tp[0];
      }
    };

  } // namespace dynd::nd::mkl
} // namespace dynd::nd
} // namespace dynd
