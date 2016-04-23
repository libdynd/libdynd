//
// Copyright (C) 2011-16 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#pragma once

#include <dynd/callables/base_callable.hpp>
#include <dynd/kernels/ifft_kernel.hpp>
#include <dynd/types/option_type.hpp>

namespace dynd {
namespace nd {
  namespace mkl {

    template <typename T>
    class ifft_callable : public base_callable {
      typedef T complex_type;
      typedef typename T::value_type real_type;

    public:
      ifft_callable()
          : base_callable(ndt::type("(Fixed**N * complex[float64], scale: ?float64) -> Fixed**N * complex[float64]")) {}

      ndt::type resolve(base_callable *DYND_UNUSED(caller), char *DYND_UNUSED(data), call_graph &cg,
                        const ndt::type &DYND_UNUSED(dst_tp), size_t DYND_UNUSED(nsrc), const ndt::type *src_tp,
                        size_t DYND_UNUSED(nkwd), const array *kwds,
                        const std::map<std::string, ndt::type> &DYND_UNUSED(tp_vars)) {
        size_t ndim = src_tp[0].get_ndim();
        real_type scale = kwds[0].is_na() ? 1 : kwds[0].as<real_type>();

        cg.emplace_back([ndim, scale](kernel_builder &kb, kernel_request_t kernreq, char *DYND_UNUSED(data),
                                      const char *DYND_UNUSED(dst_arrmeta), size_t DYND_UNUSED(narg),
                                      const char *const *src_arrmeta) {
          kb.emplace_back<ifft_kernel<complex_type>>(kernreq, ndim, src_arrmeta[0], scale);
        });

        return src_tp[0];
      }
    };

  } // namespace dynd::nd::mkl
} // namespace dynd::nd
} // namespace dynd
