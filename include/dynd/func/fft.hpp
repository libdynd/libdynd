//
// Copyright (C) 2011-15 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#pragma once

#include <dynd/func/multidispatch.hpp>
#include <dynd/kernels/fft.hpp>

#ifdef DYND_FFTW

namespace dynd {
namespace nd {
  namespace decl {

    struct fftw : arrfunc<fftw> {
      typedef kernels::fftw_ck<fftw_complex, fftw_complex, FFTW_FORWARD> CKT;

      static nd::arrfunc as_arrfunc() { return nd::as_arrfunc<CKT>(); }
    };

    struct rfftw : arrfunc<rfftw> {
      typedef kernels::fftw_ck<fftw_complex, double> CKT;

      static nd::arrfunc as_arrfunc() { return nd::as_arrfunc<CKT>(); }
    };

    struct ifftw : arrfunc<ifftw> {
      typedef kernels::fftw_ck<fftw_complex, fftw_complex> CKT;

      static nd::arrfunc as_arrfunc() { return nd::as_arrfunc<CKT>(); }
    };

    class irfftw : public arrfunc<irfftw> {
    public:
      typedef kernels::fftw_ck<double, fftw_complex> CKT;

      static nd::arrfunc as_arrfunc() { return nd::as_arrfunc<CKT>(); }
    };
  }

} // namespace dynd::decl::nd

namespace nd {

  extern decl::fftw fftw;
  extern decl::rfftw rfftw;

  extern decl::ifftw ifftw;
  extern decl::irfftw irfftw;

} // namespace dynd::nd

namespace nd {
  namespace decl {

    struct fft : arrfunc<fft> {
      static nd::arrfunc as_arrfunc()
      {
        std::vector<nd::arrfunc> children;

#ifdef DYND_FFTW
        children.push_back(nd::fftw);
#endif

//        std::cout << "--" << std::endl;
  //      std::cout << ndt::type("cuda_device[Fixed**N * float64]") << std::endl;
    //    std::cout << "--" << std::endl;

#ifdef DYND_CUDA
        typedef kernels::fft_ck<kernel_request_cuda_device, complex<double>,
                                complex<double>> ck_type;
        children.push_back(nd::as_arrfunc<ck_type>());
#endif

        return functional::multidispatch(
            ndt::type(
                "(M[Fixed**N * complex[float64]], shape: ?N * int64, axes: ?Fixed * int64, "
                "flags: ?int32) -> M[Fixed**N * complex[float64]]"),
            children);
      }
    };

    struct rfft : arrfunc<rfft> {
      static nd::arrfunc as_arrfunc() { return nd::rfftw; }
    };

    struct ifft : arrfunc<ifft> {
      static nd::arrfunc as_arrfunc() { return nd::ifftw; }
    };

    struct irfft : arrfunc<irfft> {
      static nd::arrfunc as_arrfunc() { return nd::irfftw; }
    };
  }
} // namespace dynd::decl::nd

namespace nd {

  extern decl::fft fft;
  extern decl::rfft rfft;

  extern decl::ifft ifft;
  extern decl::irfft irfft;
}
} // namespace dynd::nd

#endif