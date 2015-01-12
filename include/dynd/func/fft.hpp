//
// Copyright (C) 2011-14 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#pragma once

#include <dynd/func/multidispatch.hpp>
#include <dynd/kernels/fft.hpp>

#ifdef DYND_FFTW

namespace dynd {
namespace decl {
  namespace nd {

    /*
    class fftw : public arrfunc<fftw> {
    public:
        static dynd::nd::arrfunc make() {
            dynd::nd::arrfunc af[2] = {arrfunc<fftw_ck<fftwf_complex,
    fftwf_complex, FFTW_FORWARD> >::make(),
                arrfunc<fftw_ck<fftw_complex, fftw_complex, FFTW_FORWARD>
    >::make()};

            dynd::nd::arrfunc af2 = make_multidispatch_arrfunc(2, af);;
            std::cout << arrfunc<fftw_ck<fftwf_complex, fftwf_complex,
    FFTW_FORWARD> >::make().get()->func_proto << std::endl;

            return af2;
        }
    };
    */

    typedef arrfunc<kernels::fftw_ck<fftw_complex, fftw_complex, FFTW_FORWARD>> fftw;
    typedef arrfunc<kernels::fftw_ck<fftw_complex, double>> rfftw;

    typedef arrfunc<kernels::fftw_ck<fftw_complex, fftw_complex, FFTW_BACKWARD>> ifftw;
    typedef arrfunc<kernels::fftw_ck<double, fftw_complex>> irfftw;
  }
} // namespace dynd::decl::nd

namespace nd {

  extern decl::nd::fftw fftw;
  extern decl::nd::rfftw rfftw;

  extern decl::nd::ifftw ifftw;
  extern decl::nd::irfftw irfftw;

} // namespace dynd::nd

namespace decl {
  namespace nd {

    class fft : public arrfunc<fft> {
    public:
      static dynd::nd::arrfunc make()
      {
        return dynd::nd::fftw;
      }
    };

    class rfft : public arrfunc<rfft> {
    public:
      static dynd::nd::arrfunc make() { return dynd::nd::rfftw; }
    };

    class ifft : public arrfunc<ifft> {
    public:
      static dynd::nd::arrfunc make() { return dynd::nd::ifftw; }
    };

    class irfft : public arrfunc<irfft> {
    public:
      static dynd::nd::arrfunc make() { return dynd::nd::irfftw; }
    };
  }
} // namespace dynd::decl::nd

namespace nd {

  extern decl::nd::fft fft;
  extern decl::nd::rfft rfft;

  extern decl::nd::ifft ifft;
  extern decl::nd::irfft irfft;
}
} // namespace dynd::nd

#endif