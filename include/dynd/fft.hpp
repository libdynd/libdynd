//
// Copyright (C) 2011-14 Irwin Zaid, DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#ifndef _DYND__FFT_HPP_
#define _DYND__FFT_HPP_

#include <dynd/array.hpp>

namespace dynd {

namespace fftw {

nd::array fft1(const nd::array &x);
nd::array ifft1(const nd::array &x);

nd::array fft(const nd::array &x);
nd::array ifft(const nd::array &x);

} // namespace fftw

inline nd::array fft1(const nd::array &x) {
    return fftw::fft(x);
}
inline nd::array ifft1(const nd::array &x) {
    return fftw::ifft(x);
}

inline nd::array fft2(const nd::array &x) {
    return fftw::fft(x);
}

inline nd::array fft(const nd::array &x) {
    return fftw::fft(x);
}

inline nd::array ifft(const nd::array &x) {
    const ndt::type &tp = x.get_type();
    if (tp.get_flags() | type_flag_not_host_readable) {
        // dispatch to cufft or whatever
    }

#ifdef DYND_FFTW
    return fftw::ifft(x);
#else
    throw std::runtime_error("no fft available");
#endif
}

} // namespace dynd

#endif // _DYND__FFT_HPP_
