//
// Copyright (C) 2011-14 Irwin Zaid, DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#ifndef _DYND__FFT_HPP_
#define _DYND__FFT_HPP_

#ifdef DYND_FFTW
#include <fftw3.h>
#endif // DYND_FFTW

#include <dynd/array.hpp>

namespace dynd {

#ifdef DYND_FFTW
namespace fftw {

fftwf_plan fftplan(size_t ndim, fftwf_complex *src, std::vector<intptr_t> src_shape, std::vector<intptr_t> src_strides,
    fftwf_complex *dst, std::vector<intptr_t> dst_strides, int sign, unsigned int flags, bool cache = true);
fftw_plan fftplan(size_t ndim, fftw_complex *src, std::vector<intptr_t> src_shape, std::vector<intptr_t> src_strides,
    fftw_complex *dst, std::vector<intptr_t> dst_strides, int sign, unsigned int flags, bool cache = true);

nd::array fft1(const nd::array &x, unsigned int flags = FFTW_ESTIMATE, bool cache = true);
nd::array ifft1(const nd::array &x, unsigned int flags = FFTW_ESTIMATE, bool cache = true);

nd::array rfft1(const nd::array &x, unsigned int flags = FFTW_ESTIMATE, bool cache = true);
nd::array irfft1(const nd::array &x, unsigned int flags = FFTW_ESTIMATE, bool cache = true);

nd::array fft2(const nd::array &x, unsigned int flags = FFTW_ESTIMATE, bool cache = true);
nd::array ifft2(const nd::array &x, unsigned int flags = FFTW_ESTIMATE, bool cache = true);

nd::array rfft2(const nd::array &x, unsigned int flags = FFTW_ESTIMATE, bool cache = true);
nd::array irfft2(const nd::array &x, unsigned int flags = FFTW_ESTIMATE, bool cache = true);

nd::array fft(const nd::array &x, unsigned int flags = FFTW_ESTIMATE, bool cache = true);
nd::array ifft(const nd::array &x, unsigned int flags = FFTW_ESTIMATE, bool cache = true);

nd::array rfft(const nd::array &x, unsigned int flags = FFTW_ESTIMATE, bool cache = true);
nd::array irfft(const nd::array &x, unsigned int flags = FFTW_ESTIMATE, bool cache = true);

} // namespace fftw
#endif // DYND_FFTW

inline nd::array fft(const nd::array &x) {
    const ndt::type &tp = x.get_type();
    if (tp.get_flags() | type_flag_not_host_readable) {
    }

#ifdef DYND_FFTW
    return fftw::fft(x);
#else
    throw std::runtime_error("no fft available");
#endif
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

inline nd::array fft1(const nd::array &x) {
    return fft(x);
}

inline nd::array ifft1(const nd::array &x) {
    return ifft(x);
}

inline nd::array fft2(const nd::array &x) {
    return fft(x);
}

inline nd::array ifft2(const nd::array &x) {
    return ifft(x);
}

} // namespace dynd

#endif // _DYND__FFT_HPP_
