//
// Copyright (C) 2011-14 Irwin Zaid, DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#ifndef _DYND__FFT_HPP_
#define _DYND__FFT_HPP_

#ifdef DYND_FFTW

#include <fftw3.h>

// These are only available publicly as of FFTW 3.3.4, so we declare them here too
extern "C" FFTW_EXTERN int fftwf_alignment_of(float *p);
extern "C" FFTW_EXTERN int fftw_alignment_of(double *p);

#endif // DYND_FFTW

#include <dynd/array.hpp>
#include <dynd/array_range.hpp>
#include <dynd/shape_tools.hpp>

namespace dynd {

#ifdef DYND_FFTW
namespace fftw {

fftwf_plan fftplan(size_t ndim, std::vector<intptr_t> shape, fftwf_complex *src, std::vector<intptr_t> src_strides,
    fftwf_complex *dst, std::vector<intptr_t> dst_strides, int sign, unsigned int flags, bool cache = true);
fftw_plan fftplan(size_t ndim, std::vector<intptr_t> shape, fftw_complex *src, std::vector<intptr_t> src_strides,
    fftw_complex *dst, std::vector<intptr_t> dst_strides, int sign, unsigned int flags, bool cache = true);

fftwf_plan fftplan(size_t ndim, std::vector<intptr_t> shape, float *src, std::vector<intptr_t> src_strides,
    fftwf_complex *dst, std::vector<intptr_t> dst_strides, unsigned int flags, bool cache = true);
fftw_plan fftplan(size_t ndim, std::vector<intptr_t> shape, double *src, std::vector<intptr_t> src_strides,
    fftw_complex *dst, std::vector<intptr_t> dst_strides, unsigned int flags, bool cache = true);

fftwf_plan fftplan(size_t ndim, std::vector<intptr_t> shape, fftwf_complex *src, std::vector<intptr_t> src_strides,
    float *dst, std::vector<intptr_t> dst_strides, unsigned int flags, bool cache = true);
fftw_plan fftplan(size_t ndim, std::vector<intptr_t> shape, fftw_complex *src, std::vector<intptr_t> src_strides,
    double *dst, std::vector<intptr_t> dst_strides, unsigned int flags, bool cache = true);

void fftcleanup();

nd::array fft(const nd::array &x, const std::vector<intptr_t> &shape, unsigned int flags = FFTW_ESTIMATE);
nd::array ifft(const nd::array &x, const std::vector<intptr_t> &shape, unsigned int flags = FFTW_ESTIMATE);

nd::array rfft(const nd::array &x, const std::vector<intptr_t> &shape, unsigned int flags = FFTW_ESTIMATE);
nd::array irfft(const nd::array &x, const std::vector<intptr_t> &shape, unsigned int flags = FFTW_ESTIMATE);

} // namespace fftw
#endif // DYND_FFTW

#define INLINE_DECLARATIONS(FUNC) \
    inline nd::array FUNC(const nd::array &x) { \
        return FUNC(x, x.get_shape()); \
    } \
\
    inline nd::array FUNC(const nd::array &x, intptr_t ndim, const intptr_t *shape) { \
        return FUNC(x, std::vector<intptr_t>(shape, shape + ndim)); \
    } \
\
    inline nd::array FUNC(const nd::array &x, intptr_t n0) { \
        intptr_t shape[1] = {n0}; \
\
        return FUNC(x, 1, shape); \
    } \
\
    inline nd::array FUNC(const nd::array &x, intptr_t n0, intptr_t n1) { \
        intptr_t shape[2] = {n0, n1}; \
\
        return FUNC(x, 2, shape); \
    } \
\
    inline nd::array FUNC(const nd::array &x, intptr_t n0, intptr_t n1, intptr_t n2) { \
        intptr_t shape[3] = {n0, n1, n2}; \
\
        return FUNC(x, 3, shape); \
    }

/**
 *
 */
inline nd::array fft(const nd::array &x, const std::vector<intptr_t> &shape) {
    if (x.get_ndim() != (intptr_t) shape.size()) {
        std::stringstream ss;
        ss << "too many dimensions provided for fft, got " << x.get_ndim()
           << " for type " << x.get_type();
        throw std::invalid_argument(ss.str());
    }

    const ndt::type &tp = x.get_type();
    if (tp.get_flags() | type_flag_not_host_readable) {
    }

#ifdef DYND_FFTW
    return fftw::fft(x, shape);
#else
    throw std::runtime_error("no fft available");
#endif
}

INLINE_DECLARATIONS(fft)

/**
 *
 */
inline nd::array ifft(const nd::array &x, const std::vector<intptr_t> &shape) {
    if (x.get_ndim() != (intptr_t) shape.size()) {
        std::stringstream ss;
        ss << "too many dimensions provided for ifft, got " << x.get_ndim()
           << " for type " << x.get_type();
        throw std::invalid_argument(ss.str());
    }

    const ndt::type &tp = x.get_type();
    if (tp.get_flags() | type_flag_not_host_readable) {
    }

#ifdef DYND_FFTW
    return fftw::ifft(x, shape);
#else
    throw std::runtime_error("no fft available");
#endif
}

INLINE_DECLARATIONS(ifft)

/**
 *
 */
inline nd::array rfft(const nd::array &x, const std::vector<intptr_t> &shape) {
    if (x.get_ndim() != (intptr_t) shape.size()) {
        std::stringstream ss;
        ss << "too many dimensions provided for rfft, got " << x.get_ndim()
           << " for type " << x.get_type();
        throw std::invalid_argument(ss.str());
    }

    const ndt::type &tp = x.get_type();
    if (tp.get_flags() | type_flag_not_host_readable) {
    }

#ifdef DYND_FFTW
    return fftw::rfft(x, shape);
#else
    throw std::runtime_error("no fft available");
#endif
}

INLINE_DECLARATIONS(rfft)

/**
 *
 */
inline nd::array irfft(const nd::array &x, const std::vector<intptr_t> &shape) {
    if (x.get_ndim() != (intptr_t) shape.size()) {
        std::stringstream ss;
        ss << "too many dimensions provided for irfft, got " << x.get_ndim()
           << " for type " << x.get_type();
        throw std::invalid_argument(ss.str());
    }

    const ndt::type &tp = x.get_type();
    if (tp.get_flags() | type_flag_not_host_readable) {
    }

#ifdef DYND_FFTW
    return fftw::irfft(x, shape);
#else
    throw std::runtime_error("no fft available");
#endif
}

INLINE_DECLARATIONS(irfft)

#undef INLINE_DECLARATIONS

nd::array fftshift(const nd::array &x);
nd::array ifftshift(const nd::array &x);

} // namespace dynd

#endif // _DYND__FFT_HPP_
