//
// Copyright (C) 2011-14 Irwin Zaid, DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#ifndef _DYND__FFT_HPP_
#define _DYND__FFT_HPP_

#include <dynd/array.hpp>
#include <dynd/array_range.hpp>

#ifdef DYND_FFTW

#include <fftw3.h>

// These are only available publicly as of FFTW 3.3.4, so we declare them here too
extern "C" {
FFTW_EXTERN int fftwf_alignment_of(float *p);
FFTW_EXTERN int fftw_alignment_of(double *p);
}

#endif // DYND_FFTW

namespace dynd {

#ifdef DYND_FFTW
namespace fftw {

fftwf_plan fftplan(std::vector<intptr_t> shape, std::vector<intptr_t> axes,
    fftwf_complex *src, std::vector<intptr_t> src_strides,
    fftwf_complex *dst, std::vector<intptr_t> dst_strides,
    int sign, unsigned int flags, bool overwrite);
fftw_plan fftplan(std::vector<intptr_t> shape, std::vector<intptr_t> axes,
    fftw_complex *src, std::vector<intptr_t> src_strides,
    fftw_complex *dst, std::vector<intptr_t> dst_strides,
    int sign, unsigned int flags, bool overwrite);

fftwf_plan fftplan(std::vector<intptr_t> shape, float *src, std::vector<intptr_t> src_strides,
    fftwf_complex *dst, std::vector<intptr_t> dst_strides, unsigned int flags, bool overwrite);
fftw_plan fftplan(std::vector<intptr_t> shape, double *src, std::vector<intptr_t> src_strides,
    fftw_complex *dst, std::vector<intptr_t> dst_strides, unsigned int flags, bool overwrite);

fftwf_plan fftplan(std::vector<intptr_t> shape, fftwf_complex *src, std::vector<intptr_t> src_strides,
    float *dst, std::vector<intptr_t> dst_strides, unsigned int flags, bool overwrite);
fftw_plan fftplan(std::vector<intptr_t> shape, fftw_complex *src, std::vector<intptr_t> src_strides,
    double *dst, std::vector<intptr_t> dst_strides, unsigned int flags, bool overwrite);

void fftcleanup();

nd::array fft(const nd::array &x, std::vector<intptr_t> shape, std::vector<intptr_t> axes,
    unsigned int flags = FFTW_MEASURE);
nd::array ifft(const nd::array &x, std::vector<intptr_t> shape, std::vector<intptr_t> axes,
    unsigned int flags = FFTW_MEASURE);

nd::array rfft(const nd::array &x, std::vector<intptr_t> shape, unsigned int flags = FFTW_MEASURE);
nd::array irfft(const nd::array &x, std::vector<intptr_t> shape, unsigned int flags = FFTW_MEASURE);

} // namespace fftw
#endif // DYND_FFTW

#define DECL_INLINES(FUNC) \
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
 * Computes the discrete Fourier transform of a complex array.
 *
 * @param x An array of complex numbers with arbitrary dimensions.
 * @param shape The shape of the Fourier transform. If any dimension is less than
 *              the corresponding dimension of 'x', that dimension will be truncated.
 *
 * @return A complex array with dimensions specified by 'shape'.
 */
inline nd::array fft(const nd::array &x, std::vector<intptr_t> shape, std::vector<intptr_t> axes) {
    if (x.get_ndim() != static_cast<intptr_t>(shape.size()) || static_cast<intptr_t>(axes.size()) > x.get_ndim()) {
        throw std::invalid_argument("dimensions provided for fft do not match");
    }

#ifdef DYND_FFTW
    return fftw::fft(x, shape, axes);
#else
    throw std::runtime_error("fft is not implemented");
#endif
}

inline nd::array fft(const nd::array &x, std::vector<intptr_t> shape) {
    std::vector<intptr_t> axes;
    for (intptr_t i = 0; i < x.get_ndim(); ++i) {
        axes.push_back(i);
    }

    return fft(x, shape, axes);
}

DECL_INLINES(fft)

/**
 * Computes the inverse discrete Fourier transform of a complex array.
 *
 * @param x An array of complex numbers with arbitrary dimensions.
 * @param shape The shape of the inverse Fourier transform. If any dimension is less than
 *              the corresponding dimension of 'x', that dimension will be truncated.
 *
 * @return A complex array with dimensions specified by 'shape'.
 */
inline nd::array ifft(const nd::array &x, std::vector<intptr_t> shape, std::vector<intptr_t> axes) {
    if (x.get_ndim() != static_cast<intptr_t>(shape.size()) || static_cast<intptr_t>(axes.size()) > x.get_ndim()) {
        throw std::invalid_argument("dimensions provided for ifft do not match");
    }

#ifdef DYND_FFTW
    return fftw::ifft(x, shape, axes);
#else
    throw std::runtime_error("ifft is not implemented");
#endif
}

inline nd::array ifft(const nd::array &x, std::vector<intptr_t> shape) {
    std::vector<intptr_t> axes;
    for (intptr_t i = 0; i < x.get_ndim(); ++i) {
        axes.push_back(i);
    }

    return ifft(x, shape, axes);
}

DECL_INLINES(ifft)

/**
 * Computes the discrete Fourier transform of a real array.
 *
 * @param x An array of real numbers with arbitrary dimensions.
 * @param shape The shape of the Fourier transform. If any dimension is less than
 *              the corresponding dimension of 'x', that dimension will be truncated.
 *
 * @return A complex array with dimensions specified by 'shape', except the last dimension
 *         is '(shape[shape.size() - 1] / 2) + 1'.
 */
inline nd::array rfft(const nd::array &x, std::vector<intptr_t> shape) {
    if (x.get_ndim() != static_cast<intptr_t>(shape.size())) {
        throw std::invalid_argument("dimensions provided for rfft do not match");
    }

#ifdef DYND_FFTW
    return fftw::rfft(x, shape);
#else
    throw std::runtime_error("rfft is not implemented");
#endif
}

DECL_INLINES(rfft)

/**
 * Computes the discrete inverse Fourier transform of a complex array, under the
 * assumption that the result is a real array.
 *
 * @param x An array of real numbers with arbitrary dimensions.
 * @param shape The shape of the Fourier transform. If any dimension is less than
 *              the corresponding dimension of 'x', that dimension will be truncated.
 *
 * @return A complex array with dimensions specified by 'shape'. By default, the shape
 *         is the same as that of 'x', except the last dimension is '2 * (x.get_shape[x.get_ndim() - 1] - 1)'.
 */
inline nd::array irfft(const nd::array &x, std::vector<intptr_t> shape) {
    if (x.get_ndim() != static_cast<intptr_t>(shape.size())) {
        throw std::invalid_argument("dimensions provided for irfft do not match");
    }

#ifdef DYND_FFTW
    return fftw::irfft(x, shape);
#else
    throw std::runtime_error("irfft is not implemented");
#endif
}

DECL_INLINES(irfft)

#undef DECL_INLINES

/**
 * Shifts the zero-frequency element to the center of an array.
 */
nd::array fftshift(const nd::array &x);

/**
 * Inverts fftshift.
 */
nd::array ifftshift(const nd::array &x);

/**
 * Returns the sample frequencies of a discrete Fourier transform, with units of cycles per seconds.
 */
nd::array fftspace(intptr_t count, double step = 1.0);

} // namespace dynd

#endif // _DYND__FFT_HPP_
