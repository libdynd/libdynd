//
// Copyright (C) 2010-14 Irwin Zaid, Mark Wiebe, DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#ifndef _DYND__COMPLEX_H__
#define _DYND__COMPLEX_H__

#include <complex>

#include <dynd/config.hpp>
#include <dynd/typed_data_assign.hpp>

namespace dynd {

template <typename T>
class dynd_complex;

template <>
class dynd_complex<float> {
public:
    typedef float value_type;
    float m_real, m_imag;

    DYND_CUDA_HOST_DEVICE_CALLABLE inline dynd_complex(float re = 0.0f, float im = 0.0f)
        : m_real(re), m_imag(im) {}

    template <typename T>
    DYND_CUDA_HOST_DEVICE_CALLABLE inline dynd_complex(const dynd_complex<T>& rhs)
        : m_real(static_cast<float>(rhs.m_real)), m_imag(static_cast<float>(rhs.m_imag)) {}

    DYND_CUDA_HOST_DEVICE_CALLABLE inline float real() const {
        return m_real;
    }
    DYND_CUDA_HOST_DEVICE_CALLABLE inline float imag() const {
        return m_imag;
    }

    DYND_CUDA_HOST_DEVICE_CALLABLE inline bool operator==(const dynd_complex<float>& rhs) const {
        return (real() == rhs.real()) && (imag() == rhs.imag());
    }

    DYND_CUDA_HOST_DEVICE_CALLABLE inline bool operator!=(const dynd_complex<float>& rhs) const {
        return !operator==(rhs);
    }
};

std::ostream& operator<<(std::ostream& out, const dynd_complex<float>& val);

template <>
class dynd_complex<double> {
public:
    typedef double value_type;
    double m_real, m_imag;

    DYND_CUDA_HOST_DEVICE_CALLABLE inline dynd_complex(double re = 0.0, double im = 0.0)
        : m_real(re), m_imag(im) {}

    template <typename T>
    DYND_CUDA_HOST_DEVICE_CALLABLE inline dynd_complex(const dynd_complex<T>& rhs)
        : m_real(rhs.m_real), m_imag(rhs.m_imag) {}

    DYND_CUDA_HOST_DEVICE_CALLABLE inline double real() const {
        return m_real;
    }
    DYND_CUDA_HOST_DEVICE_CALLABLE inline double imag() const {
        return m_imag;
    }

    DYND_CUDA_HOST_DEVICE_CALLABLE inline bool operator==(const dynd_complex<double>& rhs) const {
        return (real() == rhs.real()) && (imag() == rhs.imag());
    }

    DYND_CUDA_HOST_DEVICE_CALLABLE inline bool operator!=(const dynd_complex<double>& rhs) const {
        return !operator==(rhs);
    }
};

std::ostream& operator<<(std::ostream& out, const dynd_complex<double>& val);

} // namespace dynd

#endif // _DYND__COMPLEX_H__
