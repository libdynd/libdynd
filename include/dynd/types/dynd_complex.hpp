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

    DYND_CUDA_HOST_DEVICE inline dynd_complex(float re = 0.0f, float im = 0.0f)
        : m_real(re), m_imag(im) {}

    template <typename T>
    DYND_CUDA_HOST_DEVICE inline dynd_complex(const dynd_complex<T>& rhs)
        : m_real(static_cast<float>(rhs.m_real)), m_imag(static_cast<float>(rhs.m_imag)) {}

    DYND_CUDA_HOST_DEVICE inline float real() const {
        return m_real;
    }
    DYND_CUDA_HOST_DEVICE inline float imag() const {
        return m_imag;
    }

    DYND_CUDA_HOST_DEVICE inline bool operator==(const dynd_complex<float>& rhs) const {
        return (real() == rhs.real()) && (imag() == rhs.imag());
    }

    DYND_CUDA_HOST_DEVICE inline bool operator!=(const dynd_complex<float>& rhs) const {
        return !operator==(rhs);
    }

    DYND_CUDA_HOST_DEVICE inline dynd_complex<float> operator+(const dynd_complex<float>& rhs) const {
        return dynd_complex<float>(m_real + rhs.m_real, m_imag + rhs.m_imag);
    }

    DYND_CUDA_HOST_DEVICE inline dynd_complex<float> operator-(const dynd_complex<float>& rhs) const {
        return dynd_complex<float>(m_real - rhs.m_real, m_imag - rhs.m_imag);
    }

    DYND_CUDA_HOST_DEVICE inline dynd_complex<float> operator*(const dynd_complex<float>& rhs) const {
        return dynd_complex<float>(m_real * rhs.m_real - m_imag * rhs.m_imag,
                                    m_real * rhs.m_imag + rhs.m_real * m_imag);
    }

    DYND_CUDA_HOST_DEVICE inline dynd_complex<float> operator*(float rhs) const {
        return dynd_complex<float>(m_real * rhs, m_imag * rhs);
    }

    DYND_CUDA_HOST_DEVICE inline dynd_complex<float> operator/(const dynd_complex<float>& rhs) const {
        float denom = rhs.m_real * rhs.m_real + rhs.m_imag + rhs.m_imag;
        return dynd_complex<float>((m_real * rhs.m_real + m_imag * rhs.m_imag) / denom,
                                    (rhs.m_real * m_imag - m_real * rhs.m_imag) / denom);
    }

    DYND_CUDA_HOST_DEVICE inline dynd_complex<float> operator/(float rhs) const {
        return dynd_complex<float>(m_real / rhs, m_imag / rhs);
    }
};

inline dynd_complex<float> operator*(float lhs, const dynd_complex<float>& rhs) {
    return dynd_complex<float>(lhs * rhs.m_real, lhs * rhs.m_imag);
}

inline dynd_complex<float> operator/(float lhs, const dynd_complex<float>& rhs) {
    float denom = rhs.m_real * rhs.m_real + rhs.m_imag + rhs.m_imag;
    return dynd_complex<float>((lhs * rhs.m_real) / denom,
                                (- lhs * rhs.m_imag) / denom);
}

std::ostream& operator<<(std::ostream& out, const dynd_complex<float>& val);

template <>
class dynd_complex<double> {
public:
    typedef double value_type;
    double m_real, m_imag;

    DYND_CUDA_HOST_DEVICE inline dynd_complex(double re = 0.0, double im = 0.0)
        : m_real(re), m_imag(im) {}

    template <typename T>
    DYND_CUDA_HOST_DEVICE inline dynd_complex(const dynd_complex<T>& rhs)
        : m_real(rhs.m_real), m_imag(rhs.m_imag) {}

    DYND_CUDA_HOST_DEVICE inline double real() const {
        return m_real;
    }
    DYND_CUDA_HOST_DEVICE inline double imag() const {
        return m_imag;
    }

    DYND_CUDA_HOST_DEVICE inline bool operator==(const dynd_complex<double>& rhs) const {
        return (real() == rhs.real()) && (imag() == rhs.imag());
    }

    DYND_CUDA_HOST_DEVICE inline bool operator!=(const dynd_complex<double>& rhs) const {
        return !operator==(rhs);
    }

    DYND_CUDA_HOST_DEVICE inline dynd_complex<double> operator+(const dynd_complex<double>& rhs) const {
        return dynd_complex<double>(m_real + rhs.m_real, m_imag + rhs.m_imag);
    }

    DYND_CUDA_HOST_DEVICE inline dynd_complex<double> operator-(const dynd_complex<double>& rhs) const {
        return dynd_complex<double>(m_real - rhs.m_real, m_imag - rhs.m_imag);
    }

    DYND_CUDA_HOST_DEVICE inline dynd_complex<double> operator*(const dynd_complex<double>& rhs) const {
        return dynd_complex<double>(m_real * rhs.m_real - m_imag * rhs.m_imag,
                                    m_real * rhs.m_imag + rhs.m_real * m_imag);
    }

    DYND_CUDA_HOST_DEVICE inline dynd_complex<double> operator*(double rhs) const {
        return dynd_complex<double>(m_real * rhs, m_imag * rhs);
    }

    DYND_CUDA_HOST_DEVICE inline dynd_complex<double> operator/(const dynd_complex<double>& rhs) const {
        double denom = rhs.m_real * rhs.m_real + rhs.m_imag + rhs.m_imag;
        return dynd_complex<double>((m_real * rhs.m_real + m_imag * rhs.m_imag) / denom,
                                    (rhs.m_real * m_imag - m_real * rhs.m_imag) / denom);
    }

    DYND_CUDA_HOST_DEVICE inline dynd_complex<double> operator/(double rhs) const {
        return dynd_complex<double>(m_real / rhs, m_imag / rhs);
    }
};

inline dynd_complex<double> operator*(double lhs, const dynd_complex<double>& rhs) {
    return dynd_complex<double>(lhs * rhs.m_real, lhs * rhs.m_imag);
}

inline dynd_complex<double> operator/(double lhs, const dynd_complex<double>& rhs) {
    double denom = rhs.m_real * rhs.m_real + rhs.m_imag + rhs.m_imag;
    return dynd_complex<double>((lhs * rhs.m_real) / denom,
                                (- lhs * rhs.m_imag) / denom);
}

std::ostream& operator<<(std::ostream& out, const dynd_complex<double>& val);

} // namespace dynd

#endif // _DYND__COMPLEX_H__
