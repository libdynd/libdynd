//
// Copyright (C) 2010-14 Irwin Zaid, Mark Wiebe, DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#ifndef _DYND__COMPLEX_H__
#define _DYND__COMPLEX_H__

#include <cmath>
#include <complex>
#include <limits>

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

    template <typename T>
    inline dynd_complex(const std::complex<T>& rhs)
        : m_real(rhs.real()), m_imag(rhs.imag()) {}

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

    inline operator std::complex<float>() const {
        return std::complex<float>(m_real, m_imag);
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

    template <typename T>
    inline dynd_complex(const std::complex<T>& rhs)
        : m_real(rhs.real()), m_imag(rhs.imag()) {}

    DYND_CUDA_HOST_DEVICE inline double real() const {
        return m_real;
    }
    DYND_CUDA_HOST_DEVICE inline double imag() const {
        return m_imag;
    }

    DYND_CUDA_HOST_DEVICE inline bool operator==(const double& rhs) const {
        return (real() == rhs) && (imag() == 0.0);
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

    inline operator std::complex<double>() const {
        return std::complex<double>(m_real, m_imag);
    }
};

// Metaprogram for determining if a type is a valid C++ scalar
// of a particular type.
template<typename T> struct is_complex_scalar {enum {value = false};};
template <> struct is_complex_scalar<char> {enum {value = true};};
template <> struct is_complex_scalar<signed char> {enum {value = true};};
template <> struct is_complex_scalar<short> {enum {value = true};};
template <> struct is_complex_scalar<int> {enum {value = true};};
template <> struct is_complex_scalar<long> {enum {value = true};};
template <> struct is_complex_scalar<long long> {enum {value = true};};
template <> struct is_complex_scalar<unsigned char> {enum {value = true};};
template <> struct is_complex_scalar<unsigned short> {enum {value = true};};
template <> struct is_complex_scalar<unsigned int> {enum {value = true};};
template <> struct is_complex_scalar<unsigned long> {enum {value = true};};
template <> struct is_complex_scalar<unsigned long long> {enum {value = true};};
template <> struct is_complex_scalar<float> {enum {value = true};};
template <> struct is_complex_scalar<double> {enum {value = true};};

template<class T>
inline typename enable_if<is_complex_scalar<T>::value, bool>::type operator==(const T& lhs, const dynd_complex<double>& rhs) {
    return dynd_complex<double>(lhs) == dynd_complex<double>(rhs);
}

inline dynd_complex<double> operator*(double lhs, const dynd_complex<double>& rhs) {
    return dynd_complex<double>(lhs * rhs.m_real, lhs * rhs.m_imag);
}

inline dynd_complex<double> operator/(double lhs, const dynd_complex<double>& rhs) {
    double denom = rhs.m_real * rhs.m_real + rhs.m_imag + rhs.m_imag;
    return dynd_complex<double>((lhs * rhs.m_real) / denom,
                                (- lhs * rhs.m_imag) / denom);
}

std::ostream& operator<<(std::ostream& out, const dynd_complex<double>& val);

template <typename T>
T abs(dynd_complex<T> z) {
    return static_cast<T>(hypot(z.real(), z.imag()));
}

template <typename T>
T arg(dynd_complex<T> z) {
    return atan2(z.imag(), z.real());
}

template <typename T>
inline dynd_complex<T> exp(dynd_complex<T> z) {
    using namespace std;
    T x, c, s;
    T r = z.real(), i = z.imag();
    dynd_complex<T> ret;

    if (isfinite(r)) {
        x = std::exp(r);

        c = std::cos(i);
        s = std::sin(i);

        if (isfinite(i)) {
            ret = dynd_complex<T>(x * c, x * s);
        } else {
            ret = dynd_complex<T>(
                std::numeric_limits<T>::quiet_NaN(),
                copysign(std::numeric_limits<T>::quiet_NaN(), i));
        }
    } else if (DYND_ISNAN(r)) {
        // r is nan
        if (i == 0) {
            ret = dynd_complex<T>(r, 0);
        } else {
            ret = dynd_complex<T>(
                r, copysign(std::numeric_limits<T>::quiet_NaN(), i));
        }
    } else {
        // r is +- inf
        if (r > 0) {
            if (i == 0) {
                ret = dynd_complex<T>(r, i);
            } else if (isfinite(i)) {
                c = std::cos(i);
                s = std::sin(i);

                ret = dynd_complex<T>(r * c, r * s);
            } else {
                // x = +inf, y = +-inf | nan
                ret = dynd_complex<T>(r, std::numeric_limits<T>::quiet_NaN());
            }
        } else {
            if (isfinite(i)) {
                x = std::exp(r);
                c = std::cos(i);
                s = std::sin(i);

                ret = dynd_complex<T>(x * c, x * s);
            } else {
                // x = -inf, y = nan | +i inf
                ret = dynd_complex<T>(0, 0);
            }
        }
    }

    return ret;
}

template <typename T>
dynd_complex<T> log(dynd_complex<T> z) {
    return dynd_complex<T>(std::log(abs(z)), arg(z));
}

template <typename T>
inline dynd_complex<T> sqrt(dynd_complex<T> z) {
    using namespace std;
    // We risk spurious overflow for components >= DBL_MAX / (1 + sqrt(2))
    const T thresh =
        std::numeric_limits<T>::max() / (1 + ::sqrt(static_cast<T>(2)));

    dynd_complex<T> result;
    T a = z.real(), b = z.imag();
    T t;
    bool scale;

    // Handle special cases.
    if (a == 0 && b == 0) {
        return dynd_complex<T>(0, b);
    }
    if (isinf(b)) {
        return dynd_complex<T>(std::numeric_limits<T>::infinity(), b);
    }
    if (DYND_ISNAN(a)) {
        t = (b - b) / (b - b); // raise invalid if b is not a NaN
        return dynd_complex<T>(a, t); // return NaN + NaN i
    }
    if (isinf(a)) {
         // csqrt(inf + NaN i) = inf + NaN i
         // csqrt(inf + y i) = inf + 0 i
         // csqrt(-inf + NaN i) = NaN +- inf i
         // csqrt(-inf + y i) = 0 + inf i
        if (signbit(a)) {
            return dynd_complex<T>(std::fabs(b - b), copysign(a, b));
        } else {
            return dynd_complex<T>(a, copysign(b - b, b));
        }
    }
    // The remaining special case (b is NaN) is handled below

    // Scale to avoid overflow
    if (std::fabs(a) >= thresh || std::fabs(b) >= thresh) {
        a *= 0.25;
        b *= 0.25;
        scale = true;
    } else {
        scale = false;
    }

    // Algorithm 312, CACM vol 10, Oct 1967
    if (a >= 0) {
        t = std::sqrt((a + hypot(a, b)) * 0.5);
        result = dynd_complex<T>(t, b / (2 * t));
    } else {
        t = std::sqrt((-a + hypot(a, b)) * 0.5);
        result = dynd_complex<T>(std::fabs(b) / (2 * t), copysign(t, b));
    }

    // Rescale
    if (scale) {
        return dynd_complex<T>(result.real() * 2, result.imag());
    } else {
        return result;
    }
}

template <typename T>
dynd_complex<T> pow(dynd_complex<T> x, dynd_complex<T> y) {
    T yr = y.real(), yi = y.imag();

    dynd_complex<T> b = log(x);
    T br = b.real(), bi = b.imag();

    return exp(dynd_complex<T>(br * yr - bi * yi, br * yi + bi * yr));
}

template <typename T>
dynd_complex<T> pow(dynd_complex<T> x, T y) {
    dynd_complex<T> b = log(x);
    T br = b.real(), bi = b.imag();

    return exp(dynd_complex<T>(br * y, bi * y));
}

template <typename T>
dynd_complex<T> cos(dynd_complex<T> z) {
    T x = z.real(), y = z.imag();
    return dynd_complex<T>(std::cos(x) * std::cosh(y), -(std::sin(x) * std::sinh(y)));
}

template <typename T>
dynd_complex<T> sin(dynd_complex<T> z) {
    T x = z.real(), y = z.imag();
    return dynd_complex<T>(std::sin(x) * std::cosh(y), std::cos(x) * std::sinh(y));
}

} // namespace dynd

#endif // _DYND__COMPLEX_H__
