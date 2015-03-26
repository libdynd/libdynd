//
// Copyright (C) 2011-15 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#pragma once

#include <cmath>
#include <complex>
#include <limits>

#include <dynd/config.hpp>
#include <dynd/math.hpp>
#include <dynd/typed_data_assign.hpp>

namespace dynd {

template <typename T>
class complex {
public:
  T m_real, m_imag;
  typedef T value_type;

  DYND_CUDA_HOST_DEVICE complex(const T &re = 0.0, const T &im = 0.0)
      : m_real(re), m_imag(im)
  {
  }

  template <typename U>
  DYND_CUDA_HOST_DEVICE complex(const complex<U> &rhs)
      : m_real(static_cast<T>(rhs.m_real)), m_imag(static_cast<T>(rhs.m_imag))
  {
  }

  complex(const std::complex<T> &rhs)
      : m_real(rhs.real()), m_imag(rhs.imag())
  {
  }

  DYND_CUDA_HOST_DEVICE T real() const { return m_real; }
  DYND_CUDA_HOST_DEVICE T imag() const { return m_imag; }

  DYND_CUDA_HOST_DEVICE complex<T> &operator=(const complex<T> &rhs)
  {
    m_real = rhs.m_real;
    m_imag = rhs.m_imag;

    return *this;
  }

  DYND_CUDA_HOST_DEVICE complex<T> &operator+=(const complex<T> &rhs)
  {
    m_real += rhs.m_real;
    m_imag += rhs.m_imag;

    return *this;
  }

  DYND_CUDA_HOST_DEVICE complex<T> &operator+=(const T &rhs)
  {
    m_real += rhs;

    return *this;
  }

  DYND_CUDA_HOST_DEVICE complex<T> &operator-=(const complex<T> &rhs)
  {
    m_real -= rhs.m_real;
    m_imag -= rhs.m_imag;

    return *this;
  }

  DYND_CUDA_HOST_DEVICE complex<T> &operator-=(const T &rhs)
  {
    m_real -= rhs;

    return *this;
  }

  DYND_CUDA_HOST_DEVICE complex<T> operator*=(const complex<T> &rhs)
  {
    new (this) complex<T>(m_real * rhs.m_real - m_imag * rhs.m_imag,
                               m_real * rhs.m_imag + rhs.m_real * m_imag);

    return *this;
  }

  DYND_CUDA_HOST_DEVICE complex<T> operator*=(const T &rhs)
  {
    m_real *= rhs;
    m_imag *= rhs;

    return *this;
  }

  DYND_CUDA_HOST_DEVICE complex<T> &operator/=(const complex<T> &rhs)
  {
    T denom = rhs.m_real * rhs.m_real + rhs.m_imag * rhs.m_imag;
    new (this)
        complex<T>((m_real * rhs.m_real + m_imag * rhs.m_imag) / denom,
                        (rhs.m_real * m_imag - m_real * rhs.m_imag) / denom);

    return *this;
  }

  DYND_CUDA_HOST_DEVICE complex<T> &operator/=(const T &rhs)
  {
    m_real /= rhs;
    m_imag /= rhs;

    return *this;
  }

  operator std::complex<T>() const { return std::complex<T>(m_real, m_imag); }
};

template <typename T, typename U>
DYND_CUDA_HOST_DEVICE bool operator==(complex<T> lhs, complex<U> rhs)
{
  return (lhs.m_real == rhs.m_real) && (lhs.m_imag == rhs.m_imag);
}

template <typename T, typename U>
DYND_CUDA_HOST_DEVICE typename std::enable_if<
    std::is_integral<U>::value || std::is_floating_point<U>::value, bool>::type
operator==(complex<T> lhs, U rhs)
{
  return (lhs.m_real == rhs) && !lhs.m_imag;
}

template <typename T, typename U>
DYND_CUDA_HOST_DEVICE typename std::enable_if<
    std::is_integral<T>::value || std::is_floating_point<T>::value, bool>::type
operator==(T lhs, complex<U> rhs)
{
  return rhs == lhs;
}

template <typename T>
bool operator==(complex<T> lhs, std::complex<T> rhs)
{
  return (lhs.m_real == rhs.real()) && (lhs.m_imag == rhs.imag());
}

template <typename T>
bool operator==(std::complex<T> lhs, complex<T> rhs)
{
  return rhs == lhs;
}

template <typename T, typename U>
DYND_CUDA_HOST_DEVICE bool operator!=(complex<T> lhs, complex<U> rhs)
{
  return !(lhs == rhs);
}

template <typename T>
DYND_CUDA_HOST_DEVICE complex<T> operator+(const complex<T> &rhs)
{
  return complex<T>(+rhs.m_real, +rhs.m_imag);
}

template <typename T>
DYND_CUDA_HOST_DEVICE complex<T> operator-(const complex<T> &rhs)
{
  return complex<T>(-rhs.m_real, -rhs.m_imag);
}

template <typename T>
DYND_CUDA_HOST_DEVICE complex<T> operator+(complex<T> lhs,
                                                complex<T> rhs)
{
  return lhs += rhs;
}

template <typename T, typename U>
DYND_CUDA_HOST_DEVICE complex<typename std::common_type<T, U>::type>
operator+(complex<T> lhs, complex<U> rhs)
{
  return static_cast<complex<typename std::common_type<T, U>::type>>(lhs) +
         static_cast<complex<typename std::common_type<T, U>::type>>(rhs);
}

template <typename T>
DYND_CUDA_HOST_DEVICE complex<T> operator+(complex<T> lhs, T rhs)
{
  return lhs += rhs;
}

template <typename T, typename U>
DYND_CUDA_HOST_DEVICE typename std::enable_if<
    std::is_integral<U>::value || std::is_floating_point<U>::value,
    complex<typename std::common_type<T, U>::type>>::type
operator+(complex<T> lhs, U rhs)
{
  return static_cast<complex<typename std::common_type<T, U>::type>>(lhs) +
         static_cast<typename std::common_type<T, U>::type>(rhs);
}

template <typename T>
DYND_CUDA_HOST_DEVICE complex<T> operator+(T lhs, complex<T> rhs)
{
  return complex<T>(lhs + rhs.m_real, rhs.m_imag);
}

template <typename T, typename U>
DYND_CUDA_HOST_DEVICE typename std::enable_if<
    std::is_integral<T>::value || std::is_floating_point<T>::value,
    complex<typename std::common_type<T, U>::type>>::type
operator+(T lhs, complex<U> rhs)
{
  return static_cast<typename std::common_type<T, U>::type>(lhs) +
         static_cast<complex<typename std::common_type<T, U>::type>>(rhs);
}

template <typename T>
DYND_CUDA_HOST_DEVICE complex<T> operator-(complex<T> lhs,
                                                complex<T> rhs)
{
  return lhs -= rhs;
}

template <typename T, typename U>
DYND_CUDA_HOST_DEVICE complex<typename std::common_type<T, U>::type>
operator-(complex<T> lhs, complex<U> rhs)
{
  return static_cast<complex<typename std::common_type<T, U>::type>>(lhs) -
         static_cast<complex<typename std::common_type<T, U>::type>>(rhs);
}

template <typename T>
DYND_CUDA_HOST_DEVICE complex<T> operator-(complex<T> lhs, T rhs)
{
  return lhs -= rhs;
}

template <typename T, typename U>
DYND_CUDA_HOST_DEVICE typename std::enable_if<
    std::is_integral<U>::value || std::is_floating_point<U>::value,
    complex<typename std::common_type<T, U>::type>>::type
operator-(complex<T> lhs, U rhs)
{
  return static_cast<complex<typename std::common_type<T, U>::type>>(lhs) -
         static_cast<typename std::common_type<T, U>::type>(rhs);
}

template <typename T>
DYND_CUDA_HOST_DEVICE complex<T> operator-(T lhs, complex<T> rhs)
{
  return complex<T>(lhs - rhs.m_real, -rhs.m_imag);
}

template <typename T, typename U>
DYND_CUDA_HOST_DEVICE typename std::enable_if<
    std::is_integral<T>::value || std::is_floating_point<T>::value,
    complex<typename std::common_type<T, U>::type>>::type
operator-(T lhs, complex<U> rhs)
{
  return static_cast<typename std::common_type<T, U>::type>(lhs) -
         static_cast<complex<typename std::common_type<T, U>::type>>(rhs);
}

template <typename T>
DYND_CUDA_HOST_DEVICE complex<T> operator*(complex<T> lhs,
                                                complex<T> rhs)
{
  return lhs *= rhs;
}

template <typename T, typename U>
DYND_CUDA_HOST_DEVICE complex<typename std::common_type<T, U>::type>
operator*(complex<T> lhs, complex<U> rhs)
{
  return static_cast<complex<typename std::common_type<T, U>::type>>(lhs) *
         static_cast<complex<typename std::common_type<T, U>::type>>(rhs);
}

template <typename T>
DYND_CUDA_HOST_DEVICE complex<T> operator*(complex<T> lhs, T rhs)
{
  return lhs *= rhs;
}

template <typename T, typename U>
DYND_CUDA_HOST_DEVICE typename std::enable_if<
    std::is_integral<U>::value || std::is_floating_point<U>::value,
    complex<typename std::common_type<T, U>::type>>::type
operator*(complex<T> lhs, U rhs)
{
  return static_cast<complex<typename std::common_type<T, U>::type>>(lhs) *
         static_cast<typename std::common_type<T, U>::type>(rhs);
}

template <typename T>
DYND_CUDA_HOST_DEVICE complex<T> operator*(T lhs, complex<T> rhs)
{
  return complex<T>(lhs * rhs.m_real, lhs * rhs.m_imag);
}

template <typename T, typename U>
DYND_CUDA_HOST_DEVICE typename std::enable_if<
    std::is_integral<T>::value || std::is_floating_point<T>::value,
    complex<typename std::common_type<T, U>::type>>::type
operator*(T lhs, complex<U> rhs)
{
  return static_cast<typename std::common_type<T, U>::type>(lhs) *
         static_cast<complex<typename std::common_type<T, U>::type>>(rhs);
}

template <typename T>
DYND_CUDA_HOST_DEVICE complex<T> operator/(complex<T> lhs,
                                                complex<T> rhs)
{
  return lhs /= rhs;
}

template <typename T, typename U>
DYND_CUDA_HOST_DEVICE complex<typename std::common_type<T, U>::type>
operator/(complex<T> lhs, complex<U> rhs)
{
  return static_cast<complex<typename std::common_type<T, U>::type>>(lhs) /
         static_cast<complex<typename std::common_type<T, U>::type>>(rhs);
}

template <typename T>
DYND_CUDA_HOST_DEVICE complex<T> operator/(complex<T> lhs, T rhs)
{
  return lhs /= rhs;
}

template <typename T, typename U>
DYND_CUDA_HOST_DEVICE typename std::enable_if<
    std::is_integral<U>::value || std::is_floating_point<U>::value,
    complex<typename std::common_type<T, U>::type>>::type
operator/(complex<T> lhs, U rhs)
{
  return static_cast<complex<typename std::common_type<T, U>::type>>(lhs) /
         static_cast<typename std::common_type<T, U>::type>(rhs);
}

template <typename T>
DYND_CUDA_HOST_DEVICE complex<T> operator/(T lhs, complex<T> rhs)
{
  T denom = rhs.m_real * rhs.m_real + rhs.m_imag * rhs.m_imag;
  return complex<T>(lhs * rhs.m_real / denom, -lhs * rhs.m_imag / denom);
}

template <typename T, typename U>
DYND_CUDA_HOST_DEVICE typename std::enable_if<
    std::is_integral<T>::value || std::is_floating_point<T>::value,
    complex<typename std::common_type<T, U>::type>>::type
operator/(T lhs, complex<U> rhs)
{
  return static_cast<typename std::common_type<T, U>::type>(lhs) /
         static_cast<complex<typename std::common_type<T, U>::type>>(rhs);
}

template <typename T>
std::ostream &operator<<(std::ostream &out, const complex<T> &val)
{
  return (out << "(" << val.m_real << " + " << val.m_imag << "j)");
}

template <typename T>
DYND_CUDA_HOST_DEVICE complex<T> _i(); // complex<T>(0, 1)

template <>
DYND_CUDA_HOST_DEVICE inline complex<float> _i<float>()
{
  return complex<float>(0.0f, 1.0f);
}

template <>
DYND_CUDA_HOST_DEVICE inline complex<double> _i<double>()
{
  return complex<double>(0.0, 1.0);
}

template <typename T>
DYND_CUDA_HOST_DEVICE T abs(complex<T> z)
{
  return static_cast<T>(hypot(z.real(), z.imag()));
}

template <typename T>
DYND_CUDA_HOST_DEVICE T arg(complex<T> z)
{
  return atan2(z.imag(), z.real());
}

template <typename T>
DYND_CUDA_HOST_DEVICE complex<T> exp(complex<T> z)
{
  T x, c, s;
  T r = z.real(), i = z.imag();
  complex<T> ret;

  if (isfinite(r)) {
    x = exp(r);

    c = cos(i);
    s = sin(i);

    if (isfinite(i)) {
      ret = complex<T>(x * c, x * s);
    } else {
      ret = complex<T>(_nan<T>(NULL),
                            copysign(_nan<T>(NULL), i));
    }
  } else if (isnan(r)) {
    // r is nan
    if (i == 0) {
      ret = complex<T>(r, 0);
    } else {
      ret =
          complex<T>(r, copysign(_nan<T>(NULL), i));
    }
  } else {
    // r is +- inf
    if (r > 0) {
      if (i == 0) {
        ret = complex<T>(r, i);
      } else if (isfinite(i)) {
        c = cos(i);
        s = sin(i);

        ret = complex<T>(r * c, r * s);
      } else {
        // x = +inf, y = +-inf | nan
        ret = complex<T>(r, _nan<T>(NULL));
      }
    } else {
      if (isfinite(i)) {
        x = exp(r);
        c = cos(i);
        s = sin(i);

        ret = complex<T>(x * c, x * s);
      } else {
        // x = -inf, y = nan | +i inf
        ret = complex<T>(0, 0);
      }
    }
  }

  return ret;
}

template <typename T>
DYND_CUDA_HOST_DEVICE complex<T> log(complex<T> z)
{
  return complex<T>(log(abs(z)), arg(z));
}

template <typename T>
DYND_CUDA_HOST_DEVICE inline complex<T> sqrt(complex<T> z)
{
  using namespace std;
  // We risk spurious overflow for components >= DBL_MAX / (1 + sqrt(2))
  const T thresh =
      std::numeric_limits<T>::max() / (1 + ::sqrt(static_cast<T>(2)));

  complex<T> result;
  T a = z.real(), b = z.imag();
  T t;
  bool scale;

  // Handle special cases.
  if (a == 0 && b == 0) {
    return complex<T>(0, b);
  }
  if (isinf(b)) {
    return complex<T>(std::numeric_limits<T>::infinity(), b);
  }
  if (isnan(a)) {
    t = (b - b) / (b - b);        // raise invalid if b is not a NaN
    return complex<T>(a, t); // return NaN + NaN i
  }
  if (isinf(a)) {
    // csqrt(inf + NaN i) = inf + NaN i
    // csqrt(inf + y i) = inf + 0 i
    // csqrt(-inf + NaN i) = NaN +- inf i
    // csqrt(-inf + y i) = 0 + inf i
    if (signbit(a)) {
      return complex<T>(std::fabs(b - b), copysign(a, b));
    } else {
      return complex<T>(a, copysign(b - b, b));
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
    result = complex<T>(t, b / (2 * t));
  } else {
    t = std::sqrt((-a + hypot(a, b)) * 0.5);
    result = complex<T>(std::fabs(b) / (2 * t), copysign(t, b));
  }

  // Rescale
  if (scale) {
    return complex<T>(result.real() * 2, result.imag());
  } else {
    return result;
  }
}

template <typename T>
DYND_CUDA_HOST_DEVICE complex<T> pow(complex<T> x, complex<T> y)
{
  T yr = y.real(), yi = y.imag();

  complex<T> b = log(x);
  T br = b.real(), bi = b.imag();

  return exp(complex<T>(br * yr - bi * yi, br * yi + bi * yr));
}

template <typename T>
DYND_CUDA_HOST_DEVICE complex<T> pow(complex<T> x, T y)
{
  complex<T> b = log(x);
  T br = b.real(), bi = b.imag();

  return exp(complex<T>(br * y, bi * y));
}

template <typename T>
DYND_CUDA_HOST_DEVICE complex<T> cos(complex<T> z)
{
  T x = z.real(), y = z.imag();
  return complex<T>(cos(x) * cosh(y),
                         -(sin(x) * sinh(y)));
}

template <typename T>
DYND_CUDA_HOST_DEVICE complex<T> sin(complex<T> z)
{
  T x = z.real(), y = z.imag();
  return complex<T>(sin(x) * cosh(y),
                         cos(x) * sinh(y));
}

} // namespace dynd

namespace std {

template <typename T, typename U>
struct common_type<dynd::complex<T>, dynd::complex<U>> {
    typedef dynd::complex<typename std::common_type<T, U>::type> type;
};

template <typename T, typename U>
struct common_type<T, dynd::complex<U>> {
    typedef dynd::complex<typename std::common_type<T, U>::type> type;
};

template <typename T, typename U>
struct common_type<dynd::complex<T>, U> {
    typedef dynd::complex<typename std::common_type<T, U>::type> type;
};

}