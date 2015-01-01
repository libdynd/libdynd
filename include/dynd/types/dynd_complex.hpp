//
// Copyright (C) 2011-14 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#pragma once

#include <cmath>
#include <complex>
#include <limits>

#include <dynd/config.hpp>
#include <dynd/typed_data_assign.hpp>

namespace dynd {

template <typename T>
class dynd_complex {
public:
  T m_real, m_imag;
  typedef T value_type;

  DYND_CUDA_HOST_DEVICE dynd_complex(const T &re = 0.0, const T &im = 0.0)
      : m_real(re), m_imag(im)
  {
  }

  template <typename U>
  explicit DYND_CUDA_HOST_DEVICE dynd_complex(const dynd_complex<U> &rhs)
      : m_real(static_cast<T>(rhs.m_real)), m_imag(static_cast<T>(rhs.m_imag))
  {
  }

  explicit dynd_complex(const std::complex<T> &rhs)
      : m_real(rhs.real()), m_imag(rhs.imag())
  {
  }

  DYND_CUDA_HOST_DEVICE T real() const { return m_real; }
  DYND_CUDA_HOST_DEVICE T imag() const { return m_imag; }

  DYND_CUDA_HOST_DEVICE dynd_complex<T> &operator=(const dynd_complex<T> &rhs)
  {
    m_real = rhs.m_real;
    m_imag = rhs.m_imag;

    return *this;
  }

  DYND_CUDA_HOST_DEVICE dynd_complex<T> &operator+=(const dynd_complex<T> &rhs)
  {
    m_real += rhs.m_real;
    m_imag += rhs.m_imag;

    return *this;
  }

  DYND_CUDA_HOST_DEVICE dynd_complex<T> &operator+=(const T &rhs)
  {
    m_real += rhs;

    return *this;
  }

  DYND_CUDA_HOST_DEVICE dynd_complex<T> &operator-=(const dynd_complex<T> &rhs)
  {
    m_real -= rhs.m_real;
    m_imag -= rhs.m_imag;

    return *this;
  }

  DYND_CUDA_HOST_DEVICE dynd_complex<T> &operator-=(const T &rhs)
  {
    m_real -= rhs;

    return *this;
  }

  DYND_CUDA_HOST_DEVICE dynd_complex<T> operator*=(const dynd_complex<T> &rhs)
  {
    new (this) dynd_complex<T>(m_real * rhs.m_real - m_imag * rhs.m_imag,
                               m_real * rhs.m_imag + rhs.m_real * m_imag);

    return *this;
  }

  DYND_CUDA_HOST_DEVICE dynd_complex<T> operator*=(const T &rhs)
  {
    m_real *= rhs;
    m_imag *= rhs;

    return *this;
  }

  DYND_CUDA_HOST_DEVICE dynd_complex<T> &operator/=(const dynd_complex<T> &rhs)
  {
    T denom = rhs.m_real * rhs.m_real + rhs.m_imag * rhs.m_imag;
    new (this)
        dynd_complex<T>((m_real * rhs.m_real + m_imag * rhs.m_imag) / denom,
                        (rhs.m_real * m_imag - m_real * rhs.m_imag) / denom);

    return *this;
  }

  DYND_CUDA_HOST_DEVICE dynd_complex<T> &operator/=(const T &rhs)
  {
    m_real /= rhs;
    m_imag /= rhs;

    return *this;
  }

  operator std::complex<T>() const { return std::complex<T>(m_real, m_imag); }

  /*
    template <typename U>
    DYND_CUDA_HOST_DEVICE operator dynd_complex<U>() const
    {
      return dynd_complex<U>(m_real, m_imag);
    }
  */
};

template <typename T, typename U>
DYND_CUDA_HOST_DEVICE bool operator==(dynd_complex<T> lhs, dynd_complex<U> rhs)
{
  return (lhs.m_real == rhs.m_real) && (lhs.m_imag == rhs.m_imag);
}

template <typename T, typename U>
DYND_CUDA_HOST_DEVICE typename std::enable_if<
    std::is_integral<U>::value || std::is_floating_point<U>::value, bool>::type
operator==(dynd_complex<T> lhs, U rhs)
{
  return (lhs.m_real == rhs) && !lhs.m_imag;
}

template <typename T, typename U>
DYND_CUDA_HOST_DEVICE typename std::enable_if<
    std::is_integral<T>::value || std::is_floating_point<T>::value, bool>::type
operator==(T lhs, dynd_complex<U> rhs)
{
  return rhs == lhs;
}

template <typename T>
bool operator==(dynd_complex<T> lhs, std::complex<T> rhs)
{
  return (lhs.m_real == rhs.real()) && (lhs.m_imag == rhs.imag());
}

template <typename T>
bool operator==(std::complex<T> lhs, dynd_complex<T> rhs)
{
  return rhs == lhs;
}

template <typename T, typename U>
DYND_CUDA_HOST_DEVICE bool operator!=(dynd_complex<T> lhs, dynd_complex<U> rhs)
{
  return !(lhs == rhs);
}

template <typename T>
DYND_CUDA_HOST_DEVICE dynd_complex<T> operator-(const dynd_complex<T> &rhs)
{
  return dynd_complex<T>(-rhs.m_real, -rhs.m_imag);
}

template <typename T>
DYND_CUDA_HOST_DEVICE dynd_complex<T> operator+(dynd_complex<T> lhs,
                                                dynd_complex<T> rhs)
{
  return lhs += rhs;
}

template <typename T, typename U>
DYND_CUDA_HOST_DEVICE dynd_complex<typename std::common_type<T, U>::type>
operator+(dynd_complex<T> lhs, dynd_complex<U> rhs)
{
  return static_cast<dynd_complex<typename std::common_type<T, U>::type>>(lhs) +
         static_cast<dynd_complex<typename std::common_type<T, U>::type>>(rhs);
}

template <typename T>
DYND_CUDA_HOST_DEVICE dynd_complex<T> operator+(dynd_complex<T> lhs, T rhs)
{
  return lhs += rhs;
}

template <typename T, typename U>
DYND_CUDA_HOST_DEVICE typename std::enable_if<
    std::is_integral<U>::value || std::is_floating_point<U>::value,
    dynd_complex<typename std::common_type<T, U>::type>>::type
operator+(dynd_complex<T> lhs, U rhs)
{
  return static_cast<dynd_complex<typename std::common_type<T, U>::type>>(lhs) +
         static_cast<typename std::common_type<T, U>::type>(rhs);
}

template <typename T>
DYND_CUDA_HOST_DEVICE dynd_complex<T> operator+(T lhs, dynd_complex<T> rhs)
{
  return dynd_complex<T>(lhs + rhs.m_real, rhs.m_imag);
}

template <typename T, typename U>
DYND_CUDA_HOST_DEVICE typename std::enable_if<
    std::is_integral<T>::value || std::is_floating_point<T>::value,
    dynd_complex<typename std::common_type<T, U>::type>>::type
operator+(T lhs, dynd_complex<U> rhs)
{
  return static_cast<typename std::common_type<T, U>::type>(lhs) +
         static_cast<dynd_complex<typename std::common_type<T, U>::type>>(rhs);
}

template <typename T>
DYND_CUDA_HOST_DEVICE dynd_complex<T> operator-(dynd_complex<T> lhs,
                                                dynd_complex<T> rhs)
{
  return lhs -= rhs;
}

template <typename T, typename U>
DYND_CUDA_HOST_DEVICE dynd_complex<typename std::common_type<T, U>::type>
operator-(dynd_complex<T> lhs, dynd_complex<U> rhs)
{
  return static_cast<dynd_complex<typename std::common_type<T, U>::type>>(lhs) -
         static_cast<dynd_complex<typename std::common_type<T, U>::type>>(rhs);
}

template <typename T>
DYND_CUDA_HOST_DEVICE dynd_complex<T> operator-(dynd_complex<T> lhs, T rhs)
{
  return lhs -= rhs;
}

template <typename T, typename U>
DYND_CUDA_HOST_DEVICE typename std::enable_if<
    std::is_integral<U>::value || std::is_floating_point<U>::value,
    dynd_complex<typename std::common_type<T, U>::type>>::type
operator-(dynd_complex<T> lhs, U rhs)
{
  return static_cast<dynd_complex<typename std::common_type<T, U>::type>>(lhs) -
         static_cast<typename std::common_type<T, U>::type>(rhs);
}

template <typename T>
DYND_CUDA_HOST_DEVICE dynd_complex<T> operator-(T lhs, dynd_complex<T> rhs)
{
  return dynd_complex<T>(lhs - rhs.m_real, -rhs.m_imag);
}

template <typename T, typename U>
DYND_CUDA_HOST_DEVICE typename std::enable_if<
    std::is_integral<T>::value || std::is_floating_point<T>::value,
    dynd_complex<typename std::common_type<T, U>::type>>::type
operator-(T lhs, dynd_complex<U> rhs)
{
  return static_cast<typename std::common_type<T, U>::type>(lhs) -
         static_cast<dynd_complex<typename std::common_type<T, U>::type>>(rhs);
}

template <typename T>
DYND_CUDA_HOST_DEVICE dynd_complex<T> operator*(dynd_complex<T> lhs,
                                                dynd_complex<T> rhs)
{
  return lhs *= rhs;
}

template <typename T, typename U>
DYND_CUDA_HOST_DEVICE dynd_complex<typename std::common_type<T, U>::type>
operator*(dynd_complex<T> lhs, dynd_complex<U> rhs)
{
  return static_cast<dynd_complex<typename std::common_type<T, U>::type>>(lhs) *
         static_cast<dynd_complex<typename std::common_type<T, U>::type>>(rhs);
}

template <typename T>
DYND_CUDA_HOST_DEVICE dynd_complex<T> operator*(dynd_complex<T> lhs, T rhs)
{
  return lhs *= rhs;
}

template <typename T, typename U>
DYND_CUDA_HOST_DEVICE typename std::enable_if<
    std::is_integral<U>::value || std::is_floating_point<U>::value,
    dynd_complex<typename std::common_type<T, U>::type>>::type
operator*(dynd_complex<T> lhs, U rhs)
{
  return static_cast<dynd_complex<typename std::common_type<T, U>::type>>(lhs) *
         static_cast<typename std::common_type<T, U>::type>(rhs);
}

template <typename T>
DYND_CUDA_HOST_DEVICE dynd_complex<T> operator*(T lhs, dynd_complex<T> rhs)
{
  return dynd_complex<T>(lhs * rhs.m_real, lhs * rhs.m_imag);
}

template <typename T, typename U>
DYND_CUDA_HOST_DEVICE typename std::enable_if<
    std::is_integral<T>::value || std::is_floating_point<T>::value,
    dynd_complex<typename std::common_type<T, U>::type>>::type
operator*(T lhs, dynd_complex<U> rhs)
{
  return static_cast<typename std::common_type<T, U>::type>(lhs) *
         static_cast<dynd_complex<typename std::common_type<T, U>::type>>(rhs);
}

template <typename T>
DYND_CUDA_HOST_DEVICE dynd_complex<T> operator/(dynd_complex<T> lhs,
                                                dynd_complex<T> rhs)
{
  return lhs /= rhs;
}

template <typename T, typename U>
DYND_CUDA_HOST_DEVICE dynd_complex<typename std::common_type<T, U>::type>
operator/(dynd_complex<T> lhs, dynd_complex<U> rhs)
{
  return static_cast<dynd_complex<typename std::common_type<T, U>::type>>(lhs) /
         static_cast<dynd_complex<typename std::common_type<T, U>::type>>(rhs);
}

template <typename T>
DYND_CUDA_HOST_DEVICE dynd_complex<T> operator/(dynd_complex<T> lhs, T rhs)
{
  return lhs /= rhs;
}

template <typename T, typename U>
DYND_CUDA_HOST_DEVICE typename std::enable_if<
    std::is_integral<U>::value || std::is_floating_point<U>::value,
    dynd_complex<typename std::common_type<T, U>::type>>::type
operator/(dynd_complex<T> lhs, U rhs)
{
  return static_cast<dynd_complex<typename std::common_type<T, U>::type>>(lhs) /
         static_cast<typename std::common_type<T, U>::type>(rhs);
}

template <typename T>
DYND_CUDA_HOST_DEVICE dynd_complex<T> operator/(T lhs, dynd_complex<T> rhs)
{
  T denom = rhs.m_real * rhs.m_real + rhs.m_imag * rhs.m_imag;
  return dynd_complex<T>(lhs * rhs.m_real / denom, -lhs * rhs.m_imag / denom);
}

template <typename T, typename U>
DYND_CUDA_HOST_DEVICE typename std::enable_if<
    std::is_integral<T>::value || std::is_floating_point<T>::value,
    dynd_complex<typename std::common_type<T, U>::type>>::type
operator/(T lhs, dynd_complex<U> rhs)
{
  return static_cast<typename std::common_type<T, U>::type>(lhs) /
         static_cast<dynd_complex<typename std::common_type<T, U>::type>>(rhs);
}

template <typename T>
std::ostream &operator<<(std::ostream &out, const dynd_complex<T> &val)
{
  return (out << "(" << val.m_real << " + " << val.m_imag << "j)");
}

template <typename T>
T abs(dynd_complex<T> z)
{
  return static_cast<T>(hypot(z.real(), z.imag()));
}

template <typename T>
T arg(dynd_complex<T> z)
{
  return atan2(z.imag(), z.real());
}

template <typename T>
inline dynd_complex<T> exp(dynd_complex<T> z)
{
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
      ret = dynd_complex<T>(std::numeric_limits<T>::quiet_NaN(),
                            copysign(std::numeric_limits<T>::quiet_NaN(), i));
    }
  } else if (DYND_ISNAN(r)) {
    // r is nan
    if (i == 0) {
      ret = dynd_complex<T>(r, 0);
    } else {
      ret =
          dynd_complex<T>(r, copysign(std::numeric_limits<T>::quiet_NaN(), i));
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
dynd_complex<T> log(dynd_complex<T> z)
{
  return dynd_complex<T>(std::log(abs(z)), arg(z));
}

template <typename T>
inline dynd_complex<T> sqrt(dynd_complex<T> z)
{
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
    t = (b - b) / (b - b);        // raise invalid if b is not a NaN
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
dynd_complex<T> pow(dynd_complex<T> x, dynd_complex<T> y)
{
  T yr = y.real(), yi = y.imag();

  dynd_complex<T> b = log(x);
  T br = b.real(), bi = b.imag();

  return exp(dynd_complex<T>(br * yr - bi * yi, br * yi + bi * yr));
}

template <typename T>
dynd_complex<T> pow(dynd_complex<T> x, T y)
{
  dynd_complex<T> b = log(x);
  T br = b.real(), bi = b.imag();

  return exp(dynd_complex<T>(br * y, bi * y));
}

template <typename T>
dynd_complex<T> cos(dynd_complex<T> z)
{
  T x = z.real(), y = z.imag();
  return dynd_complex<T>(std::cos(x) * std::cosh(y),
                         -(std::sin(x) * std::sinh(y)));
}

template <typename T>
dynd_complex<T> sin(dynd_complex<T> z)
{
  T x = z.real(), y = z.imag();
  return dynd_complex<T>(std::sin(x) * std::cosh(y),
                         std::cos(x) * std::sinh(y));
}

} // namespace dynd

namespace std {

template <typename T, typename U>
struct common_type<dynd::dynd_complex<T>, dynd::dynd_complex<U>> {
    typedef dynd::dynd_complex<typename std::common_type<T, U>::type> type;
};

template <typename T, typename U>
struct common_type<T, dynd::dynd_complex<U>> {
    typedef dynd::dynd_complex<typename std::common_type<T, U>::type> type;
};

template <typename T, typename U>
struct common_type<dynd::dynd_complex<T>, U> {
    typedef dynd::dynd_complex<typename std::common_type<T, U>::type> type;
};

}