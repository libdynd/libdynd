//
// Copyright (C) 2011-15 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#pragma once

#include <cmath>
#include <complex>
#include <limits>

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

  complex(const std::complex<T> &rhs) : m_real(rhs.real()), m_imag(rhs.imag())
  {
  }

  DYND_CUDA_HOST_DEVICE T real() const
  {
    return m_real;
  }
  DYND_CUDA_HOST_DEVICE T imag() const
  {
    return m_imag;
  }

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
    new (this) complex<T>((m_real * rhs.m_real + m_imag * rhs.m_imag) / denom,
                          (rhs.m_real * m_imag - m_real * rhs.m_imag) / denom);

    return *this;
  }

  DYND_CUDA_HOST_DEVICE complex<T> &operator/=(const T &rhs)
  {
    m_real /= rhs;
    m_imag /= rhs;

    return *this;
  }

  operator std::complex<T>() const
  {
    return std::complex<T>(m_real, m_imag);
  }

  template <typename U,
            typename = std::enable_if<std::is_integral<T>::value ||
                                      std::is_floating_point<T>::value>>
  explicit operator U() const
  {
    return static_cast<U>(m_real);
  }
};

typedef complex<float32> complex64;
typedef complex<float64> complex128;

template <typename T>
DYND_CUDA_HOST_DEVICE complex<T> operator/(bool1 lhs, complex<T> rhs)
{
  return static_cast<complex<T>>(lhs) / rhs;
}

template <typename T>
DYND_CUDA_HOST_DEVICE complex<T> operator/(int128 lhs, complex<T> rhs)
{
  return static_cast<complex<T>>(lhs) / rhs;
}

template <typename T>
DYND_CUDA_HOST_DEVICE complex<T> operator/(complex<T> lhs, bool1 rhs)
{
  return lhs / static_cast<complex<T>>(rhs);
}

template <typename T>
DYND_CUDA_HOST_DEVICE complex<T> operator/(complex<T> lhs, int128 rhs)
{
  return lhs / static_cast<complex<T>>(rhs);
}

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
DYND_CUDA_HOST_DEVICE complex<T> operator+(complex<T> lhs, complex<T> rhs)
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
DYND_CUDA_HOST_DEVICE complex<T> operator-(complex<T> lhs, complex<T> rhs)
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
DYND_CUDA_HOST_DEVICE complex<T> operator*(complex<T> lhs, complex<T> rhs)
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
DYND_CUDA_HOST_DEVICE complex<T> operator/(complex<T> lhs, complex<T> rhs)
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
