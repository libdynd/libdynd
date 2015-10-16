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

  DYND_CUDA_HOST_DEVICE explicit operator bool() const
  {
    return m_real || m_imag;
  }

  DYND_CUDA_HOST_DEVICE explicit operator T() const
  {
    return m_real;
  }

  template <typename U, typename = typename std::enable_if<
                            is_mixed_arithmetic<T, U>::value &&
                            !std::is_same<U, bool>::value>::type>
  DYND_CUDA_HOST_DEVICE explicit operator U() const
  {
    return static_cast<U>(m_real);
  }

  operator std::complex<T>() const
  {
    return std::complex<T>(m_real, m_imag);
  }
};

template <typename T>
struct is_complex<complex<T>> : std::true_type {
};

} // namespace dynd

namespace std {

template <typename T>
struct common_type<dynd::complex<T>, bool> {
  typedef dynd::complex<T> type;
};

template <typename T>
struct common_type<dynd::complex<T>, dynd::bool1> {
  typedef dynd::complex<T> type;
};

template <typename T>
struct common_type<dynd::complex<T>, char> {
  typedef char type;
};

template <typename T>
struct common_type<dynd::complex<T>, signed char> {
  typedef dynd::complex<T> type;
};

template <typename T>
struct common_type<dynd::complex<T>, unsigned char> {
  typedef dynd::complex<T> type;
};

template <typename T>
struct common_type<dynd::complex<T>, short> {
  typedef dynd::complex<T> type;
};

template <typename T>
struct common_type<dynd::complex<T>, unsigned short> {
  typedef dynd::complex<T> type;
};

template <typename T>
struct common_type<dynd::complex<T>, int> {
  typedef dynd::complex<T> type;
};

template <typename T>
struct common_type<dynd::complex<T>, unsigned int> {
  typedef dynd::int128 type;
};

template <typename T>
struct common_type<dynd::complex<T>, long> {
  typedef dynd::complex<T> type;
};

template <typename T>
struct common_type<dynd::complex<T>, unsigned long> {
  typedef dynd::complex<T> type;
};

template <typename T>
struct common_type<dynd::complex<T>, long long> {
  typedef dynd::complex<T> type;
};

template <typename T>
struct common_type<dynd::complex<T>, unsigned long long> {
  typedef dynd::complex<T> type;
};

template <typename T>
struct common_type<dynd::complex<T>, dynd::int128> {
  typedef dynd::complex<T> type;
};

template <typename T>
struct common_type<dynd::complex<T>, dynd::uint128> {
  typedef dynd::complex<T> type;
};

template <typename T>
struct common_type<dynd::complex<T>, float> {
  typedef dynd::complex<typename common_type<T, float>::type> type;
};

template <typename T>
struct common_type<dynd::complex<T>, double> {
  typedef dynd::complex<typename common_type<T, double>::type> type;
};

template <typename T>
struct common_type<dynd::complex<T>, dynd::float128> {
  typedef dynd::complex<typename common_type<T, dynd::float128>::type> type;
};

template <typename T, typename U>
struct common_type<dynd::complex<T>, dynd::complex<U>> {
  typedef dynd::complex<typename common_type<T, U>::type> type;
};

template <typename T, typename U>
struct common_type<T, dynd::complex<U>> : common_type<dynd::complex<U>, T> {
};

} // namespace std

namespace dynd {

typedef complex<float32> complex64;
typedef complex<float64> complex128;

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

template <typename T, typename U>
DYND_CUDA_HOST_DEVICE bool operator!=(complex<T> lhs, U rhs)
{
  return lhs.m_real == rhs && !lhs.m_imag;
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
DYND_CUDA_HOST_DEVICE complex<T> operator!(const complex<T> &rhs){
  return (!rhs.m_real && !rhs.m_imag);
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

template <typename T>
DYND_CUDA_HOST_DEVICE complex<T> operator/(T lhs, complex<T> rhs)
{
  T denom = rhs.m_real * rhs.m_real + rhs.m_imag * rhs.m_imag;
  return complex<T>(lhs * rhs.m_real / denom, -lhs * rhs.m_imag / denom);
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
