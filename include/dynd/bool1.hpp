//
// Copyright (C) 2011-15 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#pragma once

#include <iostream>
#include <dynd/visibility.hpp>

namespace dynd {

// A boolean class that is just one byte
class DYND_API bool1 {
  char m_value;

public:
  DYND_CUDA_HOST_DEVICE bool1() = default;

  DYND_CUDA_HOST_DEVICE explicit bool1(bool value) : m_value(value)
  {
  }

  operator bool() const
  {
    return m_value != 0;
  }

  DYND_CUDA_HOST_DEVICE bool1 &operator=(bool rhs)
  {
    m_value = rhs;
    return *this;
  }

  DYND_CUDA_HOST_DEVICE bool1 operator+() const
  {
    return *this;
  }

  DYND_CUDA_HOST_DEVICE bool1 operator-() const
  {
    return *this;
  }

  DYND_CUDA_HOST_DEVICE bool1 operator!() const {
    return bool1(m_value == 0);
  }

  DYND_CUDA_HOST_DEVICE bool1 operator~() const {
    return bool1(m_value == 0);
  }

  DYND_CUDA_HOST_DEVICE bool1 operator&&(bool1 &rhs) {
    return bool1(m_value && rhs.m_value);
  }

  DYND_CUDA_HOST_DEVICE bool1 operator||(bool1 &rhs) {
    return bool1(m_value || rhs.m_value);
  }

  DYND_CUDA_HOST_DEVICE bool1 &operator+=(bool1 rhs)
  {
    m_value += rhs.m_value;
    return *this;
  }

  DYND_CUDA_HOST_DEVICE bool1 &operator-=(bool1 rhs)
  {
    m_value -= rhs.m_value;
    return *this;
  }

  DYND_CUDA_HOST_DEVICE bool1 &operator*=(bool1 rhs)
  {
    m_value *= rhs.m_value;
    return *this;
  }

  DYND_CUDA_HOST_DEVICE bool1 &operator/=(bool1 rhs)
  {
    m_value /= rhs.m_value;
    return *this;
  }

  friend int operator+(bool1 lhs, bool1 rhs);
  friend int operator-(bool1 lhs, bool1 rhs);
  friend int operator*(bool1 lhs, bool1 rhs);
  friend int operator/(bool1 lhs, bool1 rhs);
};

template <>
struct is_integral<bool1> : std::true_type {
};

DYND_CUDA_HOST_DEVICE inline int operator+(bool1 lhs, bool1 rhs)
{
  return lhs.m_value + rhs.m_value;
}

DYND_CUDA_HOST_DEVICE inline int operator-(bool1 lhs, bool1 rhs)
{
  return lhs.m_value - rhs.m_value;
}

DYND_CUDA_HOST_DEVICE inline int operator*(bool1 lhs, bool1 rhs)
{
  return lhs.m_value * rhs.m_value;
}

DYND_CUDA_HOST_DEVICE inline int operator/(bool1 lhs, bool1 rhs)
{
  return lhs.m_value / rhs.m_value;
}

} // namespace dynd

namespace std {

template <>
struct common_type<dynd::bool1, bool> : common_type<char, bool> {
};

template <>
struct common_type<dynd::bool1, dynd::bool1> {
  typedef dynd::bool1 type;
};

template <>
struct common_type<dynd::bool1, char> : common_type<char, char> {
};

template <>
struct common_type<dynd::bool1, signed char> : common_type<char, signed char> {
};

template <>
struct common_type<dynd::bool1, unsigned char> : common_type<char, unsigned char> {
};

template <>
struct common_type<dynd::bool1, short> : common_type<char, short> {
};

template <>
struct common_type<dynd::bool1, unsigned short> : common_type<char, unsigned short> {
};

template <>
struct common_type<dynd::bool1, int> : common_type<char, int> {
};

template <>
struct common_type<dynd::bool1, unsigned int> : common_type<char, unsigned int> {
};

template <>
struct common_type<dynd::bool1, long> : common_type<char, long> {
};

template <>
struct common_type<dynd::bool1, unsigned long> : common_type<char, unsigned long> {
};

template <>
struct common_type<dynd::bool1, long long> : common_type<char, long long> {
};

template <>
struct common_type<dynd::bool1, unsigned long long> : common_type<char, unsigned long long> {
};

template <>
struct common_type<dynd::bool1, float> : common_type<char, float> {
};

template <>
struct common_type<dynd::bool1, double> : common_type<char, double> {
};

template <typename T>
struct common_type<T, dynd::bool1> : common_type<dynd::bool1, T> {
};

} // namespace std

namespace dynd {

DYND_CUDA_HOST_DEVICE inline bool operator<(bool1 lhs, bool1 rhs)
{
  return static_cast<bool>(lhs) < static_cast<bool>(rhs);
}

template <typename T>
DYND_CUDA_HOST_DEVICE typename std::enable_if<std::is_arithmetic<T>::value, bool>::type operator<(bool1 lhs, T rhs)
{
  return static_cast<T>(lhs) < rhs;
}

template <typename T>
DYND_CUDA_HOST_DEVICE typename std::enable_if<std::is_arithmetic<T>::value, bool>::type operator<(T lhs, bool1 rhs)
{
  return lhs < static_cast<T>(rhs);
}

DYND_CUDA_HOST_DEVICE inline bool operator<=(bool1 lhs, bool1 rhs)
{
  return static_cast<bool>(lhs) <= static_cast<bool>(rhs);
}

template <typename T>
DYND_CUDA_HOST_DEVICE typename std::enable_if<std::is_arithmetic<T>::value, bool>::type operator<=(bool1 lhs, T rhs)
{
  return static_cast<T>(lhs) <= rhs;
}

template <typename T>
DYND_CUDA_HOST_DEVICE typename std::enable_if<std::is_arithmetic<T>::value, bool>::type operator<=(T lhs, bool1 rhs)
{
  return lhs <= static_cast<T>(rhs);
}

DYND_CUDA_HOST_DEVICE inline bool operator==(bool1 lhs, bool1 rhs)
{
  return static_cast<bool>(lhs) == static_cast<bool>(rhs);
}

template <typename T>
DYND_CUDA_HOST_DEVICE typename std::enable_if<std::is_arithmetic<T>::value, bool>::type operator==(bool1 lhs, T rhs)
{
  return static_cast<T>(lhs) == rhs;
}

template <typename T>
DYND_CUDA_HOST_DEVICE typename std::enable_if<std::is_arithmetic<T>::value, bool>::type operator==(T lhs, bool1 rhs)
{
  return lhs == static_cast<T>(rhs);
}

DYND_CUDA_HOST_DEVICE inline bool operator!=(bool1 lhs, bool1 rhs)
{
  return static_cast<bool>(lhs) != static_cast<bool>(rhs);
}

template <typename T>
DYND_CUDA_HOST_DEVICE typename std::enable_if<std::is_arithmetic<T>::value, bool>::type operator!=(bool1 lhs, T rhs)
{
  return static_cast<T>(lhs) != rhs;
}

template <typename T>
DYND_CUDA_HOST_DEVICE typename std::enable_if<std::is_arithmetic<T>::value, bool>::type operator!=(T lhs, bool1 rhs)
{
  return lhs != static_cast<T>(rhs);
}

DYND_CUDA_HOST_DEVICE inline bool operator>=(bool1 lhs, bool1 rhs)
{
  return static_cast<bool>(lhs) >= static_cast<bool>(rhs);
}

template <typename T>
DYND_CUDA_HOST_DEVICE typename std::enable_if<std::is_arithmetic<T>::value, bool>::type operator>=(bool1 lhs, T rhs)
{
  return static_cast<T>(lhs) >= rhs;
}

template <typename T>
DYND_CUDA_HOST_DEVICE typename std::enable_if<std::is_arithmetic<T>::value, bool>::type operator>=(T lhs, bool1 rhs)
{
  return lhs >= static_cast<T>(rhs);
}

DYND_CUDA_HOST_DEVICE inline bool operator>(bool1 lhs, bool1 rhs)
{
  return static_cast<bool>(lhs) > static_cast<bool>(rhs);
}

template <typename T>
DYND_CUDA_HOST_DEVICE typename std::enable_if<std::is_arithmetic<T>::value, bool>::type operator>(bool1 lhs, T rhs)
{
  return static_cast<T>(lhs) > rhs;
}

template <typename T>
DYND_CUDA_HOST_DEVICE typename std::enable_if<std::is_arithmetic<T>::value, bool>::type operator>(T lhs, bool1 rhs)
{
  return lhs > static_cast<T>(rhs);
}

inline std::ostream &operator<<(std::ostream &o, const bool1 &rhs)
{
  return o << static_cast<bool>(rhs);
}

} // namespace dynd
