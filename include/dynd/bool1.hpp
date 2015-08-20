//
// Copyright (C) 2011-15 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#pragma once

#include <iostream>

namespace dynd {

// A boolean class that is just one byte
class bool1 {
  char m_value;

public:
  DYND_CUDA_HOST_DEVICE bool1() = default;

  DYND_CUDA_HOST_DEVICE explicit bool1(bool value) : m_value(value)
  {
  }

  operator bool() const
  {
    return m_value ? true : false;
  }

  DYND_CUDA_HOST_DEVICE bool1 &operator=(bool value)
  {
    m_value = value;
    return *this;
  }

  DYND_CUDA_HOST_DEVICE bool1 &operator/=(bool1 rhs)
  {
    m_value /= rhs.m_value;
    return *this;
  }

  friend int operator/(bool1 lhs, bool1 rhs);
};

DYND_CUDA_HOST_DEVICE inline bool operator+(bool1 lhs, bool1 rhs)
{
  return static_cast<bool>(lhs) && static_cast<bool>(rhs);
}

DYND_CUDA_HOST_DEVICE inline int operator/(bool1 lhs, bool1 rhs)
{
  return lhs.m_value / rhs.m_value;
}

template <typename T>
DYND_CUDA_HOST_DEVICE typename std::enable_if<
    std::is_arithmetic<T>::value,
    typename std::common_type<bool, T>::type>::type
operator/(bool1 lhs, T rhs)
{
  return static_cast<typename std::common_type<bool, T>::type>(lhs) / rhs;
}

template <typename T>
DYND_CUDA_HOST_DEVICE typename std::enable_if<
    std::is_arithmetic<T>::value,
    typename std::common_type<bool, T>::type>::type
operator/(T lhs, bool1 rhs)
{
  return lhs / static_cast<typename std::common_type<T, bool>::type>(rhs);
}

DYND_CUDA_HOST_DEVICE inline bool operator<(bool1 lhs, bool1 rhs)
{
  return static_cast<bool>(lhs) < static_cast<bool>(rhs);
}

template <typename T>
DYND_CUDA_HOST_DEVICE typename std::enable_if<std::is_arithmetic<T>::value,
                                              bool>::type
operator<(bool1 lhs, T rhs)
{
  return static_cast<T>(lhs) < rhs;
}

template <typename T>
DYND_CUDA_HOST_DEVICE typename std::enable_if<std::is_arithmetic<T>::value,
                                              bool>::type
operator<(T lhs, bool1 rhs)
{
  return lhs < static_cast<T>(rhs);
}

DYND_CUDA_HOST_DEVICE inline bool operator<=(bool1 lhs, bool1 rhs)
{
  return static_cast<bool>(lhs) <= static_cast<bool>(rhs);
}

template <typename T>
DYND_CUDA_HOST_DEVICE typename std::enable_if<std::is_arithmetic<T>::value,
                                              bool>::type
operator<=(bool1 lhs, T rhs)
{
  return static_cast<T>(lhs) <= rhs;
}

template <typename T>
DYND_CUDA_HOST_DEVICE typename std::enable_if<std::is_arithmetic<T>::value,
                                              bool>::type
operator<=(T lhs, bool1 rhs)
{
  return lhs <= static_cast<T>(rhs);
}

DYND_CUDA_HOST_DEVICE inline bool operator==(bool1 lhs, bool1 rhs)
{
  return static_cast<bool>(lhs) == static_cast<bool>(rhs);
}

template <typename T>
DYND_CUDA_HOST_DEVICE typename std::enable_if<std::is_arithmetic<T>::value,
                                              bool>::type
operator==(bool1 lhs, T rhs)
{
  return static_cast<T>(lhs) == rhs;
}

template <typename T>
DYND_CUDA_HOST_DEVICE typename std::enable_if<std::is_arithmetic<T>::value,
                                              bool>::type
operator==(T lhs, bool1 rhs)
{
  return lhs == static_cast<T>(rhs);
}

DYND_CUDA_HOST_DEVICE inline bool operator!=(bool1 lhs, bool1 rhs)
{
  return static_cast<bool>(lhs) != static_cast<bool>(rhs);
}

template <typename T>
DYND_CUDA_HOST_DEVICE typename std::enable_if<std::is_arithmetic<T>::value,
                                              bool>::type
operator!=(bool1 lhs, T rhs)
{
  return static_cast<T>(lhs) != rhs;
}

template <typename T>
DYND_CUDA_HOST_DEVICE typename std::enable_if<std::is_arithmetic<T>::value,
                                              bool>::type
operator!=(T lhs, bool1 rhs)
{
  return lhs != static_cast<T>(rhs);
}

DYND_CUDA_HOST_DEVICE inline bool operator>=(bool1 lhs, bool1 rhs)
{
  return static_cast<bool>(lhs) >= static_cast<bool>(rhs);
}

template <typename T>
DYND_CUDA_HOST_DEVICE typename std::enable_if<std::is_arithmetic<T>::value,
                                              bool>::type
operator>=(bool1 lhs, T rhs)
{
  return static_cast<T>(lhs) >= rhs;
}

template <typename T>
DYND_CUDA_HOST_DEVICE typename std::enable_if<std::is_arithmetic<T>::value,
                                              bool>::type
operator>=(T lhs, bool1 rhs)
{
  return lhs >= static_cast<T>(rhs);
}

DYND_CUDA_HOST_DEVICE inline bool operator>(bool1 lhs, bool1 rhs)
{
  return static_cast<bool>(lhs) > static_cast<bool>(rhs);
}

template <typename T>
DYND_CUDA_HOST_DEVICE typename std::enable_if<std::is_arithmetic<T>::value,
                                              bool>::type
operator>(bool1 lhs, T rhs)
{
  return static_cast<T>(lhs) > rhs;
}

template <typename T>
DYND_CUDA_HOST_DEVICE typename std::enable_if<std::is_arithmetic<T>::value,
                                              bool>::type
operator>(T lhs, bool1 rhs)
{
  return lhs > static_cast<T>(rhs);
}

inline std::ostream &operator<<(std::ostream &o, const bool1 &DYND_UNUSED(rhs))
{
  return (o << "<bool1 printing unimplemented>");
}

} // namespace dynd
