//
// Copyright (C) 2011-15 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#pragma once

namespace dynd {

// A boolean class that is just one byte
class bool1 {
  char m_value;

public:
  DYND_CUDA_HOST_DEVICE bool1() = default;

  DYND_CUDA_HOST_DEVICE explicit bool1(bool value) : m_value(value) {}

  // Special case complex conversion to avoid ambiguous overload
  /*
    template <class T>
    DYND_CUDA_HOST_DEVICE explicit bool1(complex<T> value)
        : m_value(value != complex<T>(0))
    {
    }
  */

  operator bool() const { return static_cast<bool>(m_value); }

  /*
    template <typename T,
              typename = typename std::enable_if<is_arithmetic<T>::value>::type>
    DYND_CUDA_HOST_DEVICE explicit operator T() const
    {
      return static_cast<T>(static_cast<bool>(m_value));
    }
  */

  DYND_CUDA_HOST_DEVICE bool1 &operator=(bool value)
  {
    m_value = value;
    return *this;
  }

  /*

    DYND_CUDA_HOST_DEVICE bool1 &operator=(bool1 value) {
      m_value = value.m_value;
      return *this;
    }
  */
};

DYND_CUDA_HOST_DEVICE inline bool operator<(bool1 lhs, bool1 rhs)
{
  return static_cast<bool>(lhs) < static_cast<bool>(rhs);
}

template <typename T>
DYND_CUDA_HOST_DEVICE typename std::enable_if<std::is_arithmetic<T>::value,
                                              bool>::type
operator<(bool1 lhs, T rhs)
{
  return static_cast<typename std::common_type<bool1, T>::type>(lhs) < rhs;
}

template <typename T>
DYND_CUDA_HOST_DEVICE typename std::enable_if<std::is_arithmetic<T>::value,
                                              bool>::type
operator<(T lhs, bool1 rhs)
{
  return lhs < static_cast<bool>(rhs);
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
  return static_cast<bool>(lhs) <= rhs;
}

template <typename T>
DYND_CUDA_HOST_DEVICE typename std::enable_if<std::is_arithmetic<T>::value,
                                              bool>::type
operator<=(T lhs, bool1 rhs)
{
  return lhs <= static_cast<bool>(rhs);
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
  return static_cast<bool>(lhs) == rhs;
}

template <typename T>
DYND_CUDA_HOST_DEVICE typename std::enable_if<std::is_arithmetic<T>::value,
                                              bool>::type
operator==(T lhs, bool1 rhs)
{
  return lhs == static_cast<bool>(rhs);
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
  return static_cast<bool>(lhs) != rhs;
}

template <typename T>
DYND_CUDA_HOST_DEVICE typename std::enable_if<std::is_arithmetic<T>::value,
                                              bool>::type
operator!=(T lhs, bool1 rhs)
{
  return lhs != static_cast<bool>(rhs);
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
  return static_cast<bool>(lhs) >= rhs;
}

template <typename T>
DYND_CUDA_HOST_DEVICE typename std::enable_if<std::is_arithmetic<T>::value,
                                              bool>::type
operator>=(T lhs, bool1 rhs)
{
  return lhs >= static_cast<bool>(rhs);
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
  return static_cast<bool>(lhs) > rhs;
}

template <typename T>
DYND_CUDA_HOST_DEVICE typename std::enable_if<std::is_arithmetic<T>::value,
                                              bool>::type
operator>(T lhs, bool1 rhs)
{
  return lhs > static_cast<bool>(rhs);
}

} // namespace dynd

namespace std {

} // namespace std