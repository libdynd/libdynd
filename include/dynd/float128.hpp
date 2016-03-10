//
// Copyright (C) 2011-15 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#pragma once

#include <iostream>
#include <stdexcept>

#if !defined(DYND_HAS_FLOAT128)

namespace dynd {

class DYNDT_API float128 {
public:
#if defined(DYND_BIG_ENDIAN)
  uint64_t m_hi, m_lo;
#else
  uint64_t m_lo, m_hi;
#endif
  float128() {}
  float128(uint64_t hi, uint64_t lo) : m_lo(lo), m_hi(hi) {}

  float128(bool1) {}

  float128(signed char value);
  float128(unsigned char value);
  float128(short value);
  float128(unsigned short value);
  float128(int value);
  float128(unsigned int value);
  float128(long value) { *this = float128((long long)value); }
  float128(unsigned long value) { *this = float128((unsigned long long)value); }
  float128(long long value);
  float128(unsigned long long value);
  float128(double value);
  float128(const int128 &value);
  float128(const uint128 &value);
  float128(const float16 &value);

  float128 &operator/=(float128 DYND_UNUSED(rhs))
  {
    throw std::runtime_error("operator/= is not implemented for float128");
  }

  operator signed char() const
  {
#ifdef __CUDA_ARCH__
    DYND_TRIGGER_ASSERT_RETURN_ZERO("float128 conversions are not completed");
#else
    throw std::runtime_error("float128 conversions are not completed");
#endif
  }
  operator unsigned char() const
  {
#ifdef __CUDA_ARCH__
    DYND_TRIGGER_ASSERT_RETURN_ZERO("float128 conversions are not completed");
#else
    throw std::runtime_error("float128 conversions are not completed");
#endif
  }
  operator short() const
  {
#ifdef __CUDA_ARCH__
    DYND_TRIGGER_ASSERT_RETURN_ZERO("float128 conversions are not completed");
#else
    throw std::runtime_error("float128 conversions are not completed");
#endif
  }
  operator unsigned short() const
  {
#ifdef __CUDA_ARCH__
    DYND_TRIGGER_ASSERT_RETURN_ZERO("float128 conversions are not completed");
#else
    throw std::runtime_error("float128 conversions are not completed");
#endif
  }
  operator int() const
  {
#ifdef __CUDA_ARCH__
    DYND_TRIGGER_ASSERT_RETURN_ZERO("float128 conversions are not completed");
#else
    throw std::runtime_error("float128 conversions are not completed");
#endif
  }
  operator unsigned int() const
  {
#ifdef __CUDA_ARCH__
    DYND_TRIGGER_ASSERT_RETURN_ZERO("float128 conversions are not completed");
#else
    throw std::runtime_error("float128 conversions are not completed");
#endif
  }
  operator long() const
  {
#ifdef __CUDA_ARCH__
    DYND_TRIGGER_ASSERT_RETURN_ZERO("float128 conversions are not completed");
#else
    throw std::runtime_error("float128 conversions are not completed");
#endif
  }
  operator unsigned long() const
  {
#ifdef __CUDA_ARCH__
    DYND_TRIGGER_ASSERT_RETURN_ZERO("float128 conversions are not completed");
#else
    throw std::runtime_error("float128 conversions are not completed");
#endif
  }
  operator long long() const
  {
#ifdef __CUDA_ARCH__
    DYND_TRIGGER_ASSERT_RETURN_ZERO("float128 conversions are not completed");
#else
    throw std::runtime_error("float128 conversions are not completed");
#endif
  }
  operator unsigned long long() const
  {
#ifdef __CUDA_ARCH__
    DYND_TRIGGER_ASSERT_RETURN_ZERO("float128 conversions are not completed");
#else
    throw std::runtime_error("float128 conversions are not completed");
#endif
  }

  operator float() const
  {
#ifdef __CUDA_ARCH__
    DYND_TRIGGER_ASSERT_RETURN_ZERO("float128 conversions are not completed");
#else
    throw std::runtime_error("float128 conversions are not completed");
#endif
  }

  operator double() const
  {
#ifdef __CUDA_ARCH__
    DYND_TRIGGER_ASSERT_RETURN_ZERO("float128 conversions are not completed");
#else
    throw std::runtime_error("float128 conversions are not completed");
#endif
  }

  explicit float128(const bool &rhs) : m_lo(0ULL), m_hi(rhs ? 0x3fff000000000000ULL : 0ULL) {}

  bool iszero() const { return (m_hi & 0x7fffffffffffffffULL) == 0 && m_lo == 0; }

  bool signbit_() const { return (m_hi & 0x8000000000000000ULL) != 0; }

  bool isnan_() const
  {
    return (m_hi & 0x7fff000000000000ULL) == 0x7fff000000000000ULL &&
           ((m_hi & 0x0000ffffffffffffULL) != 0ULL || m_lo != 0ULL);
  }

  bool isinf_() const { return (m_hi & 0x7fffffffffffffffULL) == 0x7fff000000000000ULL && (m_lo == 0ULL); }

  bool isfinite_() const { return (m_hi & 0x7fff000000000000ULL) != 0x7fff000000000000ULL; }

  /*

       bool operator==(const float128 &rhs) const
    {
      // The equality cases are as follows:
      //   - If either value is NaN, never equal.
      //   - If the values are equal, equal.
      //   - If the values are both signed zeros, equal.
      return (!isnan_() && !rhs.isnan_()) &&
             ((m_hi == rhs.m_hi && m_lo == rhs.m_lo) ||
              (((m_hi | rhs.m_hi) & 0x7fffffffffffffffULL) == 0ULL &&
               (m_lo | rhs.m_lo) == 0ULL));
    }

       bool operator!=(const float128 &rhs) const
    {
      return !operator==(rhs);
    }

  */

  bool less_nonan(const float128 &rhs) const
  {
    if (signbit_()) {
      if (rhs.signbit_()) {
        return m_hi > rhs.m_hi || (m_hi == rhs.m_hi && m_lo > rhs.m_lo);
      }
      else {
        // Signed zeros are equal, have to check for it
        return (m_hi != 0x8000000000000000ULL) || (m_lo != 0LL) || (rhs.m_hi != 0LL) || rhs.m_lo != 0LL;
      }
    }
    else {
      if (rhs.signbit_()) {
        return false;
      }
      else {
        return m_hi < rhs.m_hi || (m_hi == rhs.m_hi && m_lo < rhs.m_lo);
      }
    }
  }

  bool less_equal_nonan(const float128 &rhs) const
  {
    if (signbit_()) {
      if (rhs.signbit_()) {
        return m_hi > rhs.m_hi || (m_hi == rhs.m_hi && m_lo >= rhs.m_lo);
      }
      else {
        return true;
      }
    }
    else {
      if (rhs.signbit_()) {
        // Signed zeros are equal, have to check for it
        return (m_hi == 0x8000000000000000ULL) && (m_lo == 0LL) && (rhs.m_hi == 0LL) && rhs.m_lo == 0LL;
      }
      else {
        return m_hi < rhs.m_hi || (m_hi == rhs.m_hi && m_lo <= rhs.m_lo);
      }
    }
  }

  /*

       bool operator<(const float128 &rhs) const
    {
      return !isnan_() && !rhs.isnan_() && less_nonan(rhs);
    }

       bool operator>(const float128 &rhs) const
    {
      return rhs.operator<(*this);
    }

       bool operator<=(const float128 &rhs) const
    {
      return !isnan_() && !rhs.isnan_() && less_equal_nonan(rhs);
    }

       bool operator>=(const float128 &rhs) const
    {
      return rhs.operator<=(*this);
    }

  */

  float128 operator+() const { return *this; }

  float128 operator-() const { return float128(-static_cast<double>(*this)); }

  bool operator!() const { return ((0x7fffffffffffffffULL | m_hi) == 0) && (m_lo == 0); }

  explicit operator bool() const { return (m_lo != 0) || ((0x7fffffffffffffffULL | m_hi) != 0); }
};

template <>
struct is_floating_point<float128> : std::true_type {
};

inline float128 operator+(const float128 &DYND_UNUSED(lhs), const float128 &DYND_UNUSED(rhs))
{
  throw std::runtime_error("addition for float128 is not implemented");
}

inline float128 operator*(float128 DYND_UNUSED(lhs), float128 DYND_UNUSED(rhs))
{
  throw std::runtime_error("operator* for float128 is not implemented");
}

inline float128 operator/(float128 DYND_UNUSED(lhs), float128 DYND_UNUSED(rhs))
{
  throw std::runtime_error("operator/ for float128 is not implemented");
}

inline bool operator<(const float128 &lhs, const float128 &rhs)
{
  return static_cast<double>(lhs) < static_cast<double>(rhs);
}

template <typename T>
typename std::enable_if<is_arithmetic<T>::value, bool>::type operator<(const float128 &lhs, const T &rhs)
{
  return static_cast<double>(lhs) < static_cast<double>(rhs);
}

template <typename T>
typename std::enable_if<is_arithmetic<T>::value, bool>::type operator<(const T &lhs, const float128 &rhs)
{
  return static_cast<double>(lhs) < static_cast<double>(rhs);
}

inline bool operator<=(const float128 &lhs, const float128 &rhs)
{
  return static_cast<double>(lhs) <= static_cast<double>(rhs);
}

template <typename T>
typename std::enable_if<is_arithmetic<T>::value, bool>::type operator<=(const float128 &lhs, const T &rhs)
{
  return static_cast<double>(lhs) <= static_cast<double>(rhs);
}

template <typename T>
typename std::enable_if<is_arithmetic<T>::value, bool>::type operator<=(const T &lhs, const float128 &rhs)
{
  return static_cast<double>(lhs) <= static_cast<double>(rhs);
}

inline bool operator==(const float128 &lhs, const float128 &rhs)
{
  return static_cast<double>(lhs) == static_cast<double>(rhs);
}

template <typename T>
typename std::enable_if<is_arithmetic<T>::value, bool>::type operator==(const float128 &lhs, const T &rhs)
{
  return static_cast<double>(lhs) == static_cast<double>(rhs);
}

template <typename T>
typename std::enable_if<is_arithmetic<T>::value, bool>::type operator==(const T &lhs, const float128 &rhs)
{
  return static_cast<double>(lhs) == static_cast<double>(rhs);
}

inline bool operator!=(const float128 &lhs, const float128 &rhs)
{
  return static_cast<double>(lhs) != static_cast<double>(rhs);
}

template <typename T>
typename std::enable_if<is_arithmetic<T>::value, bool>::type operator!=(const float128 &lhs, const T &rhs)
{
  return static_cast<double>(lhs) != static_cast<double>(rhs);
}

template <typename T>
typename std::enable_if<is_arithmetic<T>::value, bool>::type operator!=(const T &lhs, const float128 &rhs)
{
  return static_cast<double>(lhs) != static_cast<double>(rhs);
}

inline bool operator>=(const float128 &lhs, const float128 &rhs)
{
  return static_cast<double>(lhs) >= static_cast<double>(rhs);
}

template <typename T>
typename std::enable_if<is_arithmetic<T>::value, bool>::type operator>=(const float128 &lhs, const T &rhs)
{
  return static_cast<double>(lhs) >= static_cast<double>(rhs);
}

template <typename T>
typename std::enable_if<is_arithmetic<T>::value, bool>::type operator>=(const T &lhs, const float128 &rhs)
{
  return static_cast<double>(lhs) >= static_cast<double>(rhs);
}

inline bool operator>(const float128 &lhs, const float128 &rhs)
{
  return static_cast<double>(lhs) > static_cast<double>(rhs);
}

template <typename T>
typename std::enable_if<is_arithmetic<T>::value, bool>::type operator>(const float128 &lhs, const T &rhs)
{
  return static_cast<double>(lhs) > static_cast<double>(rhs);
}

template <typename T>
typename std::enable_if<is_arithmetic<T>::value, bool>::type operator>(const T &lhs, const float128 &rhs)
{
  return static_cast<double>(lhs) > static_cast<double>(rhs);
}

inline std::ostream &operator<<(std::ostream &o, const float128 &DYND_UNUSED(rhs))
{
  return (o << "<float128 printing unimplemented>");
}

inline float128 floor(float128 value) { return static_cast<double>(value); }

} // namespace dynd

namespace std {

template <>
struct common_type<dynd::float128, bool> {
  typedef dynd::float128 type;
};

template <>
struct common_type<dynd::float128, dynd::bool1> {
  typedef dynd::float128 type;
};

template <>
struct common_type<dynd::float128, char> {
  typedef dynd::float128 type;
};

template <>
struct common_type<dynd::float128, signed char> {
  typedef dynd::float128 type;
};

template <>
struct common_type<dynd::float128, unsigned char> {
  typedef dynd::float128 type;
};

template <>
struct common_type<dynd::float128, short> {
  typedef dynd::float128 type;
};

template <>
struct common_type<dynd::float128, unsigned short> {
  typedef dynd::float128 type;
};

template <>
struct common_type<dynd::float128, int> {
  typedef dynd::float128 type;
};

template <>
struct common_type<dynd::float128, unsigned int> {
  typedef dynd::float128 type;
};

template <>
struct common_type<dynd::float128, long> {
  typedef dynd::float128 type;
};

template <>
struct common_type<dynd::float128, unsigned long> {
  typedef dynd::float128 type;
};

template <>
struct common_type<dynd::float128, long long> {
  typedef dynd::float128 type;
};

template <>
struct common_type<dynd::float128, unsigned long long> {
  typedef dynd::float128 type;
};

template <>
struct common_type<dynd::float128, dynd::int128> {
  typedef dynd::float128 type;
};

template <>
struct common_type<dynd::float128, dynd::uint128> {
  typedef dynd::float128 type;
};

template <>
struct common_type<dynd::float128, float> {
  typedef dynd::float128 type;
};

template <>
struct common_type<dynd::float128, double> {
  typedef dynd::float128 type;
};

template <>
struct common_type<dynd::float128, dynd::float128> {
  typedef dynd::float128 type;
};

template <typename T>
struct common_type<T, dynd::float128> : common_type<dynd::float128, T> {
};

} // namespace std

#endif // !defined(DYND_HAS_FLOAT128)
