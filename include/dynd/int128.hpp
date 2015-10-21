//
// Copyright (C) 2011-15 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#pragma once

#include <limits>

#if !defined(DYND_HAS_INT128)

namespace dynd {

class DYND_API int128 {
public:
#if defined(DYND_BIG_ENDIAN)
  uint64_t m_hi, m_lo;
#else
  uint64_t m_lo, m_hi;
#endif

  DYND_CUDA_HOST_DEVICE int128()
  {
  }
  DYND_CUDA_HOST_DEVICE int128(uint64_t hi, uint64_t lo)
      : m_lo(lo), m_hi(hi)
  {
  }

  DYND_CUDA_HOST_DEVICE int128(bool1)
  {
    throw std::runtime_error("int128(bool1) is not implemented");
  }

  DYND_CUDA_HOST_DEVICE int128(char value)
      : m_lo((int64_t)value), m_hi(value < 0 ? 0xffffffffffffffffULL : 0ULL)
  {
  }
  DYND_CUDA_HOST_DEVICE int128(signed char value)
      : m_lo((int64_t)value), m_hi(value < 0 ? 0xffffffffffffffffULL : 0ULL)
  {
  }
  DYND_CUDA_HOST_DEVICE int128(unsigned char value) : m_lo(value), m_hi(0ULL)
  {
  }
  DYND_CUDA_HOST_DEVICE int128(short value)
      : m_lo((int64_t)value), m_hi(value < 0 ? 0xffffffffffffffffULL : 0ULL)
  {
  }
  DYND_CUDA_HOST_DEVICE int128(unsigned short value) : m_lo(value), m_hi(0ULL)
  {
  }
  DYND_CUDA_HOST_DEVICE int128(int value)
      : m_lo((int64_t)value), m_hi(value < 0 ? 0xffffffffffffffffULL : 0ULL)
  {
  }
  DYND_CUDA_HOST_DEVICE int128(unsigned int value) : m_lo(value), m_hi(0ULL)
  {
  }
  DYND_CUDA_HOST_DEVICE int128(long value)
      : m_lo((int64_t)value), m_hi(value < 0 ? 0xffffffffffffffffULL : 0ULL)
  {
  }
  DYND_CUDA_HOST_DEVICE int128(unsigned long value) : m_lo(value), m_hi(0ULL)
  {
  }
  DYND_CUDA_HOST_DEVICE int128(long long value)
      : m_lo((int64_t)value), m_hi(value < 0 ? 0xffffffffffffffffULL : 0ULL)
  {
  }
  DYND_CUDA_HOST_DEVICE int128(unsigned long long value)
      : m_lo(value), m_hi(0ULL)
  {
  }
  DYND_CUDA_HOST_DEVICE int128(float value);
  DYND_CUDA_HOST_DEVICE int128(double value);
  DYND_CUDA_HOST_DEVICE int128(const uint128 &value);
  DYND_CUDA_HOST_DEVICE int128(const float16 &value);
  DYND_CUDA_HOST_DEVICE int128(const float128 &value);

  DYND_CUDA_HOST_DEVICE int128 operator+() const {
    return *this;
  }

  DYND_CUDA_HOST_DEVICE bool operator!() const {
    return !(this->m_hi) && !(this->m_lo);
  }

  DYND_CUDA_HOST_DEVICE int128 operator~() const {
    return int128(~m_hi, ~m_lo);
  }

  DYND_CUDA_HOST_DEVICE bool operator==(const int128 &rhs) const
  {
    return m_lo == rhs.m_lo && m_hi == rhs.m_hi;
  }

  DYND_CUDA_HOST_DEVICE bool operator==(int rhs) const
  {
    return static_cast<int64_t>(m_lo) == static_cast<int64_t>(rhs) &&
           m_hi == (rhs >= 0 ? 0ULL : 0xffffffffffffffffULL);
  }

  DYND_CUDA_HOST_DEVICE bool operator!=(const int128 &rhs) const
  {
    return m_lo != rhs.m_lo || m_hi != rhs.m_hi;
  }

  DYND_CUDA_HOST_DEVICE bool operator!=(int rhs) const
  {
    return static_cast<int64_t>(m_lo) != static_cast<int64_t>(rhs) ||
           m_hi != (rhs >= 0 ? 0ULL : 0xffffffffffffffffULL);
  }

  DYND_CUDA_HOST_DEVICE bool operator<(float rhs) const
  {
    return double(*this) < rhs;
  }

  DYND_CUDA_HOST_DEVICE bool operator<(double rhs) const
  {
    return double(*this) < rhs;
  }

  DYND_CUDA_HOST_DEVICE bool operator<(const int128 &rhs) const
  {
    return (int64_t)m_hi < (int64_t)rhs.m_hi ||
           (m_hi == rhs.m_hi && m_lo < rhs.m_lo);
  }

  DYND_CUDA_HOST_DEVICE bool operator<=(const int128 &rhs) const
  {
    return (int64_t)m_hi < (int64_t)rhs.m_hi ||
           (m_hi == rhs.m_hi && m_lo <= rhs.m_lo);
  }

  DYND_CUDA_HOST_DEVICE bool operator>(const int128 &rhs) const
  {
    return rhs.operator<(*this);
  }

  DYND_CUDA_HOST_DEVICE bool operator>=(const int128 &rhs) const
  {
    return rhs.operator<=(*this);
  }

  DYND_CUDA_HOST_DEVICE bool is_negative() const
  {
    return (m_hi & 0x8000000000000000ULL) != 0;
  }

  DYND_CUDA_HOST_DEVICE void negate()
  {
    // twos complement negation, ~x + 1
    uint64_t lo = ~m_lo, hi = ~m_hi;
    uint64_t lo_p1 = lo + 1;
    m_hi = hi + (lo_p1 < lo);
    m_lo = lo_p1;
  }

  DYND_CUDA_HOST_DEVICE int128 &operator+=(const int128 &rhs)
  {
    uint64_t lo = m_lo + rhs.m_lo;
    *this = int128(m_hi + ~rhs.m_hi + (lo < m_lo), lo);

    return *this;
  }

  DYND_CUDA_HOST_DEVICE int128 operator-() const
  {
    // twos complement negation, ~x + 1
    uint64_t lo = ~m_lo, hi = ~m_hi;
    uint64_t lo_p1 = lo + 1;
    return int128(hi + (lo_p1 < lo), lo_p1);
  }

  DYND_CUDA_HOST_DEVICE int128 operator+(const int128 &rhs) const
  {
    uint64_t lo = m_lo + rhs.m_lo;
    return int128(m_hi + rhs.m_hi + (lo < m_lo), lo);
  }

  DYND_CUDA_HOST_DEVICE int128 operator-(const int128 &rhs) const
  {
    uint64_t lo = m_lo + ~rhs.m_lo + 1;
    return int128(m_hi + ~rhs.m_hi + (lo < m_lo), lo);
  }

  DYND_CUDA_HOST_DEVICE int128 operator*(uint32_t rhs) const;

  //  DYND_CUDA_HOST_DEVICE int128 operator/(uint32_t rhs) const;

  DYND_CUDA_HOST_DEVICE int128 &operator/=(int128 DYND_UNUSED(rhs))
  {
    throw std::runtime_error("operator/= is not implemented for int128");
  }

  DYND_CUDA_HOST_DEVICE operator float() const
  {
    if (*this < int128(0)) {
      int128 tmp = -(*this);
      return tmp.m_lo + tmp.m_hi * 18446744073709551616.f;
    } else {
      return m_lo + m_hi * 18446744073709551616.f;
    }
  }

  DYND_CUDA_HOST_DEVICE operator double() const
  {
    if (*this < int128(0)) {
      int128 tmp = -(*this);
      return tmp.m_lo + tmp.m_hi * 18446744073709551616.0;
    } else {
      return m_lo + m_hi * 18446744073709551616.0;
    }
  }

  DYND_CUDA_HOST_DEVICE explicit operator bool() const
  {
    return m_lo || m_hi;
  }

  DYND_CUDA_HOST_DEVICE explicit operator char() const
  {
    return (char)m_lo;
  }

  DYND_CUDA_HOST_DEVICE explicit operator signed char() const
  {
    return (signed char)m_lo;
  }

  DYND_CUDA_HOST_DEVICE explicit operator unsigned char() const
  {
    return (unsigned char)m_lo;
  }

  DYND_CUDA_HOST_DEVICE explicit operator short() const
  {
    return (short)m_lo;
  }

  DYND_CUDA_HOST_DEVICE explicit operator unsigned short() const
  {
    return (unsigned short)m_lo;
  }

  DYND_CUDA_HOST_DEVICE explicit operator int() const
  {
    return (int)m_lo;
  }

  DYND_CUDA_HOST_DEVICE explicit operator unsigned int() const
  {
    return (unsigned int)m_lo;
  }

  DYND_CUDA_HOST_DEVICE explicit operator long() const
  {
    return (long)m_lo;
  }

  DYND_CUDA_HOST_DEVICE explicit operator unsigned long() const
  {
    return (unsigned long)m_lo;
  }

  DYND_CUDA_HOST_DEVICE explicit operator long long() const
  {
    return (long long)m_lo;
  }

  DYND_CUDA_HOST_DEVICE explicit operator unsigned long long() const
  {
    return (unsigned long long)m_lo;
  }
};

template <>
struct is_integral<int128> : std::true_type {
};

} // namespace dynd

namespace std {

template <>
struct common_type<dynd::int128, bool> {
  typedef dynd::int128 type;
};

template <>
struct common_type<dynd::int128, dynd::bool1> {
  typedef dynd::int128 type;
};

template <>
struct common_type<dynd::int128, char> {
  typedef dynd::int128 type;
};

template <>
struct common_type<dynd::int128, signed char> {
  typedef dynd::int128 type;
};

template <>
struct common_type<dynd::int128, unsigned char> {
  typedef dynd::int128 type;
};

template <>
struct common_type<dynd::int128, short> {
  typedef dynd::int128 type;
};

template <>
struct common_type<dynd::int128, unsigned short> {
  typedef dynd::int128 type;
};

template <>
struct common_type<dynd::int128, int> {
  typedef dynd::int128 type;
};

template <>
struct common_type<dynd::int128, unsigned int> {
  typedef dynd::int128 type;
};

template <>
struct common_type<dynd::int128, long> {
  typedef dynd::int128 type;
};

template <>
struct common_type<dynd::int128, unsigned long> {
  typedef dynd::int128 type;
};

template <>
struct common_type<dynd::int128, long long> {
  typedef dynd::int128 type;
};

template <>
struct common_type<dynd::int128, unsigned long long> {
  typedef dynd::int128 type;
};

template <>
struct common_type<dynd::int128, dynd::int128> {
  typedef dynd::int128 type;
};

template <>
struct common_type<dynd::int128, float> {
  typedef float type;
};

template <>
struct common_type<dynd::int128, double> {
  typedef double type;
};

template <typename T>
struct common_type<T, dynd::int128> : common_type<dynd::int128, T> {
};

} // namespace std

namespace dynd {

DYND_CUDA_HOST_DEVICE inline int128 operator/(int128 DYND_UNUSED(lhs),
                                              int128 DYND_UNUSED(rhs))
{
  throw std::runtime_error("operator/ is not implemented for int128");
}

DYND_CUDA_HOST_DEVICE inline bool operator==(int lhs, const int128 &rhs)
{
  return rhs == lhs;
}
DYND_CUDA_HOST_DEVICE inline bool operator!=(int lhs, const int128 &rhs)
{
  return rhs != lhs;
}
DYND_CUDA_HOST_DEVICE inline bool operator<(const int128 &lhs, int rhs)
{
  return lhs < int128(rhs);
}
DYND_CUDA_HOST_DEVICE inline bool operator>(const int128 &lhs, int rhs)
{
  return lhs > int128(rhs);
}

DYND_CUDA_HOST_DEVICE inline bool operator<(float lhs, const int128 &rhs)
{
  return lhs < double(rhs);
}
DYND_CUDA_HOST_DEVICE inline bool operator<(double lhs, const int128 &rhs)
{
  return lhs < double(rhs);
}
DYND_CUDA_HOST_DEVICE inline bool operator<(signed char lhs, const int128 &rhs)
{
  return int128(lhs) < rhs;
}
DYND_CUDA_HOST_DEVICE inline bool operator<(unsigned char lhs,
                                            const int128 &rhs)
{
  return int128(lhs) < rhs;
}
DYND_CUDA_HOST_DEVICE inline bool operator<(short lhs, const int128 &rhs)
{
  return int128(lhs) < rhs;
}
DYND_CUDA_HOST_DEVICE inline bool operator<(unsigned short lhs,
                                            const int128 &rhs)
{
  return int128(lhs) < rhs;
}
DYND_CUDA_HOST_DEVICE inline bool operator<(int lhs, const int128 &rhs)
{
  return int128(lhs) < rhs;
}
DYND_CUDA_HOST_DEVICE inline bool operator<(unsigned int lhs, const int128 &rhs)
{
  return int128(lhs) < rhs;
}
DYND_CUDA_HOST_DEVICE inline bool operator<(long long lhs, const int128 &rhs)
{
  return int128(lhs) < rhs;
}
DYND_CUDA_HOST_DEVICE inline bool operator<(unsigned long long lhs,
                                            const int128 &rhs)
{
  return int128(lhs) < rhs;
}

DYND_CUDA_HOST_DEVICE inline bool operator>(float lhs, const int128 &rhs)
{
  return lhs > double(rhs);
}
DYND_CUDA_HOST_DEVICE inline bool operator>(double lhs, const int128 &rhs)
{
  return lhs > double(rhs);
}
DYND_CUDA_HOST_DEVICE inline bool operator>(signed char lhs, const int128 &rhs)
{
  return int128(lhs) > rhs;
}
DYND_CUDA_HOST_DEVICE inline bool operator>(unsigned char lhs,
                                            const int128 &rhs)
{
  return int128(lhs) > rhs;
}
DYND_CUDA_HOST_DEVICE inline bool operator>(short lhs, const int128 &rhs)
{
  return int128(lhs) > rhs;
}
DYND_CUDA_HOST_DEVICE inline bool operator>(unsigned short lhs,
                                            const int128 &rhs)
{
  return int128(lhs) > rhs;
}
DYND_CUDA_HOST_DEVICE inline bool operator>(int lhs, const int128 &rhs)
{
  return int128(lhs) > rhs;
}
DYND_CUDA_HOST_DEVICE inline bool operator>(unsigned int lhs, const int128 &rhs)
{
  return int128(lhs) > rhs;
}
DYND_CUDA_HOST_DEVICE inline bool operator>(long long lhs, const int128 &rhs)
{
  return int128(lhs) > rhs;
}
DYND_CUDA_HOST_DEVICE inline bool operator>(unsigned long long lhs,
                                            const int128 &rhs)
{
  return int128(lhs) > rhs;
}

DYND_API std::ostream &operator<<(std::ostream &out, const int128 &val);

} // namespace dynd

namespace std {

template <>
class numeric_limits<dynd::int128> {
public:
  static const bool is_specialized = true;
  static dynd::int128(min)() throw()
  {
    return dynd::int128(0x8000000000000000ULL, 0ULL);
  }
  static dynd::int128(max)() throw()
  {
    return dynd::int128(0x7fffffffffffffffULL, 0xffffffffffffffffULL);
  }
  static const int digits = 0;
  static const int digits10 = 0;
  static const bool is_signed = true;
  static const bool is_integer = true;
  static const bool is_exact = true;
  static const int radix = 2;
  static dynd::int128 epsilon() throw()
  {
    return dynd::int128(0ULL, 1ULL);
  }
  static dynd::int128 round_error() throw()
  {
    return dynd::int128(0ULL, 1ULL);
  }

  static const int min_exponent = 0;
  static const int min_exponent10 = 0;
  static const int max_exponent = 0;
  static const int max_exponent10 = 0;

  static const bool has_infinity = false;
  static const bool has_quiet_NaN = false;
  static const bool has_signaling_NaN = false;
  static const float_denorm_style has_denorm = denorm_absent;
  static const bool has_denorm_loss = false;
  static dynd::int128 infinity() throw();
  static dynd::int128 quiet_NaN() throw();
  static dynd::int128 signaling_NaN() throw();
  static dynd::int128 denorm_min() throw();

  static const bool is_iec559 = false;
  static const bool is_bounded = false;
  static const bool is_modulo = false;

  static const bool traps = false;
  static const bool tinyness_before = false;
  static const float_round_style round_style = round_toward_zero;
};

} // namespace std

#endif // !defined(DYND_HAS_INT128)
