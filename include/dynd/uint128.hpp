//
// Copyright (C) 2011-15 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#pragma once

#include <limits>
#include <iostream>

#if !defined(DYND_HAS_UINT128)

namespace dynd {

#if !defined(DYND_HAS_INT128)
class int128;
#endif

class DYND_API uint128 {
public:
#if defined(DYND_BIG_ENDIAN)
  uint64_t m_hi, m_lo;
#else
  uint64_t m_lo, m_hi;
#endif
  DYND_CUDA_HOST_DEVICE uint128()
  {
  }
  DYND_CUDA_HOST_DEVICE uint128(uint64_t hi, uint64_t lo) : m_lo(lo), m_hi(hi)
  {
  }

  DYND_CUDA_HOST_DEVICE uint128(bool1)
  {
    throw std::runtime_error("uint128(bool1) is not implemented");
  }

  DYND_CUDA_HOST_DEVICE uint128(char value) : m_lo(value), m_hi(0ULL)
  {
  }
  DYND_CUDA_HOST_DEVICE uint128(signed char value) : m_lo(value), m_hi(0ULL)
  {
  }
  DYND_CUDA_HOST_DEVICE uint128(unsigned char value) : m_lo(value), m_hi(0ULL)
  {
  }
  DYND_CUDA_HOST_DEVICE uint128(short value) : m_lo(value), m_hi(0ULL)
  {
  }
  DYND_CUDA_HOST_DEVICE uint128(unsigned short value) : m_lo(value), m_hi(0ULL)
  {
  }
  DYND_CUDA_HOST_DEVICE uint128(int value) : m_lo(value), m_hi(0ULL)
  {
  }
  DYND_CUDA_HOST_DEVICE uint128(unsigned int value) : m_lo(value), m_hi(0ULL)
  {
  }
  DYND_CUDA_HOST_DEVICE uint128(long value) : m_lo(value), m_hi(0ULL)
  {
  }
  DYND_CUDA_HOST_DEVICE uint128(unsigned long value) : m_lo(value), m_hi(0ULL)
  {
  }
  DYND_CUDA_HOST_DEVICE uint128(long long value) : m_lo(value), m_hi(0ULL)
  {
  }
  DYND_CUDA_HOST_DEVICE uint128(unsigned long long value)
      : m_lo(value), m_hi(0ULL)
  {
  }
  DYND_CUDA_HOST_DEVICE uint128(float value);
  DYND_CUDA_HOST_DEVICE uint128(double value);
  DYND_CUDA_HOST_DEVICE uint128(const int128 &value);
  DYND_CUDA_HOST_DEVICE uint128(const float16 &value);
  DYND_CUDA_HOST_DEVICE uint128(const float128 &value);

  DYND_CUDA_HOST_DEVICE bool operator==(const uint128 &rhs) const
  {
    return m_hi == rhs.m_hi && m_lo == rhs.m_lo;
  }

  DYND_CUDA_HOST_DEVICE bool operator==(uint64_t rhs) const
  {
    return m_hi == 0 && m_lo == rhs;
  }

  DYND_CUDA_HOST_DEVICE bool operator==(int rhs) const
  {
    return rhs >= 0 && m_hi == 0u && m_lo == static_cast<unsigned int>(rhs);
  }

  DYND_CUDA_HOST_DEVICE bool operator==(unsigned int rhs) const
  {
    return m_hi == 0u && m_lo == rhs;
  }

  DYND_CUDA_HOST_DEVICE uint128 operator+() const {
    return *this;
  }

  DYND_CUDA_HOST_DEVICE uint128 operator-() const {
  DYND_ALLOW_UNSIGNED_UNARY_MINUS
    if (this->m_lo != 0) {
      return uint128(!this->m_hi, -this->m_lo);
    } else {
      return uint128(-this->m_hi, 0);
    }
  DYND_END_ALLOW_UNSIGNED_UNARY_MINUS
  }

  DYND_CUDA_HOST_DEVICE bool operator!() const {
    return (m_hi != 0) && (m_lo != 0);
  }

  DYND_CUDA_HOST_DEVICE uint128 operator~() const {
    return uint128(~m_hi, ~m_lo);
  }

  DYND_CUDA_HOST_DEVICE bool operator!=(const uint128 &rhs) const
  {
    return m_hi != rhs.m_hi || m_lo != rhs.m_lo;
  }

  DYND_CUDA_HOST_DEVICE bool operator!=(uint64_t rhs) const
  {
    return m_hi != 0 || m_lo != rhs;
  }

  DYND_CUDA_HOST_DEVICE bool operator!=(int rhs) const
  {
    return rhs < 0 || m_hi != 0u || m_lo != static_cast<unsigned int>(rhs);
  }

  DYND_CUDA_HOST_DEVICE bool operator!=(unsigned int rhs) const
  {
    return m_hi != 0u || m_lo != rhs;
  }

  DYND_CUDA_HOST_DEVICE bool operator<(float rhs) const
  {
    return double(*this) < rhs;
  }

  DYND_CUDA_HOST_DEVICE bool operator<(double rhs) const
  {
    return double(*this) < rhs;
  }

  DYND_CUDA_HOST_DEVICE bool operator<(const uint128 &rhs) const
  {
    return m_hi < rhs.m_hi || (m_hi == rhs.m_hi && m_lo < rhs.m_lo);
  }

  DYND_CUDA_HOST_DEVICE bool operator<=(const uint128 &rhs) const
  {
    return m_hi < rhs.m_hi || (m_hi == rhs.m_hi && m_lo <= rhs.m_lo);
  }

  DYND_CUDA_HOST_DEVICE bool operator>(const uint128 &rhs) const
  {
    return rhs.operator<(*this);
  }

  DYND_CUDA_HOST_DEVICE bool operator>=(const uint128 &rhs) const
  {
    return rhs.operator<=(*this);
  }

  DYND_CUDA_HOST_DEVICE uint128 operator+(const uint128 &rhs) const
  {
    uint64_t lo = m_lo + rhs.m_lo;
    return uint128(m_hi + rhs.m_hi + (lo < m_lo), lo);
  }

  DYND_CUDA_HOST_DEVICE uint128 operator+(uint64_t rhs) const
  {
    uint64_t lo = m_lo + rhs;
    return uint128(m_hi + (lo < m_lo), lo);
  }

  DYND_CUDA_HOST_DEVICE uint128 operator+(uint32_t rhs) const
  {
    uint64_t lo = m_lo + static_cast<uint64_t>(rhs);
    return uint128(m_hi + (lo < m_lo), lo);
  }

  DYND_CUDA_HOST_DEVICE uint128 operator-(const uint128 &rhs) const
  {
    uint64_t lo = m_lo + ~rhs.m_lo + 1;
    return uint128(m_hi + ~rhs.m_hi + (lo < m_lo), lo);
  }

  DYND_CUDA_HOST_DEVICE uint128 operator-(uint64_t rhs) const
  {
    uint64_t lo = m_lo + ~rhs + 1;
    return uint128(m_hi + 0xffffffffffffffffULL + (lo < m_lo), lo);
  }

  DYND_CUDA_HOST_DEVICE uint128 operator*(uint32_t rhs) const;
  DYND_CUDA_HOST_DEVICE uint128 operator/(uint32_t rhs) const;

  DYND_CUDA_HOST_DEVICE void divrem(uint32_t rhs, uint32_t &out_rem);

  DYND_CUDA_HOST_DEVICE uint128 &operator/=(uint128 DYND_UNUSED(rhs))
  {
    throw std::runtime_error("operator/= is not implemented for uint128");
  }

  DYND_CUDA_HOST_DEVICE explicit operator bool() const
  {
    return m_lo || m_hi;
  }

  DYND_CUDA_HOST_DEVICE operator float() const
  {
    return m_lo + m_hi * 18446744073709551616.f;
  }

  DYND_CUDA_HOST_DEVICE operator double() const
  {
    return m_lo + m_hi * 18446744073709551616.0;
  }

  DYND_CUDA_HOST_DEVICE operator char() const
  {
    return static_cast<char>(m_lo);
  }
  DYND_CUDA_HOST_DEVICE operator signed char() const
  {
    return static_cast<signed char>(m_lo);
  }
  DYND_CUDA_HOST_DEVICE operator unsigned char() const
  {
    return static_cast<unsigned char>(m_lo);
  }
  DYND_CUDA_HOST_DEVICE operator short() const
  {
    return static_cast<short>(m_lo);
  }
  DYND_CUDA_HOST_DEVICE operator unsigned short() const
  {
    return static_cast<unsigned short>(m_lo);
  }
  DYND_CUDA_HOST_DEVICE operator int() const
  {
    return static_cast<int>(m_lo);
  }
  DYND_CUDA_HOST_DEVICE operator unsigned int() const
  {
    return static_cast<unsigned int>(m_lo);
  }
  DYND_CUDA_HOST_DEVICE operator long() const
  {
    return static_cast<long>(m_lo);
  }
  DYND_CUDA_HOST_DEVICE operator unsigned long() const
  {
    return static_cast<unsigned long>(m_lo);
  }
  DYND_CUDA_HOST_DEVICE operator long long() const
  {
    return static_cast<long long>(m_lo);
  }
  DYND_CUDA_HOST_DEVICE operator unsigned long long() const
  {
    return static_cast<unsigned long long>(m_lo);
  }
};

template <>
struct is_integral<uint128> : std::true_type {
};

DYND_CUDA_HOST_DEVICE inline uint128 operator/(uint128 DYND_UNUSED(lhs),
                                               uint128 DYND_UNUSED(rhs))
{
  throw std::runtime_error("operator/ is not implemented for int128");
}

} // namespace dynd

namespace std {

template <>
struct common_type<dynd::uint128, bool> {
  typedef dynd::uint128 type;
};

template <>
struct common_type<dynd::uint128, dynd::bool1> {
  typedef dynd::uint128 type;
};

template <>
struct common_type<dynd::uint128, char> {
  typedef dynd::uint128 type;
};

template <>
struct common_type<dynd::uint128, signed char> {
  typedef dynd::uint128 type;
};

template <>
struct common_type<dynd::uint128, unsigned char> {
  typedef dynd::uint128 type;
};

template <>
struct common_type<dynd::uint128, short> {
  typedef dynd::uint128 type;
};

template <>
struct common_type<dynd::uint128, unsigned short> {
  typedef dynd::uint128 type;
};

template <>
struct common_type<dynd::uint128, int> {
  typedef dynd::uint128 type;
};

template <>
struct common_type<dynd::uint128, unsigned int> {
  typedef dynd::uint128 type;
};

template <>
struct common_type<dynd::uint128, long> {
  typedef dynd::uint128 type;
};

template <>
struct common_type<dynd::uint128, unsigned long> {
  typedef dynd::uint128 type;
};

template <>
struct common_type<dynd::uint128, long long> {
  typedef dynd::uint128 type;
};

template <>
struct common_type<dynd::uint128, unsigned long long> {
  typedef dynd::uint128 type;
};

template <>
struct common_type<dynd::uint128, dynd::int128> {
  typedef dynd::uint128 type;
};

template <>
struct common_type<dynd::uint128, dynd::uint128> {
  typedef dynd::uint128 type;
};

template <>
struct common_type<dynd::uint128, float> {
  typedef float type;
};

template <>
struct common_type<dynd::uint128, double> {
  typedef double type;
};

template <typename T>
struct common_type<T, dynd::uint128> : common_type<dynd::uint128, T> {
};

} // namespace std

namespace dynd {

DYND_CUDA_HOST_DEVICE inline bool operator==(unsigned int lhs,
                                             const uint128 &rhs)
{
  return rhs.m_hi == 0u && lhs == rhs.m_lo;
}
DYND_CUDA_HOST_DEVICE inline bool operator!=(unsigned int lhs,
                                             const uint128 &rhs)
{
  return rhs.m_hi != 0u || lhs != rhs.m_lo;
}

DYND_CUDA_HOST_DEVICE inline bool operator<(float lhs, const uint128 &rhs)
{
  return lhs < double(rhs);
}
DYND_CUDA_HOST_DEVICE inline bool operator<(double lhs, const uint128 &rhs)
{
  return lhs < double(rhs);
}
DYND_CUDA_HOST_DEVICE inline bool operator<(signed char lhs, const uint128 &rhs)
{
  return lhs < 0 || uint128(lhs) < rhs;
}
DYND_CUDA_HOST_DEVICE inline bool operator<(unsigned char lhs,
                                            const uint128 &rhs)
{
  return uint128(lhs) < rhs;
}
DYND_CUDA_HOST_DEVICE inline bool operator<(short lhs, const uint128 &rhs)
{
  return lhs < 0 || uint128(lhs) < rhs;
}
DYND_CUDA_HOST_DEVICE inline bool operator<(unsigned short lhs,
                                            const uint128 &rhs)
{
  return uint128(lhs) < rhs;
}
DYND_CUDA_HOST_DEVICE inline bool operator<(int lhs, const uint128 &rhs)
{
  return lhs < 0 || uint128(lhs) < rhs;
}
DYND_CUDA_HOST_DEVICE inline bool operator<(unsigned int lhs,
                                            const uint128 &rhs)
{
  return uint128(lhs) < rhs;
}
DYND_CUDA_HOST_DEVICE inline bool operator<(long lhs, const uint128 &rhs)
{
  return lhs < 0 || uint128(lhs) < rhs;
}
DYND_CUDA_HOST_DEVICE inline bool operator<(unsigned long lhs,
                                            const uint128 &rhs)
{
  return uint128(lhs) < rhs;
}
DYND_CUDA_HOST_DEVICE inline bool operator<(long long lhs, const uint128 &rhs)
{
  return lhs < 0 || uint128(lhs) < rhs;
}
DYND_CUDA_HOST_DEVICE inline bool operator<(unsigned long long lhs,
                                            const uint128 &rhs)
{
  return uint128(lhs) < rhs;
}

template<typename T, typename dynd::detail::enable_for<T, unsigned char, unsigned short, unsigned int, unsigned long, unsigned long long>::type = 0>
DYND_CUDA_HOST_DEVICE inline void operator+=(uint128 DYND_UNUSED(lhs), T DYND_UNUSED(rhs)) {
  throw std::runtime_error("operator += is not implemented for uint128");
}

DYND_API std::ostream &operator<<(std::ostream &out, const uint128 &val);

} // namespace dynd

namespace std {

template <>
class numeric_limits<dynd::uint128> {
public:
  static const bool is_specialized = true;
  static dynd::uint128(min)() throw()
  {
    return dynd::uint128(0ULL, 0ULL);
  }
  static dynd::uint128(max)() throw()
  {
    return dynd::uint128(0xffffffffffffffffULL, 0xffffffffffffffffULL);
  }
  static const int digits = 0;
  static const int digits10 = 0;
  static const bool is_signed = true;
  static const bool is_integer = true;
  static const bool is_exact = true;
  static const int radix = 2;
  static dynd::uint128 epsilon() throw()
  {
    return dynd::uint128(0ULL, 1ULL);
  }
  static dynd::uint128 round_error() throw()
  {
    return dynd::uint128(0ULL, 1ULL);
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
  static dynd::uint128 infinity() throw();
  static dynd::uint128 quiet_NaN() throw();
  static dynd::uint128 signaling_NaN() throw();
  static dynd::uint128 denorm_min() throw();

  static const bool is_iec559 = false;
  static const bool is_bounded = false;
  static const bool is_modulo = false;

  static const bool traps = false;
  static const bool tinyness_before = false;
  static const float_round_style round_style = round_toward_zero;
};

} // namespace std

#endif // !defined(DYND_HAS_UINT128)
