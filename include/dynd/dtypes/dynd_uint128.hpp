//
// Copyright (C) 2010-13 Mark Wiebe, DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#ifndef _DYND__UINT128_H__
#define _DYND__UINT128_H__

#include <dynd/config.hpp>
#include <dynd/dtype_assign.hpp>

#include <iostream>

#if !defined(DYND_HAS_UINT128)

namespace dynd {

#if !defined(DYND_HAS_FLOAT128)
class dynd_float128;
#endif
#if !defined(DYND_HAS_INT128)
class dynd_int128;
#endif

class dynd_uint128 {
public:
#if defined(DYND_BIG_ENDIAN)
    uint64_t m_hi, m_lo;
#else
    uint64_t m_lo, m_hi;
#endif
    inline dynd_uint128() {}
    inline dynd_uint128(uint64_t hi, uint64_t lo)
        : m_lo(lo), m_hi(hi) {}

    inline dynd_uint128(char value)
        : m_lo(value), m_hi(0ULL) {}
    inline dynd_uint128(signed char value)
        : m_lo(value), m_hi(0ULL) {}
    inline dynd_uint128(unsigned char value)
        : m_lo(value), m_hi(0ULL) {}
    inline dynd_uint128(short value)
        : m_lo(value), m_hi(0ULL) {}
    inline dynd_uint128(unsigned short value)
        : m_lo(value), m_hi(0ULL) {}
    inline dynd_uint128(int value)
        : m_lo(value), m_hi(0ULL) {}
    inline dynd_uint128(unsigned int value)
        : m_lo(value), m_hi(0ULL) {}
    inline dynd_uint128(long long value)
        : m_lo(value), m_hi(0ULL) {}
    inline dynd_uint128(unsigned long long value)
        : m_lo(value), m_hi(0ULL) {}
    dynd_uint128(float value);
    dynd_uint128(double value);
    dynd_uint128(const dynd_int128& value);
    dynd_uint128(const dynd_float128& value);

    inline bool operator==(const dynd_uint128& rhs) const {
        return m_hi == rhs.m_hi && m_lo == rhs.m_lo;
    }

    inline bool operator==(uint64_t rhs) const {
        return m_hi == 0 && m_lo == rhs;
    }

    inline bool operator!=(const dynd_uint128& rhs) const {
        return m_hi != rhs.m_hi || m_lo != rhs.m_lo;
    }

    inline bool operator!=(uint64_t rhs) const {
        return m_hi != 0 || m_lo != rhs;
    }

    inline bool operator<(float rhs) const {
        return double(*this) < rhs;
    }

    inline bool operator<(double rhs) const {
        return double(*this) < rhs;
    }

    inline bool operator<(const dynd_uint128& rhs) const {
        return m_hi < rhs.m_hi ||
                        (m_hi == rhs.m_hi && m_lo < rhs.m_lo);
    }

    inline bool operator<=(const dynd_uint128& rhs) const {
        return m_hi < rhs.m_hi ||
                        (m_hi == rhs.m_hi && m_lo <= rhs.m_lo);
    }

    inline bool operator>(const dynd_uint128& rhs) const {
        return rhs.operator<(*this);
    }

    inline bool operator>=(const dynd_uint128& rhs) const {
        return rhs.operator<=(*this);
    }

    inline dynd_uint128 operator+(const dynd_uint128& rhs) const {
        uint64_t lo = m_lo + rhs.m_lo;
        return dynd_uint128(m_hi + rhs.m_hi + (lo < m_lo), lo);
    }

    inline dynd_uint128 operator-(const dynd_uint128& rhs) const {
        uint64_t lo = m_lo + ~rhs.m_lo + 1;
        return dynd_uint128(m_hi + ~rhs.m_hi + (lo < m_lo), lo);
    }

    dynd_uint128 operator*(uint32_t rhs) const;
    dynd_uint128 operator/(uint32_t rhs) const;

    void divrem(uint32_t rhs, uint32_t& out_rem);

    operator float() const {
        return m_lo + m_hi * 18446744073709551616.f;
    }

    operator double() const {
        return m_lo + m_hi * 18446744073709551616.0;
    }

    operator char() const {
        return (char)m_lo;
    }
    operator signed char() const {
        return (signed char)m_lo;
    }
    operator unsigned char() const {
        return (unsigned char)m_lo;
    }
    operator short() const {
        return (short)m_lo;
    }
    operator unsigned short() const {
        return (unsigned short)m_lo;
    }
    operator int() const {
        return (int)m_lo;
    }
    operator unsigned int() const {
        return (unsigned int)m_lo;
    }
    operator long() const {
        return (long)m_lo;
    }
    operator unsigned long() const {
        return (unsigned long)m_lo;
    }
    operator long long() const {
        return (long long)m_lo;
    }
    operator unsigned long long() const {
        return (unsigned long long)m_lo;
    }
};

inline bool operator<(float lhs, const dynd_uint128& rhs) {
    return lhs < double(rhs);
}
inline bool operator<(double lhs, const dynd_uint128& rhs) {
    return lhs < double(rhs);
}
inline bool operator<(signed char lhs, const dynd_uint128& rhs) {
    return lhs < 0 || dynd_uint128(lhs) < rhs;
}
inline bool operator<(unsigned char lhs, const dynd_uint128& rhs) {
    return dynd_uint128(lhs) < rhs;
}
inline bool operator<(short lhs, const dynd_uint128& rhs) {
    return lhs < 0 || dynd_uint128(lhs) < rhs;
}
inline bool operator<(unsigned short lhs, const dynd_uint128& rhs) {
    return dynd_uint128(lhs) < rhs;
}
inline bool operator<(int lhs, const dynd_uint128& rhs) {
    return lhs < 0 || dynd_uint128(lhs) < rhs;
}
inline bool operator<(unsigned int lhs, const dynd_uint128& rhs) {
    return dynd_uint128(lhs) < rhs;
}
inline bool operator<(long long lhs, const dynd_uint128& rhs) {
    return lhs < 0 || dynd_uint128(lhs) < rhs;
}
inline bool operator<(unsigned long long lhs, const dynd_uint128& rhs) {
    return dynd_uint128(lhs) < rhs;
}

std::ostream& operator<<(std::ostream& out, const dynd_uint128& val);

} // namespace dynd

namespace std {

template<>
class numeric_limits<::dynd::dynd_uint128> {
public:
  static const bool is_specialized = true;
  static ::dynd::dynd_uint128 min() throw() {
    return ::dynd::dynd_uint128(0ULL, 0ULL);
  }
  static ::dynd::dynd_uint128 max() throw() {
    return ::dynd::dynd_uint128(0xffffffffffffffffULL, 0xffffffffffffffffULL);
  }
  static const int  digits = 0;
  static const int  digits10 = 0;
  static const bool is_signed = true;
  static const bool is_integer = true;
  static const bool is_exact = true;
  static const int radix = 2;
  static ::dynd::dynd_uint128 epsilon() throw() {
    return ::dynd::dynd_uint128(0ULL, 1ULL);
  }
  static ::dynd::dynd_uint128 round_error() throw() {
    return ::dynd::dynd_uint128(0ULL, 1ULL);
  }

  static const int  min_exponent = 0;
  static const int  min_exponent10 = 0;
  static const int  max_exponent = 0;
  static const int  max_exponent10 = 0;

  static const bool has_infinity = false;
  static const bool has_quiet_NaN = false;
  static const bool has_signaling_NaN = false;
  static const float_denorm_style has_denorm = denorm_absent;
  static const bool has_denorm_loss = false;
  static ::dynd::dynd_uint128 infinity() throw();
  static ::dynd::dynd_uint128 quiet_NaN() throw();
  static ::dynd::dynd_uint128 signaling_NaN() throw();
  static ::dynd::dynd_uint128 denorm_min() throw();

  static const bool is_iec559 = false;
  static const bool is_bounded = false;
  static const bool is_modulo = false;

  static const bool traps = false;
  static const bool tinyness_before = false;
  static const float_round_style round_style = round_toward_zero;
};
    
} // namespace std

#endif // !defined(DYND_HAS_UINT128)


#endif // _DYND__UINT128_H__