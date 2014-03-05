//
// Copyright (C) 2010-13 Mark Wiebe, DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#ifndef _DYND__INT128_H__
#define _DYND__INT128_H__

#include <dynd/config.hpp>
#include <dynd/typed_data_assign.hpp>

#include <limits>

#if !defined(DYND_HAS_INT128)

namespace dynd {

#if !defined(DYND_HAS_FLOAT128)
class dynd_float128;
#endif
#if !defined(DYND_HAS_UINT128)
class dynd_uint128;
#endif

class dynd_float16;

class dynd_int128 {
public:
#if defined(DYND_BIG_ENDIAN)
    uint64_t m_hi, m_lo;
#else
    uint64_t m_lo, m_hi;
#endif
    DYND_CUDA_HOST_DEVICE inline dynd_int128() {}
    DYND_CUDA_HOST_DEVICE inline dynd_int128(uint64_t hi, uint64_t lo)
        : m_lo(lo), m_hi(hi) {}

    DYND_CUDA_HOST_DEVICE inline dynd_int128(char value)
        : m_lo((int64_t)value), m_hi(value < 0 ? 0xffffffffffffffffULL : 0ULL) {}
    DYND_CUDA_HOST_DEVICE inline dynd_int128(signed char value)
        : m_lo((int64_t)value), m_hi(value < 0 ? 0xffffffffffffffffULL : 0ULL) {}
    DYND_CUDA_HOST_DEVICE inline dynd_int128(unsigned char value)
        : m_lo(value), m_hi(0ULL) {}
    DYND_CUDA_HOST_DEVICE inline dynd_int128(short value)
        : m_lo((int64_t)value), m_hi(value < 0 ? 0xffffffffffffffffULL : 0ULL) {}
    DYND_CUDA_HOST_DEVICE inline dynd_int128(unsigned short value)
        : m_lo(value), m_hi(0ULL) {}
    DYND_CUDA_HOST_DEVICE inline dynd_int128(int value)
        : m_lo((int64_t)value), m_hi(value < 0 ? 0xffffffffffffffffULL : 0ULL) {}
    DYND_CUDA_HOST_DEVICE inline dynd_int128(unsigned int value)
        : m_lo(value), m_hi(0ULL) {}
    DYND_CUDA_HOST_DEVICE inline dynd_int128(long value)
        : m_lo((int64_t)value), m_hi(value < 0 ? 0xffffffffffffffffULL : 0ULL) {}
    DYND_CUDA_HOST_DEVICE inline dynd_int128(unsigned long value)
        : m_lo(value), m_hi(0ULL) {}
    DYND_CUDA_HOST_DEVICE inline dynd_int128(long long value)
        : m_lo((int64_t)value), m_hi(value < 0 ? 0xffffffffffffffffULL : 0ULL) {}
    DYND_CUDA_HOST_DEVICE inline dynd_int128(unsigned long long value)
        : m_lo(value), m_hi(0ULL) {}
    DYND_CUDA_HOST_DEVICE dynd_int128(float value);
    DYND_CUDA_HOST_DEVICE dynd_int128(double value);
    DYND_CUDA_HOST_DEVICE dynd_int128(const dynd_uint128& value);
    DYND_CUDA_HOST_DEVICE dynd_int128(const dynd_float16& value);
    DYND_CUDA_HOST_DEVICE dynd_int128(const dynd_float128& value);

    DYND_CUDA_HOST_DEVICE inline bool operator==(const dynd_int128& rhs) const {
        return m_lo == rhs.m_lo && m_hi == rhs.m_hi;
    }

    DYND_CUDA_HOST_DEVICE inline bool operator!=(const dynd_int128& rhs) const {
        return m_lo != rhs.m_lo || m_hi != rhs.m_hi;
    }

    DYND_CUDA_HOST_DEVICE inline bool operator<(float rhs) const {
        return double(*this) < rhs;
    }

    DYND_CUDA_HOST_DEVICE inline bool operator<(double rhs) const {
        return double(*this) < rhs;
    }

    DYND_CUDA_HOST_DEVICE inline bool operator<(const dynd_int128& rhs) const {
        return (int64_t)m_hi < (int64_t)rhs.m_hi ||
                        (m_hi == rhs.m_hi && m_lo < rhs.m_lo);
    }

    DYND_CUDA_HOST_DEVICE inline bool operator<=(const dynd_int128& rhs) const {
        return (int64_t)m_hi < (int64_t)rhs.m_hi ||
                        (m_hi == rhs.m_hi && m_lo <= rhs.m_lo);
    }

    DYND_CUDA_HOST_DEVICE inline bool operator>(const dynd_int128& rhs) const {
        return rhs.operator<(*this);
    }

    DYND_CUDA_HOST_DEVICE inline bool operator>=(const dynd_int128& rhs) const {
        return rhs.operator<=(*this);
    }

    DYND_CUDA_HOST_DEVICE inline void negate() {
        // twos complement negation, ~x + 1
        uint64_t lo = ~m_lo, hi = ~m_hi;
        uint64_t lo_p1 = lo + 1;
        m_hi = hi + (lo_p1 < lo);
        m_lo = lo_p1;
    }

    DYND_CUDA_HOST_DEVICE inline dynd_int128 operator-() const {
        // twos complement negation, ~x + 1
        uint64_t lo = ~m_lo, hi = ~m_hi;
        uint64_t lo_p1 = lo + 1;
        return dynd_int128(hi + (lo_p1 < lo), lo_p1);
    }

    DYND_CUDA_HOST_DEVICE inline dynd_int128 operator+(const dynd_int128& rhs) const {
        uint64_t lo = m_lo + rhs.m_lo;
        return dynd_int128(m_hi + rhs.m_hi + (lo < m_lo), lo);
    }

    DYND_CUDA_HOST_DEVICE inline dynd_int128 operator-(const dynd_int128& rhs) const {
        uint64_t lo = m_lo + ~rhs.m_lo + 1;
        return dynd_int128(m_hi + ~rhs.m_hi + (lo < m_lo), lo);
    }

    DYND_CUDA_HOST_DEVICE inline dynd_int128 operator*(uint32_t rhs) const;

    DYND_CUDA_HOST_DEVICE inline dynd_int128 operator/(uint32_t rhs) const;

    DYND_CUDA_HOST_DEVICE operator float() const {
        if (*this < dynd_int128(0)) {
            dynd_int128 tmp = -(*this);
            return tmp.m_lo + tmp.m_hi * 18446744073709551616.f;
        } else {
            return m_lo + m_hi * 18446744073709551616.f;
        }
    }

    DYND_CUDA_HOST_DEVICE operator double() const {
        if (*this < dynd_int128(0)) {
            dynd_int128 tmp = -(*this);
            return tmp.m_lo + tmp.m_hi * 18446744073709551616.0;
        } else {
            return m_lo + m_hi * 18446744073709551616.0;
        }
    }

    DYND_CUDA_HOST_DEVICE operator char() const {
        return (char)m_lo;
    }
    DYND_CUDA_HOST_DEVICE operator signed char() const {
        return (signed char)m_lo;
    }
    DYND_CUDA_HOST_DEVICE operator unsigned char() const {
        return (unsigned char)m_lo;
    }
    DYND_CUDA_HOST_DEVICE operator short() const {
        return (short)m_lo;
    }
    DYND_CUDA_HOST_DEVICE operator unsigned short() const {
        return (unsigned short)m_lo;
    }
    DYND_CUDA_HOST_DEVICE operator int() const {
        return (int)m_lo;
    }
    DYND_CUDA_HOST_DEVICE operator unsigned int() const {
        return (unsigned int)m_lo;
    }
    DYND_CUDA_HOST_DEVICE operator long() const {
        return (long)m_lo;
    }
    DYND_CUDA_HOST_DEVICE operator unsigned long() const {
        return (unsigned long)m_lo;
    }
    DYND_CUDA_HOST_DEVICE operator long long() const {
        return (long long)m_lo;
    }
    DYND_CUDA_HOST_DEVICE operator unsigned long long() const {
        return (unsigned long long)m_lo;
    }
};

DYND_CUDA_HOST_DEVICE inline bool operator<(float lhs, const dynd_int128& rhs) {
    return lhs < double(rhs);
}
DYND_CUDA_HOST_DEVICE inline bool operator<(double lhs, const dynd_int128& rhs) {
    return lhs < double(rhs);
}
DYND_CUDA_HOST_DEVICE inline bool operator<(signed char lhs, const dynd_int128& rhs) {
    return dynd_int128(lhs) < rhs;
}
DYND_CUDA_HOST_DEVICE inline bool operator<(unsigned char lhs, const dynd_int128& rhs) {
    return dynd_int128(lhs) < rhs;
}
DYND_CUDA_HOST_DEVICE inline bool operator<(short lhs, const dynd_int128& rhs) {
    return dynd_int128(lhs) < rhs;
}
DYND_CUDA_HOST_DEVICE inline bool operator<(unsigned short lhs, const dynd_int128& rhs) {
    return dynd_int128(lhs) < rhs;
}
DYND_CUDA_HOST_DEVICE inline bool operator<(int lhs, const dynd_int128& rhs) {
    return dynd_int128(lhs) < rhs;
}
DYND_CUDA_HOST_DEVICE inline bool operator<(unsigned int lhs, const dynd_int128& rhs) {
    return dynd_int128(lhs) < rhs;
}
DYND_CUDA_HOST_DEVICE inline bool operator<(long long lhs, const dynd_int128& rhs) {
    return dynd_int128(lhs) < rhs;
}
DYND_CUDA_HOST_DEVICE inline bool operator<(unsigned long long lhs, const dynd_int128& rhs) {
    return dynd_int128(lhs) < rhs;
}

std::ostream& operator<<(std::ostream& out, const dynd_int128& val);

} // namespace dynd

namespace std {

template<>
class numeric_limits<dynd::dynd_int128> {
public:
  static const bool is_specialized = true;
  static ::dynd::dynd_int128 min() throw() {
    return ::dynd::dynd_int128(0x8000000000000000ULL, 0ULL);
  }
  static ::dynd::dynd_int128 max() throw() {
    return ::dynd::dynd_int128(0x7fffffffffffffffULL, 0xffffffffffffffffULL);
  }
  static const int  digits = 0;
  static const int  digits10 = 0;
  static const bool is_signed = true;
  static const bool is_integer = true;
  static const bool is_exact = true;
  static const int radix = 2;
  static ::dynd::dynd_int128 epsilon() throw() {
    return ::dynd::dynd_int128(0ULL, 1ULL);
  }
  static ::dynd::dynd_int128 round_error() throw() {
    return ::dynd::dynd_int128(0ULL, 1ULL);
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
  static ::dynd::dynd_int128 infinity() throw();
  static ::dynd::dynd_int128 quiet_NaN() throw();
  static ::dynd::dynd_int128 signaling_NaN() throw();
  static ::dynd::dynd_int128 denorm_min() throw();

  static const bool is_iec559 = false;
  static const bool is_bounded = false;
  static const bool is_modulo = false;

  static const bool traps = false;
  static const bool tinyness_before = false;
  static const float_round_style round_style = round_toward_zero;
};
    
} // namespace std

#endif // !defined(DYND_HAS_INT128)


#endif // _DYND__INT128_H__
