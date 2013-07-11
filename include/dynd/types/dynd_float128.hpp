//
// Copyright (C) 2010-13 Mark Wiebe, DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#ifndef _DYND__FLOAT128_H__
#define _DYND__FLOAT128_H__

#include <dynd/config.hpp>
#include <dynd/dtype_assign.hpp>

#include <iostream>
#include <stdexcept>

#if !defined(DYND_HAS_FLOAT128)

namespace dynd {

#if !defined(DYND_HAS_INT128)
class dynd_int128;
#endif
#if !defined(DYND_HAS_UINT128)
class dynd_uint128;
#endif

class dynd_float16;

class dynd_float128 {
public:
#if defined(DYND_BIG_ENDIAN)
    uint64_t m_hi, m_lo;
#else
    uint64_t m_lo, m_hi;
#endif
    inline dynd_float128() {}
    inline dynd_float128(uint64_t hi, uint64_t lo)
        : m_lo(lo), m_hi(hi) {}
    dynd_float128(signed char value);
    dynd_float128(unsigned char value);
    dynd_float128(short value);
    dynd_float128(unsigned short value);
    dynd_float128(int value);
    dynd_float128(unsigned int value);
    inline dynd_float128(long value) {
        *this = dynd_float128((long long)value);
    }
    inline dynd_float128(unsigned long value) {
        *this = dynd_float128((unsigned long long)value);
    }
    dynd_float128(long long value);
    dynd_float128(unsigned long long value);
    dynd_float128(double value);
    dynd_float128(const dynd_int128& value);
    dynd_float128(const dynd_uint128& value);
    dynd_float128(const dynd_float16& value);

    operator signed char() const {
        throw std::runtime_error("float128 conversions are not completed");
    }
    operator unsigned char() const {
        throw std::runtime_error("float128 conversions are not completed");
    }
    operator short() const {
        throw std::runtime_error("float128 conversions are not completed");
    }
    operator unsigned short() const {
        throw std::runtime_error("float128 conversions are not completed");
    }
    operator int() const {
        throw std::runtime_error("float128 conversions are not completed");
    }
    operator unsigned int() const {
        throw std::runtime_error("float128 conversions are not completed");
    }
    operator long() const {
        throw std::runtime_error("float128 conversions are not completed");
    }
    operator unsigned long() const {
        throw std::runtime_error("float128 conversions are not completed");
    }
    operator long long() const {
        throw std::runtime_error("float128 conversions are not completed");
    }
    operator unsigned long long() const {
        throw std::runtime_error("float128 conversions are not completed");
    }
    operator double() const {
        throw std::runtime_error("float128 conversions are not completed");
    }


    inline explicit dynd_float128(const bool& rhs)
        : m_lo(0ULL), m_hi(rhs ? 0x3fff000000000000ULL : 0ULL) {}

    inline bool iszero() const {
        return (m_hi&0x7fffffffffffffffULL) == 0 && m_lo == 0;
    }

    inline bool signbit() const {
        return (m_hi&0x8000000000000000ULL) != 0;
    }

    inline bool isnan() const {
        return (m_hi&0x7fff000000000000ULL) == 0x7fff000000000000ULL &&
               ((m_hi&0x0000ffffffffffffULL) != 0ULL || m_lo != 0ULL);
    }

    inline bool isinf() const {
        return (m_hi&0x7fffffffffffffffULL) == 0x7fff000000000000ULL &&
               (m_lo == 0ULL);
    }

    bool isfinite() const {
        return (m_hi&0x7fff000000000000ULL) != 0x7fff000000000000ULL;
    }

    inline bool operator==(const dynd_float128& rhs) const {
        // The equality cases are as follows:
        //   - If either value is NaN, never equal.
        //   - If the values are equal, equal.
        //   - If the values are both signed zeros, equal.
        return (!isnan() && !rhs.isnan()) &&
               ((m_hi == rhs.m_hi && m_lo == rhs.m_lo) ||
                (((m_hi | rhs.m_hi) & 0x7fffffffffffffffULL) == 0ULL &&
                    (m_lo | rhs.m_lo) == 0ULL));
    }

    inline bool operator!=(const dynd_float128& rhs) const {
        return !operator==(rhs);
    }

    bool less_nonan(const dynd_float128& rhs) const {
        if (signbit()) {
            if (rhs.signbit()) {
                return m_hi > rhs.m_hi ||
                    (m_hi == rhs.m_hi && m_lo > rhs.m_lo);
            } else {
                // Signed zeros are equal, have to check for it
                return (m_hi != 0x8000000000000000ULL) || (m_lo != 0LL) ||
                       (rhs.m_hi != 0LL) || rhs.m_lo != 0LL;
            }
        } else {
            if (rhs.signbit()) {
                return false;
            } else {
                return m_hi < rhs.m_hi ||
                    (m_hi == rhs.m_hi && m_lo < rhs.m_lo);
            }
        }
    }

    bool less_equal_nonan(const dynd_float128& rhs) const {
        if (signbit()) {
            if (rhs.signbit()) {
                return m_hi > rhs.m_hi ||
                    (m_hi == rhs.m_hi && m_lo >= rhs.m_lo);
            } else {
                return true;
            }
        } else {
            if (rhs.signbit()) {
                // Signed zeros are equal, have to check for it
                return (m_hi == 0x8000000000000000ULL) && (m_lo == 0LL) &&
                       (rhs.m_hi == 0LL) && rhs.m_lo == 0LL;
            } else {
                return m_hi < rhs.m_hi ||
                    (m_hi == rhs.m_hi && m_lo <= rhs.m_lo);
            }
        }
    }

    inline bool operator<(const dynd_float128& rhs) const {
        return !isnan() && !rhs.isnan() && less_nonan(rhs);
    }

    inline bool operator>(const dynd_float128& rhs) const {
        return rhs.operator<(*this);
    }

    inline bool operator<=(const dynd_float128& rhs) const {
        return !isnan() && !rhs.isnan() && less_equal_nonan(rhs);
    }

    inline bool operator>=(const dynd_float128& rhs) const {
        return rhs.operator<=(*this);
    }

};

inline std::ostream& operator<<(std::ostream& o, const dynd_float128& DYND_UNUSED(rhs)) {
    return (o << "<float128 printing unimplemented>");
}

} // namespace dynd

#endif // !defined(DYND_HAS_FLOAT128)

#endif // _DYND__FLOAT128_H__
