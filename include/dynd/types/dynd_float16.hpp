//
// Copyright (C) 2010-13 Mark Wiebe, DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#ifndef _DYND__FLOAT16_H__
#define _DYND__FLOAT16_H__

#include <dynd/config.hpp>
#include <dynd/typed_data_assign.hpp>

// Half-precision constants, in bits form
#define DYND_FLOAT16_ZERO   (0x0000u)
#define DYND_FLOAT16_PZERO  (0x0000u)
#define DYND_FLOAT16_NZERO  (0x8000u)
#define DYND_FLOAT16_ONE    (0x3c00u)
#define DYND_FLOAT16_NEGONE (0xbc00u)
#define DYND_FLOAT16_PINF   (0x7c00u)
#define DYND_FLOAT16_NINF   (0xfc00u)
#define DYND_FLOAT16_NAN    (0x7e00u)
#define DYND_FLOAT16_MAX    (0x7bffu)

namespace dynd {

#if !defined(DYND_HAS_FLOAT128)
class dynd_float128;
#endif
#if !defined(DYND_HAS_INT128)
class dynd_int128;
#endif
#if !defined(DYND_HAS_UINT128)
class dynd_uint128;
#endif

// Bit-level conversions
DYND_CUDA_HOST_DEVICE uint16_t float_to_halfbits(float value, assign_error_mode errmode);
DYND_CUDA_HOST_DEVICE uint16_t double_to_halfbits(double value, assign_error_mode errmode);
DYND_CUDA_HOST_DEVICE float halfbits_to_float(uint16_t value);
DYND_CUDA_HOST_DEVICE double halfbits_to_double(uint16_t value);

class dynd_float16 {
    uint16_t m_bits;

public:
    class raw_bits_tag {};
    DYND_CUDA_HOST_DEVICE inline dynd_float16(uint16_t bits, raw_bits_tag)
        : m_bits(bits) {}

    DYND_CUDA_HOST_DEVICE inline dynd_float16() {}
    DYND_CUDA_HOST_DEVICE inline dynd_float16(const dynd_float16& rhs)
        : m_bits(rhs.m_bits) {}
    DYND_CUDA_HOST_DEVICE inline dynd_float16(float f, assign_error_mode errmode)
        : m_bits(float_to_halfbits(f, errmode)) {}
    DYND_CUDA_HOST_DEVICE inline dynd_float16(double d, assign_error_mode errmode)
        : m_bits(double_to_halfbits(d, errmode)) {}
    DYND_CUDA_HOST_DEVICE inline explicit dynd_float16(bool rhs)
        : m_bits(rhs ? DYND_FLOAT16_ONE : DYND_FLOAT16_ZERO) {}

    DYND_CUDA_HOST_DEVICE inline dynd_float16(float f)
        : m_bits(float_to_halfbits(f, assign_error_none)) {}
    DYND_CUDA_HOST_DEVICE inline dynd_float16(double d)
        : m_bits(double_to_halfbits(d, assign_error_none)) {}
    DYND_CUDA_HOST_DEVICE inline dynd_float16(int32_t value)
        : m_bits(float_to_halfbits((float)value, assign_error_none)) {}
    DYND_CUDA_HOST_DEVICE inline dynd_float16(uint32_t value)
        : m_bits(float_to_halfbits((float)value, assign_error_none)) {}
    DYND_CUDA_HOST_DEVICE inline dynd_float16(int64_t value)
        : m_bits(float_to_halfbits((float)value, assign_error_none)) {}
    DYND_CUDA_HOST_DEVICE inline dynd_float16(uint64_t value)
        : m_bits(float_to_halfbits((float)value, assign_error_none)) {}
    DYND_CUDA_HOST_DEVICE dynd_float16(const dynd_int128& value);
    DYND_CUDA_HOST_DEVICE dynd_float16(const dynd_uint128& value);
    DYND_CUDA_HOST_DEVICE dynd_float16(const dynd_float128& value);

    DYND_CUDA_HOST_DEVICE inline operator float() const {
        return halfbits_to_float(m_bits);
    }
    DYND_CUDA_HOST_DEVICE inline operator double() const {
        return halfbits_to_double(m_bits);
    }
    DYND_CUDA_HOST_DEVICE inline operator int8_t() const {
        return (int8_t)halfbits_to_float(m_bits);
    }
    DYND_CUDA_HOST_DEVICE inline operator uint8_t() const {
        return (uint8_t)halfbits_to_float(m_bits);
    }
    DYND_CUDA_HOST_DEVICE inline operator int16_t() const {
        return (int16_t)halfbits_to_float(m_bits);
    }
    DYND_CUDA_HOST_DEVICE inline operator uint16_t() const {
        return (uint16_t)halfbits_to_float(m_bits);
    }
    DYND_CUDA_HOST_DEVICE inline operator int32_t() const {
        return (int32_t)halfbits_to_float(m_bits);
    }
    DYND_CUDA_HOST_DEVICE inline operator uint32_t() const {
        return (uint32_t)halfbits_to_float(m_bits);
    }
    DYND_CUDA_HOST_DEVICE inline operator int64_t() const {
        return (int64_t)halfbits_to_float(m_bits);
    }
    DYND_CUDA_HOST_DEVICE inline operator uint64_t() const {
        return (uint64_t)halfbits_to_float(m_bits);
    }
    DYND_CUDA_HOST_DEVICE operator dynd_int128() const;
    DYND_CUDA_HOST_DEVICE operator dynd_uint128() const;
    DYND_CUDA_HOST_DEVICE operator dynd_float128() const;


    DYND_CUDA_HOST_DEVICE inline uint16_t bits() const {
        return m_bits;
    }

    DYND_CUDA_HOST_DEVICE inline bool iszero() const {
        return (m_bits&0x7fff) == 0;
    }

    DYND_CUDA_HOST_DEVICE inline bool signbit_() const {
        return (m_bits&0x8000u) != 0;
    }

    DYND_CUDA_HOST_DEVICE inline bool isnan_() const {
        return ((m_bits&0x7c00u) == 0x7c00u) && ((m_bits&0x03ffu) != 0x0000u);
    }

    DYND_CUDA_HOST_DEVICE inline bool isinf_() const {
        return ((m_bits&0x7fffu) == 0x7c00u);
    }

    DYND_CUDA_HOST_DEVICE inline bool isfinite_() const {
        return ((m_bits&0x7c00u) != 0x7c00u);
    }

    DYND_CUDA_HOST_DEVICE inline bool operator==(const dynd_float16& rhs) const {
        // The equality cases are as follows:
        //   - If either value is NaN, never equal.
        //   - If the values are equal, equal.
        //   - If the values are both signed zeros, equal.
        return (!isnan_() && !rhs.isnan_()) &&
               (m_bits == rhs.m_bits || ((m_bits | rhs.m_bits) & 0x7fff) == 0);
    }

    DYND_CUDA_HOST_DEVICE inline bool operator!=(const dynd_float16& rhs) const {
        return !operator==(rhs);
    }

    DYND_CUDA_HOST_DEVICE bool less_nonan(const dynd_float16& rhs) const {
        if (signbit_()) {
            if (rhs.signbit_()) {
                return m_bits > rhs.m_bits;
            } else {
                // Signed zeros are equal, have to check for it
                return (m_bits != 0x8000u) || (rhs.m_bits != 0x0000u);
            }
        } else {
            if (rhs.signbit_()) {
                return false;
            } else {
                return m_bits < rhs.m_bits;
            }
        }
    }

    DYND_CUDA_HOST_DEVICE bool less_equal_nonan(const dynd_float16& rhs) const {
        if (signbit_()) {
            if (rhs.signbit_()) {
                return m_bits >= rhs.m_bits;
            } else {
                return true;
            }
        } else {
            if (rhs.signbit_()) {
                // Signed zeros are equal, have to check for it
                return (m_bits == 0x0000u) && (rhs.m_bits == 0x8000u);
            } else {
                return m_bits <= rhs.m_bits;
            }
        }
    }

    DYND_CUDA_HOST_DEVICE inline bool operator<(const dynd_float16& rhs) const {
        return !isnan_() && !rhs.isnan_() && less_nonan(rhs);
    }

    DYND_CUDA_HOST_DEVICE inline bool operator>(const dynd_float16& rhs) const {
        return rhs.operator<(*this);
    }

    DYND_CUDA_HOST_DEVICE inline bool operator<=(const dynd_float16& rhs) const {
        return !isnan_() && !rhs.isnan_() && less_equal_nonan(rhs);
    }

    DYND_CUDA_HOST_DEVICE inline bool operator>=(const dynd_float16& rhs) const {
        return rhs.operator<=(*this);
    }

    DYND_CUDA_HOST_DEVICE friend dynd_float16 float16_from_bits(uint16_t bits);
};

// Metaprogram for determining if a type is a valid C++ scalar
// of a particular type.
template<typename T> struct is_float16_scalar {enum {value = false};};
template <> struct is_float16_scalar<char> {enum {value = true};};
template <> struct is_float16_scalar<signed char> {enum {value = true};};
template <> struct is_float16_scalar<short> {enum {value = true};};
template <> struct is_float16_scalar<int> {enum {value = true};};
template <> struct is_float16_scalar<long> {enum {value = true};};
template <> struct is_float16_scalar<long long> {enum {value = true};};
template <> struct is_float16_scalar<unsigned char> {enum {value = true};};
template <> struct is_float16_scalar<unsigned short> {enum {value = true};};
template <> struct is_float16_scalar<unsigned int> {enum {value = true};};
template <> struct is_float16_scalar<unsigned long> {enum {value = true};};
template <> struct is_float16_scalar<unsigned long long> {enum {value = true};};
template <> struct is_float16_scalar<float> {enum {value = true};};
template <> struct is_float16_scalar<double> {enum {value = true};};

template<class T>
inline typename enable_if<is_float16_scalar<T>::value, bool>::type operator<(const T& lhs, const dynd_float16& rhs) {
    return double(lhs) < double(rhs);
}
template<class T>
inline typename enable_if<is_float16_scalar<T>::value, bool>::type operator>(const T& lhs, const dynd_float16& rhs) {
    return double(lhs) > double(rhs);
}
template<class T>
inline typename enable_if<is_float16_scalar<T>::value, bool>::type operator<=(const T& lhs, const dynd_float16& rhs) {
    return double(lhs) <= double(rhs);
}
template<class T>
inline typename enable_if<is_float16_scalar<T>::value, bool>::type operator>=(const T& lhs, const dynd_float16& rhs) {
    return double(lhs) >= double(rhs);
}
template<class T>
inline typename enable_if<is_float16_scalar<T>::value, bool>::type operator==(const T& lhs, const dynd_float16& rhs) {
    return double(lhs) == double(rhs);
}
template<class T>
inline typename enable_if<is_float16_scalar<T>::value, bool>::type operator!=(const T& lhs, const dynd_float16& rhs) {
    return double(lhs) != double(rhs);
}

/**
 * Constructs a dynd_float16 from a uint16_t
 * containing its bits.
 */
DYND_CUDA_HOST_DEVICE inline dynd_float16 float16_from_bits(uint16_t bits) {
    return dynd_float16(bits, dynd_float16::raw_bits_tag());
}

} // namespace dynd

#endif // _DYND__FLOAT16_H__
