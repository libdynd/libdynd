//
// Copyright (C) 2011-15 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#pragma once

// Half-precision constants, in bits form
#define DYND_FLOAT16_ZERO (0x0000u)
#define DYND_FLOAT16_PZERO (0x0000u)
#define DYND_FLOAT16_NZERO (0x8000u)
#define DYND_FLOAT16_ONE (0x3c00u)
#define DYND_FLOAT16_NEGONE (0xbc00u)
#define DYND_FLOAT16_PINF (0x7c00u)
#define DYND_FLOAT16_NINF (0xfc00u)
#define DYND_FLOAT16_NAN (0x7e00u)
#define DYND_FLOAT16_MAX (0x7bffu)

namespace dynd {

// Bit-level conversions
DYND_CUDA_HOST_DEVICE DYND_API uint16_t float_to_halfbits(float value);

DYND_CUDA_HOST_DEVICE DYND_API uint16_t double_to_halfbits(double value);

DYND_CUDA_HOST_DEVICE DYND_API float halfbits_to_float(uint16_t value);

DYND_CUDA_HOST_DEVICE DYND_API double halfbits_to_double(uint16_t value);

class DYND_API float16 {
  uint16_t m_bits;

public:
  class raw_bits_tag {
  };
  DYND_CUDA_HOST_DEVICE explicit float16(uint16_t bits, raw_bits_tag)
      : m_bits(bits)
  {
  }

  DYND_CUDA_HOST_DEVICE float16()
  {
  }

  DYND_CUDA_HOST_DEVICE explicit float16(bool1 rhs)
      : m_bits(rhs ? DYND_FLOAT16_ONE : DYND_FLOAT16_ZERO)
  {
  }

  DYND_CUDA_HOST_DEVICE explicit float16(bool rhs)
      : m_bits(rhs ? DYND_FLOAT16_ONE : DYND_FLOAT16_ZERO)
  {
  }

  DYND_CUDA_HOST_DEVICE explicit float16(int8 value)
      : float16(static_cast<float32>(value))
  {
  }

  DYND_CUDA_HOST_DEVICE explicit float16(int16 value)
      : float16(static_cast<float32>(value))
  {
  }

  DYND_CUDA_HOST_DEVICE explicit float16(int32 value)
      : float16(static_cast<float32>(value))
  {
  }

  DYND_CUDA_HOST_DEVICE explicit float16(int64 value)
      : float16(static_cast<float32>(value))
  {
  }

  DYND_CUDA_HOST_DEVICE explicit float16(int128 value);

  DYND_CUDA_HOST_DEVICE explicit float16(uint16 value)
      : float16(static_cast<float32>(value))
  {
  }

  DYND_CUDA_HOST_DEVICE explicit float16(uint32 value)
      : float16(static_cast<float32>(value))
  {
  }

  DYND_CUDA_HOST_DEVICE explicit float16(uint64 value)
      : float16(static_cast<float32>(value))
  {
  }

  DYND_CUDA_HOST_DEVICE explicit float16(uint128 value);

  DYND_CUDA_HOST_DEVICE
  explicit float16(float f) : m_bits(float_to_halfbits(f))
  {
  }

  DYND_CUDA_HOST_DEVICE
  explicit float16(double d) : m_bits(double_to_halfbits(d))
  {
  }

  DYND_CUDA_HOST_DEVICE explicit float16(const float128 &value);

  DYND_CUDA_HOST_DEVICE float16(const float16 &rhs) : m_bits(rhs.m_bits)
  {
  }

  DYND_CUDA_HOST_DEVICE explicit operator int8() const
  {
    return static_cast<int8>(halfbits_to_float(m_bits));
  }

  DYND_CUDA_HOST_DEVICE explicit operator int16() const
  {
    DYND_IGNORE_MAYBE_UNINITIALIZED
    return static_cast<int16>(halfbits_to_float(m_bits));
    DYND_END_IGNORE_MAYBE_UNINITIALIZED
  }

  DYND_CUDA_HOST_DEVICE explicit operator int32() const
  {
    return static_cast<int32>(halfbits_to_float(m_bits));
  }

  DYND_CUDA_HOST_DEVICE explicit operator int64() const
  {
    return static_cast<int64>(halfbits_to_float(m_bits));
  }

  DYND_CUDA_HOST_DEVICE explicit operator int128() const;

  DYND_CUDA_HOST_DEVICE explicit operator uint8() const
  {
    return static_cast<uint8>(halfbits_to_float(m_bits));
  }

  DYND_CUDA_HOST_DEVICE explicit operator uint16() const
  {
    return static_cast<uint16>(halfbits_to_float(m_bits));
  }

  DYND_CUDA_HOST_DEVICE explicit operator uint32() const
  {
    return static_cast<uint32>(halfbits_to_float(m_bits));
  }

  DYND_CUDA_HOST_DEVICE explicit operator uint64() const
  {
    return static_cast<uint64>(halfbits_to_float(m_bits));
  }

  DYND_CUDA_HOST_DEVICE explicit operator uint128() const;

  DYND_CUDA_HOST_DEVICE operator float32() const
  {
    DYND_IGNORE_MAYBE_UNINITIALIZED
    return halfbits_to_float(m_bits);
    DYND_END_IGNORE_MAYBE_UNINITIALIZED
  }

  DYND_CUDA_HOST_DEVICE explicit operator float64() const
  {
    return halfbits_to_double(m_bits);
  }

  DYND_CUDA_HOST_DEVICE explicit operator float128() const;

  DYND_CUDA_HOST_DEVICE uint16_t bits() const
  {
    return m_bits;
  }

  DYND_CUDA_HOST_DEVICE bool iszero() const
  {
    return (m_bits & 0x7fff) == 0;
  }

  DYND_CUDA_HOST_DEVICE bool signbit_() const
  {
    return (m_bits & 0x8000u) != 0;
  }

  DYND_CUDA_HOST_DEVICE bool isnan_() const
  {
    return ((m_bits & 0x7c00u) == 0x7c00u) && ((m_bits & 0x03ffu) != 0x0000u);
  }

  DYND_CUDA_HOST_DEVICE bool isinf_() const
  {
    return ((m_bits & 0x7fffu) == 0x7c00u);
  }

  DYND_CUDA_HOST_DEVICE bool isfinite_() const
  {
    return ((m_bits & 0x7c00u) != 0x7c00u);
  }

  /*
    DYND_CUDA_HOST_DEVICE bool operator==(const float16 &rhs) const
    {
      // The equality cases are as follows:
      //   - If either value is NaN, never equal.
      //   - If the values are equal, equal.
      //   - If the values are both signed zeros, equal.
      return (!isnan_() && !rhs.isnan_()) &&
             (m_bits == rhs.m_bits || ((m_bits | rhs.m_bits) & 0x7fff) == 0);
    }

    DYND_CUDA_HOST_DEVICE bool operator!=(const float16 &rhs) const
    {
      return !operator==(rhs);
    }
  */

  DYND_CUDA_HOST_DEVICE bool less_nonan(const float16 &rhs) const
  {
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

  DYND_CUDA_HOST_DEVICE bool less_equal_nonan(const float16 &rhs) const
  {
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

  /*
    DYND_CUDA_HOST_DEVICE bool operator<(const float16 &rhs) const
    {
      return !isnan_() && !rhs.isnan_() && less_nonan(rhs);
    }

    DYND_CUDA_HOST_DEVICE bool operator>(const float16 &rhs) const
    {
      return rhs.operator<(*this);
    }

    DYND_CUDA_HOST_DEVICE bool operator<=(const float16 &rhs) const
    {
      return !isnan_() && !rhs.isnan_() && less_equal_nonan(rhs);
    }

    DYND_CUDA_HOST_DEVICE bool operator>=(const float16 &rhs) const
    {
      return rhs.operator<=(*this);
    }
  */

  DYND_CUDA_HOST_DEVICE friend float16 float16_from_bits(uint16_t bits);

  float16 operator+() const {
    return *this;
  }

  float16 operator-() const
  {
    return float16(-static_cast<float>(*this));
  }

  bool operator!() const
  {
    return (0x7fffu | m_bits) == 0;
  }

  DYND_CUDA_HOST_DEVICE explicit operator bool() const {
    return (0x7ffu | m_bits) != 0;
  }

};

DYND_CUDA_HOST_DEVICE inline bool isfinite(float16 value)
{
  return value.isfinite_();
}

inline float16 operator+(const float16 &DYND_UNUSED(lhs),
                         const float16 &DYND_UNUSED(rhs))
{
  throw std::runtime_error("+ is not implemented for float16");
}

inline bool operator<(const float16 &lhs, const float16 &rhs)
{
  return static_cast<double>(lhs) < static_cast<double>(rhs);
}

template <typename T>
typename std::enable_if<
    is_arithmetic<T>::value && !std::is_same<T, float128>::value, bool>::type
operator<(const float16 &lhs, const T &rhs)
{
  return static_cast<double>(lhs) < static_cast<double>(rhs);
}

template <typename T>
typename std::enable_if<
    is_arithmetic<T>::value && !std::is_same<T, float128>::value, bool>::type
operator<(const T &lhs, const float16 &rhs)
{
  return static_cast<double>(lhs) < static_cast<double>(rhs);
}

inline bool operator<=(const float16 &lhs, const float16 &rhs)
{
  return static_cast<double>(lhs) <= static_cast<double>(rhs);
}

template <typename T>
typename std::enable_if<is_arithmetic<T>::value, bool>::type
operator<=(const float16 &lhs, const T &rhs)
{
  return static_cast<double>(lhs) <= static_cast<double>(rhs);
}

template <typename T>
typename std::enable_if<is_arithmetic<T>::value, bool>::type
operator<=(const T &lhs, const float16 &rhs)
{
  return static_cast<double>(lhs) <= static_cast<double>(rhs);
}

inline bool operator==(const float16 &lhs, const float16 &rhs)
{
  return static_cast<double>(lhs) == static_cast<double>(rhs);
}

template <typename T>
typename std::enable_if<
    is_arithmetic<T>::value && !std::is_same<T, float128>::value, bool>::type
operator==(const float16 &lhs, const T &rhs)
{
  return static_cast<double>(lhs) == static_cast<double>(rhs);
}

template <typename T>
typename std::enable_if<
    is_arithmetic<T>::value && !std::is_same<T, float128>::value, bool>::type
operator==(const T &lhs, const float16 &rhs)
{
  return static_cast<double>(lhs) == static_cast<double>(rhs);
}

inline bool operator!=(const float16 &lhs, const float16 &rhs)
{
  return static_cast<double>(lhs) != static_cast<double>(rhs);
}

template <typename T>
typename std::enable_if<
    is_arithmetic<T>::value && !std::is_same<T, float128>::value, bool>::type
operator!=(const float16 &lhs, const T &rhs)
{
  return static_cast<double>(lhs) != static_cast<double>(rhs);
}

template <typename T>
typename std::enable_if<
    is_arithmetic<T>::value && !std::is_same<T, float128>::value, bool>::type
operator!=(const T &lhs, const float16 &rhs)
{
  return static_cast<double>(lhs) != static_cast<double>(rhs);
}

inline bool operator>=(const float16 &lhs, const float16 &rhs)
{
  return static_cast<double>(lhs) >= static_cast<double>(rhs);
}

template <typename T>
typename std::enable_if<is_arithmetic<T>::value, bool>::type
operator>=(const float16 &lhs, const T &rhs)
{
  return static_cast<double>(lhs) >= static_cast<double>(rhs);
}

template <typename T>
typename std::enable_if<is_arithmetic<T>::value, bool>::type
operator>=(const T &lhs, const float16 &rhs)
{
  return static_cast<double>(lhs) >= static_cast<double>(rhs);
}

inline bool operator>(const float16 &lhs, const float16 &rhs)
{
  return static_cast<double>(lhs) > static_cast<double>(rhs);
}

template <typename T>
typename std::enable_if<
    is_arithmetic<T>::value && !std::is_same<T, float128>::value, bool>::type
operator>(const float16 &lhs, const T &rhs)
{
  return static_cast<double>(lhs) > static_cast<double>(rhs);
}

template <typename T>
typename std::enable_if<
    is_arithmetic<T>::value && !std::is_same<T, float128>::value, bool>::type
operator>(const T &lhs, const float16 &rhs)
{
  return static_cast<double>(lhs) > static_cast<double>(rhs);
}

/**
 * Constructs a float16 from a uint16_t
 * containing its bits.
 */
DYND_CUDA_HOST_DEVICE inline float16 float16_from_bits(uint16_t bits)
{
  return float16(bits, float16::raw_bits_tag());
}

inline float16 floor(float16 value)
{
  return float16(std::floor(static_cast<float>(value)));
}

inline std::ostream &operator<<(std::ostream &o,
                                const float16 &DYND_UNUSED(rhs))
{
  return (o << "<float16 printing unimplemented>");
}

} // namespace dynd
