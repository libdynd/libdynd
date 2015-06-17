//
// Copyright (C) 2011-15 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#pragma once

#include <dynd/config.hpp>
#include <dynd/diagnostics.hpp>
#include <dynd/typed_data_assign.hpp>

#include <iostream>
#include <stdexcept>

#if !defined(DYND_HAS_FLOAT128)

namespace dynd {

#if !defined(DYND_HAS_INT128)
class int128;
#endif
#if !defined(DYND_HAS_UINT128)
class uint128;
#endif

class float16;

class float128 {
public:
#if defined(DYND_BIG_ENDIAN)
  uint64_t m_hi, m_lo;
#else
  uint64_t m_lo, m_hi;
#endif
  DYND_CUDA_HOST_DEVICE float128() {}
  DYND_CUDA_HOST_DEVICE float128(uint64_t hi, uint64_t lo) : m_lo(lo), m_hi(hi)
  {
  }

  float128(bool1) {}

  DYND_CUDA_HOST_DEVICE float128(signed char value);
  DYND_CUDA_HOST_DEVICE float128(unsigned char value);
  DYND_CUDA_HOST_DEVICE float128(short value);
  DYND_CUDA_HOST_DEVICE float128(unsigned short value);
  DYND_CUDA_HOST_DEVICE float128(int value);
  DYND_CUDA_HOST_DEVICE float128(unsigned int value);
  DYND_CUDA_HOST_DEVICE float128(long value)
  {
    *this = float128((long long)value);
  }
  DYND_CUDA_HOST_DEVICE float128(unsigned long value)
  {
    *this = float128((unsigned long long)value);
  }
  DYND_CUDA_HOST_DEVICE float128(long long value);
  DYND_CUDA_HOST_DEVICE float128(unsigned long long value);
  DYND_CUDA_HOST_DEVICE float128(double value);
  DYND_CUDA_HOST_DEVICE float128(const int128 &value);
  DYND_CUDA_HOST_DEVICE float128(const uint128 &value);
  DYND_CUDA_HOST_DEVICE float128(const float16 &value);

  DYND_CUDA_HOST_DEVICE operator signed char() const
  {
#ifdef __CUDA_ARCH__
    DYND_TRIGGER_ASSERT_RETURN_ZERO("float128 conversions are not completed");
#else
    throw std::runtime_error("float128 conversions are not completed");
#endif
  }
  DYND_CUDA_HOST_DEVICE operator unsigned char() const
  {
#ifdef __CUDA_ARCH__
    DYND_TRIGGER_ASSERT_RETURN_ZERO("float128 conversions are not completed");
#else
    throw std::runtime_error("float128 conversions are not completed");
#endif
  }
  DYND_CUDA_HOST_DEVICE operator short() const
  {
#ifdef __CUDA_ARCH__
    DYND_TRIGGER_ASSERT_RETURN_ZERO("float128 conversions are not completed");
#else
    throw std::runtime_error("float128 conversions are not completed");
#endif
  }
  DYND_CUDA_HOST_DEVICE operator unsigned short() const
  {
#ifdef __CUDA_ARCH__
    DYND_TRIGGER_ASSERT_RETURN_ZERO("float128 conversions are not completed");
#else
    throw std::runtime_error("float128 conversions are not completed");
#endif
  }
  DYND_CUDA_HOST_DEVICE operator int() const
  {
#ifdef __CUDA_ARCH__
    DYND_TRIGGER_ASSERT_RETURN_ZERO("float128 conversions are not completed");
#else
    throw std::runtime_error("float128 conversions are not completed");
#endif
  }
  DYND_CUDA_HOST_DEVICE operator unsigned int() const
  {
#ifdef __CUDA_ARCH__
    DYND_TRIGGER_ASSERT_RETURN_ZERO("float128 conversions are not completed");
#else
    throw std::runtime_error("float128 conversions are not completed");
#endif
  }
  DYND_CUDA_HOST_DEVICE operator long() const
  {
#ifdef __CUDA_ARCH__
    DYND_TRIGGER_ASSERT_RETURN_ZERO("float128 conversions are not completed");
#else
    throw std::runtime_error("float128 conversions are not completed");
#endif
  }
  DYND_CUDA_HOST_DEVICE operator unsigned long() const
  {
#ifdef __CUDA_ARCH__
    DYND_TRIGGER_ASSERT_RETURN_ZERO("float128 conversions are not completed");
#else
    throw std::runtime_error("float128 conversions are not completed");
#endif
  }
  DYND_CUDA_HOST_DEVICE operator long long() const
  {
#ifdef __CUDA_ARCH__
    DYND_TRIGGER_ASSERT_RETURN_ZERO("float128 conversions are not completed");
#else
    throw std::runtime_error("float128 conversions are not completed");
#endif
  }
  DYND_CUDA_HOST_DEVICE operator unsigned long long() const
  {
#ifdef __CUDA_ARCH__
    DYND_TRIGGER_ASSERT_RETURN_ZERO("float128 conversions are not completed");
#else
    throw std::runtime_error("float128 conversions are not completed");
#endif
  }
  DYND_CUDA_HOST_DEVICE operator double() const
  {
#ifdef __CUDA_ARCH__
    DYND_TRIGGER_ASSERT_RETURN_ZERO("float128 conversions are not completed");
#else
    throw std::runtime_error("float128 conversions are not completed");
#endif
  }

  DYND_CUDA_HOST_DEVICE inline explicit float128(const bool &rhs)
      : m_lo(0ULL), m_hi(rhs ? 0x3fff000000000000ULL : 0ULL)
  {
  }

  DYND_CUDA_HOST_DEVICE inline bool iszero() const
  {
    return (m_hi & 0x7fffffffffffffffULL) == 0 && m_lo == 0;
  }

  DYND_CUDA_HOST_DEVICE inline bool signbit_() const
  {
    return (m_hi & 0x8000000000000000ULL) != 0;
  }

  DYND_CUDA_HOST_DEVICE inline bool isnan_() const
  {
    return (m_hi & 0x7fff000000000000ULL) == 0x7fff000000000000ULL &&
           ((m_hi & 0x0000ffffffffffffULL) != 0ULL || m_lo != 0ULL);
  }

  DYND_CUDA_HOST_DEVICE inline bool isinf_() const
  {
    return (m_hi & 0x7fffffffffffffffULL) == 0x7fff000000000000ULL &&
           (m_lo == 0ULL);
  }

  DYND_CUDA_HOST_DEVICE bool isfinite_() const
  {
    return (m_hi & 0x7fff000000000000ULL) != 0x7fff000000000000ULL;
  }

  DYND_CUDA_HOST_DEVICE inline bool operator==(const float128 &rhs) const
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

  DYND_CUDA_HOST_DEVICE inline bool operator!=(const float128 &rhs) const
  {
    return !operator==(rhs);
  }

  DYND_CUDA_HOST_DEVICE bool less_nonan(const float128 &rhs) const
  {
    if (signbit_()) {
      if (rhs.signbit_()) {
        return m_hi > rhs.m_hi || (m_hi == rhs.m_hi && m_lo > rhs.m_lo);
      } else {
        // Signed zeros are equal, have to check for it
        return (m_hi != 0x8000000000000000ULL) || (m_lo != 0LL) ||
               (rhs.m_hi != 0LL) || rhs.m_lo != 0LL;
      }
    } else {
      if (rhs.signbit_()) {
        return false;
      } else {
        return m_hi < rhs.m_hi || (m_hi == rhs.m_hi && m_lo < rhs.m_lo);
      }
    }
  }

  DYND_CUDA_HOST_DEVICE bool less_equal_nonan(const float128 &rhs) const
  {
    if (signbit_()) {
      if (rhs.signbit_()) {
        return m_hi > rhs.m_hi || (m_hi == rhs.m_hi && m_lo >= rhs.m_lo);
      } else {
        return true;
      }
    } else {
      if (rhs.signbit_()) {
        // Signed zeros are equal, have to check for it
        return (m_hi == 0x8000000000000000ULL) && (m_lo == 0LL) &&
               (rhs.m_hi == 0LL) && rhs.m_lo == 0LL;
      } else {
        return m_hi < rhs.m_hi || (m_hi == rhs.m_hi && m_lo <= rhs.m_lo);
      }
    }
  }

  DYND_CUDA_HOST_DEVICE inline bool operator<(const float128 &rhs) const
  {
    return !isnan_() && !rhs.isnan_() && less_nonan(rhs);
  }

  DYND_CUDA_HOST_DEVICE inline bool operator>(const float128 &rhs) const
  {
    return rhs.operator<(*this);
  }

  DYND_CUDA_HOST_DEVICE inline bool operator<=(const float128 &rhs) const
  {
    return !isnan_() && !rhs.isnan_() && less_equal_nonan(rhs);
  }

  DYND_CUDA_HOST_DEVICE inline bool operator>=(const float128 &rhs) const
  {
    return rhs.operator<=(*this);
  }
};

inline std::ostream &operator<<(std::ostream &o,
                                const float128 &DYND_UNUSED(rhs))
{
  return (o << "<float128 printing unimplemented>");
}

} // namespace dynd

#endif // !defined(DYND_HAS_FLOAT128)
