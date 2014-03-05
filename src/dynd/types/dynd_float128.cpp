//
// Copyright (C) 2010-13 Mark Wiebe, DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <dynd/types/dynd_float128.hpp>
#include <dynd/types/dynd_float16.hpp>

#include <sstream>

#if !defined(DYND_HAS_FLOAT128)

using namespace std;
using namespace dynd;

namespace {
    DYND_CUDA_HOST_DEVICE inline uint8_t leading_zerobits(uint8_t value) {
        return (value&0xf0u) ?
                    ((value&0xc0u) ? ((value&0x80u) ? 0u : 1u)
                                   : ((value&0x20u) ? 2u : 3u))
                             :
                    ((value&0x0cu) ? ((value&0x08u) ? 4u : 5u)
                                   : ((value&0x02u) ? 6u : 7u));
    }

    DYND_CUDA_HOST_DEVICE inline uint8_t leading_zerobits(uint16_t value) {
        return (value&0xff00u) ? (leading_zerobits((uint8_t)(value >> 8)))
                              : (leading_zerobits((uint8_t)value) + 8u);
    }

    DYND_CUDA_HOST_DEVICE inline uint8_t leading_zerobits(uint32_t value) {
        return (value&0xffff0000u) ? (leading_zerobits((uint16_t)(value >> 16)))
                              : (leading_zerobits((uint16_t)value) + 16u);
    }

    DYND_CUDA_HOST_DEVICE inline uint8_t leading_zerobits(uint64_t value) {
        return (value&0xffffffff00000000ULL)
                            ? (leading_zerobits((uint32_t)(value >> 32)))
                            : (leading_zerobits((uint32_t)value) + 32u);
    }
} // anonymous namespace

dynd::dynd_float128::dynd_float128(signed char value)
{
    if (value == 0) {
        m_hi = 0LL;
    } else {
        if (value < 0) {
            m_hi = 0x8000000000000000ULL;
            value = -value;
        } else {
            m_hi = 0LL;
        }
        uint8_t lead = leading_zerobits(uint8_t(value));
        m_hi += uint64_t(16838 + 8 - lead) << 48;
        m_hi += (uint64_t(value) << (41 + lead)) & 0x0000ffffffffffffULL;
    }
    m_lo = 0LL;
}

dynd::dynd_float128::dynd_float128(unsigned char value)
{
    if (value == 0) {
        m_hi = 0LL;
    } else {
        uint8_t lead = leading_zerobits(uint8_t(value));
        m_hi = uint64_t(16838 + 8 - lead) << 48;
        m_hi += (uint64_t(value) << (41 + lead)) & 0x0000ffffffffffffULL;
    }
    m_lo = 0LL;
}

dynd::dynd_float128::dynd_float128(short value)
{
    if (value == 0) {
        m_hi = 0LL;
    } else {
        if (value < 0) {
            m_hi = 0x8000000000000000ULL;
            value = -value;
        } else {
            m_hi = 0LL;
        }
        uint8_t lead = leading_zerobits(uint16_t(value));
        m_hi += uint64_t(16838 + 16 - lead) << 48;
        m_hi += (uint64_t(value) << (33 + lead)) & 0x0000ffffffffffffULL;
    }
    m_lo = 0LL;
}

dynd::dynd_float128::dynd_float128(unsigned short value)
{
    if (value == 0) {
        m_hi = 0LL;
    } else {
        uint8_t lead = leading_zerobits(uint16_t(value));
        m_hi = uint64_t(16838 + 16 - lead) << 48;
        m_hi += (uint64_t(value) << (33 + lead)) & 0x0000ffffffffffffULL;
    }
    m_lo = 0LL;
}

dynd::dynd_float128::dynd_float128(int value)
{
    if (value == 0) {
        m_hi = 0LL;
    } else {
        if (value < 0) {
            m_hi = 0x8000000000000000ULL;
            value = -value;
        } else {
            m_hi = 0LL;
        }
        uint8_t lead = leading_zerobits(uint32_t(value));
        m_hi += uint64_t(16838 + 32 - lead) << 48;
        m_hi += (uint64_t(value) << (17 + lead)) & 0x0000ffffffffffffULL;
    }
    m_lo = 0LL;
}

dynd::dynd_float128::dynd_float128(unsigned int value)
{
    if (value == 0) {
        m_hi = 0LL;
    } else {
        uint8_t lead = leading_zerobits(uint32_t(value));
        m_hi = uint64_t(16838 + 32 - lead) << 48;
        m_hi += (uint64_t(value) << (17 + lead)) & 0x0000ffffffffffffULL;
    }
    m_lo = 0LL;
}

dynd::dynd_float128::dynd_float128(long long value)
{
    if (value == 0) {
        m_hi = 0LL;
        m_lo = 0LL;
    } else {
        if (value < 0) {
            m_hi = 0x8000000000000000ULL;
            value = -value;
        } else {
            m_hi = 0LL;
        }
        uint8_t lead = leading_zerobits(uint64_t(value));
        m_hi += uint64_t(16838 + 64 - lead) << 48;
        m_hi += (uint64_t(value) << (-15 + lead)) & 0x0000ffffffffffffULL;
        m_lo = uint64_t(value) << (49 + lead);
    }
}

dynd::dynd_float128::dynd_float128(unsigned long long value)
{
    if (value == 0) {
        m_hi = 0LL;
        m_lo = 0LL;
    } else {
        uint8_t lead = leading_zerobits(uint64_t(value));
        m_hi = uint64_t(16838 + 64 - lead) << 48;
        m_hi += (uint64_t(value) << (-15 + lead)) & 0x0000ffffffffffffULL;
        m_lo = uint64_t(value) << (49 + lead);
    }
}

dynd::dynd_float128::dynd_float128(const dynd_int128& DYND_UNUSED(value))
{
#ifdef DYND_CUDA_DEVICE_ARCH
    DYND_TRIGGER_ASSERT("dynd int128 to float128 conversion isn't implemented");
#else
    throw runtime_error("dynd int128 to float128 conversion isn't implemented");
#endif
}

dynd::dynd_float128::dynd_float128(const dynd_uint128& DYND_UNUSED(value))
{
#ifdef DYND_CUDA_DEVICE_ARCH
    DYND_TRIGGER_ASSERT("dynd int128 to float128 conversion isn't implemented");
#else
    throw runtime_error("dynd uint128 to float128 conversion isn't implemented");
#endif
}

dynd::dynd_float128::dynd_float128(const dynd_float16& value)
{
    *this = dynd_float128(double(value));
}

dynd::dynd_float128::dynd_float128(double value)
{
    union { double d; uint64_t dbits; } conv;
    conv.d = value;
    uint64_t d = conv.dbits;
    uint64_t d_exp, d_sig;
    uint64_t q_sgn, q_exp, q_sig;

    d_exp = (d&0x7ff0000000000000ULL);
    q_sgn = (d&0x8000000000000000ULL);
    switch (d_exp) {
        case 0LL: // 0 or subnormal
            d_sig = (d&0x000fffffffffffffULL);
            // Signed zero
            if (d_sig == 0) {
                m_hi = q_sgn;
                m_lo = 0LL;
                break;
            }
            // Subnormal
            d_sig <<= 1;
            while ((d_sig&0x0010000000000000ULL) == 0ULL) {
                d_sig <<= 1;
                d_exp++;
            }
            q_exp = (16383 - 1023 - d_exp) << 48;
            q_sig = (d_sig&0x000fffffffffffffULL);
            m_hi = q_sgn + q_exp + (q_sig >> 4);
            m_lo = q_sig << 60;
            break;
        case 0x7ff0000000000000ULL: // inf or NaN
            // All-ones exponent and a copy of the significand
            d_sig = (d&0x000fffffffffffffULL);
            m_hi = q_sgn + 0x7ff0000000000000ULL + (d_sig >> 4);
            m_lo = d_sig << 60;
            break;
        default: // normalized
            // Just need to adjust the exponent and shift
            m_hi = q_sgn + (((d&0x7fffffffffffffffULL) >> 4) + 0x3c00000000000000ULL);
            m_lo = d << 60;
            break;
    }
}

#endif // !defined(DYND_HAS_FLOAT128)

