//
// Copyright (C) 2011-16 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <stdexcept>
#include <sstream>
#include <cmath>

#include <dynd/config.hpp>

#if !defined(DYND_HAS_UINT128)

using namespace std;
using namespace dynd;

dynd::uint128::uint128(float value)
{
    if (value < 0) {
        m_hi = m_lo = 0;
    } else {
        if (value >= 18446744073709551616.0) {
            m_hi = (uint64_t)((double)value / 18446744073709551616.0);
            m_lo = (uint64_t)fmod((double)value, 18446744073709551616.0);
        } else {
            m_hi = 0;
            m_lo = (uint64_t)value;
        }
    }
}

dynd::uint128::uint128(double value)
{
    if (value < 0) {
        m_hi = m_lo = 0;
    } else {
        if (value >= 18446744073709551616.0) {
            m_hi = (uint64_t)(value / 18446744073709551616.0);
            m_lo = (uint64_t)fmod(value, 18446744073709551616.0);
        } else {
            m_hi = 0;
            m_lo = (uint64_t)value;
        }
    }
}

#if defined(DYND_HAS_INT128)
dynd::uint128::uint128(const int128& value)
    : m_lo((uint64_t)value), m_hi((uint64_t)(value >> 64))
{
}
#else
dynd::uint128::uint128(const int128& value)
    : m_lo(value.m_lo), m_hi(value.m_hi)
{
}
#endif

dynd::uint128::uint128(const float16& value)
    : m_lo(int(float(value))), m_hi(0LL)
{
}

dynd::uint128::uint128(const float128& DYND_UNUSED(value))
{
#ifndef __CUDA_ARCH__
    throw runtime_error("dynd float128 to uint128 conversion is not implemented");
#endif
}

uint128 dynd::uint128::operator*(uint32_t rhs) const
{
    // Split the multiplication into three
    // First product
    uint64_t lo_partial = (m_lo&0x00000000ffffffffULL) * rhs;
    // Second product
    uint64_t tmp = ((m_lo&0xffffffff00000000ULL) >> 32) * rhs;
    uint64_t lo = lo_partial + (tmp << 32);
    uint64_t hi = (tmp >> 32) + (lo < lo_partial);
    // Third product
    hi += m_hi * rhs;
    return uint128(hi, lo);
}


uint128 dynd::uint128::operator/(uint32_t rhs) const
{
    // Split the division into three
    // First division (bits 127..64)
    uint64_t hi_div = m_hi / rhs;
    uint64_t hi_rem = m_hi % rhs;
    // Second division (bits 63..32)
    uint64_t mid_val = (hi_rem << 32) | (m_lo >> 32);
    uint64_t mid_div = mid_val / rhs;
    uint64_t mid_rem = mid_val % rhs;
    // Third division (bits 31..0)
    uint64_t low_val = (mid_rem << 32) | (m_lo&0x00000000ffffffffULL);
    uint64_t low_div = low_val / rhs;
    return uint128(hi_div, (mid_div << 32) | low_div);
}

void dynd::uint128::divrem(uint32_t rhs, uint32_t& out_rem)
{
    // Split the division into three
    // First division (bits 127..64)
    uint64_t hi_div = m_hi / rhs;
    uint64_t hi_rem = m_hi % rhs;
    // Second division (bits 63..32)
    uint64_t mid_val = (hi_rem << 32) | (m_lo >> 32);
    uint64_t mid_div = mid_val / rhs;
    uint64_t mid_rem = mid_val % rhs;
    // Third division (bits 31..0)
    uint64_t low_val = (mid_rem << 32) | (m_lo&0x00000000ffffffffULL);
    uint64_t low_div = low_val / rhs;
    out_rem = low_val % rhs;
    m_hi = hi_div;
    m_lo = (mid_div << 32) | low_div;
}

std::ostream& dynd::operator<<(ostream& out, const uint128& val)
{
    if (val == uint128(0ULL)) {
        return (out << '0');
    }

    string buffer(40, '\0');
    size_t idx = 39;
    uint128 tmp = val;
    uint32_t digit = 0;
    do {
        tmp.divrem(10u, digit);
        buffer[idx--] = digit + '0';
    } while (tmp != uint128(0ULL));
    return (out << &buffer[idx+1]);
}

#endif // !defined(DYND_HAS_UINT128)
