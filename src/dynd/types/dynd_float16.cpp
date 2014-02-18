//
// Copyright (C) 2010-13 Mark Wiebe, DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <dynd/types/dynd_float16.hpp>
#include <dynd/types/dynd_float128.hpp>
#include <dynd/types/dynd_int128.hpp>
#include <dynd/types/dynd_uint128.hpp>

#include <sstream>
#include <stdexcept>

using namespace std;
using namespace dynd;

// This chooses between 'ties to even' and 'ties away from zero'.
#define DYND_FLOAT16_ROUND_TIES_TO_EVEN 1

uint16_t dynd::float_to_halfbits(float value, assign_error_mode errmode)
{
    union { float f; uint32_t fbits; } conv;
    conv.f = value;
    uint32_t f = conv.fbits;

    uint32_t f_exp, f_sig;
    uint16_t h_sgn, h_exp, h_sig;

    h_sgn = (uint16_t) ((f&0x80000000u) >> 16);
    f_exp = (f&0x7f800000u);

    // Exponent overflow/NaN converts to signed inf/NaN
    if (f_exp >= 0x47800000u) {
        if (f_exp == 0x7f800000u) {
            // Inf or NaN
            f_sig = (f&0x007fffffu);
            if (f_sig != 0) {
                // NaN - propagate the flag in the significand...
                uint16_t ret = (uint16_t) (0x7c00u + (f_sig >> 13));
                // ...but make sure it stays a NaN
                if (ret == 0x7c00u) {
                    ret++;
                }
                return h_sgn + ret;
            } else {
                // signed inf
                return (uint16_t) (h_sgn + 0x7c00u);
            }
        } else {
            // overflow to signed inf
            if (errmode >= assign_error_overflow) {
#ifndef __CUDA_ARCH__
                stringstream ss;
                ss << "overflow converting float32 " << value << " to float16";
                throw overflow_error(ss.str());
#endif
            }
            return (uint16_t) (h_sgn + 0x7c00u);
        }
    }

    // Exponent underflow converts to a subnormal float16 or signed zero
    if (f_exp <= 0x38000000u) {
        // Signed zeros, subnormal floats, and floats with small
        // exponents all convert to signed zero halfs.
        if (f_exp < 0x33000000u) {
            // If f != 0, it underflowed to 0
            if (errmode >= assign_error_inexact && (f&0x7fffffff) != 0) {
#ifndef __CUDA_ARCH__
                stringstream ss;
                ss << "underflow converting float32 " << value << " to float16";
                throw runtime_error(ss.str());
#endif
            }
            return h_sgn;
        }
        // Make the subnormal significand
        f_exp >>= 23;
        f_sig = (0x00800000u + (f&0x007fffffu));
        // If it's not exactly represented, it underflowed
        if (errmode >= assign_error_inexact && (f_sig&(((uint32_t)1 << (126 - f_exp)) - 1)) != 0) {
#ifndef __CUDA_ARCH__
            stringstream ss;
            ss << "underflow converting float32 " << value << " to float16";
            throw runtime_error(ss.str());
#endif
        }
        f_sig >>= (113 - f_exp);
        // Handle rounding by adding 1 to the bit beyond dynd_float16 precision
#if DYND_FLOAT16_ROUND_TIES_TO_EVEN
        // If the last bit in the float16 significand is 0 (already even), and
        // the remaining bit pattern is 1000...0, then we do not add one
        // to the bit after the dynd_float16 significand.  In all other cases, we do.
        if ((f_sig&0x00003fffu) != 0x00001000u) {
            f_sig += 0x00001000u;
        }
#else
        f_sig += 0x00001000u;
#endif
        h_sig = (uint16_t) (f_sig >> 13);
        // If the rounding causes a bit to spill into h_exp, it will
        // increment h_exp from zero to one and h_sig will be zero.
        // This is the correct result.
        return (uint16_t) (h_sgn + h_sig);
    }

    // Regular case with no overflow or underflow
    h_exp = (uint16_t) ((f_exp - 0x38000000u) >> 13);
    // Handle rounding by adding 1 to the bit beyond dynd_float16 precision
    f_sig = (f&0x007fffffu);
#if DYND_FLOAT16_ROUND_TIES_TO_EVEN
    // If the last bit in the dynd_float16 significand is 0 (already even), and
    // the remaining bit pattern is 1000...0, then we do not add one
    // to the bit after the dynd_float16 significand.  In all other cases, we do.
    if ((f_sig&0x00003fffu) != 0x00001000u) {
        f_sig += 0x00001000u;
    }
#else
    f_sig += 0x00001000u;
#endif
    h_sig = (uint16_t) (f_sig >> 13);
    // If the rounding causes a bit to spill into h_exp, it will
    // increment h_exp by one and h_sig will be zero.  This is the
    // correct result.  h_exp may increment to 15, at greatest, in
    // which case the result overflows to a signed inf.
    h_sig += h_exp;
    if (h_sig == 0x7c00u && errmode >= assign_error_overflow) {
#ifndef __CUDA_ARCH__
        stringstream ss;
        ss << "overflow converting float32 " << value << " to float16";
        throw overflow_error(ss.str());
#endif
    }
    return h_sgn + h_sig;
}

uint16_t dynd::double_to_halfbits(double value, assign_error_mode errmode)
{
    union { double d; uint64_t dbits; } conv;
    conv.d = value;
    uint64_t d = conv.dbits;

    uint64_t d_exp, d_sig;
    uint16_t h_sgn, h_exp, h_sig;

    h_sgn = (d&0x8000000000000000ULL) >> 48;
    d_exp = (d&0x7ff0000000000000ULL);

    // Exponent overflow/NaN converts to signed inf/NaN
    if (d_exp >= 0x40f0000000000000ULL) {
        if (d_exp == 0x7ff0000000000000ULL) {
            // Inf or NaN
            d_sig = (d&0x000fffffffffffffULL);
            if (d_sig != 0) {
                // NaN - propagate the flag in the significand...
                uint16_t ret = (uint16_t) (0x7c00u + (d_sig >> 42));
                // ...but make sure it stays a NaN
                if (ret == 0x7c00u) {
                    ret++;
                }
                return h_sgn + ret;
            } else {
                // signed inf
                return h_sgn + 0x7c00u;
            }
        } else {
            // overflow to signed inf
            if (errmode >= assign_error_overflow) {
#ifndef __CUDA_ARCH__
                stringstream ss;
                ss << "overflow converting float64 " << value << " to float16";
                throw overflow_error(ss.str());
#endif
            }
            return h_sgn + 0x7c00u;
        }
    }

    // Exponent underflow converts to subnormal dynd_float16 or signed zero
    if (d_exp <= 0x3f00000000000000ULL) {
        // Signed zeros, subnormal floats, and floats with small
        // exponents all convert to signed zero halfs.
        if (d_exp < 0x3e60000000000000ULL) {
            // If d != 0, it underflowed to 0
            if (errmode >= assign_error_inexact && (d&0x7fffffffffffffffULL) != 0) {
#ifndef __CUDA_ARCH__
                stringstream ss;
                ss << "underflow converting float32 " << value << " to float16";
                throw runtime_error(ss.str());
#endif
            }
            return h_sgn;
        }
        // Make the subnormal significand
        d_exp >>= 52;
        d_sig = (0x0010000000000000ULL + (d&0x000fffffffffffffULL));
        // If it's not exactly represented, it underflowed
        if (errmode >= assign_error_inexact && (d_sig&(((uint64_t)1 << (1051 - d_exp)) - 1)) != 0) {
#ifndef __CUDA_ARCH__
            stringstream ss;
            ss << "underflow converting float32 " << value << " to float16";
            throw runtime_error(ss.str());
#endif
        }
        d_sig >>= (1009 - d_exp);
        // Handle rounding by adding 1 to the bit beyond dynd_float16 precision
#if DYND_FLOAT16_ROUND_TIES_TO_EVEN
        // If the last bit in the dynd_float16 significand is 0 (already even), and
        // the remaining bit pattern is 1000...0, then we do not add one
        // to the bit after the dynd_float16 significand.  In all other cases, we do.
        if ((d_sig&0x000007ffffffffffULL) != 0x0000020000000000ULL) {
            d_sig += 0x0000020000000000ULL;
        }
#else
        d_sig += 0x0000020000000000ULL;
#endif
        h_sig = (uint16_t) (d_sig >> 42);
        // If the rounding causes a bit to spill into h_exp, it will
        // increment h_exp from zero to one and h_sig will be zero.
        // This is the correct result.
        return h_sgn + h_sig;
    }

    // Regular case with no overflow or underflow
    h_exp = (uint16_t) ((d_exp - 0x3f00000000000000ULL) >> 42);
    // Handle rounding by adding 1 to the bit beyond dynd_float16 precision
    d_sig = (d&0x000fffffffffffffULL);
#if DYND_FLOAT16_ROUND_TIES_TO_EVEN
    // If the last bit in the dynd_float16 significand is 0 (already even), and
    // the remaining bit pattern is 1000...0, then we do not add one
    // to the bit after the dynd_float16 significand.  In all other cases, we do.
    if ((d_sig&0x000007ffffffffffULL) != 0x0000020000000000ULL) {
        d_sig += 0x0000020000000000ULL;
    }
#else
    d_sig += 0x0000020000000000ULL;
#endif
    h_sig = (uint16_t) (d_sig >> 42);

    // If the rounding causes a bit to spill into h_exp, it will
    // increment h_exp by one and h_sig will be zero.  This is the
    // correct result.  h_exp may increment to 15, at greatest, in
    // which case the result overflows to a signed inf.
    h_sig += h_exp;
    if (h_sig == 0x7c00u && errmode >= assign_error_overflow) {
#ifndef __CUDA_ARCH__
        stringstream ss;
        ss << "overflow converting float64 " << value << " to float16";
        throw overflow_error(ss.str());
#endif
    }
    return h_sgn + h_sig;
}

float dynd::halfbits_to_float(uint16_t h)
{
    union { float f; uint32_t fbits; } conv;
    uint16_t h_exp, h_sig;
    uint32_t f_sgn, f_exp, f_sig;

    h_exp = (h&0x7c00u);
    f_sgn = ((uint32_t)h&0x8000u) << 16;
    switch (h_exp) {
        case 0x0000u: // 0 or subnormal
            h_sig = (h&0x03ffu);
            // Signed zero
            if (h_sig == 0) {
                conv.fbits = f_sgn;
                return conv.f;
            }
            // Subnormal
            h_sig <<= 1;
            while ((h_sig&0x0400u) == 0) {
                h_sig <<= 1;
                h_exp++;
            }
            f_exp = ((uint32_t)(127 - 15 - h_exp)) << 23;
            f_sig = ((uint32_t)(h_sig&0x03ffu)) << 13;
            conv.fbits = f_sgn + f_exp + f_sig;
            return conv.f;
        case 0x7c00u: // inf or NaN
            // All-ones exponent and a copy of the significand
            conv.fbits = f_sgn + 0x7f800000u + (((uint32_t)(h&0x03ffu)) << 13);
            return conv.f;
        default: // normalized
            // Just need to adjust the exponent and shift
            conv.fbits = f_sgn + (((uint32_t)(h&0x7fffu) + 0x1c000u) << 13);
            return conv.f;
    }
}

double dynd::halfbits_to_double(uint16_t h)
{
    union { double d; uint64_t dbits; } conv;
    uint16_t h_exp, h_sig;
    uint64_t d_sgn, d_exp, d_sig;

    h_exp = (h&0x7c00u);
    d_sgn = ((uint64_t)h&0x8000u) << 48;
    switch (h_exp) {
        case 0x0000u: // 0 or subnormal
            h_sig = (h&0x03ffu);
            // Signed zero
            if (h_sig == 0) {
                conv.dbits = d_sgn;
                return conv.d;
            }
            // Subnormal
            h_sig <<= 1;
            while ((h_sig&0x0400u) == 0) {
                h_sig <<= 1;
                h_exp++;
            }
            d_exp = ((uint64_t)(1023 - 15 - h_exp)) << 52;
            d_sig = ((uint64_t)(h_sig&0x03ffu)) << 42;
            conv.dbits = d_sgn + d_exp + d_sig;
            return conv.d;
        case 0x7c00u: // inf or NaN
            // All-ones exponent and a copy of the significand
            conv.dbits = d_sgn + 0x7ff0000000000000ULL +
                                (((uint64_t)(h&0x03ffu)) << 42);
            return conv.d;
        default: // normalized
            // Just need to adjust the exponent and shift
            conv.dbits = d_sgn + (((uint64_t)(h&0x7fffu) + 0xfc000u) << 42);
            return conv.d;
    }
}

dynd::dynd_float16::dynd_float16(const dynd_int128& value)
{
    m_bits = double_to_halfbits((double)value, assign_error_none);
}

dynd::dynd_float16::dynd_float16(const dynd_uint128& value)
{
    m_bits = double_to_halfbits((double)value, assign_error_none);
}

dynd::dynd_float16::dynd_float16(const dynd_float128& value)
{
    m_bits = double_to_halfbits(double(value), assign_error_none);
}

dynd::dynd_float16::operator dynd_int128() const
{
    return dynd_int128(int32_t(*this));
}
dynd::dynd_float16::operator dynd_uint128() const
{
    return dynd_uint128(uint32_t(*this));
}
dynd::dynd_float16::operator dynd_float128() const
{
    return dynd_float128(double(*this));
}
