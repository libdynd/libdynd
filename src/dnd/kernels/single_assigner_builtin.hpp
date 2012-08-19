//
// Copyright (C) 2011-12, Dynamic NDArray Developers
// BSD 2-Clause License, see LICENSE.txt
//

// This file is an internal implementation detail of built-in value assignment
// for aligned values in native byte order.

#include <dnd/fpstatus.hpp>
#include <cmath>
#include <complex>
#include <limits>

#include <dnd/dtype.hpp>
#include <dnd/diagnostics.hpp>

#if defined(_MSC_VER)
// Tell the visual studio compiler we're accessing the FPU flags
#pragma fenv_access(on)
#endif

namespace dnd {

template<class dst_type, class src_type, dtype_kind_t dst_kind, dtype_kind_t src_kind, assign_error_mode errmode>
struct single_assigner_builtin_base;

// Any assignment with no error checking
template<class dst_type, class src_type, dtype_kind_t dst_kind, dtype_kind_t src_kind>
struct single_assigner_builtin_base<dst_type, src_type, dst_kind, src_kind, assign_error_none>
{
    static void assign(dst_type *dst, const src_type *src) {
        DND_TRACE_ASSIGNMENT(static_cast<dst_type>(*src), dst_type, *src, src_type);

        *dst = static_cast<dst_type>(*src);
    }
};

// Complex floating point -> non-complex with no error checking
template<class dst_type, class src_real_type>
struct single_assigner_builtin_base<dst_type, std::complex<src_real_type>, int_kind, complex_kind, assign_error_none>
{
    static void assign(dst_type *dst, const std::complex<src_real_type> *src) {
        DND_TRACE_ASSIGNMENT(static_cast<dst_type>(src->real()), dst_type, *src, std::complex<src_real_type>);

        *dst = static_cast<dst_type>(src->real());
    }
};
template<class dst_type, class src_real_type>
struct single_assigner_builtin_base<dst_type, std::complex<src_real_type>, uint_kind, complex_kind, assign_error_none>
{
    static void assign(dst_type *dst, const std::complex<src_real_type> *src) {
        DND_TRACE_ASSIGNMENT(static_cast<dst_type>(src->real()), dst_type, *src, std::complex<src_real_type>);

        *dst = static_cast<dst_type>(src->real());
    }
};
template<class dst_type, class src_real_type>
struct single_assigner_builtin_base<dst_type, std::complex<src_real_type>, real_kind, complex_kind, assign_error_none>
{
    static void assign(dst_type *dst, const std::complex<src_real_type> *src) {
        DND_TRACE_ASSIGNMENT(static_cast<dst_type>(src->real()), dst_type, *src, std::complex<src_real_type>);

        *dst = static_cast<dst_type>(src->real());
    }
};

// Anything -> boolean with overflow checking
template<class src_type, dtype_kind_t src_kind>
struct single_assigner_builtin_base<dnd_bool, src_type, bool_kind, src_kind, assign_error_overflow>
{
    static void assign(dnd_bool *dst, const src_type *src) {
        src_type s = *src;

        DND_TRACE_ASSIGNMENT((bool)(s != src_type(0)), dnd_bool, s, src_type);

        if (s == src_type(0)) {
            *dst = false;
        } else if (s == src_type(1)) {
            *dst = true;
        } else {
            throw std::runtime_error("overflow while assigning to boolean value");
        }
    }
};

// Anything -> boolean with other error checking
template<class src_type, dtype_kind_t src_kind>
struct single_assigner_builtin_base<dnd_bool, src_type, bool_kind, src_kind, assign_error_fractional>
    : public single_assigner_builtin_base<dnd_bool, src_type, bool_kind, src_kind, assign_error_overflow> {};
template<class src_type, dtype_kind_t src_kind>
struct single_assigner_builtin_base<dnd_bool, src_type, bool_kind, src_kind, assign_error_inexact>
    : public single_assigner_builtin_base<dnd_bool, src_type, bool_kind, src_kind, assign_error_overflow> {};

// Boolean -> anything with other error checking
template<class dst_type, dtype_kind_t dst_kind>
struct single_assigner_builtin_base<dst_type, dnd_bool, dst_kind, bool_kind, assign_error_overflow>
    : public single_assigner_builtin_base<dst_type, dnd_bool, dst_kind, bool_kind, assign_error_none> {};
template<class dst_type, dtype_kind_t dst_kind>
struct single_assigner_builtin_base<dst_type, dnd_bool, dst_kind, bool_kind, assign_error_fractional>
    : public single_assigner_builtin_base<dst_type, dnd_bool, dst_kind, bool_kind, assign_error_none> {};
template<class dst_type, dtype_kind_t dst_kind>
struct single_assigner_builtin_base<dst_type, dnd_bool, dst_kind, bool_kind, assign_error_inexact>
    : public single_assigner_builtin_base<dst_type, dnd_bool, dst_kind, bool_kind, assign_error_none> {};

// Boolean -> boolean with other error checking
template<>
struct single_assigner_builtin_base<dnd_bool, dnd_bool, bool_kind, bool_kind, assign_error_overflow>
    : public single_assigner_builtin_base<dnd_bool, dnd_bool, bool_kind, bool_kind, assign_error_none> {};
template<>
struct single_assigner_builtin_base<dnd_bool, dnd_bool, bool_kind, bool_kind, assign_error_fractional>
    : public single_assigner_builtin_base<dnd_bool, dnd_bool, bool_kind, bool_kind, assign_error_none> {};
template<>
struct single_assigner_builtin_base<dnd_bool, dnd_bool, bool_kind, bool_kind, assign_error_inexact>
    : public single_assigner_builtin_base<dnd_bool, dnd_bool, bool_kind, bool_kind, assign_error_none> {};

// Signed int -> signed int with overflow checking
template<class dst_type, class src_type>
struct single_assigner_builtin_base<dst_type, src_type, int_kind, int_kind, assign_error_overflow>
{
    static void assign(dst_type *dst, const src_type *src) {
        src_type s = *src;

        if (s < std::numeric_limits<dst_type>::min() || s > std::numeric_limits<dst_type>::max()) {
            throw std::runtime_error("overflow while assigning signed integer to signed integer");
        }
        *dst = static_cast<dst_type>(s);
    }
};

// Signed int -> signed int with other error checking
template<class dst_type, class src_type>
struct single_assigner_builtin_base<dst_type, src_type, int_kind, int_kind, assign_error_fractional>
    : public single_assigner_builtin_base<dst_type, src_type, int_kind, int_kind, assign_error_overflow> {};
template<class dst_type, class src_type>
struct single_assigner_builtin_base<dst_type, src_type, int_kind, int_kind, assign_error_inexact>
    : public single_assigner_builtin_base<dst_type, src_type, int_kind, int_kind, assign_error_overflow> {};

// Unsigned int -> signed int with overflow checking just when sizeof(dst) <= sizeof(src)
template<class dst_type, class src_type, bool dst_le>
struct single_assigner_builtin_unsigned_to_signed_overflow_base
    : public single_assigner_builtin_base<dst_type, src_type, int_kind, uint_kind, assign_error_none> {};
template<class dst_type, class src_type>
struct single_assigner_builtin_unsigned_to_signed_overflow_base<dst_type, src_type, true>
{
    static void assign(dst_type *dst, const src_type *src) {
        src_type s = *src;

        DND_TRACE_ASSIGNMENT(static_cast<dst_type>(s), dst_type, s, src_type);

        if (s > static_cast<src_type>(std::numeric_limits<dst_type>::max())) {
            throw std::runtime_error("overflow while assigning unsigned integer signed integer");
        }
        *dst = static_cast<dst_type>(s);
    }
};
template<class dst_type, class src_type>
struct single_assigner_builtin_base<dst_type, src_type, int_kind, uint_kind, assign_error_overflow>
    : public single_assigner_builtin_unsigned_to_signed_overflow_base<dst_type,
                                                    src_type, sizeof(dst_type) <= sizeof(src_type)> {};

// Unsigned int -> signed int with other error checking
template<class dst_type, class src_type>
struct single_assigner_builtin_base<dst_type, src_type, int_kind, uint_kind, assign_error_fractional>
    : public single_assigner_builtin_base<dst_type, src_type, int_kind, uint_kind, assign_error_overflow> {};
template<class dst_type, class src_type>
struct single_assigner_builtin_base<dst_type, src_type, int_kind, uint_kind, assign_error_inexact>
    : public single_assigner_builtin_base<dst_type, src_type, int_kind, uint_kind, assign_error_overflow> {};

// Signed int -> unsigned int with positive overflow checking just when sizeof(dst) < sizeof(src)
template<class dst_type, class src_type, bool dst_lt>
struct single_assigner_builtin_signed_to_unsigned_overflow_base
{
    static void assign(dst_type *dst, const src_type *src) {
        src_type s = *src;

        DND_TRACE_ASSIGNMENT(static_cast<dst_type>(s), dst_type, s, src_type);

        if (s < 0) {
            throw std::runtime_error("overflow while assigning signed integer to unsigned integer");
        }
        *dst = static_cast<dst_type>(s);
    }
};
template<class dst_type, class src_type>
struct single_assigner_builtin_signed_to_unsigned_overflow_base<dst_type, src_type, true>
{
    static void assign(dst_type *dst, const src_type *src) {
        src_type s = *src;

        DND_TRACE_ASSIGNMENT(static_cast<dst_type>(s), dst_type, s, src_type);

        if (s < 0 || s > static_cast<src_type>(std::numeric_limits<dst_type>::max())) {
            throw std::runtime_error("overflow while assigning signed integer to unsigned integer");
        }
        *dst = static_cast<dst_type>(s);
    }
};
template<class dst_type, class src_type>
struct single_assigner_builtin_base<dst_type, src_type, uint_kind, int_kind, assign_error_overflow>
    : public single_assigner_builtin_signed_to_unsigned_overflow_base<dst_type,
                                                    src_type, sizeof(dst_type) < sizeof(src_type)> {};

// Signed int -> unsigned int with other error checking
template<class dst_type, class src_type>
struct single_assigner_builtin_base<dst_type, src_type, uint_kind, int_kind, assign_error_fractional>
    : public single_assigner_builtin_base<dst_type, src_type, uint_kind, int_kind, assign_error_overflow> {};
template<class dst_type, class src_type>
struct single_assigner_builtin_base<dst_type, src_type, uint_kind, int_kind, assign_error_inexact>
    : public single_assigner_builtin_base<dst_type, src_type, uint_kind, int_kind, assign_error_overflow> {};

// Unsigned int -> unsigned int with overflow checking
template<class dst_type, class src_type>
struct single_assigner_builtin_base<dst_type, src_type, uint_kind, uint_kind, assign_error_overflow>
{
    static void assign(dst_type *dst, const src_type *src) {
        src_type s = *src;

        DND_TRACE_ASSIGNMENT(static_cast<dst_type>(s), dst_type, s, src_type);

        if (s > std::numeric_limits<dst_type>::max()) {
            throw std::runtime_error("overflow while assigning unsigned integer to unsigned integer");
        }
        *dst = static_cast<dst_type>(s);
    }
};

// Unsigned int -> unsigned int with other error checking
template<class dst_type, class src_type>
struct single_assigner_builtin_base<dst_type, src_type, uint_kind, uint_kind, assign_error_fractional>
    : public single_assigner_builtin_base<dst_type, src_type, uint_kind, uint_kind, assign_error_overflow> {};
template<class dst_type, class src_type>
struct single_assigner_builtin_base<dst_type, src_type, uint_kind, uint_kind, assign_error_inexact>
    : public single_assigner_builtin_base<dst_type, src_type, uint_kind, uint_kind, assign_error_overflow> {};

// Signed int -> floating point with inexact checking
template<class dst_type, class src_type>
struct single_assigner_builtin_base<dst_type, src_type, real_kind, int_kind, assign_error_inexact>
{
    static void assign(dst_type *dst, const src_type *src) {
        src_type s = *src;
        dst_type d = static_cast<dst_type>(s);

        DND_TRACE_ASSIGNMENT(d, dst_type, s, src_type);

        if (static_cast<src_type>(d) != s) {
            std::stringstream ss;
            ss << "inexact value while assigning " << make_dtype<src_type>() << " to " << make_dtype<dst_type>();
            throw std::runtime_error(ss.str());
        }
        *dst = d;
    }
};

// Signed int -> floating point with other checking
template<class dst_type, class src_type>
struct single_assigner_builtin_base<dst_type, src_type, real_kind, int_kind, assign_error_overflow>
    : public single_assigner_builtin_base<dst_type, src_type, real_kind, int_kind, assign_error_none> {};
template<class dst_type, class src_type>
struct single_assigner_builtin_base<dst_type, src_type, real_kind, int_kind, assign_error_fractional>
    : public single_assigner_builtin_base<dst_type, src_type, real_kind, int_kind, assign_error_none> {};

// Signed int -> complex floating point with inexact checking
template<class dst_real_type, class src_type>
struct single_assigner_builtin_base<std::complex<dst_real_type>, src_type, complex_kind, int_kind, assign_error_inexact>
{
    static void assign(std::complex<dst_real_type> *dst, const src_type *src) {
        src_type s = *src;
        dst_real_type d = static_cast<dst_real_type>(s);

        DND_TRACE_ASSIGNMENT(d, std::complex<dst_real_type>, s, src_type);

        if (static_cast<src_type>(d) != s) {
            throw std::runtime_error("inexact value while assigning signed integer to complex floating point");
        }
        *dst = d;
    }
};

// Signed int -> complex floating point with other checking
template<class dst_real_type, class src_type>
struct single_assigner_builtin_base<std::complex<dst_real_type>, src_type, complex_kind, int_kind, assign_error_overflow>
    : public single_assigner_builtin_base<std::complex<dst_real_type>, src_type, complex_kind, int_kind, assign_error_none> {};
template<class dst_real_type, class src_type>
struct single_assigner_builtin_base<std::complex<dst_real_type>, src_type, complex_kind, int_kind, assign_error_fractional>
    : public single_assigner_builtin_base<std::complex<dst_real_type>, src_type, complex_kind, int_kind, assign_error_none> {};

// Unsigned int -> floating point with inexact checking
template<class dst_type, class src_type>
struct single_assigner_builtin_base<dst_type, src_type, real_kind, uint_kind, assign_error_inexact>
{
    static void assign(dst_type *dst, const src_type *src) {
        src_type s = *src;
        dst_type d = static_cast<dst_type>(s);

        DND_TRACE_ASSIGNMENT(d, dst_type, s, src_type);

        if (static_cast<src_type>(d) != s) {
            throw std::runtime_error("inexact value while assigning unsigned integer to floating point");
        }
        *dst = d;
    }
};

// Unsigned int -> floating point with other checking
template<class dst_type, class src_type>
struct single_assigner_builtin_base<dst_type, src_type, real_kind, uint_kind, assign_error_overflow>
    : public single_assigner_builtin_base<dst_type, src_type, real_kind, uint_kind, assign_error_none> {};
template<class dst_type, class src_type>
struct single_assigner_builtin_base<dst_type, src_type, real_kind, uint_kind, assign_error_fractional>
    : public single_assigner_builtin_base<dst_type, src_type, real_kind, uint_kind, assign_error_none> {};

// Unsigned int -> complex floating point with inexact checking
template<class dst_real_type, class src_type>
struct single_assigner_builtin_base<std::complex<dst_real_type>, src_type, complex_kind, uint_kind, assign_error_inexact>
{
    static void assign(std::complex<dst_real_type> *dst, const src_type *src) {
        src_type s = *src;
        dst_real_type d = static_cast<dst_real_type>(s);

        DND_TRACE_ASSIGNMENT(d, std::complex<dst_real_type>, s, src_type);

        if (static_cast<src_type>(d) != s) {
            throw std::runtime_error("inexact value while assigning unsigned integer to floating point");
        }
        *dst = d;
    }
};

// Unsigned int -> complex floating point with other checking
template<class dst_real_type, class src_type>
struct single_assigner_builtin_base<std::complex<dst_real_type>, src_type, complex_kind, uint_kind, assign_error_overflow>
    : public single_assigner_builtin_base<std::complex<dst_real_type>, src_type, complex_kind, uint_kind, assign_error_none> {};
template<class dst_real_type, class src_type>
struct single_assigner_builtin_base<std::complex<dst_real_type>, src_type, complex_kind, uint_kind, assign_error_fractional>
    : public single_assigner_builtin_base<std::complex<dst_real_type>, src_type, complex_kind, uint_kind, assign_error_none> {};

// Floating point -> signed int with overflow checking
template<class dst_type, class src_type>
struct single_assigner_builtin_base<dst_type, src_type, int_kind, real_kind, assign_error_overflow>
{
    static void assign(dst_type *dst, const src_type *src) {
        src_type s = *src;

        DND_TRACE_ASSIGNMENT(static_cast<dst_type>(s), dst_type, s, src_type);

        if (s < std::numeric_limits<dst_type>::min() || s > std::numeric_limits<dst_type>::max()) {
            throw std::runtime_error("overflow while assigning floating point to signed integer");
        }
        *dst = static_cast<dst_type>(s);
    }
};

// Floating point -> signed int with fractional checking
template<class dst_type, class src_type>
struct single_assigner_builtin_base<dst_type, src_type, int_kind, real_kind, assign_error_fractional>
{
    static void assign(dst_type *dst, const src_type *src) {
        src_type s = *src;

        DND_TRACE_ASSIGNMENT(static_cast<dst_type>(s), dst_type, s, src_type);

        if (s < std::numeric_limits<dst_type>::min() || s > std::numeric_limits<dst_type>::max()) {
            throw std::runtime_error("overflow while assigning floating point to signed integer");
        }

        if (std::floor(s) != s) {
            throw std::runtime_error("fractional part lost while assigning floating point to signed integer");
        }
        *dst = static_cast<dst_type>(s);
    }
};

// Floating point -> signed int with other checking
template<class dst_type, class src_type>
struct single_assigner_builtin_base<dst_type, src_type, int_kind, real_kind, assign_error_inexact>
    : public single_assigner_builtin_base<dst_type, src_type, int_kind, real_kind, assign_error_fractional> {};

// Complex floating point -> signed int with overflow checking
template<class dst_type, class src_real_type>
struct single_assigner_builtin_base<dst_type, std::complex<src_real_type>, int_kind, complex_kind, assign_error_overflow>
{
    static void assign(dst_type *dst, const std::complex<src_real_type> *src) {
        std::complex<src_real_type> s = *src;

        DND_TRACE_ASSIGNMENT(static_cast<dst_type>(s.real()), dst_type, s, std::complex<src_real_type>);

        if (s.imag() != 0) {
            throw std::runtime_error("loss of imaginary component while assigning complex floating point to signed integer");
        }

        if (s.real() < std::numeric_limits<dst_type>::min() || s.real() > std::numeric_limits<dst_type>::max()) {
            throw std::runtime_error("overflow while assigning complex floating point to signed integer");
        }
        *dst = static_cast<dst_type>(s.real());
    }
};

// Complex floating point -> signed int with fractional checking
template<class dst_type, class src_real_type>
struct single_assigner_builtin_base<dst_type, std::complex<src_real_type>, int_kind, complex_kind, assign_error_fractional>
{
    static void assign(dst_type *dst, const std::complex<src_real_type> *src) {
        std::complex<src_real_type> s = *src;

        DND_TRACE_ASSIGNMENT(static_cast<dst_type>(s.real()), dst_type, s, std::complex<src_real_type>);

        if (s.imag() != 0) {
            throw std::runtime_error("loss of imaginary component while assigning complex floating point to signed integer");
        }

        if (s.real() < std::numeric_limits<dst_type>::min() || s.real() > std::numeric_limits<dst_type>::max()) {
            throw std::runtime_error("overflow while assigning complex floating point to signed integer");
        }

        if (std::floor(s.real()) != s.real()) {
            throw std::runtime_error("fractional part lost while assigning complex floating point to signed integer");
        }
        *dst = static_cast<dst_type>(s.real());
    }
};

// Complex floating point -> signed int with other checking
template<class dst_type, class src_real_type>
struct single_assigner_builtin_base<dst_type, std::complex<src_real_type>, int_kind, complex_kind, assign_error_inexact>
    : public single_assigner_builtin_base<dst_type, std::complex<src_real_type>, int_kind, complex_kind, assign_error_fractional> {};

// Floating point -> unsigned int with overflow checking
template<class dst_type, class src_type>
struct single_assigner_builtin_base<dst_type, src_type, uint_kind, real_kind, assign_error_overflow>
{
    static void assign(dst_type *dst, const src_type *src) {
        src_type s = *src;

        DND_TRACE_ASSIGNMENT(static_cast<dst_type>(s), dst_type, s, src_type);

        if (s < 0 || s > std::numeric_limits<dst_type>::max()) {
            throw std::runtime_error("overflow while assigning floating point to unsigned integer");
        }
        *dst = static_cast<dst_type>(s);
    }
};

// Floating point -> unsigned int with fractional checking
template<class dst_type, class src_type>
struct single_assigner_builtin_base<dst_type, src_type, uint_kind, real_kind, assign_error_fractional>
{
    static void assign(dst_type *dst, const src_type *src) {
        src_type s = *src;

        DND_TRACE_ASSIGNMENT(static_cast<dst_type>(s), dst_type, s, src_type);

        if (s < 0 || s > std::numeric_limits<dst_type>::max()) {
            throw std::runtime_error("overflow while assigning floating point to unsigned integer");
        }

        if (std::floor(s) != s) {
            throw std::runtime_error("fractional part lost while assigning floating point to unsigned integer");
        }
        *dst = static_cast<dst_type>(s);
    }
};

// Floating point -> unsigned int with other checking
template<class dst_type, class src_type>
struct single_assigner_builtin_base<dst_type, src_type, uint_kind, real_kind, assign_error_inexact>
    : public single_assigner_builtin_base<dst_type, src_type, uint_kind, real_kind, assign_error_fractional> {};

// Complex floating point -> unsigned int with overflow checking
template<class dst_type, class src_real_type>
struct single_assigner_builtin_base<dst_type, std::complex<src_real_type>, uint_kind, complex_kind, assign_error_overflow>
{
    static void assign(dst_type *dst, const std::complex<src_real_type> *src) {
        std::complex<src_real_type> s = *src;

        DND_TRACE_ASSIGNMENT(static_cast<dst_type>(s.real()), dst_type, s, std::complex<src_real_type>);

        if (s.imag() != 0) {
            throw std::runtime_error("loss of imaginary component while assigning complex floating point to unsigned integer");
        }

        if (s.real() < 0 || s.real() > std::numeric_limits<dst_type>::max()) {
            throw std::runtime_error("overflow while assigning complex floating point to unsigned integer");
        }
        *dst = static_cast<dst_type>(s.real());
    }
};

// Complex floating point -> unsigned int with fractional checking
template<class dst_type, class src_real_type>
struct single_assigner_builtin_base<dst_type, std::complex<src_real_type>, uint_kind, complex_kind, assign_error_fractional>
{
    static void assign(dst_type *dst, const std::complex<src_real_type> *src) {
        std::complex<src_real_type> s = *src;

        DND_TRACE_ASSIGNMENT(static_cast<dst_type>(s.real()), dst_type, s, std::complex<src_real_type>);

        if (s.imag() != 0) {
            throw std::runtime_error("loss of imaginary component while assigning complex floating point to unsigned integer");
        }

        if (s.real() < 0 || s.real() > std::numeric_limits<dst_type>::max()) {
            throw std::runtime_error("overflow while assigning complex floating point to unsigned integer");
        }

        if (std::floor(s.real()) != s.real()) {
            throw std::runtime_error("fractional part lost while assigning complex floating point to unsigned integer");
        }
        *dst = static_cast<dst_type>(s.real());
    }
};

// Complex floating point -> unsigned int with other checking
template<class dst_type, class src_real_type>
struct single_assigner_builtin_base<dst_type, std::complex<src_real_type>, uint_kind, complex_kind, assign_error_inexact>
    : public single_assigner_builtin_base<dst_type, std::complex<src_real_type>, uint_kind, complex_kind, assign_error_fractional> {};

// float -> float with no checking
template<>
struct single_assigner_builtin_base<float, float, real_kind, real_kind, assign_error_overflow>
    : public single_assigner_builtin_base<float, float, real_kind, real_kind, assign_error_none> {};
template<>
struct single_assigner_builtin_base<float, float, real_kind, real_kind, assign_error_fractional>
    : public single_assigner_builtin_base<float, float, real_kind, real_kind, assign_error_none> {};
template<>
struct single_assigner_builtin_base<float, float, real_kind, real_kind, assign_error_inexact>
    : public single_assigner_builtin_base<float, float, real_kind, real_kind, assign_error_none> {};

// complex<float> -> complex<float> with no checking
template<>
struct single_assigner_builtin_base<std::complex<float>, std::complex<float>, complex_kind, complex_kind, assign_error_overflow>
    : public single_assigner_builtin_base<std::complex<float>, std::complex<float>, complex_kind, complex_kind, assign_error_none> {};
template<>
struct single_assigner_builtin_base<std::complex<float>, std::complex<float>, complex_kind, complex_kind, assign_error_fractional>
    : public single_assigner_builtin_base<std::complex<float>, std::complex<float>, complex_kind, complex_kind, assign_error_none> {};
template<>
struct single_assigner_builtin_base<std::complex<float>, std::complex<float>, complex_kind, complex_kind, assign_error_inexact>
    : public single_assigner_builtin_base<std::complex<float>, std::complex<float>, complex_kind, complex_kind, assign_error_none> {};

// float -> double with no checking
template<>
struct single_assigner_builtin_base<double, float, real_kind, real_kind, assign_error_overflow>
    : public single_assigner_builtin_base<double, float, real_kind, real_kind, assign_error_none> {};
template<>
struct single_assigner_builtin_base<double, float, real_kind, real_kind, assign_error_fractional>
    : public single_assigner_builtin_base<double, float, real_kind, real_kind, assign_error_none> {};
template<>
struct single_assigner_builtin_base<double, float, real_kind, real_kind, assign_error_inexact>
    : public single_assigner_builtin_base<double, float, real_kind, real_kind, assign_error_none> {};

// complex<float> -> complex<double> with no checking
template<>
struct single_assigner_builtin_base<std::complex<double>, std::complex<float>, complex_kind, complex_kind, assign_error_overflow>
    : public single_assigner_builtin_base<std::complex<double>, std::complex<float>, complex_kind, complex_kind, assign_error_none> {};
template<>
struct single_assigner_builtin_base<std::complex<double>, std::complex<float>, complex_kind, complex_kind, assign_error_fractional>
    : public single_assigner_builtin_base<std::complex<double>, std::complex<float>, complex_kind, complex_kind, assign_error_none> {};
template<>
struct single_assigner_builtin_base<std::complex<double>, std::complex<float>, complex_kind, complex_kind, assign_error_inexact>
    : public single_assigner_builtin_base<std::complex<double>, std::complex<float>, complex_kind, complex_kind, assign_error_none> {};

// double -> double with no checking
template<>
struct single_assigner_builtin_base<double, double, real_kind, real_kind, assign_error_overflow>
    : public single_assigner_builtin_base<double, double, real_kind, real_kind, assign_error_none> {};
template<>
struct single_assigner_builtin_base<double, double, real_kind, real_kind, assign_error_fractional>
    : public single_assigner_builtin_base<double, double, real_kind, real_kind, assign_error_none> {};
template<>
struct single_assigner_builtin_base<double, double, real_kind, real_kind, assign_error_inexact>
    : public single_assigner_builtin_base<double, double, real_kind, real_kind, assign_error_none> {};

// complex<double> -> complex<double> with no checking
template<>
struct single_assigner_builtin_base<std::complex<double>, std::complex<double>, complex_kind, complex_kind, assign_error_overflow>
    : public single_assigner_builtin_base<std::complex<double>, std::complex<double>, complex_kind, complex_kind, assign_error_none> {};
template<>
struct single_assigner_builtin_base<std::complex<double>, std::complex<double>, complex_kind, complex_kind, assign_error_fractional>
    : public single_assigner_builtin_base<std::complex<double>, std::complex<double>, complex_kind, complex_kind, assign_error_none> {};
template<>
struct single_assigner_builtin_base<std::complex<double>, std::complex<double>, complex_kind, complex_kind, assign_error_inexact>
    : public single_assigner_builtin_base<std::complex<double>, std::complex<double>, complex_kind, complex_kind, assign_error_none> {};

// double -> float with overflow checking
template<>
struct single_assigner_builtin_base<float, double, real_kind, real_kind, assign_error_overflow>
{
    static void assign(float *dst, const double *src) {
        DND_TRACE_ASSIGNMENT(static_cast<float>(*src), float, *src, double);

        clear_fp_status();
        *dst = static_cast<float>(*src);
        if (is_overflow_fp_status()) {
            throw std::runtime_error("overflow while assigning double to float");
        }
    }
};

// double -> float with fractional checking
template<>
struct single_assigner_builtin_base<float, double, real_kind, real_kind, assign_error_fractional>
    : public single_assigner_builtin_base<float, double, real_kind, real_kind, assign_error_overflow> {};


// double -> float with inexact checking
template<>
struct single_assigner_builtin_base<float, double, real_kind, real_kind, assign_error_inexact>
{
    static void assign(float *dst, const double *src) {
        DND_TRACE_ASSIGNMENT(static_cast<float>(*src), float, *src, double);

        double s = *src;
        float d;
        clear_fp_status();
        d = static_cast<float>(s);
        if (is_overflow_fp_status()) {
            throw std::runtime_error("overflow while assigning double to float");
        }
        // The inexact status didn't work as it should have, so converting back to double and comparing
        //if (is_inexact_fp_status()) {
        //    throw std::runtime_error("inexact precision loss while assigning double to float");
        //}
        if (d != s) {
            throw std::runtime_error("inexact precision loss while assigning double to float");
        }
        *dst = d;
    }
};

// complex<double> -> complex<float> with overflow checking
template<>
struct single_assigner_builtin_base<std::complex<float>, std::complex<double>, complex_kind, complex_kind, assign_error_overflow>
{
    static void assign(std::complex<float> *dst, const std::complex<double> *src) {
        DND_TRACE_ASSIGNMENT(static_cast<std::complex<float> >(*src), std::complex<float>, *src, std::complex<double>);

        clear_fp_status();
        *dst = static_cast<std::complex<float> >(*src);
        if (is_overflow_fp_status()) {
            throw std::runtime_error("overflow while assigning complex double to complex float");
        }
    }
};

// complex<double> -> complex<float> with fractional checking
template<>
struct single_assigner_builtin_base<std::complex<float>, std::complex<double>, complex_kind, complex_kind, assign_error_fractional>
    : public single_assigner_builtin_base<std::complex<float>, std::complex<double>, complex_kind, complex_kind, assign_error_overflow> {};


// complex<double> -> complex<float> with inexact checking
template<>
struct single_assigner_builtin_base<std::complex<float>, std::complex<double>, complex_kind, complex_kind, assign_error_inexact>
{
    static void assign(std::complex<float> *dst, const std::complex<double> *src) {
        DND_TRACE_ASSIGNMENT(static_cast<std::complex<float> >(*src), std::complex<float>, *src, std::complex<double>);

        std::complex<double> s = *src;
        std::complex<float> d;
        clear_fp_status();
        d = static_cast<std::complex<float> >(s);
        if (is_overflow_fp_status()) {
            throw std::runtime_error("overflow while assigning complex double to complex float");
        }
        // The inexact status didn't work as it should have, so converting back to double and comparing
        //if (is_inexact_fp_status()) {
        //    throw std::runtime_error("inexact precision loss while assigning double to float");
        //}
        if (d.real() != s.real() || d.imag() != s.imag()) {
            throw std::runtime_error("inexact precision loss while assigning complex double to complex float");
        }
        *dst = d;
    }
};

// complex<T> -> T with overflow checking
template<typename real_type>
struct single_assigner_builtin_base<real_type, std::complex<real_type>, real_kind, complex_kind, assign_error_overflow>
{
    static void assign(real_type *dst, const std::complex<real_type> *src) {
        std::complex<real_type> s = *src;

        DND_TRACE_ASSIGNMENT(static_cast<float>(s.real()), real_type, s, std::complex<real_type>);

        if (s.imag() != 0) {
            throw std::runtime_error("loss of imaginary component while assigning complex floating point to real floating point");
        }

        *dst = s.real();
    }
};

// complex<T> -> T with fractional checking
template<typename real_type>
struct single_assigner_builtin_base<real_type, std::complex<real_type>, real_kind, complex_kind, assign_error_fractional>
    : public single_assigner_builtin_base<real_type, std::complex<real_type>, real_kind, complex_kind, assign_error_overflow> {};

// complex<T> -> T with inexact checking
template<typename real_type>
struct single_assigner_builtin_base<real_type, std::complex<real_type>, real_kind, complex_kind, assign_error_inexact>
    : public single_assigner_builtin_base<real_type, std::complex<real_type>, real_kind, complex_kind, assign_error_overflow> {};

// T -> complex<T>
template<typename real_type>
struct single_assigner_builtin_base<std::complex<real_type>, real_type, complex_kind, real_kind, assign_error_none>
{
    static void assign(std::complex<real_type> *dst, const real_type *src) {
        DND_TRACE_ASSIGNMENT(static_cast<std::complex<real_type> >(*src), std::complex<real_type>, *src, real_type);

        *dst = *src;
    }
};
template<typename real_type, assign_error_mode errmode>
struct single_assigner_builtin_base<std::complex<real_type>, real_type, complex_kind, real_kind, errmode>
    : public single_assigner_builtin_base<std::complex<real_type>, real_type, complex_kind, real_kind, assign_error_none> {};

// float -> complex<double>
template<>
struct single_assigner_builtin_base<std::complex<double>, float, complex_kind, real_kind, assign_error_none>
{
    static void assign(std::complex<double> *dst, const float *src) {
        DND_TRACE_ASSIGNMENT(static_cast<std::complex<double> >(*src), std::complex<double>, *src, float);

        *dst = *src;
    }
};
template<assign_error_mode errmode>
struct single_assigner_builtin_base<std::complex<double>, float, complex_kind, real_kind, errmode>
    : public single_assigner_builtin_base<std::complex<double>, float, complex_kind, real_kind, assign_error_none> {};

// complex<float> -> double with overflow checking
template<>
struct single_assigner_builtin_base<double, std::complex<float>, real_kind, complex_kind, assign_error_overflow>
{
    static void assign(double *dst, const std::complex<float> *src) {
        std::complex<float> s = *src;

        DND_TRACE_ASSIGNMENT(static_cast<double>(s.real()), double, s, std::complex<float>);

        if (s.imag() != 0) {
            throw std::runtime_error("loss of imaginary component while assigning complex floating point to real floating point");
        }

        *dst = s.real();
    }
};

// complex<float> -> double with fractional checking
template<>
struct single_assigner_builtin_base<double, std::complex<float>, real_kind, complex_kind, assign_error_fractional>
    : public single_assigner_builtin_base<double, std::complex<float>, real_kind, complex_kind, assign_error_overflow> {};

// complex<float> -> double with inexact checking
template<>
struct single_assigner_builtin_base<double, std::complex<float>, real_kind, complex_kind, assign_error_inexact>
    : public single_assigner_builtin_base<double, std::complex<float>, real_kind, complex_kind, assign_error_overflow> {};

// complex<double> -> float with overflow checking
template<>
struct single_assigner_builtin_base<float, std::complex<double>, real_kind, complex_kind, assign_error_overflow>
{
    static void assign(float *dst, const std::complex<double> *src) {
        std::complex<double> s = *src;
        float d;

        DND_TRACE_ASSIGNMENT(static_cast<float>(s.real()), float, s, std::complex<double>);

        if (s.imag() != 0) {
            throw std::runtime_error("loss of imaginary component while assigning complex floating point to real floating point");
        }

        clear_fp_status();
        d = static_cast<float>(s.real());
        if (is_overflow_fp_status()) {
            throw std::runtime_error("overflow while assigning complex floating point to real floating point");
        }

        *dst = d;
    }
};

// complex<double> -> float with fractional checking
template<>
struct single_assigner_builtin_base<float, std::complex<double>, real_kind, complex_kind, assign_error_fractional>
    : public single_assigner_builtin_base<float, std::complex<double>, real_kind, complex_kind, assign_error_overflow> {};

// complex<double> -> float with inexact checking
template<>
struct single_assigner_builtin_base<float, std::complex<double>, real_kind, complex_kind, assign_error_inexact>
{
    static void assign(float *dst, const std::complex<double> *src) {
        std::complex<double> s = *src;
        float d;

        DND_TRACE_ASSIGNMENT(static_cast<float>(s.real()), float, s, std::complex<double>);

        if (s.imag() != 0) {
            throw std::runtime_error("loss of imaginary component while assigning complex floating point to real floating point");
        }

        clear_fp_status();
        d = static_cast<float>(s.real());
        if (is_overflow_fp_status()) {
            throw std::runtime_error("overflow while assigning complex floating point to real floating point");
        }

        if (d != s.real()) {
            throw std::runtime_error("inexact precision loss while assigning complex floating point to real floating point");
        }

        *dst = d;
    }
};

// double -> complex<float> with overflow checking
template<>
struct single_assigner_builtin_base<std::complex<float>, double, complex_kind, real_kind, assign_error_overflow>
{
    static void assign(std::complex<float> *dst, const double *src) {
        double s = *src;
        float d;

        DND_TRACE_ASSIGNMENT(static_cast<std::complex<float> >(s), std::complex<float>, s, double);

        clear_fp_status();
        d = static_cast<float>(s);
        if (is_overflow_fp_status()) {
            throw std::runtime_error("overflow while assigning real floating point to complex floating point");
        }

        *dst = d;
    }
};

// double -> complex<float> with fractional checking
template<>
struct single_assigner_builtin_base<std::complex<float>, double, complex_kind, real_kind, assign_error_fractional>
    : public single_assigner_builtin_base<std::complex<float>, double, complex_kind, real_kind, assign_error_overflow> {};

// double -> complex<float> with inexact checking
template<>
struct single_assigner_builtin_base<std::complex<float>, double, complex_kind, real_kind, assign_error_inexact>
{
    static void assign(std::complex<float> *dst, const double *src) {
        double s = *src;
        float d;

        DND_TRACE_ASSIGNMENT(static_cast<std::complex<float> >(s), std::complex<float>, s, double);

        clear_fp_status();
        d = static_cast<float>(s);
        if (is_overflow_fp_status()) {
            throw std::runtime_error("overflow while assigning real floating point to complex floating point");
        }

        if (d != s) {
            throw std::runtime_error("inexact precision loss while assigning complex floating point to real floating point");
        }

        *dst = d;
    }
};





// This is the interface exposed for use outside of this file
template <class dst_type, class src_type, assign_error_mode errmode>
struct single_assigner_builtin
    : public single_assigner_builtin_base<dst_type, src_type,
                        dtype_kind_of<dst_type>::value, dtype_kind_of<src_type>::value, errmode>
{};
template <class same_type, assign_error_mode errmode>
struct single_assigner_builtin<same_type, same_type, errmode>
{
    static void assign(same_type *dst, const same_type *src) {
        *dst = *src;
    }
};

} // namespace dnd

