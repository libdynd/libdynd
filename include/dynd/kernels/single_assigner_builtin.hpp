//
// Copyright (C) 2011-15 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

// This file is an internal implementation detail of built-in value assignment
// for aligned values in native byte order.

#include <dynd/fpstatus.hpp>
#include <cmath>
#include <complex>
#include <limits>

#include <dynd/config.hpp>
#include <dynd/type.hpp>
#include <dynd/diagnostics.hpp>

#if defined(_MSC_VER)
// Tell the visual studio compiler we're accessing the FPU flags
#pragma fenv_access(on)
#endif

namespace dynd {

template<class dst_type, class src_type, assign_error_mode errmode>
struct single_assigner_builtin_base_error {
    static void assign(dst_type *DYND_UNUSED(dst), const src_type *DYND_UNUSED(src)) {
        //DYND_TRACE_ASSIGNMENT(static_cast<float>(*src), float, *src, double);

        std::stringstream ss;
        ss << "assignment from " << ndt::make_type<src_type>() << " to " << ndt::make_type<dst_type>();
        ss << "with error mode " << errmode << " is not implemented";
        throw std::runtime_error(ss.str());
    }
};

template<class dst_type, class src_type>
struct single_assigner_builtin_base_error<dst_type, src_type, assign_error_nocheck> {
    DYND_CUDA_HOST_DEVICE static void assign(dst_type *DYND_UNUSED(dst), const src_type *DYND_UNUSED(src)) {
        //DYND_TRACE_ASSIGNMENT(static_cast<float>(*src), float, *src, double);

#ifdef __CUDA_ARCH__
        DYND_TRIGGER_ASSERT("assignment is not implemented for CUDA global memory");
#else
        std::stringstream ss;
        ss << "assignment from " << ndt::make_type<src_type>() << " to " << ndt::make_type<dst_type>();
        ss << "with error mode " << assign_error_nocheck << " is not implemented";
        throw std::runtime_error(ss.str());
#endif
    }
};

template<class dst_type, class src_type, type_kind_t dst_kind, type_kind_t src_kind, assign_error_mode errmode>
struct single_assigner_builtin_base : public single_assigner_builtin_base_error<dst_type, src_type, errmode> {};

// Any assignment with no error checking
template<class dst_type, class src_type, type_kind_t dst_kind, type_kind_t src_kind>
struct single_assigner_builtin_base<dst_type, src_type, dst_kind, src_kind, assign_error_nocheck>
{
    DYND_CUDA_HOST_DEVICE static void assign(dst_type *dst, const src_type *src) {
        DYND_TRACE_ASSIGNMENT(static_cast<dst_type>(*src), dst_type, *src, src_type);

        *dst = static_cast<dst_type>(*src);
    }
};

// Complex floating point -> non-complex with no error checking
template<class dst_type, class src_real_type>
struct single_assigner_builtin_base<dst_type, complex<src_real_type>, sint_kind, complex_kind, assign_error_nocheck>
{
    DYND_CUDA_HOST_DEVICE static void assign(dst_type *dst, const complex<src_real_type> *src) {
        DYND_TRACE_ASSIGNMENT(static_cast<dst_type>(src->real()), dst_type, *src, complex<src_real_type>);

        *dst = static_cast<dst_type>(src->real());
    }
};
template<class dst_type, class src_real_type>
struct single_assigner_builtin_base<dst_type, complex<src_real_type>, uint_kind, complex_kind, assign_error_nocheck>
{
    DYND_CUDA_HOST_DEVICE static void assign(dst_type *dst, const complex<src_real_type> *src) {
        DYND_TRACE_ASSIGNMENT(static_cast<dst_type>(src->real()), dst_type, *src, complex<src_real_type>);

        *dst = static_cast<dst_type>(src->real());
    }
};
template<class src_real_type>
struct single_assigner_builtin_base<float, complex<src_real_type>, real_kind, complex_kind, assign_error_nocheck>
{
    DYND_CUDA_HOST_DEVICE static void assign(float *dst, const complex<src_real_type> *src) {
        DYND_TRACE_ASSIGNMENT(static_cast<float>(src->real()), dst_type, *src, complex<src_real_type>);

        *dst = static_cast<float>(src->real());
    }
};
template<class src_real_type>
struct single_assigner_builtin_base<double, complex<src_real_type>, real_kind, complex_kind, assign_error_nocheck>
{
    DYND_CUDA_HOST_DEVICE static void assign(double *dst, const complex<src_real_type> *src) {
        DYND_TRACE_ASSIGNMENT(static_cast<double>(src->real()), dst_type, *src, complex<src_real_type>);

        *dst = static_cast<double>(src->real());
    }
};


// Anything -> boolean with no checking
template<class src_type, type_kind_t src_kind>
struct single_assigner_builtin_base<dynd_bool, src_type, bool_kind, src_kind, assign_error_nocheck>
{
    DYND_CUDA_HOST_DEVICE static void assign(dynd_bool *dst, const src_type *src) {
        DYND_TRACE_ASSIGNMENT((bool)(s != src_type(0)), dynd_bool, s, src_type);

        *dst = (*src != src_type(0));
    }
};

// Anything -> boolean with overflow checking
template<class src_type, type_kind_t src_kind>
struct single_assigner_builtin_base<dynd_bool, src_type, bool_kind, src_kind, assign_error_overflow>
{
    static void assign(dynd_bool *dst, const src_type *src) {
        src_type s = *src;

        DYND_TRACE_ASSIGNMENT((bool)(s != src_type(0)), dynd_bool, s, src_type);

        if (s == src_type(0)) {
            *dst = false;
        } else if (s == src_type(1)) {
            *dst = true;
        } else {
            std::stringstream ss;
            ss << "overflow while assigning " << ndt::make_type<src_type>() << " value ";
            ss << s << " to " << ndt::make_type<dynd_bool>();
            throw std::overflow_error(ss.str());
        }
    }
};

// Anything -> boolean with other error checking
template<class src_type, type_kind_t src_kind>
struct single_assigner_builtin_base<dynd_bool, src_type, bool_kind, src_kind, assign_error_fractional>
    : public single_assigner_builtin_base<dynd_bool, src_type, bool_kind, src_kind, assign_error_overflow> {};
template<class src_type, type_kind_t src_kind>
struct single_assigner_builtin_base<dynd_bool, src_type, bool_kind, src_kind, assign_error_inexact>
    : public single_assigner_builtin_base<dynd_bool, src_type, bool_kind, src_kind, assign_error_overflow> {};

// Boolean -> anything with other error checking
template<class dst_type, type_kind_t dst_kind>
struct single_assigner_builtin_base<dst_type, dynd_bool, dst_kind, bool_kind, assign_error_overflow>
    : public single_assigner_builtin_base<dst_type, dynd_bool, dst_kind, bool_kind, assign_error_nocheck> {};
template<class dst_type, type_kind_t dst_kind>
struct single_assigner_builtin_base<dst_type, dynd_bool, dst_kind, bool_kind, assign_error_fractional>
    : public single_assigner_builtin_base<dst_type, dynd_bool, dst_kind, bool_kind, assign_error_nocheck> {};
template<class dst_type, type_kind_t dst_kind>
struct single_assigner_builtin_base<dst_type, dynd_bool, dst_kind, bool_kind, assign_error_inexact>
    : public single_assigner_builtin_base<dst_type, dynd_bool, dst_kind, bool_kind, assign_error_nocheck> {};

// Boolean -> boolean with other error checking
template<>
struct single_assigner_builtin_base<dynd_bool, dynd_bool, bool_kind, bool_kind, assign_error_overflow>
    : public single_assigner_builtin_base<dynd_bool, dynd_bool, bool_kind, bool_kind, assign_error_nocheck> {};
template<>
struct single_assigner_builtin_base<dynd_bool, dynd_bool, bool_kind, bool_kind, assign_error_fractional>
    : public single_assigner_builtin_base<dynd_bool, dynd_bool, bool_kind, bool_kind, assign_error_nocheck> {};
template<>
struct single_assigner_builtin_base<dynd_bool, dynd_bool, bool_kind, bool_kind, assign_error_inexact>
    : public single_assigner_builtin_base<dynd_bool, dynd_bool, bool_kind, bool_kind, assign_error_nocheck> {};

// Signed int -> signed int with overflow checking just when sizeof(dst) < sizeof(src)
template<class dst_type, class src_type, bool dst_lt>
struct single_assigner_builtin_signed_to_signed_overflow_base
    : public single_assigner_builtin_base<dst_type, src_type, sint_kind, sint_kind, assign_error_nocheck> {};

// Signed int -> signed int with overflow checking
template<class dst_type, class src_type>
struct single_assigner_builtin_signed_to_signed_overflow_base<dst_type, src_type, true>
{
    static void assign(dst_type *dst, const src_type *src) {
        src_type s = *src;

        if (s < static_cast<src_type>(std::numeric_limits<dst_type>::min()) ||
                        s > static_cast<src_type>(std::numeric_limits<dst_type>::max())) {
            std::stringstream ss;
            ss << "overflow while assigning " << ndt::make_type<src_type>() << " value ";
            ss << s << " to " << ndt::make_type<dst_type>();
            throw std::overflow_error(ss.str());
        }
        *dst = static_cast<dst_type>(s);
    }
};

// Signed int -> signed int with overflow checking
template<class dst_type, class src_type>
struct single_assigner_builtin_base<dst_type, src_type, sint_kind, sint_kind, assign_error_overflow>
    : public single_assigner_builtin_signed_to_signed_overflow_base<dst_type, src_type, sizeof(dst_type) < sizeof(src_type)> {};

// Signed int -> signed int with other error checking
template<class dst_type, class src_type>
struct single_assigner_builtin_base<dst_type, src_type, sint_kind, sint_kind, assign_error_fractional>
    : public single_assigner_builtin_signed_to_signed_overflow_base<dst_type, src_type, sizeof(dst_type) < sizeof(src_type)> {};
template<class dst_type, class src_type>
struct single_assigner_builtin_base<dst_type, src_type, sint_kind, sint_kind, assign_error_inexact>
    : public single_assigner_builtin_signed_to_signed_overflow_base<dst_type, src_type, sizeof(dst_type) < sizeof(src_type)> {};

// Unsigned int -> signed int with overflow checking just when sizeof(dst) <= sizeof(src)
template<class dst_type, class src_type, bool dst_le>
struct single_assigner_builtin_unsigned_to_signed_overflow_base
    : public single_assigner_builtin_base<dst_type, src_type, sint_kind, uint_kind, assign_error_nocheck> {};
template<class dst_type, class src_type>
struct single_assigner_builtin_unsigned_to_signed_overflow_base<dst_type, src_type, true>
{
    static void assign(dst_type *dst, const src_type *src) {
        src_type s = *src;

        DYND_TRACE_ASSIGNMENT(static_cast<dst_type>(s), dst_type, s, src_type);

        if (s > static_cast<src_type>(std::numeric_limits<dst_type>::max())) {
            std::stringstream ss;
            ss << "overflow while assigning " << ndt::make_type<src_type>() << " value ";
            ss << s << " to " << ndt::make_type<dst_type>();
            throw std::overflow_error(ss.str());
        }
        *dst = static_cast<dst_type>(s);
    }
};
template<class dst_type, class src_type>
struct single_assigner_builtin_base<dst_type, src_type, sint_kind, uint_kind, assign_error_overflow>
    : public single_assigner_builtin_unsigned_to_signed_overflow_base<dst_type,
                                                    src_type, sizeof(dst_type) <= sizeof(src_type)> {};

// Unsigned int -> signed int with other error checking
template<class dst_type, class src_type>
struct single_assigner_builtin_base<dst_type, src_type, sint_kind, uint_kind, assign_error_fractional>
    : public single_assigner_builtin_base<dst_type, src_type, sint_kind, uint_kind, assign_error_overflow> {};
template<class dst_type, class src_type>
struct single_assigner_builtin_base<dst_type, src_type, sint_kind, uint_kind, assign_error_inexact>
    : public single_assigner_builtin_base<dst_type, src_type, sint_kind, uint_kind, assign_error_overflow> {};

// Signed int -> unsigned int with positive overflow checking just when sizeof(dst) < sizeof(src)
template<class dst_type, class src_type, bool dst_lt>
struct single_assigner_builtin_signed_to_unsigned_overflow_base
{
    static void assign(dst_type *dst, const src_type *src) {
        src_type s = *src;

        DYND_TRACE_ASSIGNMENT(static_cast<dst_type>(s), dst_type, s, src_type);

        if (s < src_type(0)) {
            std::stringstream ss;
            ss << "overflow while assigning " << ndt::make_type<src_type>() << " value ";
            ss << s << " to " << ndt::make_type<dst_type>();
            throw std::overflow_error(ss.str());
        }
        *dst = static_cast<dst_type>(s);
    }
};
template<class dst_type, class src_type>
struct single_assigner_builtin_signed_to_unsigned_overflow_base<dst_type, src_type, true>
{
    static void assign(dst_type *dst, const src_type *src) {
        src_type s = *src;

        DYND_TRACE_ASSIGNMENT(static_cast<dst_type>(s), dst_type, s, src_type);

        if ((s < src_type(0)) || (static_cast<src_type>(std::numeric_limits<dst_type>::max()) < s)) {
            std::stringstream ss;
            ss << "overflow while assigning " << ndt::make_type<src_type>() << " value ";
            ss << s << " to " << ndt::make_type<dst_type>();
            throw std::overflow_error(ss.str());
        }
        *dst = static_cast<dst_type>(s);
    }
};
template<class dst_type, class src_type>
struct single_assigner_builtin_base<dst_type, src_type, uint_kind, sint_kind, assign_error_overflow>
    : public single_assigner_builtin_signed_to_unsigned_overflow_base<dst_type,
                                                    src_type, sizeof(dst_type) < sizeof(src_type)> {};

// Signed int -> unsigned int with other error checking
template<class dst_type, class src_type>
struct single_assigner_builtin_base<dst_type, src_type, uint_kind, sint_kind, assign_error_fractional>
    : public single_assigner_builtin_base<dst_type, src_type, uint_kind, sint_kind, assign_error_overflow> {};
template<class dst_type, class src_type>
struct single_assigner_builtin_base<dst_type, src_type, uint_kind, sint_kind, assign_error_inexact>
    : public single_assigner_builtin_base<dst_type, src_type, uint_kind, sint_kind, assign_error_overflow> {};

// Unsigned int -> unsigned int with overflow checking just when sizeof(dst) < sizeof(src)
template<class dst_type, class src_type, bool dst_lt>
struct single_assigner_builtin_unsigned_to_unsigned_overflow_base
    : public single_assigner_builtin_base<dst_type, src_type, uint_kind, uint_kind, assign_error_nocheck> {};

// Unsigned int -> unsigned int with overflow checking
template<class dst_type, class src_type>
struct single_assigner_builtin_unsigned_to_unsigned_overflow_base<dst_type, src_type, true>
{
    static void assign(dst_type *dst, const src_type *src) {
        src_type s = *src;

        DYND_TRACE_ASSIGNMENT(static_cast<dst_type>(s), dst_type, s, src_type);

        if (std::numeric_limits<dst_type>::max() < s) {
            std::stringstream ss;
            ss << "overflow while assigning " << ndt::make_type<src_type>() << " value ";
            ss << s << " to " << ndt::make_type<dst_type>();
            throw std::overflow_error(ss.str());
        }
        *dst = static_cast<dst_type>(s);
    }
};

template<class dst_type, class src_type>
struct single_assigner_builtin_base<dst_type, src_type, uint_kind, uint_kind, assign_error_overflow>
    : public single_assigner_builtin_unsigned_to_unsigned_overflow_base<dst_type, src_type, sizeof(dst_type) < sizeof(src_type)> {};

// Unsigned int -> unsigned int with other error checking
template<class dst_type, class src_type>
struct single_assigner_builtin_base<dst_type, src_type, uint_kind, uint_kind, assign_error_fractional>
    : public single_assigner_builtin_unsigned_to_unsigned_overflow_base<dst_type, src_type, sizeof(dst_type) < sizeof(src_type)> {};
template<class dst_type, class src_type>
struct single_assigner_builtin_base<dst_type, src_type, uint_kind, uint_kind, assign_error_inexact>
    : public single_assigner_builtin_unsigned_to_unsigned_overflow_base<dst_type, src_type, sizeof(dst_type) < sizeof(src_type)> {};

// Signed int -> floating point with inexact checking
template<class dst_type, class src_type>
struct single_assigner_builtin_base<dst_type, src_type, real_kind, sint_kind, assign_error_inexact>
{
    static void assign(dst_type *dst, const src_type *src) {
        src_type s = *src;
        dst_type d = static_cast<dst_type>(s);

        DYND_TRACE_ASSIGNMENT(d, dst_type, s, src_type);

        if (static_cast<src_type>(d) != s) {
            std::stringstream ss;
            ss << "inexact value while assigning " << ndt::make_type<src_type>() << " value ";
            ss << s << " to " << ndt::make_type<dst_type>() << " value " << d;
            throw std::runtime_error(ss.str());
        }
        *dst = d;
    }
};

// Signed int -> floating point with other checking
template<class dst_type, class src_type>
struct single_assigner_builtin_base<dst_type, src_type, real_kind, sint_kind, assign_error_overflow>
    : public single_assigner_builtin_base<dst_type, src_type, real_kind, sint_kind, assign_error_nocheck> {};
template<class dst_type, class src_type>
struct single_assigner_builtin_base<dst_type, src_type, real_kind, sint_kind, assign_error_fractional>
    : public single_assigner_builtin_base<dst_type, src_type, real_kind, sint_kind, assign_error_nocheck> {};

// Signed int -> complex floating point with no checking
template<class dst_real_type, class src_type>
struct single_assigner_builtin_base<complex<dst_real_type>, src_type, complex_kind, sint_kind, assign_error_nocheck>
{
    DYND_CUDA_HOST_DEVICE static void assign(complex<dst_real_type> *dst, const src_type *src) {
        DYND_TRACE_ASSIGNMENT(d, complex<dst_real_type>, *src, src_type);

        *dst = static_cast<dst_real_type>(*src);
    }
};

// Signed int -> complex floating point with inexact checking
template<class dst_real_type, class src_type>
struct single_assigner_builtin_base<complex<dst_real_type>, src_type, complex_kind, sint_kind, assign_error_inexact>
{
    static void assign(complex<dst_real_type> *dst, const src_type *src) {
        src_type s = *src;
        dst_real_type d = static_cast<dst_real_type>(s);

        DYND_TRACE_ASSIGNMENT(d, complex<dst_real_type>, s, src_type);

        if (static_cast<src_type>(d) != s) {
            std::stringstream ss;
            ss << "inexact value while assigning " << ndt::make_type<src_type>() << " value ";
            ss << s << " to " << ndt::make_type<complex<dst_real_type> >() << " value " << d;
            throw std::runtime_error(ss.str());
        }
        *dst = d;
    }
};

// Signed int -> complex floating point with other checking
template<class dst_real_type, class src_type>
struct single_assigner_builtin_base<complex<dst_real_type>, src_type, complex_kind, sint_kind, assign_error_overflow>
    : public single_assigner_builtin_base<complex<dst_real_type>, src_type, complex_kind, sint_kind, assign_error_nocheck> {};
template<class dst_real_type, class src_type>
struct single_assigner_builtin_base<complex<dst_real_type>, src_type, complex_kind, sint_kind, assign_error_fractional>
    : public single_assigner_builtin_base<complex<dst_real_type>, src_type, complex_kind, sint_kind, assign_error_nocheck> {};

// Unsigned int -> floating point with inexact checking
template<class dst_type, class src_type>
struct single_assigner_builtin_base<dst_type, src_type, real_kind, uint_kind, assign_error_inexact>
{
    static void assign(dst_type *dst, const src_type *src) {
        src_type s = *src;
        dst_type d = static_cast<dst_type>(s);

        DYND_TRACE_ASSIGNMENT(d, dst_type, s, src_type);

        if (static_cast<src_type>(d) != s) {
            std::stringstream ss;
            ss << "inexact value while assigning " << ndt::make_type<src_type>() << " value ";
            ss << s << " to " << ndt::make_type<dst_type>() << " value " << d;
            throw std::runtime_error(ss.str());
        }
        *dst = d;
    }
};

// Unsigned int -> floating point with other checking
template<class dst_type, class src_type>
struct single_assigner_builtin_base<dst_type, src_type, real_kind, uint_kind, assign_error_overflow>
    : public single_assigner_builtin_base<dst_type, src_type, real_kind, uint_kind, assign_error_nocheck> {};
template<class dst_type, class src_type>
struct single_assigner_builtin_base<dst_type, src_type, real_kind, uint_kind, assign_error_fractional>
    : public single_assigner_builtin_base<dst_type, src_type, real_kind, uint_kind, assign_error_nocheck> {};

// Unsigned int -> complex floating point with no checking
template<class dst_real_type, class src_type>
struct single_assigner_builtin_base<complex<dst_real_type>, src_type, complex_kind, uint_kind, assign_error_nocheck>
{
    DYND_CUDA_HOST_DEVICE static void assign(complex<dst_real_type> *dst, const src_type *src) {
        DYND_TRACE_ASSIGNMENT(d, complex<dst_real_type>, *src, src_type);

        *dst = static_cast<dst_real_type>(*src);
    }
};

// Unsigned int -> complex floating point with inexact checking
template<class dst_real_type, class src_type>
struct single_assigner_builtin_base<complex<dst_real_type>, src_type, complex_kind, uint_kind, assign_error_inexact>
{
    static void assign(complex<dst_real_type> *dst, const src_type *src) {
        src_type s = *src;
        dst_real_type d = static_cast<dst_real_type>(s);

        DYND_TRACE_ASSIGNMENT(d, complex<dst_real_type>, s, src_type);

        if (static_cast<src_type>(d) != s) {
            std::stringstream ss;
            ss << "inexact value while assigning " << ndt::make_type<src_type>() << " value ";
            ss << s << " to " << ndt::make_type<complex<dst_real_type> >() << " value " << d;
            throw std::runtime_error(ss.str());
        }
        *dst = d;
    }
};

// Unsigned int -> complex floating point with other checking
template<class dst_real_type, class src_type>
struct single_assigner_builtin_base<complex<dst_real_type>, src_type, complex_kind, uint_kind, assign_error_overflow>
    : public single_assigner_builtin_base<complex<dst_real_type>, src_type, complex_kind, uint_kind, assign_error_nocheck> {};
template<class dst_real_type, class src_type>
struct single_assigner_builtin_base<complex<dst_real_type>, src_type, complex_kind, uint_kind, assign_error_fractional>
    : public single_assigner_builtin_base<complex<dst_real_type>, src_type, complex_kind, uint_kind, assign_error_nocheck> {};

// Floating point -> signed int with overflow checking
template<class dst_type, class src_type>
struct single_assigner_builtin_base<dst_type, src_type, sint_kind, real_kind, assign_error_overflow>
{
    static void assign(dst_type *dst, const src_type *src) {
        src_type s = *src;

        DYND_TRACE_ASSIGNMENT(static_cast<dst_type>(s), dst_type, s, src_type);

        if (s < std::numeric_limits<dst_type>::min() || std::numeric_limits<dst_type>::max() < s) {
            std::stringstream ss;
            ss << "overflow while assigning " << ndt::make_type<src_type>() << " value ";
            ss << s << " to " << ndt::make_type<dst_type>();
            throw std::overflow_error(ss.str());
        }
        *dst = static_cast<dst_type>(s);
    }
};

// Floating point -> signed int with fractional checking
template<class dst_type, class src_type>
struct single_assigner_builtin_base<dst_type, src_type, sint_kind, real_kind, assign_error_fractional>
{
    static void assign(dst_type *dst, const src_type *src) {
        src_type s = *src;

        DYND_TRACE_ASSIGNMENT(static_cast<dst_type>(s), dst_type, s, src_type);

        if (s < std::numeric_limits<dst_type>::min() || std::numeric_limits<dst_type>::max() < s) {
            std::stringstream ss;
            ss << "overflow while assigning " << ndt::make_type<src_type>() << " value ";
            ss << s << " to " << ndt::make_type<dst_type>();
            throw std::overflow_error(ss.str());
        }

        if (std::floor(s) != s) {
            std::stringstream ss;
            ss << "fractional part lost while assigning " << ndt::make_type<src_type>() << " value ";
            ss << s << " to " << ndt::make_type<dst_type>();
            throw std::runtime_error(ss.str());
        }
        *dst = static_cast<dst_type>(s);
    }
};

// Floating point -> signed int with other checking
template<class dst_type, class src_type>
struct single_assigner_builtin_base<dst_type, src_type, sint_kind, real_kind, assign_error_inexact>
    : public single_assigner_builtin_base<dst_type, src_type, sint_kind, real_kind, assign_error_fractional> {};

// Complex floating point -> signed int with overflow checking
template<class dst_type, class src_real_type>
struct single_assigner_builtin_base<dst_type, complex<src_real_type>, sint_kind, complex_kind, assign_error_overflow>
{
    static void assign(dst_type *dst, const complex<src_real_type> *src) {
        complex<src_real_type> s = *src;

        DYND_TRACE_ASSIGNMENT(static_cast<dst_type>(s.real()), dst_type, s, complex<src_real_type>);

        if (s.imag() != 0) {
            std::stringstream ss;
            ss << "loss of imaginary component while assigning " << ndt::make_type<complex<src_real_type> >() << " value ";
            ss << s << " to " << ndt::make_type<dst_type>();
            throw std::runtime_error(ss.str());
        }

        if (s.real() < std::numeric_limits<dst_type>::min() || std::numeric_limits<dst_type>::max() < s.real()) {
            std::stringstream ss;
            ss << "overflow while assigning " << ndt::make_type<complex<src_real_type> >() << " value ";
            ss << s << " to " << ndt::make_type<dst_type>();
            throw std::overflow_error(ss.str());
        }
        *dst = static_cast<dst_type>(s.real());
    }
};

// Complex floating point -> signed int with fractional checking
template<class dst_type, class src_real_type>
struct single_assigner_builtin_base<dst_type, complex<src_real_type>, sint_kind, complex_kind, assign_error_fractional>
{
    static void assign(dst_type *dst, const complex<src_real_type> *src) {
        complex<src_real_type> s = *src;

        DYND_TRACE_ASSIGNMENT(static_cast<dst_type>(s.real()), dst_type, s, complex<src_real_type>);

        if (s.imag() != 0) {
            std::stringstream ss;
            ss << "loss of imaginary component while assigning " << ndt::make_type<complex<src_real_type> >() << " value ";
            ss << s << " to " << ndt::make_type<dst_type>();
            throw std::runtime_error(ss.str());
        }

        if (s.real() < std::numeric_limits<dst_type>::min() || std::numeric_limits<dst_type>::max() < s.real()) {
            std::stringstream ss;
            ss << "overflow while assigning " << ndt::make_type<complex<src_real_type> >() << " value ";
            ss << s << " to " << ndt::make_type<dst_type>();
            throw std::overflow_error(ss.str());
        }

        if (std::floor(s.real()) != s.real()) {
            std::stringstream ss;
            ss << "fractional part lost while assigning " << ndt::make_type<complex<src_real_type> >() << " value ";
            ss << s << " to " << ndt::make_type<dst_type>();
            throw std::runtime_error(ss.str());
        }
        *dst = static_cast<dst_type>(s.real());
    }
};

// Complex floating point -> signed int with other checking
template<class dst_type, class src_real_type>
struct single_assigner_builtin_base<dst_type, complex<src_real_type>, sint_kind, complex_kind, assign_error_inexact>
    : public single_assigner_builtin_base<dst_type, complex<src_real_type>, sint_kind, complex_kind, assign_error_fractional> {};

// Floating point -> unsigned int with overflow checking
template<class dst_type, class src_type>
struct single_assigner_builtin_base<dst_type, src_type, uint_kind, real_kind, assign_error_overflow>
{
    static void assign(dst_type *dst, const src_type *src) {
        src_type s = *src;

        DYND_TRACE_ASSIGNMENT(static_cast<dst_type>(s), dst_type, s, src_type);

        if (s < 0 || std::numeric_limits<dst_type>::max() < s) {
            std::stringstream ss;
            ss << "overflow while assigning " << ndt::make_type<src_type>() << " value ";
            ss << s << " to " << ndt::make_type<dst_type>();
            throw std::overflow_error(ss.str());
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

        DYND_TRACE_ASSIGNMENT(static_cast<dst_type>(s), dst_type, s, src_type);

        if (s < 0 || std::numeric_limits<dst_type>::max() < s) {
            std::stringstream ss;
            ss << "overflow while assigning " << ndt::make_type<src_type>() << " value ";
            ss << s << " to " << ndt::make_type<dst_type>();
            throw std::overflow_error(ss.str());
        }

        if (std::floor(s) != s) {
            std::stringstream ss;
            ss << "fractional part lost while assigning " << ndt::make_type<src_type>() << " value ";
            ss << s << " to " << ndt::make_type<dst_type>();
            throw std::runtime_error(ss.str());
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
struct single_assigner_builtin_base<dst_type, complex<src_real_type>, uint_kind, complex_kind, assign_error_overflow>
{
    static void assign(dst_type *dst, const complex<src_real_type> *src) {
        complex<src_real_type> s = *src;

        DYND_TRACE_ASSIGNMENT(static_cast<dst_type>(s.real()), dst_type, s, complex<src_real_type>);

        if (s.imag() != 0) {
            std::stringstream ss;
            ss << "loss of imaginary component while assigning " << ndt::make_type<complex<src_real_type> >() << " value ";
            ss << s << " to " << ndt::make_type<dst_type>();
            throw std::runtime_error(ss.str());
        }

        if (s.real() < 0 || std::numeric_limits<dst_type>::max() < s.real()) {
            std::stringstream ss;
            ss << "overflow while assigning " << ndt::make_type<complex<src_real_type> >() << " value ";
            ss << s << " to " << ndt::make_type<dst_type>();
            throw std::overflow_error(ss.str());
        }
        *dst = static_cast<dst_type>(s.real());
    }
};

// Complex floating point -> unsigned int with fractional checking
template<class dst_type, class src_real_type>
struct single_assigner_builtin_base<dst_type, complex<src_real_type>, uint_kind, complex_kind, assign_error_fractional>
{
    static void assign(dst_type *dst, const complex<src_real_type> *src) {
        complex<src_real_type> s = *src;

        DYND_TRACE_ASSIGNMENT(static_cast<dst_type>(s.real()), dst_type, s, complex<src_real_type>);

        if (s.imag() != 0) {
            std::stringstream ss;
            ss << "loss of imaginary component while assigning " << ndt::make_type<complex<src_real_type> >() << " value ";
            ss << s << " to " << ndt::make_type<dst_type>();
            throw std::runtime_error(ss.str());
        }

        if (s.real() < 0 || std::numeric_limits<dst_type>::max() < s.real()) {
            std::stringstream ss;
            ss << "overflow while assigning " << ndt::make_type<complex<src_real_type> >() << " value ";
            ss << s << " to " << ndt::make_type<dst_type>();
            throw std::overflow_error(ss.str());
        }

        if (std::floor(s.real()) != s.real()) {
            std::stringstream ss;
            ss << "fractional part lost while assigning " << ndt::make_type<complex<src_real_type> >() << " value ";
            ss << s << " to " << ndt::make_type<dst_type>();
            throw std::runtime_error(ss.str());
        }
        *dst = static_cast<dst_type>(s.real());
    }
};

// Complex floating point -> unsigned int with other checking
template<class dst_type, class src_real_type>
struct single_assigner_builtin_base<dst_type, complex<src_real_type>, uint_kind, complex_kind, assign_error_inexact>
    : public single_assigner_builtin_base<dst_type, complex<src_real_type>, uint_kind, complex_kind, assign_error_fractional> {};

// float -> float with no checking
template<>
struct single_assigner_builtin_base<float, float, real_kind, real_kind, assign_error_overflow>
    : public single_assigner_builtin_base<float, float, real_kind, real_kind, assign_error_nocheck> {};
template<>
struct single_assigner_builtin_base<float, float, real_kind, real_kind, assign_error_fractional>
    : public single_assigner_builtin_base<float, float, real_kind, real_kind, assign_error_nocheck> {};
template<>
struct single_assigner_builtin_base<float, float, real_kind, real_kind, assign_error_inexact>
    : public single_assigner_builtin_base<float, float, real_kind, real_kind, assign_error_nocheck> {};

// complex<float> -> complex<float> with no checking
template<>
struct single_assigner_builtin_base<complex<float>, complex<float>, complex_kind, complex_kind, assign_error_overflow>
    : public single_assigner_builtin_base<complex<float>, complex<float>, complex_kind, complex_kind, assign_error_nocheck> {};
template<>
struct single_assigner_builtin_base<complex<float>, complex<float>, complex_kind, complex_kind, assign_error_fractional>
    : public single_assigner_builtin_base<complex<float>, complex<float>, complex_kind, complex_kind, assign_error_nocheck> {};
template<>
struct single_assigner_builtin_base<complex<float>, complex<float>, complex_kind, complex_kind, assign_error_inexact>
    : public single_assigner_builtin_base<complex<float>, complex<float>, complex_kind, complex_kind, assign_error_nocheck> {};

// float -> double with no checking
template<>
struct single_assigner_builtin_base<double, float, real_kind, real_kind, assign_error_overflow>
    : public single_assigner_builtin_base<double, float, real_kind, real_kind, assign_error_nocheck> {};
template<>
struct single_assigner_builtin_base<double, float, real_kind, real_kind, assign_error_fractional>
    : public single_assigner_builtin_base<double, float, real_kind, real_kind, assign_error_nocheck> {};
template<>
struct single_assigner_builtin_base<double, float, real_kind, real_kind, assign_error_inexact>
    : public single_assigner_builtin_base<double, float, real_kind, real_kind, assign_error_nocheck> {};

// complex<float> -> complex<double> with no checking
template<>
struct single_assigner_builtin_base<complex<double>, complex<float>, complex_kind, complex_kind, assign_error_overflow>
    : public single_assigner_builtin_base<complex<double>, complex<float>, complex_kind, complex_kind, assign_error_nocheck> {};
template<>
struct single_assigner_builtin_base<complex<double>, complex<float>, complex_kind, complex_kind, assign_error_fractional>
    : public single_assigner_builtin_base<complex<double>, complex<float>, complex_kind, complex_kind, assign_error_nocheck> {};
template<>
struct single_assigner_builtin_base<complex<double>, complex<float>, complex_kind, complex_kind, assign_error_inexact>
    : public single_assigner_builtin_base<complex<double>, complex<float>, complex_kind, complex_kind, assign_error_nocheck> {};

// double -> double with no checking
template<>
struct single_assigner_builtin_base<double, double, real_kind, real_kind, assign_error_overflow>
    : public single_assigner_builtin_base<double, double, real_kind, real_kind, assign_error_nocheck> {};
template<>
struct single_assigner_builtin_base<double, double, real_kind, real_kind, assign_error_fractional>
    : public single_assigner_builtin_base<double, double, real_kind, real_kind, assign_error_nocheck> {};
template<>
struct single_assigner_builtin_base<double, double, real_kind, real_kind, assign_error_inexact>
    : public single_assigner_builtin_base<double, double, real_kind, real_kind, assign_error_nocheck> {};

// complex<double> -> complex<double> with no checking
template<>
struct single_assigner_builtin_base<complex<double>, complex<double>, complex_kind, complex_kind, assign_error_overflow>
    : public single_assigner_builtin_base<complex<double>, complex<double>, complex_kind, complex_kind, assign_error_nocheck> {};
template<>
struct single_assigner_builtin_base<complex<double>, complex<double>, complex_kind, complex_kind, assign_error_fractional>
    : public single_assigner_builtin_base<complex<double>, complex<double>, complex_kind, complex_kind, assign_error_nocheck> {};
template<>
struct single_assigner_builtin_base<complex<double>, complex<double>, complex_kind, complex_kind, assign_error_inexact>
    : public single_assigner_builtin_base<complex<double>, complex<double>, complex_kind, complex_kind, assign_error_nocheck> {};

// double -> float with overflow checking
template<>
struct single_assigner_builtin_base<float, double, real_kind, real_kind, assign_error_overflow>
{
    static void assign(float *dst, const double *src) {
        DYND_TRACE_ASSIGNMENT(static_cast<float>(*src), float, *src, double);

#if defined(DYND_USE_FPSTATUS)
        clear_fp_status();
        *dst = static_cast<float>(*src);
        if (is_overflow_fp_status()) {
            std::stringstream ss;
            ss << "overflow while assigning " << ndt::make_type<double>() << " value ";
            ss << *src << " to " << ndt::make_type<float>();
            throw std::overflow_error(ss.str());
        }
#else
        double s = *src;
        if (isfinite(s) && (s < -std::numeric_limits<float>::max() ||
                            s > std::numeric_limits<float>::max())) {
            std::stringstream ss;
            ss << "overflow while assigning " << ndt::make_type<double>() << " value ";
            ss << *src << " to " << ndt::make_type<float>();
            throw std::overflow_error(ss.str());
        }
        *dst = static_cast<float>(s);
#endif // DYND_USE_FPSTATUS
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
        DYND_TRACE_ASSIGNMENT(static_cast<float>(*src), float, *src, double);

        double s = *src;
        float d;
#if defined(DYND_USE_FPSTATUS)
        clear_fp_status();
        d = static_cast<float>(s);
        if (is_overflow_fp_status()) {
            std::stringstream ss;
            ss << "overflow while assigning " << ndt::make_type<double>() << " value ";
            ss << *src << " to " << ndt::make_type<float>();
            throw std::overflow_error(ss.str());
        }
#else
        if (isfinite(s) && (s < -std::numeric_limits<float>::max() ||
                            s > std::numeric_limits<float>::max())) {
            std::stringstream ss;
            ss << "overflow while assigning " << ndt::make_type<double>() << " value ";
            ss << *src << " to " << ndt::make_type<float>();
            throw std::runtime_error(ss.str());
        }
        d = static_cast<float>(s);
#endif // DYND_USE_FPSTATUS

        // The inexact status didn't work as it should have, so converting back to double and comparing
        //if (is_inexact_fp_status()) {
        //    throw std::runtime_error("inexact precision loss while assigning double to float");
        //}
        if (d != s) {
            std::stringstream ss;
            ss << "inexact precision loss while assigning " << ndt::make_type<double>() << " value ";
            ss << *src << " to " << ndt::make_type<float>();
            throw std::runtime_error(ss.str());
        }
        *dst = d;
    }
};


// complex<double> -> complex<float> with overflow checking
template<>
struct single_assigner_builtin_base<complex<float>, complex<double>, complex_kind, complex_kind, assign_error_overflow>
{
    static void assign(complex<float> *dst, const complex<double> *src) {
        DYND_TRACE_ASSIGNMENT(static_cast<complex<float> >(*src), complex<float>, *src, complex<double>);

#if defined(DYND_USE_FPSTATUS)
        clear_fp_status();
        *dst = static_cast<complex<float> >(*src);
        if (is_overflow_fp_status()) {
            std::stringstream ss;
            ss << "overflow while assigning " << ndt::make_type<complex<double> >() << " value ";
            ss << *src << " to " << ndt::make_type<complex<float> >();
            throw std::overflow_error(ss.str());
        }
#else
        complex<double>(s) = *src;
        if (s.real() < -std::numeric_limits<float>::max() || s.real() > std::numeric_limits<float>::max() ||
                    s.imag() < -std::numeric_limits<float>::max() || s.imag() > std::numeric_limits<float>::max()) {
            std::stringstream ss;
            ss << "overflow while assigning " << ndt::make_type<complex<double> >() << " value ";
            ss << *src << " to " << ndt::make_type<complex<float> >();
            throw std::overflow_error(ss.str());
        }
        *dst = static_cast<complex<float> >(s);
#endif // DYND_USE_FPSTATUS
    }
};

// complex<double> -> complex<float> with fractional checking
template<>
struct single_assigner_builtin_base<complex<float>, complex<double>, complex_kind, complex_kind, assign_error_fractional>
    : public single_assigner_builtin_base<complex<float>, complex<double>, complex_kind, complex_kind, assign_error_overflow> {};


// complex<double> -> complex<float> with inexact checking
template<>
struct single_assigner_builtin_base<complex<float>, complex<double>, complex_kind, complex_kind, assign_error_inexact>
{
    static void assign(complex<float> *dst, const complex<double> *src) {
        DYND_TRACE_ASSIGNMENT(static_cast<complex<float> >(*src), complex<float>, *src, complex<double>);

        complex<double> s = *src;
        complex<float> d;

#if defined(DYND_USE_FPSTATUS)
        clear_fp_status();
        d = static_cast<complex<float> >(s);
        if (is_overflow_fp_status()) {
            std::stringstream ss;
            ss << "overflow while assigning " << ndt::make_type<complex<double> >() << " value ";
            ss << *src << " to " << ndt::make_type<complex<float> >();
            throw std::overflow_error(ss.str());
        }
#else
        if (s.real() < -std::numeric_limits<float>::max() || s.real() > std::numeric_limits<float>::max() ||
                    s.imag() < -std::numeric_limits<float>::max() || s.imag() > std::numeric_limits<float>::max()) {
            std::stringstream ss;
            ss << "overflow while assigning " << ndt::make_type<complex<double> >() << " value ";
            ss << *src << " to " << ndt::make_type<complex<float> >();
            throw std::overflow_error(ss.str());
        }
        d = static_cast<complex<float> >(s);
#endif // DYND_USE_FPSTATUS

        // The inexact status didn't work as it should have, so converting back to double and comparing
        //if (is_inexact_fp_status()) {
        //    throw std::runtime_error("inexact precision loss while assigning double to float");
        //}
        if (d.real() != s.real() || d.imag() != s.imag()) {
            std::stringstream ss;
            ss << "inexact precision loss while assigning " << ndt::make_type<complex<double> >() << " value ";
            ss << *src << " to " << ndt::make_type<complex<float> >();
            throw std::runtime_error(ss.str());
        }
        *dst = d;
    }
};

// complex<T> -> T with overflow checking
template<typename real_type>
struct single_assigner_builtin_base<real_type, complex<real_type>, real_kind, complex_kind, assign_error_overflow>
{
    static void assign(real_type *dst, const complex<real_type> *src) {
        complex<real_type> s = *src;

        DYND_TRACE_ASSIGNMENT(static_cast<float>(s.real()), real_type, s, complex<real_type>);

        if (s.imag() != 0) {
            std::stringstream ss;
            ss << "loss of imaginary component while assigning " << ndt::make_type<complex<real_type> >() << " value ";
            ss << *src << " to " << ndt::make_type<real_type>();
            throw std::runtime_error(ss.str());
        }

        *dst = s.real();
    }
};

// complex<T> -> T with fractional checking
template<typename real_type>
struct single_assigner_builtin_base<real_type, complex<real_type>, real_kind, complex_kind, assign_error_fractional>
    : public single_assigner_builtin_base<real_type, complex<real_type>, real_kind, complex_kind, assign_error_overflow> {};

// complex<T> -> T with inexact checking
template<typename real_type>
struct single_assigner_builtin_base<real_type, complex<real_type>, real_kind, complex_kind, assign_error_inexact>
    : public single_assigner_builtin_base<real_type, complex<real_type>, real_kind, complex_kind, assign_error_overflow> {};



// double -> complex<float>
template<>
struct single_assigner_builtin_base<complex<float>, double, complex_kind, real_kind, assign_error_nocheck>
{
    DYND_CUDA_HOST_DEVICE static void assign(complex<float> *dst, const double *src) {
        DYND_TRACE_ASSIGNMENT(static_cast<complex<float> >(*src), complex<float>, *src, double);

        *dst = static_cast<float>(*src);
    }
};
// T -> complex<T>
template<typename real_type>
struct single_assigner_builtin_base<complex<real_type>, real_type, complex_kind, real_kind, assign_error_nocheck>
{
    DYND_CUDA_HOST_DEVICE static void assign(complex<real_type> *dst, const real_type *src) {
        DYND_TRACE_ASSIGNMENT(static_cast<complex<real_type> >(*src), complex<real_type>, *src, real_type);

        *dst = *src;
    }
};
template<typename real_type, assign_error_mode errmode>
struct single_assigner_builtin_base<complex<real_type>, real_type, complex_kind, real_kind, errmode>
    : public single_assigner_builtin_base<complex<real_type>, real_type, complex_kind, real_kind, assign_error_nocheck> {};

// float -> complex<double>
template<>
struct single_assigner_builtin_base<complex<double>, float, complex_kind, real_kind, assign_error_nocheck>
{
    DYND_CUDA_HOST_DEVICE static void assign(complex<double> *dst, const float *src) {
        DYND_TRACE_ASSIGNMENT(static_cast<complex<double> >(*src), complex<double>, *src, float);

        *dst = *src;
    }
};
template<assign_error_mode errmode>
struct single_assigner_builtin_base<complex<double>, float, complex_kind, real_kind, errmode>
    : public single_assigner_builtin_base<complex<double>, float, complex_kind, real_kind, assign_error_nocheck> {};

// complex<float> -> double with overflow checking
template<>
struct single_assigner_builtin_base<double, complex<float>, real_kind, complex_kind, assign_error_overflow>
{
    static void assign(double *dst, const complex<float> *src) {
        complex<float> s = *src;

        DYND_TRACE_ASSIGNMENT(static_cast<double>(s.real()), double, s, complex<float>);

        if (s.imag() != 0) {
            std::stringstream ss;
            ss << "loss of imaginary component while assigning " << ndt::make_type<complex<float> >() << " value ";
            ss << *src << " to " << ndt::make_type<double>();
            throw std::runtime_error(ss.str());
        }

        *dst = s.real();
    }
};

// complex<float> -> double with fractional checking
template<>
struct single_assigner_builtin_base<double, complex<float>, real_kind, complex_kind, assign_error_fractional>
    : public single_assigner_builtin_base<double, complex<float>, real_kind, complex_kind, assign_error_overflow> {};

// complex<float> -> double with inexact checking
template<>
struct single_assigner_builtin_base<double, complex<float>, real_kind, complex_kind, assign_error_inexact>
    : public single_assigner_builtin_base<double, complex<float>, real_kind, complex_kind, assign_error_overflow> {};

// complex<double> -> float with overflow checking
template<>
struct single_assigner_builtin_base<float, complex<double>, real_kind, complex_kind, assign_error_overflow>
{
    static void assign(float *dst, const complex<double> *src) {
        complex<double> s = *src;
        float d;

        DYND_TRACE_ASSIGNMENT(static_cast<float>(s.real()), float, s, complex<double>);

        if (s.imag() != 0) {
            std::stringstream ss;
            ss << "loss of imaginary component while assigning " << ndt::make_type<complex<double> >() << " value ";
            ss << *src << " to " << ndt::make_type<float>();
            throw std::runtime_error(ss.str());
        }

#if defined(DYND_USE_FPSTATUS)
        clear_fp_status();
        d = static_cast<float>(s.real());
        if (is_overflow_fp_status()) {
            std::stringstream ss;
            ss << "overflow while assigning " << ndt::make_type<complex<double> >() << " value ";
            ss << *src << " to " << ndt::make_type<float>();
            throw std::overflow_error(ss.str());
        }
#else
        if (s.real() < -std::numeric_limits<float>::max() || s.real() > std::numeric_limits<float>::max()) {
            std::stringstream ss;
            ss << "overflow while assigning " << ndt::make_type<complex<double> >() << " value ";
            ss << *src << " to " << ndt::make_type<float>();
            throw std::overflow_error(ss.str());
        }
        d = static_cast<float>(s.real());
#endif // DYND_USE_FPSTATUS

        *dst = d;
    }
};

// complex<double> -> float with fractional checking
template<>
struct single_assigner_builtin_base<float, complex<double>, real_kind, complex_kind, assign_error_fractional>
    : public single_assigner_builtin_base<float, complex<double>, real_kind, complex_kind, assign_error_overflow> {};

// complex<double> -> float with inexact checking
template<>
struct single_assigner_builtin_base<float, complex<double>, real_kind, complex_kind, assign_error_inexact>
{
    static void assign(float *dst, const complex<double> *src) {
        complex<double> s = *src;
        float d;

        DYND_TRACE_ASSIGNMENT(static_cast<float>(s.real()), float, s, complex<double>);

        if (s.imag() != 0) {
            std::stringstream ss;
            ss << "loss of imaginary component while assigning " << ndt::make_type<complex<double> >() << " value ";
            ss << *src << " to " << ndt::make_type<float>();
            throw std::runtime_error(ss.str());
        }

#if defined(DYND_USE_FPSTATUS)
        clear_fp_status();
        d = static_cast<float>(s.real());
        if (is_overflow_fp_status()) {
            std::stringstream ss;
            ss << "overflow while assigning " << ndt::make_type<complex<double> >() << " value ";
            ss << s << " to " << ndt::make_type<float>();
            throw std::overflow_error(ss.str());
        }
#else
        if (s.real() < -std::numeric_limits<float>::max() || s.real() > std::numeric_limits<float>::max()) {
            std::stringstream ss;
            ss << "overflow while assigning " << ndt::make_type<complex<double> >() << " value ";
            ss << s << " to " << ndt::make_type<float>();
            throw std::overflow_error(ss.str());
        }
        d = static_cast<float>(s.real());
#endif // DYND_USE_FPSTATUS

        if (d != s.real()) {
            std::stringstream ss;
            ss << "inexact precision loss while assigning " << ndt::make_type<complex<double> >() << " value ";
            ss << *src << " to " << ndt::make_type<float>();
            throw std::runtime_error(ss.str());
        }

        *dst = d;
    }
};

// double -> complex<float> with overflow checking
template<>
struct single_assigner_builtin_base<complex<float>, double, complex_kind, real_kind, assign_error_overflow>
{
    static void assign(complex<float> *dst, const double *src) {
        double s = *src;
        float d;

        DYND_TRACE_ASSIGNMENT(static_cast<complex<float> >(s), complex<float>, s, double);

#if defined(DYND_USE_FPSTATUS)
        clear_fp_status();
        d = static_cast<float>(s);
        if (is_overflow_fp_status()) {
            std::stringstream ss;
            ss << "overflow while assigning " << ndt::make_type<double>() << " value ";
            ss << s << " to " << ndt::make_type<complex<float> >();
            throw std::overflow_error(ss.str());
        }
#else
        if (isfinite(s) && (s < -std::numeric_limits<float>::max() ||
                            s > std::numeric_limits<float>::max())) {
            std::stringstream ss;
            ss << "overflow while assigning " << ndt::make_type<double>() << " value ";
            ss << s << " to " << ndt::make_type<complex<float> >();
            throw std::overflow_error(ss.str());
        }
        d = static_cast<float>(s);
#endif // DYND_USE_FPSTATUS

        *dst = d;
    }
};

// double -> complex<float> with fractional checking
template<>
struct single_assigner_builtin_base<complex<float>, double, complex_kind, real_kind, assign_error_fractional>
    : public single_assigner_builtin_base<complex<float>, double, complex_kind, real_kind, assign_error_overflow> {};

// double -> complex<float> with inexact checking
template<>
struct single_assigner_builtin_base<complex<float>, double, complex_kind, real_kind, assign_error_inexact>
{
    static void assign(complex<float> *dst, const double *src) {
        double s = *src;
        float d;

        DYND_TRACE_ASSIGNMENT(static_cast<complex<float> >(s), complex<float>, s, double);

#if defined(DYND_USE_FPSTATUS)
        clear_fp_status();
        d = static_cast<float>(s);
        if (is_overflow_fp_status()) {
            std::stringstream ss;
            ss << "overflow while assigning " << ndt::make_type<double>() << " value ";
            ss << s << " to " << ndt::make_type<complex<float> >();
            throw std::overflow_error(ss.str());
        }
#else
        if (isfinite(s) && (s < -std::numeric_limits<float>::max() ||
                            s > std::numeric_limits<float>::max())) {
            std::stringstream ss;
            ss << "overflow while assigning " << ndt::make_type<double>() << " value ";
            ss << s << " to " << ndt::make_type<complex<float> >();
            throw std::overflow_error(ss.str());
        }
        d = static_cast<float>(s);
#endif // DYND_USE_FPSTATUS

        if (d != s) {
            std::stringstream ss;
            ss << "inexact precision loss while assigning " << ndt::make_type<double>() << " value ";
            ss << s << " to " << ndt::make_type<complex<float> >();
            throw std::runtime_error(ss.str());
        }

        *dst = d;
    }
};

#include "single_assigner_builtin_int128.hpp"
#include "single_assigner_builtin_uint128.hpp"
#include "single_assigner_builtin_float128.hpp"
#include "single_assigner_builtin_float16.hpp"



// This is the interface exposed for use outside of this file
template <class dst_type, class src_type, assign_error_mode errmode>
struct single_assigner_builtin
    : public single_assigner_builtin_base<dst_type, src_type,
                        dynd_kind_of<dst_type>::value, dynd_kind_of<src_type>::value, errmode>
{};
template <class same_type, assign_error_mode errmode>
struct single_assigner_builtin<same_type, same_type, errmode>
{
    DYND_CUDA_HOST_DEVICE static void assign(same_type *dst, const same_type *src) {
        *dst = *src;
    }
};

} // namespace dynd
