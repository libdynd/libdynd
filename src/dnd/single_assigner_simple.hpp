//
// Copyright (C) 2011 Mark Wiebe (mwwiebe@gmail.com)
// All rights reserved.
//
// This is unreleased proprietary software.
//

// This file is an internal implementation detail of built-in value assignment
// for aligned values in native byte order.

#include <dnd/fpstatus.hpp>
#include <cmath>


// Put it in an anonymous namespace
namespace {

using namespace dnd;

template<class dst_type, class src_type, dtype_kind dst_kind, dtype_kind src_kind, assign_error_mode errmode>
struct single_assigner_simple_base;

// Any assignment with no error checking
template<class dst_type, class src_type, dtype_kind dst_kind, dtype_kind src_kind>
struct single_assigner_simple_base<dst_type, src_type, dst_kind, src_kind, assign_error_none>
{
    static void assign(dst_type *dst, const src_type *src) {
        *dst = static_cast<dst_type>(*src);
    }
};

// Anything -> boolean with overflow checking
template<class src_type, dtype_kind src_kind>
struct single_assigner_simple_base<dnd_bool, src_type, bool_kind, src_kind, assign_error_overflow>
{
    static void assign(dnd_bool *dst, const src_type *src) {
        src_type s = *src;

        if (s == 0) {
            *dst = false;
        } else if (s == 1) {
            *dst = true;
        } else {
            throw std::runtime_error("overflow while assigning to boolean value");
        }
    }
};

// Anything -> boolean with other error checking
template<class src_type, dtype_kind src_kind>
struct single_assigner_simple_base<dnd_bool, src_type, bool_kind, src_kind, assign_error_fractional>
    : public single_assigner_simple_base<dnd_bool, src_type, bool_kind, src_kind, assign_error_overflow> {};
template<class src_type, dtype_kind src_kind>
struct single_assigner_simple_base<dnd_bool, src_type, bool_kind, src_kind, assign_error_inexact>
    : public single_assigner_simple_base<dnd_bool, src_type, bool_kind, src_kind, assign_error_overflow> {};

// Boolean -> anything with other error checking
template<class dst_type, dtype_kind dst_kind>
struct single_assigner_simple_base<dst_type, dnd_bool, dst_kind, bool_kind, assign_error_overflow>
    : public single_assigner_simple_base<dst_type, dnd_bool, dst_kind, bool_kind, assign_error_none> {};
template<class dst_type, dtype_kind dst_kind>
struct single_assigner_simple_base<dst_type, dnd_bool, dst_kind, bool_kind, assign_error_fractional>
    : public single_assigner_simple_base<dst_type, dnd_bool, dst_kind, bool_kind, assign_error_none> {};
template<class dst_type, dtype_kind dst_kind>
struct single_assigner_simple_base<dst_type, dnd_bool, dst_kind, bool_kind, assign_error_inexact>
    : public single_assigner_simple_base<dst_type, dnd_bool, dst_kind, bool_kind, assign_error_none> {};

// Boolean -> boolean with other error checking
template<>
struct single_assigner_simple_base<dnd_bool, dnd_bool, bool_kind, bool_kind, assign_error_overflow>
    : public single_assigner_simple_base<dnd_bool, dnd_bool, bool_kind, bool_kind, assign_error_none> {};
template<>
struct single_assigner_simple_base<dnd_bool, dnd_bool, bool_kind, bool_kind, assign_error_fractional>
    : public single_assigner_simple_base<dnd_bool, dnd_bool, bool_kind, bool_kind, assign_error_none> {};
template<>
struct single_assigner_simple_base<dnd_bool, dnd_bool, bool_kind, bool_kind, assign_error_inexact>
    : public single_assigner_simple_base<dnd_bool, dnd_bool, bool_kind, bool_kind, assign_error_none> {};

// Signed int -> signed int with overflow checking
template<class dst_type, class src_type>
struct single_assigner_simple_base<dst_type, src_type, int_kind, int_kind, assign_error_overflow>
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
struct single_assigner_simple_base<dst_type, src_type, int_kind, int_kind, assign_error_fractional>
    : public single_assigner_simple_base<dst_type, src_type, int_kind, int_kind, assign_error_overflow> {};
template<class dst_type, class src_type>
struct single_assigner_simple_base<dst_type, src_type, int_kind, int_kind, assign_error_inexact>
    : public single_assigner_simple_base<dst_type, src_type, int_kind, int_kind, assign_error_overflow> {};

// Unsigned int -> signed int with overflow checking
template<class dst_type, class src_type>
struct single_assigner_simple_base<dst_type, src_type, int_kind, uint_kind, assign_error_overflow>
{
    static void assign(dst_type *dst, const src_type *src) {
        src_type s = *src;

        if (s > std::numeric_limits<dst_type>::max()) {
            throw std::runtime_error("overflow while assigning unsigned integer signed integer");
        }
        *dst = static_cast<dst_type>(s);
    }
};

// Unsigned int -> signed int with other error checking
template<class dst_type, class src_type>
struct single_assigner_simple_base<dst_type, src_type, int_kind, uint_kind, assign_error_fractional>
    : public single_assigner_simple_base<dst_type, src_type, int_kind, uint_kind, assign_error_overflow> {};
template<class dst_type, class src_type>
struct single_assigner_simple_base<dst_type, src_type, int_kind, uint_kind, assign_error_inexact>
    : public single_assigner_simple_base<dst_type, src_type, int_kind, uint_kind, assign_error_overflow> {};

// Signed int -> unsigned int with overflow checking
template<class dst_type, class src_type>
struct single_assigner_simple_base<dst_type, src_type, uint_kind, int_kind, assign_error_overflow>
{
    static void assign(dst_type *dst, const src_type *src) {
        src_type s = *src;

        if (s < 0 || s > std::numeric_limits<dst_type>::max()) {
            throw std::runtime_error("overflow while assigning signed integer to unsigned integer");
        }
        *dst = static_cast<dst_type>(s);
    }
};

// Signed int -> unsigned int with other error checking
template<class dst_type, class src_type>
struct single_assigner_simple_base<dst_type, src_type, uint_kind, int_kind, assign_error_fractional>
    : public single_assigner_simple_base<dst_type, src_type, uint_kind, int_kind, assign_error_overflow> {};
template<class dst_type, class src_type>
struct single_assigner_simple_base<dst_type, src_type, uint_kind, int_kind, assign_error_inexact>
    : public single_assigner_simple_base<dst_type, src_type, uint_kind, int_kind, assign_error_overflow> {};

// Unsigned int -> unsigned int with overflow checking
template<class dst_type, class src_type>
struct single_assigner_simple_base<dst_type, src_type, uint_kind, uint_kind, assign_error_overflow>
{
    static void assign(dst_type *dst, const src_type *src) {
        src_type s = *src;

        if (s > std::numeric_limits<dst_type>::max()) {
            throw std::runtime_error("overflow while assigning unsigned integer to unsigned integer");
        }
        *dst = static_cast<dst_type>(s);
    }
};

// Unsigned int -> unsigned int with other error checking
template<class dst_type, class src_type>
struct single_assigner_simple_base<dst_type, src_type, uint_kind, uint_kind, assign_error_fractional>
    : public single_assigner_simple_base<dst_type, src_type, uint_kind, uint_kind, assign_error_overflow> {};
template<class dst_type, class src_type>
struct single_assigner_simple_base<dst_type, src_type, uint_kind, uint_kind, assign_error_inexact>
    : public single_assigner_simple_base<dst_type, src_type, uint_kind, uint_kind, assign_error_overflow> {};

// Signed int -> floating point with inexact checking
template<class dst_type, class src_type>
struct single_assigner_simple_base<dst_type, src_type, float_kind, int_kind, assign_error_inexact>
{
    static void assign(dst_type *dst, const src_type *src) {
        src_type s = *src;
        dst_type d = static_cast<dst_type>(s);

        if (static_cast<src_type>(d) != s) {
            throw std::runtime_error("inexact value while assigning signed integer to floating point");
        }
        *dst = d;
    }
};

// Signed int -> floating point with other checking
template<class dst_type, class src_type>
struct single_assigner_simple_base<dst_type, src_type, float_kind, int_kind, assign_error_overflow>
    : public single_assigner_simple_base<dst_type, src_type, float_kind, int_kind, assign_error_none> {};
template<class dst_type, class src_type>
struct single_assigner_simple_base<dst_type, src_type, float_kind, int_kind, assign_error_fractional>
    : public single_assigner_simple_base<dst_type, src_type, float_kind, int_kind, assign_error_none> {};

// Unsigned int -> floating point with inexact checking
template<class dst_type, class src_type>
struct single_assigner_simple_base<dst_type, src_type, float_kind, uint_kind, assign_error_inexact>
{
    static void assign(dst_type *dst, const src_type *src) {
        src_type s = *src;
        dst_type d = static_cast<dst_type>(s);

        if (static_cast<src_type>(d) != s) {
            throw std::runtime_error("inexact value while assigning unsigned integer to floating point");
        }
        *dst = d;
    }
};

// Unsigned int -> floating point with other checking
template<class dst_type, class src_type>
struct single_assigner_simple_base<dst_type, src_type, float_kind, uint_kind, assign_error_overflow>
    : public single_assigner_simple_base<dst_type, src_type, float_kind, uint_kind, assign_error_none> {};
template<class dst_type, class src_type>
struct single_assigner_simple_base<dst_type, src_type, float_kind, uint_kind, assign_error_fractional>
    : public single_assigner_simple_base<dst_type, src_type, float_kind, uint_kind, assign_error_none> {};

// Floating point -> signed int with overflow checking
template<class dst_type, class src_type>
struct single_assigner_simple_base<dst_type, src_type, int_kind, float_kind, assign_error_overflow>
{
    static void assign(dst_type *dst, const src_type *src) {
        src_type s = *src;

        if (s < std::numeric_limits<dst_type>::min() || s > std::numeric_limits<dst_type>::max()) {
            throw std::runtime_error("overflow while assigning floating point to signed integer");
        }
        *dst = static_cast<dst_type>(s);
    }
};

// Floating point -> signed int with fractional checking
template<class dst_type, class src_type>
struct single_assigner_simple_base<dst_type, src_type, int_kind, float_kind, assign_error_fractional>
{
    static void assign(dst_type *dst, const src_type *src) {
        src_type s = *src;

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
struct single_assigner_simple_base<dst_type, src_type, int_kind, float_kind, assign_error_inexact>
    : public single_assigner_simple_base<dst_type, src_type, int_kind, float_kind, assign_error_fractional> {};

// Floating point -> unsigned int with overflow checking
template<class dst_type, class src_type>
struct single_assigner_simple_base<dst_type, src_type, uint_kind, float_kind, assign_error_overflow>
{
    static void assign(dst_type *dst, const src_type *src) {
        src_type s = *src;

        if (s < std::numeric_limits<dst_type>::min() || s > std::numeric_limits<dst_type>::max()) {
            throw std::runtime_error("overflow while assigning floating point to unsigned integer");
        }
        *dst = static_cast<dst_type>(s);
    }
};

// Floating point -> unsigned int with fractional checking
template<class dst_type, class src_type>
struct single_assigner_simple_base<dst_type, src_type, uint_kind, float_kind, assign_error_fractional>
{
    static void assign(dst_type *dst, const src_type *src) {
        src_type s = *src;

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
struct single_assigner_simple_base<dst_type, src_type, uint_kind, float_kind, assign_error_inexact>
    : public single_assigner_simple_base<dst_type, src_type, uint_kind, float_kind, assign_error_fractional> {};

// float -> float with no checking
template<>
struct single_assigner_simple_base<float, float, float_kind, float_kind, assign_error_overflow>
    : public single_assigner_simple_base<float, float, float_kind, float_kind, assign_error_none> {};
template<>
struct single_assigner_simple_base<float, float, float_kind, float_kind, assign_error_fractional>
    : public single_assigner_simple_base<float, float, float_kind, float_kind, assign_error_none> {};
template<>
struct single_assigner_simple_base<float, float, float_kind, float_kind, assign_error_inexact>
    : public single_assigner_simple_base<float, float, float_kind, float_kind, assign_error_none> {};

// float -> double with no checking
template<>
struct single_assigner_simple_base<double, float, float_kind, float_kind, assign_error_overflow>
    : public single_assigner_simple_base<double, float, float_kind, float_kind, assign_error_none> {};
template<>
struct single_assigner_simple_base<double, float, float_kind, float_kind, assign_error_fractional>
    : public single_assigner_simple_base<double, float, float_kind, float_kind, assign_error_none> {};
template<>
struct single_assigner_simple_base<double, float, float_kind, float_kind, assign_error_inexact>
    : public single_assigner_simple_base<double, float, float_kind, float_kind, assign_error_none> {};
//
// double -> double with no checking
template<>
struct single_assigner_simple_base<double, double, float_kind, float_kind, assign_error_overflow>
    : public single_assigner_simple_base<double, double, float_kind, float_kind, assign_error_none> {};
template<>
struct single_assigner_simple_base<double, double, float_kind, float_kind, assign_error_fractional>
    : public single_assigner_simple_base<double, double, float_kind, float_kind, assign_error_none> {};
template<>
struct single_assigner_simple_base<double, double, float_kind, float_kind, assign_error_inexact>
    : public single_assigner_simple_base<double, double, float_kind, float_kind, assign_error_none> {};

// double -> float with overflow checking
template<>
struct single_assigner_simple_base<float, double, float_kind, float_kind, assign_error_overflow>
{
    static void assign(float *dst, const double *src) {
        clear_fp_status();
        *dst = static_cast<float>(*src);
        if (is_overflow_fp_status()) {
            throw std::runtime_error("overflow while assigning double to float");
        }
    }
};
//
// double -> float with fractional checking
template<>
struct single_assigner_simple_base<float, double, float_kind, float_kind, assign_error_fractional>
    : public single_assigner_simple_base<float, double, float_kind, float_kind, assign_error_overflow> {};


// double -> float with inexact checking
template<>
struct single_assigner_simple_base<float, double, float_kind, float_kind, assign_error_inexact>
{
    static void assign(float *dst, const double *src) {
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



// This is the interface exposed for use outside of this file
template <class dst_type, class src_type, assign_error_mode errmode>
struct single_assigner_simple
    : public single_assigner_simple_base<dst_type, src_type,
                        kind_of<dst_type>::value, kind_of<src_type>::value, errmode>
{};
template <class same_type, assign_error_mode errmode>
struct single_assigner_simple<same_type, same_type, errmode>
{
    static void assign(same_type *dst, const same_type *src) {
        *dst = *src;
    }
};

} // anonymous namespace

