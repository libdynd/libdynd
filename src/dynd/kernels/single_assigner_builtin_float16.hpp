//
// Copyright (C) 2011-14 Mark Wiebe, DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

// This file is an internal implementation detail of built-in value assignment
// for aligned values in native byte order.


// Float16 -> bool
template<>
struct single_assigner_builtin_base<dynd_bool, dynd_float16, bool_kind, real_kind, assign_error_nocheck>
{
    DYND_CUDA_HOST_DEVICE inline static void assign(dynd_bool *dst, const dynd_float16 *src) {
        // DYND_TRACE_ASSIGNMENT((bool)(!s.iszero()), dynd_bool, s, dynd_float16);

        *dst = !src->iszero();
    }
};
template<>
struct single_assigner_builtin_base<dynd_bool, dynd_float16, bool_kind, real_kind, assign_error_overflow>
{
    inline static void assign(dynd_bool *dst, const dynd_float16 *src) {
        float tmp = float(*src);
        single_assigner_builtin_base<dynd_bool, float, bool_kind, real_kind,
                                     assign_error_overflow>::assign(dst, &tmp);
    }
};
template<>
struct single_assigner_builtin_base<dynd_bool, dynd_float16, bool_kind, real_kind, assign_error_fractional>
{
    inline static void assign(dynd_bool *dst, const dynd_float16 *src) {
        float tmp = float(*src);
        single_assigner_builtin_base<dynd_bool, float, bool_kind, real_kind,
                                     assign_error_fractional>::assign(dst,
                                                                      &tmp);
    }
};
template<>
struct single_assigner_builtin_base<dynd_bool, dynd_float16, bool_kind, real_kind, assign_error_inexact>
{
    inline static void assign(dynd_bool *dst, const dynd_float16 *src) {
        float tmp = float(*src);
        single_assigner_builtin_base<dynd_bool, float, bool_kind, real_kind,
                                     assign_error_inexact>::assign(dst, &tmp);
    }
};

// Bool -> float16
template<>
struct single_assigner_builtin_base<dynd_float16, dynd_bool, real_kind, bool_kind, assign_error_nocheck>
{
    DYND_CUDA_HOST_DEVICE inline static void assign(dynd_float16 *dst, const dynd_bool *src) {
        // DYND_TRACE_ASSIGNMENT((bool)(!s.iszero()), dynd_bool, s, dynd_float16);

        *dst = float16_from_bits(*src ? DYND_FLOAT16_ONE : 0);
    }
};
template<>
struct single_assigner_builtin_base<dynd_float16, dynd_bool, real_kind, bool_kind, assign_error_overflow>
    : public single_assigner_builtin_base<dynd_float16, dynd_bool, real_kind, bool_kind, assign_error_nocheck> {};
template<>
struct single_assigner_builtin_base<dynd_float16, dynd_bool, real_kind, bool_kind, assign_error_fractional>
    : public single_assigner_builtin_base<dynd_float16, dynd_bool, real_kind, bool_kind, assign_error_nocheck> {};
template<>
struct single_assigner_builtin_base<dynd_float16, dynd_bool, real_kind, bool_kind, assign_error_inexact>
    : public single_assigner_builtin_base<dynd_float16, dynd_bool, real_kind, bool_kind, assign_error_nocheck> {};


// Anything -> float16
#define DYND_MAKE_WITH_KIND(src_kind) \
template<class src_type> \
struct single_assigner_builtin_base<dynd_float16, src_type, real_kind, src_kind, assign_error_nocheck> \
{ \
    DYND_CUDA_HOST_DEVICE inline static void assign(dynd_float16 *dst, const src_type *src) { \
        float tmp; \
        single_assigner_builtin_base<float, src_type, real_kind, src_kind, assign_error_nocheck>::assign(&tmp, src); \
        *dst = dynd_float16(tmp, assign_error_nocheck); \
    } \
}; \
template<class src_type> \
struct single_assigner_builtin_base<dynd_float16, src_type, real_kind, src_kind, assign_error_overflow> \
{ \
    inline static void assign(dynd_float16 *dst, const src_type *src) { \
        float tmp; \
        single_assigner_builtin_base<float, src_type, real_kind, src_kind, assign_error_overflow>::assign(&tmp, src); \
        *dst = dynd_float16(tmp, assign_error_overflow); \
    } \
}; \
template<class src_type> \
struct single_assigner_builtin_base<dynd_float16, src_type, real_kind, src_kind, assign_error_fractional> \
{ \
    inline static void assign(dynd_float16 *dst, const src_type *src) { \
        float tmp; \
        single_assigner_builtin_base<float, src_type, real_kind, src_kind, assign_error_fractional>::assign(&tmp, src); \
        *dst = dynd_float16(tmp, assign_error_fractional); \
    } \
}; \
template<class src_type> \
struct single_assigner_builtin_base<dynd_float16, src_type, real_kind, src_kind, assign_error_inexact> \
{ \
    inline static void assign(dynd_float16 *dst, const src_type *src) { \
        float tmp; \
        single_assigner_builtin_base<float, src_type, real_kind, src_kind, assign_error_inexact>::assign(&tmp, src); \
        *dst = dynd_float16(tmp, assign_error_inexact); \
    } \
};

DYND_MAKE_WITH_KIND(bool_kind);
DYND_MAKE_WITH_KIND(int_kind);
DYND_MAKE_WITH_KIND(uint_kind);
DYND_MAKE_WITH_KIND(real_kind);
DYND_MAKE_WITH_KIND(complex_kind);

#undef DYND_MAKE_WITH_KIND

// Float16 -> anything
#define DYND_MAKE_WITH_KIND(dst_kind) \
template<class dst_type> \
struct single_assigner_builtin_base<dst_type, dynd_float16, dst_kind, real_kind, assign_error_nocheck> \
{ \
    DYND_CUDA_HOST_DEVICE inline static void assign(dst_type *dst, const dynd_float16 *src) { \
        float tmp = float(*src); \
        single_assigner_builtin_base<dst_type, float, dst_kind, real_kind, assign_error_nocheck>::assign(dst, &tmp); \
    } \
}; \
template<class dst_type> \
struct single_assigner_builtin_base<dst_type, dynd_float16, dst_kind, real_kind, assign_error_overflow> \
{ \
    inline static void assign(dst_type *dst, const dynd_float16 *src) { \
        float tmp = float(*src); \
        single_assigner_builtin_base<dst_type, float, dst_kind, real_kind, assign_error_overflow>::assign(dst, &tmp); \
    } \
}; \
template<class dst_type> \
struct single_assigner_builtin_base<dst_type, dynd_float16, dst_kind, real_kind, assign_error_fractional> \
{ \
    inline static void assign(dst_type *dst, const dynd_float16 *src) { \
        float tmp = float(*src); \
        single_assigner_builtin_base<dst_type, float, dst_kind, real_kind, assign_error_fractional>::assign(dst, &tmp); \
    } \
}; \
template<class dst_type> \
struct single_assigner_builtin_base<dst_type, dynd_float16, dst_kind, real_kind, assign_error_inexact> \
{ \
    inline static void assign(dst_type *dst, const dynd_float16 *src) { \
        float tmp = float(*src); \
        single_assigner_builtin_base<dst_type, float, dst_kind, real_kind, assign_error_inexact>::assign(dst, &tmp); \
    } \
};

DYND_MAKE_WITH_KIND(bool_kind);
DYND_MAKE_WITH_KIND(int_kind);
DYND_MAKE_WITH_KIND(uint_kind);
DYND_MAKE_WITH_KIND(real_kind);
DYND_MAKE_WITH_KIND(complex_kind);

#undef DYND_MAKE_WITH_KIND
