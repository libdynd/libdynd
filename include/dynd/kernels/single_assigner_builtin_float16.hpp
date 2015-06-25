//
// Copyright (C) 2011-15 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

// This file is an internal implementation detail of built-in value assignment
// for aligned values in native byte order.


// Float16 -> bool
template<>
struct single_assigner_builtin_base<bool1, float16, bool_kind, real_kind, assign_error_nocheck>
{
    DYND_CUDA_HOST_DEVICE inline static void assign(bool1 *dst, const float16 *src) {
        // DYND_TRACE_ASSIGNMENT((bool)(!s.iszero()), bool1, s, float16);

        *dst = !src->iszero();
    }
};
template<>
struct single_assigner_builtin_base<bool1, float16, bool_kind, real_kind, assign_error_overflow>
{
    inline static void assign(bool1 *dst, const float16 *src) {
        float tmp = float(*src);
        single_assigner_builtin_base<bool1, float, bool_kind, real_kind,
                                     assign_error_overflow>::assign(dst, &tmp);
    }
};
template<>
struct single_assigner_builtin_base<bool1, float16, bool_kind, real_kind, assign_error_fractional>
{
    inline static void assign(bool1 *dst, const float16 *src) {
        float tmp = float(*src);
        single_assigner_builtin_base<bool1, float, bool_kind, real_kind,
                                     assign_error_fractional>::assign(dst,
                                                                      &tmp);
    }
};
template<>
struct single_assigner_builtin_base<bool1, float16, bool_kind, real_kind, assign_error_inexact>
{
    inline static void assign(bool1 *dst, const float16 *src) {
        float tmp = float(*src);
        single_assigner_builtin_base<bool1, float, bool_kind, real_kind,
                                     assign_error_inexact>::assign(dst, &tmp);
    }
};

// Bool -> float16
template<>
struct single_assigner_builtin_base<float16, bool1, real_kind, bool_kind, assign_error_nocheck>
{
    DYND_CUDA_HOST_DEVICE inline static void assign(float16 *dst, const bool1 *src) {
        // DYND_TRACE_ASSIGNMENT((bool)(!s.iszero()), bool1, s, float16);

        *dst = float16_from_bits(*src ? DYND_FLOAT16_ONE : 0);
    }
};
template<>
struct single_assigner_builtin_base<float16, bool1, real_kind, bool_kind, assign_error_overflow>
    : public single_assigner_builtin_base<float16, bool1, real_kind, bool_kind, assign_error_nocheck> {};
template<>
struct single_assigner_builtin_base<float16, bool1, real_kind, bool_kind, assign_error_fractional>
    : public single_assigner_builtin_base<float16, bool1, real_kind, bool_kind, assign_error_nocheck> {};
template<>
struct single_assigner_builtin_base<float16, bool1, real_kind, bool_kind, assign_error_inexact>
    : public single_assigner_builtin_base<float16, bool1, real_kind, bool_kind, assign_error_nocheck> {};


// Anything -> float16
#define DYND_MAKE_WITH_KIND(src_kind) \
template<class src_type> \
struct single_assigner_builtin_base<float16, src_type, real_kind, src_kind, assign_error_nocheck> \
{ \
    DYND_CUDA_HOST_DEVICE inline static void assign(float16 *DYND_UNUSED(dst), const src_type *DYND_UNUSED(src)) { \
        throw std::runtime_error("error"); \
    } \
}; \
template<class src_type> \
struct single_assigner_builtin_base<float16, src_type, real_kind, src_kind, assign_error_overflow> \
{ \
    inline static void assign(float16 *DYND_UNUSED(dst), const src_type *DYND_UNUSED(src)) { \
        throw std::runtime_error("error"); \
    } \
}; \
template<class src_type> \
struct single_assigner_builtin_base<float16, src_type, real_kind, src_kind, assign_error_fractional> \
{ \
    inline static void assign(float16 *DYND_UNUSED(dst), const src_type *DYND_UNUSED(src)) { \
        throw std::runtime_error("error"); \
    } \
}; \
template<class src_type> \
struct single_assigner_builtin_base<float16, src_type, real_kind, src_kind, assign_error_inexact> \
{ \
    inline static void assign(float16 *DYND_UNUSED(dst), const src_type *DYND_UNUSED(src)) { \
        throw std::runtime_error("error"); \
    } \
};

DYND_MAKE_WITH_KIND(bool_kind);
DYND_MAKE_WITH_KIND(real_kind);
DYND_MAKE_WITH_KIND(complex_kind);

#undef DYND_MAKE_WITH_KIND

//        float tmp = static_cast<float>(*src); 
  //      single_assigner_builtin_base<dst_type, float, dst_kind, real_kind, assign_error_nocheck>::assign(dst, &tmp); 

//        float tmp = static_cast<float>(*src); 
  //      single_assigner_builtin_base<dst_type, float, dst_kind, real_kind, assign_error_overflow>::assign(dst, &tmp); 

//        float tmp = static_cast<float>(*src); 
  //      single_assigner_builtin_base<dst_type, float, dst_kind, real_kind, assign_error_fractional>::assign(dst, &tmp); 

//        float tmp = static_cast<float>(*src); 
  //      single_assigner_builtin_base<dst_type, float, dst_kind, real_kind, assign_error_inexact>::assign(dst, &tmp); 

// Float16 -> anything
#define DYND_MAKE_WITH_KIND(dst_kind) \
template<class dst_type> \
struct single_assigner_builtin_base<dst_type, float16, dst_kind, real_kind, assign_error_nocheck> \
{ \
    DYND_CUDA_HOST_DEVICE inline static void assign(dst_type *DYND_UNUSED(dst), const float16 *DYND_UNUSED(src)) { \
        throw std::runtime_error("error"); \
    } \
}; \
template<class dst_type> \
struct single_assigner_builtin_base<dst_type, float16, dst_kind, real_kind, assign_error_overflow> \
{ \
    inline static void assign(dst_type *DYND_UNUSED(dst), const float16 *DYND_UNUSED(src)) { \
        throw std::runtime_error("error"); \
    } \
}; \
template<class dst_type> \
struct single_assigner_builtin_base<dst_type, float16, dst_kind, real_kind, assign_error_fractional> \
{ \
    inline static void assign(dst_type *DYND_UNUSED(dst), const float16 *DYND_UNUSED(src)) { \
        throw std::runtime_error("error"); \
    } \
}; \
template<class dst_type> \
struct single_assigner_builtin_base<dst_type, float16, dst_kind, real_kind, assign_error_inexact> \
{ \
    inline static void assign(dst_type *DYND_UNUSED(dst), const float16 *DYND_UNUSED(src)) { \
        throw std::runtime_error("error"); \
    } \
};

DYND_MAKE_WITH_KIND(bool_kind);
DYND_MAKE_WITH_KIND(real_kind);
DYND_MAKE_WITH_KIND(complex_kind);

#undef DYND_MAKE_WITH_KIND
