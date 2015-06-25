//
// Copyright (C) 2011-15 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

// This file is an internal implementation detail of built-in value assignment
// for aligned values in native byte order.

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

DYND_MAKE_WITH_KIND(complex_kind);

#undef DYND_MAKE_WITH_KIND