//
// Copyright (C) 2011-14 Mark Wiebe, DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

// This file is an internal implementation detail of built-in value assignment
// for aligned values in native byte order.

#if !defined(DYND_HAS_FLOAT128)

// float128 -> boolean with no checking
template<>
struct single_assigner_builtin_base<dynd_bool, dynd_float128, bool_kind, real_kind, assign_error_none>
{
    DYND_CUDA_HOST_DEVICE static void assign(dynd_bool *dst, const dynd_float128 *src, ckernel_prefix *DYND_UNUSED(extra)) {
        DYND_TRACE_ASSIGNMENT((bool)(s != src_type(0)), dynd_bool, s, src_type);

        *dst = ((src->m_lo != 0) || ((src->m_hi&0x7fffffffffffffffULL) != 0));
    }
};
template<>
struct single_assigner_builtin_base<dynd_bool, dynd_float128, bool_kind, real_kind, assign_error_overflow>
{
    static void assign(dynd_bool *dst, const dynd_float128 *src, ckernel_prefix *DYND_UNUSED(extra)) {
        // DYND_TRACE_ASSIGNMENT((bool)(s != src_type(0)), dynd_bool, s, src_type);

        if ((src->m_hi&0x7fffffffffffffffULL) == 0 && src->m_lo == 0) {
            *dst = 0;
        } else if (src->m_hi == 0x3fff000000000000ULL && src->m_lo == 0) { // 1.0 in binary128
            *dst = 1;
        } else {
            std::stringstream ss;
            ss << "overflow while assigning " << ndt::make_type<dynd_float128>();
            // TODO: ss << " value " << s;
            ss << " to " << ndt::make_type<dynd_bool>();
            throw std::runtime_error(ss.str());
        }
    }
};
template<>
struct single_assigner_builtin_base<dynd_bool, dynd_float128, bool_kind, real_kind, assign_error_fractional>
    : public single_assigner_builtin_base<dynd_bool, dynd_float128, bool_kind, real_kind, assign_error_overflow> {};
template<>
struct single_assigner_builtin_base<dynd_bool, dynd_float128, bool_kind, real_kind, assign_error_inexact>
    : public single_assigner_builtin_base<dynd_bool, dynd_float128, bool_kind, real_kind, assign_error_overflow> {};

// Bool -> float128
template<>
struct single_assigner_builtin_base<dynd_float128, dynd_bool, real_kind, bool_kind, assign_error_none>
{
    DYND_CUDA_HOST_DEVICE static void assign(dynd_float128 *dst, const dynd_bool *src, ckernel_prefix *DYND_UNUSED(extra)) {
        DYND_TRACE_ASSIGNMENT((bool)(s != src_type(0)), dynd_bool, s, src_type);

        if (*src) {
            *dst = dynd_float128(0x3fff000000000000ULL, 0ULL);
        } else {
            *dst = dynd_float128(0ULL, 0ULL);
        }
    }
};
template<>
struct single_assigner_builtin_base<dynd_float128, dynd_bool, real_kind, bool_kind, assign_error_overflow>
    : public single_assigner_builtin_base<dynd_float128, dynd_bool, real_kind, bool_kind, assign_error_none> {};
template<>
struct single_assigner_builtin_base<dynd_float128, dynd_bool, real_kind, bool_kind, assign_error_fractional>
    : public single_assigner_builtin_base<dynd_float128, dynd_bool, real_kind, bool_kind, assign_error_none> {};
template<>
struct single_assigner_builtin_base<dynd_float128, dynd_bool, real_kind, bool_kind, assign_error_inexact>
    : public single_assigner_builtin_base<dynd_float128,dynd_bool,  real_kind, bool_kind, assign_error_none> {};

// Anything -> float128
#define DYND_MAKE_WITH_KIND(src_kind) \
template<class src_type> \
struct single_assigner_builtin_base<dynd_float128, src_type, real_kind, src_kind, assign_error_none> \
    : public single_assigner_builtin_base_error<dynd_float128, src_type, assign_error_none> {}; \
template<class src_type> \
struct single_assigner_builtin_base<dynd_float128, src_type, real_kind, src_kind, assign_error_overflow> \
    : public single_assigner_builtin_base_error<dynd_float128, src_type, assign_error_overflow> {}; \
template<class src_type> \
struct single_assigner_builtin_base<dynd_float128, src_type, real_kind, src_kind, assign_error_fractional> \
    : public single_assigner_builtin_base_error<dynd_float128, src_type, assign_error_fractional> {}; \
template<class src_type> \
struct single_assigner_builtin_base<dynd_float128, src_type, real_kind, src_kind, assign_error_inexact> \
    : public single_assigner_builtin_base_error<dynd_float128, src_type, assign_error_inexact> {};

DYND_MAKE_WITH_KIND(int_kind);
DYND_MAKE_WITH_KIND(uint_kind);
DYND_MAKE_WITH_KIND(real_kind);
DYND_MAKE_WITH_KIND(complex_kind);

#undef DYND_MAKE_WITH_KIND

// Float128 -> anything
#define DYND_MAKE_WITH_KIND(dst_kind) \
template<class dst_type> \
struct single_assigner_builtin_base<dst_type, dynd_float128, dst_kind, real_kind, assign_error_none> \
    : public single_assigner_builtin_base_error<dst_type, dynd_float128, assign_error_none> {}; \
template<class dst_type> \
struct single_assigner_builtin_base<dst_type, dynd_float128, dst_kind, real_kind, assign_error_overflow> \
    : public single_assigner_builtin_base_error<dst_type, dynd_float128, assign_error_overflow> {}; \
template<class dst_type> \
struct single_assigner_builtin_base<dst_type, dynd_float128, dst_kind, real_kind, assign_error_fractional> \
    : public single_assigner_builtin_base_error<dst_type, dynd_float128, assign_error_fractional> {}; \
template<class dst_type> \
struct single_assigner_builtin_base<dst_type, dynd_float128, dst_kind, real_kind, assign_error_inexact> \
    : public single_assigner_builtin_base_error<dst_type, dynd_float128, assign_error_inexact> {};

DYND_MAKE_WITH_KIND(bool_kind);
DYND_MAKE_WITH_KIND(int_kind);
DYND_MAKE_WITH_KIND(uint_kind);
DYND_MAKE_WITH_KIND(real_kind);
DYND_MAKE_WITH_KIND(complex_kind);

#undef DYND_MAKE_WITH_KIND

// float16 -> float128
template<>
struct single_assigner_builtin_base<dynd_float128, dynd_float16, real_kind, real_kind, assign_error_none>
    : public single_assigner_builtin_base_error<dynd_float128, dynd_float16, assign_error_none> {};
template<>
struct single_assigner_builtin_base<dynd_float128, dynd_float16, real_kind, real_kind, assign_error_overflow>
    : public single_assigner_builtin_base_error<dynd_float128, dynd_float16, assign_error_overflow> {};
template<>
struct single_assigner_builtin_base<dynd_float128, dynd_float16, real_kind, real_kind, assign_error_fractional>
    : public single_assigner_builtin_base_error<dynd_float128, dynd_float16, assign_error_fractional> {};
template<>
struct single_assigner_builtin_base<dynd_float128, dynd_float16, real_kind, real_kind, assign_error_inexact>
    : public single_assigner_builtin_base_error<dynd_float128, dynd_float16, assign_error_inexact> {};


// float128 -> float16
template<>
struct single_assigner_builtin_base<dynd_float16, dynd_float128, real_kind, real_kind, assign_error_none>
    : public single_assigner_builtin_base_error<dynd_float16, dynd_float128, assign_error_none> {};
template<>
struct single_assigner_builtin_base<dynd_float16, dynd_float128, real_kind, real_kind, assign_error_overflow>
    : public single_assigner_builtin_base_error<dynd_float16, dynd_float128, assign_error_overflow> {};
template<>
struct single_assigner_builtin_base<dynd_float16, dynd_float128, real_kind, real_kind, assign_error_fractional>
    : public single_assigner_builtin_base_error<dynd_float16, dynd_float128, assign_error_fractional> {};
template<>
struct single_assigner_builtin_base<dynd_float16, dynd_float128, real_kind, real_kind, assign_error_inexact>
    : public single_assigner_builtin_base_error<dynd_float16, dynd_float128, assign_error_inexact> {};

#endif // !defined(DYND_HAS_FLOAT128)
