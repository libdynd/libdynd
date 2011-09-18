//
// Copyright (C) 2011 Mark Wiebe (mwwiebe@gmail.com)
// All rights reserved.
//
// This is unreleased proprietary software.
//
#include <dnd/dtype_assign.hpp>
#include <dnd/dtype_casting.hpp>

#include <iostream>//DEBUG
#include <stdexcept>
#include <cstring>
#include <limits>

using namespace std;
using namespace dnd;

template<int size> struct byteswapper;
template<> struct byteswapper<1> {
    static void byteswap(char *dst, const char *src) {
        *dst = *src;
    }
};
template<> struct byteswapper<2> {
    static void byteswap(char *dst, const char *src) {
        dst[0] = src[1];
        dst[1] = src[0];
    }
};
template<> struct byteswapper<4> {
    static void byteswap(char *dst, const char *src) {
        dst[0] = src[3];
        dst[1] = src[2];
        dst[2] = src[1];
        dst[3] = src[0];
    }
};
template<> struct byteswapper<8> {
    static void byteswap(char *dst, const char *src) {
        dst[0] = src[7];
        dst[1] = src[6];
        dst[2] = src[5];
        dst[3] = src[4];
        dst[4] = src[3];
        dst[5] = src[2];
        dst[6] = src[1];
        dst[7] = src[0];
    }
};

template<class T> struct is_signed_int {enum {value = false};};
template<> struct is_signed_int<int8_t> {enum {value = true};};
template<> struct is_signed_int<int16_t> {enum {value = true};};
template<> struct is_signed_int<int32_t> {enum {value = true};};
template<> struct is_signed_int<int64_t> {enum {value = true};};

template<class T> struct is_floating {enum {value = false};};
template<> struct is_floating<float> {enum {value = true};};
template<> struct is_floating<double> {enum {value = true};};

template<class T> struct is_boolean {enum {value = false};};
template<> struct is_boolean<dnd_bool> {enum {value = true};};


// The single_assigner class assigns a single item of the known dtypes, doing casting
// and byte swapping as necessary.

template<class dst_type, class src_type>
struct single_assigner_base {
    static void assign_noexcept(void *dst, const void *src) {
        const src_type *s = reinterpret_cast<const src_type *>(src);
        dst_type *d = reinterpret_cast<dst_type *>(dst);

        *d = static_cast<dst_type>(*s);
    }
};

template<class dst_type, class src_type, bool dst_byteswap, bool src_byteswap>
struct single_assigner : public single_assigner_base<dst_type, src_type> {
};

// Specializations of single_assigner with different checked types
template <class T>
struct single_assigner<T, T, false, false> : public single_assigner_base<T, T> {
    static void assign(void *dst, const void *src) {
        const T *s = reinterpret_cast<const T *>(src);
        T *d = reinterpret_cast<T *>(dst);

        *d = *s;
    }
};

template <>
struct single_assigner<dnd_bool, dnd_bool, false, false> : public single_assigner_base<dnd_bool, dnd_bool> {
    static void assign(void *dst, const void *src) {
        const dnd_bool *s = reinterpret_cast<const dnd_bool *>(src);
        dnd_bool *d = reinterpret_cast<dnd_bool *>(dst);

        *d = *s;
    }
};

template <class T>
struct single_assigner<T, dnd_bool, false, false> : public single_assigner_base<T, dnd_bool> {
    static void assign(void *dst, const void *src) {
        const dnd_bool *s = reinterpret_cast<const dnd_bool *>(src);
        T *d = reinterpret_cast<T *>(dst);

        *d = *s;
    }
};

// Any unspecialized case is unchecked
template <class dst_type, class src_type, bool dst_smaller, bool dst_equal, bool dst_signed, bool src_signed, bool dst_bool, bool dst_floating, bool src_floating>
struct checked_single_assigner {
    static void assign(void *dst, const void *src) {
        const src_type *s = reinterpret_cast<const src_type *>(src);
        dst_type d, *d_ptr = reinterpret_cast<dst_type *>(dst);

        d = static_cast<dst_type>(*s);
        *d_ptr = d;
    }
};

template <class dst_type, class src_type, bool is_signed>
struct checked_single_assigner<dst_type, src_type, true, false, is_signed, is_signed, false, false, false> {
    static void assign(void *dst, const void *src) {
        const src_type *s = reinterpret_cast<const src_type *>(src);
        dst_type d, *d_ptr = reinterpret_cast<dst_type *>(dst);

        d = static_cast<dst_type>(*s);
        if (d != *s) {
            throw std::runtime_error("overflow while assigning integer values");
        }
        *d_ptr = d;
    }
};

template <class dst_type, class src_type>
struct checked_single_assigner<dst_type, src_type, true, false, true, false, false, false, false> {
    static void assign(void *dst, const void *src) {
        const src_type *s = reinterpret_cast<const src_type *>(src);
        dst_type *d = reinterpret_cast<dst_type *>(dst);

        if (*s > numeric_limits<dst_type>::max()) {
            throw std::runtime_error("overflow while assigning integer values");
        }
        *d = static_cast<dst_type>(*s);
    }
};

template <class dst_type, class src_type>
struct checked_single_assigner<dst_type, src_type, false, true, true, false, false, false, false> {
    static void assign(void *dst, const void *src) {
        const src_type *s = reinterpret_cast<const src_type *>(src);
        dst_type *d = reinterpret_cast<dst_type *>(dst);

        if (*s > numeric_limits<dst_type>::max()) {
            throw std::runtime_error("overflow while assigning integer values");
        }
        *d = static_cast<dst_type>(*s);
    }
};

template <class dst_type, class src_type>
struct checked_single_assigner<dst_type, src_type, true, false, false, true, false, false, false> {
    static void assign(void *dst, const void *src) {
        const src_type *s = reinterpret_cast<const src_type *>(src);
        dst_type *d = reinterpret_cast<dst_type *>(dst);

        if (*s < 0 || *s > numeric_limits<dst_type>::max()) {
            throw std::runtime_error("overflow while assigning integer values");
        }
        *d = static_cast<dst_type>(*s);
    }
};

template <class dst_type, class src_type>
struct checked_single_assigner<dst_type, src_type, false, true, false, true, false, false, false> {
    static void assign(void *dst, const void *src) {
        const src_type *s = reinterpret_cast<const src_type *>(src);
        dst_type *d = reinterpret_cast<dst_type *>(dst);

        if (*s < 0) {
            throw std::runtime_error("overflow while assigning integer values");
        }
        *d = static_cast<dst_type>(*s);
    }
};

template <class src_type, bool dst_smaller, bool dst_equal, bool src_signed, bool src_floating>
struct checked_single_assigner<dnd_bool, src_type, dst_smaller, dst_equal, false, src_signed, true, false, src_floating> {
    static void assign(void *dst, const void *src) {
        const src_type *s = reinterpret_cast<const src_type *>(src);
        dnd_bool *d = reinterpret_cast<dnd_bool *>(dst);

        if (*s == 0) {
            *d = false;
        } else if (*s == 1) {
            *d = true;
        } else {
            throw std::runtime_error("overflow while assigning to boolean value");
        }
    }
};

template <class float_type, class T>
struct checked_float_assigner {
    static void assign_from_float(void *dst, const void *src) {
        const float_type *s = reinterpret_cast<const float_type *>(src);
        T d, *d_ptr = reinterpret_cast<T *>(dst);

        d = static_cast<T>(*s);
        if (static_cast<float_type>(d) != *s) {
            throw std::runtime_error("overflow while assigning from float");
        }
        *d_ptr = d;
    }
    static void assign_to_float(void *dst, const void *src) {
        const T *s = reinterpret_cast<const T *>(src);
        float_type d, *d_ptr = reinterpret_cast<float_type *>(dst);

        d = static_cast<float_type>(*s);
        if (static_cast<T>(d) != *s) {
            throw std::runtime_error("overflow while assigning to float");
        }
        *d_ptr = d;
    }
};

template <class src_type, bool dst_smaller, bool dst_equal, bool src_signed>
struct checked_single_assigner<float, src_type, dst_smaller, dst_equal, false, src_signed, false, true, false> {
    static void assign(void *dst, const void *src) {
        checked_float_assigner<float, src_type>::assign_to_float(dst, src);
    }
};

template <class src_type, bool dst_smaller, bool dst_equal, bool src_signed>
struct checked_single_assigner<double, src_type, dst_smaller, dst_equal, false, src_signed, false, true, false> {
    static void assign(void *dst, const void *src) {
        checked_float_assigner<double, src_type>::assign_to_float(dst, src);
    }
};

template <class dst_type, bool dst_smaller, bool dst_equal, bool dst_signed>
struct checked_single_assigner<dst_type, float, dst_smaller, dst_equal, dst_signed, false, false, false, true> {
    static void assign(void *dst, const void *src) {
        checked_float_assigner<float, dst_type>::assign_from_float(dst, src);
    }
};

template <class dst_type, bool dst_smaller, bool dst_equal, bool dst_signed>
struct checked_single_assigner<dst_type, double, dst_smaller, dst_equal, dst_signed, false, false, false, true> {
    static void assign(void *dst, const void *src) {
        checked_float_assigner<double, dst_type>::assign_from_float(dst, src);
    }
};

template <class dst_type, class src_type, bool dst_smaller, bool dst_equal>
struct checked_single_assigner<dst_type, src_type, dst_smaller, dst_equal, false, false, false, true, true> {
    static void assign(void *dst, const void *src) {
        checked_float_assigner<dst_type, src_type>::assign_to_float(dst, src);
    }
};

template <class dst_type, class src_type>
struct single_assigner<dst_type, src_type, false, false> :
                                    public single_assigner_base<dst_type, src_type> {
    static void assign(void *dst, const void *src) {
        checked_single_assigner<dst_type, src_type,
                                sizeof(dst_type) < sizeof(src_type),
                                sizeof(dst_type) == sizeof(src_type),
                                is_signed_int<dst_type>::value,
                                is_signed_int<src_type>::value,
                                is_boolean<dst_type>::value,
                                is_floating<dst_type>::value,
                                is_floating<src_type>::value
                                >::assign(dst, src);

    }
};

// Specializations of single_assigner with different variants of byte-swapping
template<class dst_type, class src_type>
struct single_assigner<dst_type, src_type, false, true> {
    static void assign_noexcept(void *dst, const void *src) {
        src_type s;
        dst_type *d = reinterpret_cast<dst_type *>(dst);

        byteswapper<sizeof(src_type)>::byteswap(reinterpret_cast<char *>(&s),
                                                reinterpret_cast<const char *>(src));
        *d = static_cast<dst_type>(s);
    }

    static void assign(void *dst, const void *src) {
        src_type s;
        dst_type *d = reinterpret_cast<dst_type *>(dst);

        byteswapper<sizeof(src_type)>::byteswap(reinterpret_cast<char *>(&s),
                                                reinterpret_cast<const char *>(src));
        single_assigner<dst_type, src_type, false, false>::assign(d, &s);
    }
};

template<class dst_type, class src_type>
struct single_assigner<dst_type, src_type, true, false> {
    static void assign_noexcept(void *dst, const void *src) {
        const src_type *s = reinterpret_cast<const src_type *>(src);
        dst_type d;

        d = static_cast<dst_type>(*s);
        byteswapper<sizeof(dst_type)>::byteswap(reinterpret_cast<char *>(dst),
                                                reinterpret_cast<const char *>(&d));
    }

    static void assign(void *dst, const void *src) {
        const src_type *s = reinterpret_cast<const src_type *>(src);
        dst_type d;

        single_assigner<dst_type, src_type, false, false>::assign(&d, s);
        byteswapper<sizeof(dst_type)>::byteswap(reinterpret_cast<char *>(dst),
                                                reinterpret_cast<const char *>(&d));
    }
};

template<class dst_type, class src_type>
struct single_assigner<dst_type, src_type, true, true> {
    static void assign_noexcept(void *dst, const void *src) {
        src_type s;
        dst_type d;

        byteswapper<sizeof(src_type)>::byteswap(reinterpret_cast<char *>(&s),
                                                reinterpret_cast<const char *>(src));
        d = static_cast<dst_type>(s);
        byteswapper<sizeof(dst_type)>::byteswap(reinterpret_cast<char *>(dst),
                                                reinterpret_cast<const char *>(&d));
    }

    static void assign(void *dst, const void *src) {
        src_type s;
        dst_type d;

        byteswapper<sizeof(src_type)>::byteswap(reinterpret_cast<char *>(&s),
                                                reinterpret_cast<const char *>(src));
        single_assigner<dst_type, src_type, false, false>::assign(&d, &s);
        byteswapper<sizeof(dst_type)>::byteswap(reinterpret_cast<char *>(dst),
                                                reinterpret_cast<const char *>(&d));
    }
};

#define DTYPE_ASSIGN_SRC_TO_DST_ONE_CASE(dst_type, src_type, \
                                dst_byteswapped, src_byteswapped, ASSIGN_FN) \
    case type_id_of<dst_type>::value: \
        single_assigner<dst_type, src_type, dst_byteswapped, src_byteswapped>::ASSIGN_FN(dst, src); \
        return

#define DTYPE_ASSIGN_SRC_TO_ANY_CASE(src_type, dst_byteswapped, src_byteswapped, ASSIGN_FN) \
    case type_id_of<src_type>::value: \
        switch (dst_dt.type_id()) { \
            DTYPE_ASSIGN_SRC_TO_DST_ONE_CASE(dnd_bool, src_type, dst_byteswapped, src_byteswapped, ASSIGN_FN); \
            DTYPE_ASSIGN_SRC_TO_DST_ONE_CASE(int8_t, src_type, dst_byteswapped, src_byteswapped, ASSIGN_FN); \
            DTYPE_ASSIGN_SRC_TO_DST_ONE_CASE(int16_t, src_type, dst_byteswapped, src_byteswapped, ASSIGN_FN); \
            DTYPE_ASSIGN_SRC_TO_DST_ONE_CASE(int32_t, src_type, dst_byteswapped, src_byteswapped, ASSIGN_FN); \
            DTYPE_ASSIGN_SRC_TO_DST_ONE_CASE(int64_t, src_type, dst_byteswapped, src_byteswapped, ASSIGN_FN); \
            DTYPE_ASSIGN_SRC_TO_DST_ONE_CASE(uint8_t, src_type, dst_byteswapped, src_byteswapped, ASSIGN_FN); \
            DTYPE_ASSIGN_SRC_TO_DST_ONE_CASE(uint16_t, src_type, dst_byteswapped, src_byteswapped, ASSIGN_FN); \
            DTYPE_ASSIGN_SRC_TO_DST_ONE_CASE(uint32_t, src_type, dst_byteswapped, src_byteswapped, ASSIGN_FN); \
            DTYPE_ASSIGN_SRC_TO_DST_ONE_CASE(uint64_t, src_type, dst_byteswapped, src_byteswapped, ASSIGN_FN); \
            DTYPE_ASSIGN_SRC_TO_DST_ONE_CASE(float, src_type, dst_byteswapped, src_byteswapped, ASSIGN_FN); \
            DTYPE_ASSIGN_SRC_TO_DST_ONE_CASE(double, src_type, dst_byteswapped, src_byteswapped, ASSIGN_FN); \
        } \
        break

#define DTYPE_ASSIGN_ANY_TO_ANY_SWITCH(src_byteswapped, dst_byteswapped, ASSIGN_FN) \
    switch (src_dt.type_id()) { \
        DTYPE_ASSIGN_SRC_TO_ANY_CASE(dnd_bool, dst_byteswapped, src_byteswapped, ASSIGN_FN); \
        DTYPE_ASSIGN_SRC_TO_ANY_CASE(int8_t, dst_byteswapped, src_byteswapped, ASSIGN_FN); \
        DTYPE_ASSIGN_SRC_TO_ANY_CASE(int16_t, dst_byteswapped, src_byteswapped, ASSIGN_FN); \
        DTYPE_ASSIGN_SRC_TO_ANY_CASE(int32_t, dst_byteswapped, src_byteswapped, ASSIGN_FN); \
        DTYPE_ASSIGN_SRC_TO_ANY_CASE(int64_t, dst_byteswapped, src_byteswapped, ASSIGN_FN); \
        DTYPE_ASSIGN_SRC_TO_ANY_CASE(uint8_t, dst_byteswapped, src_byteswapped, ASSIGN_FN); \
        DTYPE_ASSIGN_SRC_TO_ANY_CASE(uint16_t, dst_byteswapped, src_byteswapped, ASSIGN_FN); \
        DTYPE_ASSIGN_SRC_TO_ANY_CASE(uint32_t, dst_byteswapped, src_byteswapped, ASSIGN_FN); \
        DTYPE_ASSIGN_SRC_TO_ANY_CASE(uint64_t, dst_byteswapped, src_byteswapped, ASSIGN_FN); \
        DTYPE_ASSIGN_SRC_TO_ANY_CASE(float, dst_byteswapped, src_byteswapped, ASSIGN_FN); \
        DTYPE_ASSIGN_SRC_TO_ANY_CASE(double, dst_byteswapped, src_byteswapped, ASSIGN_FN); \
    }

void dnd::dtype_assign_noexcept(void *dst, const void *src, dtype dst_dt, dtype src_dt)
{
    if (src_dt.is_byteswapped()) {
        if (dst_dt.is_byteswapped()) {
            DTYPE_ASSIGN_ANY_TO_ANY_SWITCH(true, true, assign_noexcept);
        } else {
            DTYPE_ASSIGN_ANY_TO_ANY_SWITCH(false, true, assign_noexcept);
        }
    } else {
        if (dst_dt.is_byteswapped()) {
            DTYPE_ASSIGN_ANY_TO_ANY_SWITCH(true, false, assign_noexcept);
        } else {
            DTYPE_ASSIGN_ANY_TO_ANY_SWITCH(false, false, assign_noexcept);
        }
    }

    throw std::runtime_error("this dtype assignment isn't yet supported");
}

void dnd::dtype_assign(void *dst, const void *src, dtype dst_dt, dtype src_dt)
{
    // Do an unchecked assignment if possible
    if (can_cast_losslessly(dst_dt, src_dt)) {
        dtype_assign_noexcept(dst, src, dst_dt, src_dt);
        return;
    }

    if (src_dt.is_byteswapped()) {
        if (dst_dt.is_byteswapped()) {
            DTYPE_ASSIGN_ANY_TO_ANY_SWITCH(true, true, assign);
        } else {
            DTYPE_ASSIGN_ANY_TO_ANY_SWITCH(false, true, assign);
        }
    } else {
        if (dst_dt.is_byteswapped()) {
            DTYPE_ASSIGN_ANY_TO_ANY_SWITCH(true, false, assign);
        } else {
            DTYPE_ASSIGN_ANY_TO_ANY_SWITCH(false, false, assign);
        }
    }

    throw std::runtime_error("this dtype assignment isn't yet supported");
}

#undef DTYPE_ASSIGN_SRC_TO_DST_ONE_CASE
#undef DTYPE_ASSIGN_SRC_TO_ANY_CASE
#undef DTYPE_ASSIGN_ANY_TO_ANY_SWITCH

// Most general assignment functions, with possibly aligned and/or byte-swapped data
template<class dst_type, class src_type, bool aligned,
                        bool dst_byteswapped, bool src_byteswapped>
struct multiple_assigner {
    static void assign_noexcept(void *dst, intptr_t dst_stride,
                                const void *src, intptr_t src_stride,
                                intptr_t count,
                                auxiliary_data *)
    {
        char *dst_cached = reinterpret_cast<char *>(dst);
        const char *src_cached = reinterpret_cast<const char *>(src);
        dst_type d;
        src_type s;

        for (intptr_t i = 0; i < count; ++i) {
            memcpy(&s, src_cached, sizeof(src_type));
            single_assigner<dst_type, src_type, dst_byteswapped, src_byteswapped>::
                                                            assign_noexcept(&d, &s);
            memcpy(dst_cached, &d, sizeof(dst_type));
            dst_cached += dst_stride;
            src_cached += src_stride;
        }
    }

    static void assign(void *dst, intptr_t dst_stride,
                                const void *src, intptr_t src_stride,
                                intptr_t count,
                                auxiliary_data *)
    {
        char *dst_cached = reinterpret_cast<char *>(dst);
        const char *src_cached = reinterpret_cast<const char *>(src);
        dst_type d;
        src_type s;

        for (intptr_t i = 0; i < count; ++i) {
            memcpy(&s, src_cached, sizeof(src_type));
            single_assigner<dst_type, src_type, dst_byteswapped, src_byteswapped>::
                                                            assign(&d, &s);
            memcpy(dst_cached, &d, sizeof(dst_type));
            dst_cached += dst_stride;
            src_cached += src_stride;
        }
    }
};

// Aligned versions, with more specializations
template<class dst_type, class src_type>
struct multiple_assigner<dst_type, src_type, true, false, false> {
    static void assign_noexcept(void *dst, intptr_t dst_stride,
                                const void *src, intptr_t src_stride,
                                intptr_t count,
                                auxiliary_data *)
    {
        const src_type *src_cached = reinterpret_cast<const src_type *>(src);
        dst_type *dst_cached = reinterpret_cast<dst_type *>(dst);
        src_stride /= sizeof(src_type);
        dst_stride /= sizeof(dst_type);

        for (intptr_t i = 0; i < count; ++i) {
            *dst_cached = static_cast<dst_type>(*src_cached);
            dst_cached += dst_stride;
            src_cached += src_stride;
        }
    }

    static void assign(void *dst, intptr_t dst_stride,
                                const void *src, intptr_t src_stride,
                                intptr_t count,
                                auxiliary_data *)
    {
        const src_type *src_cached = reinterpret_cast<const src_type *>(src);
        dst_type *dst_cached = reinterpret_cast<dst_type *>(dst);
        src_stride /= sizeof(src_type);
        dst_stride /= sizeof(dst_type);

        for (intptr_t i = 0; i < count; ++i) {
            *dst_cached = static_cast<dst_type>(*src_cached);
            single_assigner<dst_type, src_type, false, false>::assign(dst_cached, src_cached);
            dst_cached += dst_stride;
            src_cached += src_stride;
        }
    }

    static void assign_noexcept_anystride_zerostride(void *dst, intptr_t dst_stride,
                                const void *src, intptr_t,
                                intptr_t count,
                                auxiliary_data *)
    {
        dst_type src_cached = static_cast<dst_type>(*reinterpret_cast<const src_type *>(src));
        dst_type *dst_cached = reinterpret_cast<dst_type *>(dst);
        dst_stride /= sizeof(dst_type);

        for (intptr_t i = 0; i < count; ++i) {
            *dst_cached = src_cached;
            dst_cached += dst_stride;
        }
    }

    static void assign_noexcept_contigstride_zerostride(void *dst, intptr_t,
                                const void *src, intptr_t,
                                intptr_t count,
                                auxiliary_data *)
    {
        dst_type src_cached = static_cast<dst_type>(*reinterpret_cast<const src_type *>(src));
        dst_type *dst_cached = reinterpret_cast<dst_type *>(dst);

        for (intptr_t i = 0; i < count; ++i, ++dst_cached) {
            *dst_cached = src_cached;
        }
    }

    static void assign_noexcept_contigstride_contigstride(void *dst, intptr_t,
                                const void *src, intptr_t,
                                intptr_t count,
                                auxiliary_data *)
    {
        const src_type *src_cached = reinterpret_cast<const src_type *>(src);
        dst_type *dst_cached = reinterpret_cast<dst_type *>(dst);

        for (intptr_t i = 0; i < count; ++i, ++dst_cached, ++src_cached) {
            *dst_cached = static_cast<dst_type>(*src_cached);
        }
    }
};

#define DTYPE_ASSIGN_SRC_TO_DST_SINGLE_CASE(dst_type, src_type, \
                                dst_byteswapped, src_byteswapped, is_aligned, ASSIGN_FN) \
    case type_id_of<dst_type>::value: \
        return make_pair( \
            &multiple_assigner<dst_type, src_type, is_aligned, dst_byteswapped, src_byteswapped>::ASSIGN_FN, \
            (auxiliary_data *)NULL); \
        break

#define DTYPE_ASSIGN_SRC_TO_ANY_CASE(src_type, is_aligned, dst_byteswapped, src_byteswapped, ASSIGN_FN) \
    case type_id_of<src_type>::value: \
        switch (dst_dt.type_id()) { \
            DTYPE_ASSIGN_SRC_TO_DST_SINGLE_CASE(dnd_bool, src_type, \
                                    dst_byteswapped, src_byteswapped, is_aligned, ASSIGN_FN); \
            DTYPE_ASSIGN_SRC_TO_DST_SINGLE_CASE(int8_t,   src_type, \
                                    dst_byteswapped, src_byteswapped, is_aligned, ASSIGN_FN); \
            DTYPE_ASSIGN_SRC_TO_DST_SINGLE_CASE(int16_t,  src_type, \
                                    dst_byteswapped, src_byteswapped, is_aligned, ASSIGN_FN); \
            DTYPE_ASSIGN_SRC_TO_DST_SINGLE_CASE(int32_t,  src_type, \
                                    dst_byteswapped, src_byteswapped, is_aligned, ASSIGN_FN); \
            DTYPE_ASSIGN_SRC_TO_DST_SINGLE_CASE(int64_t,  src_type, \
                                    dst_byteswapped, src_byteswapped, is_aligned, ASSIGN_FN); \
            DTYPE_ASSIGN_SRC_TO_DST_SINGLE_CASE(uint8_t,  src_type, \
                                    dst_byteswapped, src_byteswapped, is_aligned, ASSIGN_FN); \
            DTYPE_ASSIGN_SRC_TO_DST_SINGLE_CASE(uint16_t, src_type, \
                                    dst_byteswapped, src_byteswapped, is_aligned, ASSIGN_FN); \
            DTYPE_ASSIGN_SRC_TO_DST_SINGLE_CASE(uint32_t, src_type, \
                                    dst_byteswapped, src_byteswapped, is_aligned, ASSIGN_FN); \
            DTYPE_ASSIGN_SRC_TO_DST_SINGLE_CASE(uint64_t, src_type, \
                                    dst_byteswapped, src_byteswapped, is_aligned, ASSIGN_FN); \
            DTYPE_ASSIGN_SRC_TO_DST_SINGLE_CASE(float,    src_type, \
                                    dst_byteswapped, src_byteswapped, is_aligned, ASSIGN_FN); \
            DTYPE_ASSIGN_SRC_TO_DST_SINGLE_CASE(double,   src_type, \
                                    dst_byteswapped, src_byteswapped, is_aligned, ASSIGN_FN); \
        } \
        break

#define DTYPE_ASSIGN_ANY_TO_ANY_SWITCH(is_aligned, src_byteswapped, dst_byteswapped, ASSIGN_FN) \
    switch (src_dt.type_id()) { \
        DTYPE_ASSIGN_SRC_TO_ANY_CASE(dnd_bool, is_aligned, dst_byteswapped, src_byteswapped, ASSIGN_FN); \
        DTYPE_ASSIGN_SRC_TO_ANY_CASE(int8_t, is_aligned, dst_byteswapped, src_byteswapped, ASSIGN_FN); \
        DTYPE_ASSIGN_SRC_TO_ANY_CASE(int16_t, is_aligned, dst_byteswapped, src_byteswapped, ASSIGN_FN); \
        DTYPE_ASSIGN_SRC_TO_ANY_CASE(int32_t, is_aligned, dst_byteswapped, src_byteswapped, ASSIGN_FN); \
        DTYPE_ASSIGN_SRC_TO_ANY_CASE(int64_t, is_aligned, dst_byteswapped, src_byteswapped, ASSIGN_FN); \
        DTYPE_ASSIGN_SRC_TO_ANY_CASE(uint8_t, is_aligned, dst_byteswapped, src_byteswapped, ASSIGN_FN); \
        DTYPE_ASSIGN_SRC_TO_ANY_CASE(uint16_t, is_aligned, dst_byteswapped, src_byteswapped, ASSIGN_FN); \
        DTYPE_ASSIGN_SRC_TO_ANY_CASE(uint32_t, is_aligned, dst_byteswapped, src_byteswapped, ASSIGN_FN); \
        DTYPE_ASSIGN_SRC_TO_ANY_CASE(uint64_t, is_aligned, dst_byteswapped, src_byteswapped, ASSIGN_FN); \
        DTYPE_ASSIGN_SRC_TO_ANY_CASE(float, is_aligned, dst_byteswapped, src_byteswapped, ASSIGN_FN); \
        DTYPE_ASSIGN_SRC_TO_ANY_CASE(double, is_aligned, dst_byteswapped, src_byteswapped, ASSIGN_FN); \
    }

std::pair<unary_operation_t, auxiliary_data *> dnd::get_dtype_strided_assign_noexcept_operation(
                    dtype dst_dt, intptr_t dst_fixedstride,
                    dtype src_dt, intptr_t src_fixedstride,
                    char align_test)
{
    bool is_aligned = dst_dt.is_data_aligned(align_test) && src_dt.is_data_aligned(align_test);
    bool src_byteswapped, dst_byteswapped;

    if (is_aligned && src_dt.extended() == NULL && dst_dt.extended() == NULL) {
        src_byteswapped = src_dt.is_byteswapped();
        dst_byteswapped = dst_dt.is_byteswapped();

        if (!src_byteswapped && !dst_byteswapped) {
            if (src_fixedstride == 0) {
                if (dst_fixedstride == dst_dt.itemsize()) {
                    DTYPE_ASSIGN_ANY_TO_ANY_SWITCH(true, false, false,
                                        assign_noexcept_contigstride_zerostride);
                } else {
                    DTYPE_ASSIGN_ANY_TO_ANY_SWITCH(true, false, false,
                                        assign_noexcept_anystride_zerostride);
                }
            } else if (dst_fixedstride == dst_dt.itemsize() && src_fixedstride == src_dt.itemsize()) {
                DTYPE_ASSIGN_ANY_TO_ANY_SWITCH(true, false, false, assign_noexcept_contigstride_contigstride);
            }
        }
    } else {
        src_byteswapped = src_dt.is_byteswapped();
        dst_byteswapped = dst_dt.is_byteswapped();
    }

    if (is_aligned) {
        if (dst_byteswapped) {
            if (src_byteswapped) {
                DTYPE_ASSIGN_ANY_TO_ANY_SWITCH(true, true, true, assign_noexcept);
            } else {
                DTYPE_ASSIGN_ANY_TO_ANY_SWITCH(true, true, false, assign_noexcept);
            }
        } else {
            if (src_byteswapped) {
                DTYPE_ASSIGN_ANY_TO_ANY_SWITCH(true, false, true, assign_noexcept);
            } else {
                DTYPE_ASSIGN_ANY_TO_ANY_SWITCH(true, false, false, assign_noexcept);
            }
        }
    } else {
        if (dst_byteswapped) {
            if (src_byteswapped) {
                DTYPE_ASSIGN_ANY_TO_ANY_SWITCH(false, true, true, assign_noexcept);
            } else {
                DTYPE_ASSIGN_ANY_TO_ANY_SWITCH(false, true, false, assign_noexcept);
            }
        } else {
            if (src_byteswapped) {
                DTYPE_ASSIGN_ANY_TO_ANY_SWITCH(false, false, true, assign_noexcept);
            } else {
                DTYPE_ASSIGN_ANY_TO_ANY_SWITCH(false, false, false, assign_noexcept);
            }
        }
    }

    throw std::runtime_error("this dtype assignment isn't yet supported");
}

std::pair<unary_operation_t, auxiliary_data *> dnd::get_dtype_strided_assign_operation(
                    dtype dst_dt, intptr_t dst_fixedstride,
                    dtype src_dt, intptr_t src_fixedstride,
                    char align_test)
{
    bool is_aligned = dst_dt.is_data_aligned(align_test) && src_dt.is_data_aligned(align_test);
    bool src_byteswapped, dst_byteswapped;

    src_byteswapped = src_dt.is_byteswapped();
    dst_byteswapped = dst_dt.is_byteswapped();

    if (is_aligned) {
        if (dst_byteswapped) {
            if (src_byteswapped) {
                DTYPE_ASSIGN_ANY_TO_ANY_SWITCH(true, true, true, assign);
            } else {
                DTYPE_ASSIGN_ANY_TO_ANY_SWITCH(true, true, false, assign);
            }
        } else {
            if (src_byteswapped) {
                DTYPE_ASSIGN_ANY_TO_ANY_SWITCH(true, false, true, assign);
            } else {
                DTYPE_ASSIGN_ANY_TO_ANY_SWITCH(true, false, false, assign);
            }
        }
    } else {
        if (dst_byteswapped) {
            if (src_byteswapped) {
                DTYPE_ASSIGN_ANY_TO_ANY_SWITCH(false, true, true, assign);
            } else {
                DTYPE_ASSIGN_ANY_TO_ANY_SWITCH(false, true, false, assign);
            }
        } else {
            if (src_byteswapped) {
                DTYPE_ASSIGN_ANY_TO_ANY_SWITCH(false, false, true, assign);
            } else {
                DTYPE_ASSIGN_ANY_TO_ANY_SWITCH(false, false, false, assign);
            }
        }
    }

    throw std::runtime_error("this dtype assignment isn't yet supported");
}

void dnd::dtype_strided_assign(void *dst, intptr_t dst_stride,
                            void *src, intptr_t src_stride,
                            intptr_t count,
                            dtype dst_dt, dtype src_dt)
{
    std::pair<unary_operation_t, auxiliary_data *> op;
    op = get_dtype_strided_assign_operation(dst_dt, dst_stride, src_dt, src_stride,
                                (char)((intptr_t)dst | dst_stride | (intptr_t)src | src_stride));
    op.first(dst, dst_stride, src, src_stride, count, op.second);
}

void dnd::dtype_strided_assign_noexcept(void *dst, intptr_t dst_stride,
                            void *src, intptr_t src_stride,
                            intptr_t count,
                            dtype dst_dt, dtype src_dt)
{
    std::pair<unary_operation_t, auxiliary_data *> op;
    op = get_dtype_strided_assign_noexcept_operation(dst_dt, dst_stride, src_dt, src_stride,
                                (char)((intptr_t)dst | dst_stride | (intptr_t)src | src_stride));
    op.first(dst, dst_stride, src, src_stride, count, op.second);
}

