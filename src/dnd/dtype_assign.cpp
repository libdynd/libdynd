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
    static void assign_unchecked(void *dst, const void *src) {
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
    static void assign_unchecked(void *dst, const void *src) {
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
    static void assign_unchecked(void *dst, const void *src) {
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
    static void assign_unchecked(void *dst, const void *src) {
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

#define DTYPE_ASSIGN_SRC_TO_ANY(src_type, src_byteswapped, ASSIGN_FN) \
    switch (dst_dt.type_id()) { \
        case bool_type_id: \
            single_assigner<dnd_bool, src_type, false, src_byteswapped>::ASSIGN_FN(dst, src); \
            break; \
        case int8_type_id: \
            single_assigner<int8_t, src_type, false, src_byteswapped>::ASSIGN_FN(dst, src); \
            break; \
        case int16_type_id: \
            if (dst_dt.is_byteswapped()) \
                single_assigner<int16_t, src_type, true, src_byteswapped>::ASSIGN_FN(dst, src); \
            else \
                single_assigner<int16_t, src_type, false, src_byteswapped>::ASSIGN_FN(dst, src); \
            break; \
        case int32_type_id: \
            if (dst_dt.is_byteswapped()) \
                single_assigner<int32_t, src_type, true, src_byteswapped>::ASSIGN_FN(dst, src); \
            else \
                single_assigner<int32_t, src_type, false, src_byteswapped>::ASSIGN_FN(dst, src); \
            break; \
        case int64_type_id: \
            if (dst_dt.is_byteswapped()) \
                single_assigner<int64_t, src_type, true, src_byteswapped>::ASSIGN_FN(dst, src); \
            else \
                single_assigner<int64_t, src_type, false, src_byteswapped>::ASSIGN_FN(dst, src); \
            break; \
        case uint8_type_id: \
            single_assigner<uint8_t, src_type, false, src_byteswapped>::ASSIGN_FN(dst, src); \
            break; \
        case uint16_type_id: \
            if (dst_dt.is_byteswapped()) \
                single_assigner<uint16_t, src_type, true, src_byteswapped>::ASSIGN_FN(dst, src); \
            else \
                single_assigner<uint16_t, src_type, false, src_byteswapped>::ASSIGN_FN(dst, src); \
            break; \
        case uint32_type_id: \
            if (dst_dt.is_byteswapped()) \
                single_assigner<uint32_t, src_type, true, src_byteswapped>::ASSIGN_FN(dst, src); \
            else \
                single_assigner<uint32_t, src_type, false, src_byteswapped>::ASSIGN_FN(dst, src); \
            break; \
        case uint64_type_id: \
            if (dst_dt.is_byteswapped()) \
                single_assigner<uint64_t, src_type, true, src_byteswapped>::ASSIGN_FN(dst, src); \
            else \
                single_assigner<uint64_t, src_type, false, src_byteswapped>::ASSIGN_FN(dst, src); \
            break; \
        case float32_type_id: \
            if (dst_dt.is_byteswapped()) \
                single_assigner<float, src_type, true, src_byteswapped>::ASSIGN_FN(dst, src); \
            else \
                single_assigner<float, src_type, false, src_byteswapped>::ASSIGN_FN(dst, src); \
            break; \
        case float64_type_id: \
            if (dst_dt.is_byteswapped()) \
                single_assigner<double, src_type, true, src_byteswapped>::ASSIGN_FN(dst, src); \
            else \
                single_assigner<double, src_type, false, src_byteswapped>::ASSIGN_FN(dst, src); \
            break; \
    }

void dnd::dtype_assign_noexcept(void *dst, const void *src, dtype dst_dt, dtype src_dt)
{
    // Do the simple case if the dtypes match
    switch (src_dt.type_id()) {
        case bool_type_id:
            DTYPE_ASSIGN_SRC_TO_ANY(dnd_bool, false, assign_unchecked);
            return;
        case int8_type_id:
            DTYPE_ASSIGN_SRC_TO_ANY(int8_t, false, assign_unchecked);
            return;
        case int16_type_id:
            if (src_dt.is_byteswapped()) {
                DTYPE_ASSIGN_SRC_TO_ANY(int16_t, true, assign_unchecked);
            } else {
                DTYPE_ASSIGN_SRC_TO_ANY(int16_t, false, assign_unchecked);
            }
            return;
        case int32_type_id:
            if (src_dt.is_byteswapped()) {
                DTYPE_ASSIGN_SRC_TO_ANY(int32_t, true, assign_unchecked);
            } else {
                DTYPE_ASSIGN_SRC_TO_ANY(int32_t, false, assign_unchecked);
            }
            return;
        case int64_type_id:
            if (src_dt.is_byteswapped()) {
                DTYPE_ASSIGN_SRC_TO_ANY(int64_t, true, assign_unchecked);
            } else {
                DTYPE_ASSIGN_SRC_TO_ANY(int64_t, false, assign_unchecked);
            }
            return;
        case uint8_type_id:
            DTYPE_ASSIGN_SRC_TO_ANY(uint8_t, false, assign_unchecked);
            return;
        case uint16_type_id:
            if (src_dt.is_byteswapped()) {
                DTYPE_ASSIGN_SRC_TO_ANY(uint16_t, true, assign_unchecked);
            } else {
                DTYPE_ASSIGN_SRC_TO_ANY(uint16_t, false, assign_unchecked);
            }
            return;
        case uint32_type_id:
            if (src_dt.is_byteswapped()) {
                DTYPE_ASSIGN_SRC_TO_ANY(uint32_t, true, assign_unchecked);
            } else {
                DTYPE_ASSIGN_SRC_TO_ANY(uint32_t, false, assign_unchecked);
            }
            return;
        case uint64_type_id:
            if (src_dt.is_byteswapped()) {
                DTYPE_ASSIGN_SRC_TO_ANY(uint64_t, true, assign_unchecked);
            } else {
                DTYPE_ASSIGN_SRC_TO_ANY(uint64_t, false, assign_unchecked);
            }
            return;
        case float32_type_id:
            if (src_dt.is_byteswapped()) {
                DTYPE_ASSIGN_SRC_TO_ANY(float, true, assign_unchecked);
            } else {
                DTYPE_ASSIGN_SRC_TO_ANY(float, false, assign_unchecked);
            }
            return;
        case float64_type_id:
            if (src_dt.is_byteswapped()) {
                DTYPE_ASSIGN_SRC_TO_ANY(double, true, assign_unchecked);
            } else {
                DTYPE_ASSIGN_SRC_TO_ANY(double, false, assign_unchecked);
            }
            return;
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

    // Do the simple case if the dtypes match
    switch (src_dt.type_id()) {
        case bool_type_id:
            DTYPE_ASSIGN_SRC_TO_ANY(dnd_bool, false, assign);
            return;
        case int8_type_id:
            DTYPE_ASSIGN_SRC_TO_ANY(int8_t, false, assign);
            return;
        case int16_type_id:
            if (src_dt.is_byteswapped()) {
                DTYPE_ASSIGN_SRC_TO_ANY(int16_t, true, assign);
            } else {
                DTYPE_ASSIGN_SRC_TO_ANY(int16_t, false, assign);
            }
            return;
        case int32_type_id:
            if (src_dt.is_byteswapped()) {
                DTYPE_ASSIGN_SRC_TO_ANY(int32_t, true, assign);
            } else {
                DTYPE_ASSIGN_SRC_TO_ANY(int32_t, false, assign);
            }
            return;
        case int64_type_id:
            if (src_dt.is_byteswapped()) {
                DTYPE_ASSIGN_SRC_TO_ANY(int64_t, true, assign);
            } else {
                DTYPE_ASSIGN_SRC_TO_ANY(int64_t, false, assign);
            }
            return;
        case uint8_type_id:
            DTYPE_ASSIGN_SRC_TO_ANY(uint8_t, false, assign);
            return;
        case uint16_type_id:
            if (src_dt.is_byteswapped()) {
                DTYPE_ASSIGN_SRC_TO_ANY(uint16_t, true, assign);
            } else {
                DTYPE_ASSIGN_SRC_TO_ANY(uint16_t, false, assign);
            }
            return;
        case uint32_type_id:
            if (src_dt.is_byteswapped()) {
                DTYPE_ASSIGN_SRC_TO_ANY(uint32_t, true, assign);
            } else {
                DTYPE_ASSIGN_SRC_TO_ANY(uint32_t, false, assign);
            }
            return;
        case uint64_type_id:
            if (src_dt.is_byteswapped()) {
                DTYPE_ASSIGN_SRC_TO_ANY(uint64_t, true, assign);
            } else {
                DTYPE_ASSIGN_SRC_TO_ANY(uint64_t, false, assign);
            }
            return;
        case float32_type_id:
            if (src_dt.is_byteswapped()) {
                DTYPE_ASSIGN_SRC_TO_ANY(float, true, assign);
            } else {
                DTYPE_ASSIGN_SRC_TO_ANY(float, false, assign);
            }
            return;
        case float64_type_id:
            if (src_dt.is_byteswapped()) {
                DTYPE_ASSIGN_SRC_TO_ANY(double, true, assign);
            } else {
                DTYPE_ASSIGN_SRC_TO_ANY(double, false, assign);
            }
            return;
    }

    throw std::runtime_error("this dtype assignment isn't yet supported");
}
