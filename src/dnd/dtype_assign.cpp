//
// Copyright (C) 2011 Mark Wiebe (mwwiebe@gmail.com)
// All rights reserved.
//
// This is unreleased proprietary software.
//
#include <dnd/dtype_assign.hpp>

#include <iostream>//DEBUG
#include <stdexcept>
#include <cstring>

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


void dnd::dtype_assign(void *dst, const void *src, dtype dt)
{
    memcpy(dst, src, dt.itemsize());
}

// The single_assigner class assigns a single item of the known dtypes, doing casting
// and byte swapping as necessary.

template<class dst_type, class src_type, bool dst_byteswap = false, bool src_byteswap = false>
struct single_assigner {
    static void assign(void *dst, const void *src) {
        const src_type *s = reinterpret_cast<const src_type *>(src);
        dst_type *d = reinterpret_cast<dst_type *>(dst);

        *d = static_cast<dst_type>(*s);
    }
};

template<class dst_type, class src_type>
struct single_assigner<dst_type, src_type, false, true> {
    static void assign(void *dst, const void *src) {
        src_type s;
        dst_type *d = reinterpret_cast<dst_type *>(dst);

        byteswapper<sizeof(src_type)>::byteswap(reinterpret_cast<char *>(&s),
                                                reinterpret_cast<const char *>(src));
        *d = static_cast<dst_type>(s);
    }
};

template<class dst_type, class src_type>
struct single_assigner<dst_type, src_type, true, false> {
    static void assign(void *dst, const void *src) {
        const src_type *s = reinterpret_cast<const src_type *>(src);
        dst_type d;

        d = static_cast<dst_type>(*s);
        byteswapper<sizeof(dst_type)>::byteswap(reinterpret_cast<char *>(dst),
                                                reinterpret_cast<const char *>(&d));
    }
};

template<class dst_type, class src_type>
struct single_assigner<dst_type, src_type, true, true> {
    static void assign(void *dst, const void *src) {
        src_type s;
        dst_type d;

        byteswapper<sizeof(src_type)>::byteswap(reinterpret_cast<char *>(&s),
                                                reinterpret_cast<const char *>(src));
        d = static_cast<dst_type>(s);
        byteswapper<sizeof(dst_type)>::byteswap(reinterpret_cast<char *>(dst),
                                                reinterpret_cast<const char *>(&d));
    }
};

#define DTYPE_ASSIGN_SRC_TO_ANY(src_type, src_byteswapped) \
    switch (dst_dt.type_id()) { \
        case bool_type_id: \
            single_assigner<dnd_bool, src_type, false, src_byteswapped>::assign(dst, src); \
            break; \
        case int8_type_id: \
            single_assigner<int8_t, src_type, false, src_byteswapped>::assign(dst, src); \
            break; \
        case int16_type_id: \
            if (dst_dt.is_byteswapped()) \
                single_assigner<int16_t, src_type, true, src_byteswapped>::assign(dst, src); \
            else \
                single_assigner<int16_t, src_type, false, src_byteswapped>::assign(dst, src); \
            break; \
        case int32_type_id: \
            if (dst_dt.is_byteswapped()) \
                single_assigner<int32_t, src_type, true, src_byteswapped>::assign(dst, src); \
            else \
                single_assigner<int32_t, src_type, false, src_byteswapped>::assign(dst, src); \
            break; \
        case int64_type_id: \
            if (dst_dt.is_byteswapped()) \
                single_assigner<int64_t, src_type, true, src_byteswapped>::assign(dst, src); \
            else \
                single_assigner<int64_t, src_type, false, src_byteswapped>::assign(dst, src); \
            break; \
        case uint8_type_id: \
            single_assigner<uint8_t, src_type, false, src_byteswapped>::assign(dst, src); \
            break; \
        case uint16_type_id: \
            if (dst_dt.is_byteswapped()) \
                single_assigner<uint16_t, src_type, true, src_byteswapped>::assign(dst, src); \
            else \
                single_assigner<uint16_t, src_type, false, src_byteswapped>::assign(dst, src); \
            break; \
        case uint32_type_id: \
            if (dst_dt.is_byteswapped()) \
                single_assigner<uint32_t, src_type, true, src_byteswapped>::assign(dst, src); \
            else \
                single_assigner<uint32_t, src_type, false, src_byteswapped>::assign(dst, src); \
            break; \
        case uint64_type_id: \
            if (dst_dt.is_byteswapped()) \
                single_assigner<uint64_t, src_type, true, src_byteswapped>::assign(dst, src); \
            else \
                single_assigner<uint64_t, src_type, false, src_byteswapped>::assign(dst, src); \
            break; \
        case float32_type_id: \
            if (dst_dt.is_byteswapped()) \
                single_assigner<float, src_type, true, src_byteswapped>::assign(dst, src); \
            else \
                single_assigner<float, src_type, false, src_byteswapped>::assign(dst, src); \
            break; \
        case float64_type_id: \
            if (dst_dt.is_byteswapped()) \
                single_assigner<double, src_type, true, src_byteswapped>::assign(dst, src); \
            else \
                single_assigner<double, src_type, false, src_byteswapped>::assign(dst, src); \
            break; \
    }

void dnd::dtype_assign(void *dst, const void *src, dtype dst_dt, dtype src_dt)
{
    // Do the simple case if the dtypes match
    switch (src_dt.type_id()) {
        case bool_type_id:
            DTYPE_ASSIGN_SRC_TO_ANY(dnd_bool, false);
            return;
        case int8_type_id:
            DTYPE_ASSIGN_SRC_TO_ANY(int8_t, false);
            return;
        case int16_type_id:
            if (src_dt.is_byteswapped()) {
                DTYPE_ASSIGN_SRC_TO_ANY(int16_t, true);
            } else {
                DTYPE_ASSIGN_SRC_TO_ANY(int16_t, false);
            }
            return;
        case int32_type_id:
            if (src_dt.is_byteswapped()) {
                DTYPE_ASSIGN_SRC_TO_ANY(int32_t, true);
            } else {
                DTYPE_ASSIGN_SRC_TO_ANY(int32_t, false);
            }
            return;
        case int64_type_id:
            if (src_dt.is_byteswapped()) {
                DTYPE_ASSIGN_SRC_TO_ANY(int64_t, true);
            } else {
                DTYPE_ASSIGN_SRC_TO_ANY(int64_t, false);
            }
            return;
        case uint8_type_id:
            DTYPE_ASSIGN_SRC_TO_ANY(uint8_t, false);
            return;
        case uint16_type_id:
            if (src_dt.is_byteswapped()) {
                DTYPE_ASSIGN_SRC_TO_ANY(uint16_t, true);
            } else {
                DTYPE_ASSIGN_SRC_TO_ANY(uint16_t, false);
            }
            return;
        case uint32_type_id:
            if (src_dt.is_byteswapped()) {
                DTYPE_ASSIGN_SRC_TO_ANY(uint32_t, true);
            } else {
                DTYPE_ASSIGN_SRC_TO_ANY(uint32_t, false);
            }
            return;
        case uint64_type_id:
            if (src_dt.is_byteswapped()) {
                DTYPE_ASSIGN_SRC_TO_ANY(uint64_t, true);
            } else {
                DTYPE_ASSIGN_SRC_TO_ANY(uint64_t, false);
            }
            return;
        case float32_type_id:
            if (src_dt.is_byteswapped()) {
                DTYPE_ASSIGN_SRC_TO_ANY(float, true);
            } else {
                DTYPE_ASSIGN_SRC_TO_ANY(float, false);
            }
            return;
        case float64_type_id:
            if (src_dt.is_byteswapped()) {
                DTYPE_ASSIGN_SRC_TO_ANY(double, true);
            } else {
                DTYPE_ASSIGN_SRC_TO_ANY(double, false);
            }
            return;
    }

    throw std::runtime_error("this dtype assignment isn't yet supported");
}
