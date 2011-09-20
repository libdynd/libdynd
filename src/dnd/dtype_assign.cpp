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

#include "single_assigner_simple.hpp"

using namespace std;
using namespace dnd;

template<int size> struct sized_byteswapper;
template<> struct sized_byteswapper<1> {
    static void byteswap(char *dst, const char *src) {
        *dst = *src;
    }
};
template<> struct sized_byteswapper<2> {
    static void byteswap(char *dst, const char *src) {
        dst[0] = src[1];
        dst[1] = src[0];
    }
};
template<> struct sized_byteswapper<4> {
    static void byteswap(char *dst, const char *src) {
        dst[0] = src[3];
        dst[1] = src[2];
        dst[2] = src[1];
        dst[3] = src[0];
    }
};
template<> struct sized_byteswapper<8> {
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

template<class T> struct byteswapper : public sized_byteswapper<sizeof(T)> {};


// The single_assigner class assigns a single item of the known dtypes, doing casting
// and byte swapping as necessary.

template <class dst_type, class src_type, bool dst_byteswapped, bool src_byteswapped, assign_error_mode errmode>
struct single_assigner;

// With no byte-swapping, just call through to single_assigner_simple
template <class dst_type, class src_type, assign_error_mode errmode>
struct single_assigner<dst_type, src_type, false, false, errmode> {
    static void assign(void *dst, const void *src) {
        single_assigner_simple<dst_type, src_type, errmode>::assign(reinterpret_cast<dst_type *>(dst),
                                                                reinterpret_cast<const src_type *>(src));
    }
};

// Simply copying with the same type raises no errors in any mode
template <class T, assign_error_mode errmode>
struct single_assigner<T, T, false, false, errmode> {
    static void assign(void *dst, const void *src) {
        *reinterpret_cast<T *>(dst) = *reinterpret_cast<const T *>(src);
    }
};

// With src byte-swapped
template <class dst_type, class src_type, assign_error_mode errmode>
struct single_assigner<dst_type, src_type, false, true, errmode> {
    static void assign(void *dst, const void *src) {
        src_type s;

        byteswapper<src_type>::byteswap(reinterpret_cast<char *>(&s), reinterpret_cast<const char *>(src));
        single_assigner_simple<dst_type, src_type, errmode>::assign(reinterpret_cast<dst_type *>(dst), &s);
    }
};

// With dst byte-swapped
template <class dst_type, class src_type, assign_error_mode errmode>
struct single_assigner<dst_type, src_type, true, false, errmode> {
    static void assign(void *dst, const void *src) {
        dst_type d;

        single_assigner_simple<dst_type, src_type, errmode>::assign(&d, reinterpret_cast<const src_type *>(src));
        byteswapper<dst_type>::byteswap(reinterpret_cast<char *>(dst), reinterpret_cast<const char *>(&d));
    }
};

// With both src and dst byte-swapped
template <class dst_type, class src_type, assign_error_mode errmode>
struct single_assigner<dst_type, src_type, true, true, errmode> {
    static void assign(void *dst, const void *src) {
        src_type s;
        dst_type d;

        byteswapper<src_type>::byteswap(reinterpret_cast<char *>(&s), reinterpret_cast<const char *>(src));
        single_assigner_simple<dst_type, src_type, errmode>::assign(&d, &s);
        byteswapper<dst_type>::byteswap(reinterpret_cast<char *>(dst), reinterpret_cast<const char *>(&d));
    }
};

typedef void (*assign_function_t)(void *dst, const void *src);

static assign_function_t single_assign_table[11][11][2][2][4] =
{
#define ERROR_MODE_LEVEL(dst_type, src_type, dst_byteswapped, src_byteswapped) { \
        &single_assigner<dst_type, src_type, dst_byteswapped, src_byteswapped, assign_error_none>::assign, \
        &single_assigner<dst_type, src_type, dst_byteswapped, src_byteswapped, assign_error_overflow>::assign, \
        &single_assigner<dst_type, src_type, dst_byteswapped, src_byteswapped, assign_error_fractional>::assign, \
        &single_assigner<dst_type, src_type, dst_byteswapped, src_byteswapped, assign_error_inexact>::assign \
    }

#define SRC_BYTESWAP_LEVEL(dst_type, src_type, dst_byteswapped) { \
        ERROR_MODE_LEVEL(dst_type, src_type, dst_byteswapped, false), \
        ERROR_MODE_LEVEL(dst_type, src_type, dst_byteswapped, true) \
    }

#define DST_BYTESWAP_LEVEL(dst_type, src_type) { \
        SRC_BYTESWAP_LEVEL(dst_type, src_type, false), \
        SRC_BYTESWAP_LEVEL(dst_type, src_type, true) \
    }

#define SRC_TYPE_LEVEL(dst_type) { \
        DST_BYTESWAP_LEVEL(dst_type, dnd_bool), \
        DST_BYTESWAP_LEVEL(dst_type, int8_t), \
        DST_BYTESWAP_LEVEL(dst_type, int16_t), \
        DST_BYTESWAP_LEVEL(dst_type, int32_t), \
        DST_BYTESWAP_LEVEL(dst_type, int64_t), \
        DST_BYTESWAP_LEVEL(dst_type, uint8_t), \
        DST_BYTESWAP_LEVEL(dst_type, uint16_t), \
        DST_BYTESWAP_LEVEL(dst_type, uint32_t), \
        DST_BYTESWAP_LEVEL(dst_type, uint64_t), \
        DST_BYTESWAP_LEVEL(dst_type, float), \
        DST_BYTESWAP_LEVEL(dst_type, double) \
    }
    
    SRC_TYPE_LEVEL(dnd_bool),
    SRC_TYPE_LEVEL(int8_t),
    SRC_TYPE_LEVEL(int16_t),
    SRC_TYPE_LEVEL(int32_t),
    SRC_TYPE_LEVEL(int64_t),
    SRC_TYPE_LEVEL(uint8_t),
    SRC_TYPE_LEVEL(uint16_t),
    SRC_TYPE_LEVEL(uint32_t),
    SRC_TYPE_LEVEL(uint64_t),
    SRC_TYPE_LEVEL(float),
    SRC_TYPE_LEVEL(double)
#undef SRC_TYPE_LEVEL
#undef DST_BYTESWAP_LEVEL
#undef SRC_BYTESWAP_LEVEL
#undef ERROR_MODE_LEVEL
};

static inline assign_function_t get_single_assign_function(const dtype& dst_dt, const dtype& src_dt,
                                                                assign_error_mode errmode)
{
    int dst_type_id = dst_dt.type_id(), src_type_id = src_dt.type_id();
    // Do a table lookup for the built-in range of dtypes
    if (dst_type_id >= bool_type_id && dst_type_id <= float64_type_id &&
            src_type_id >= bool_type_id && src_type_id <= float64_type_id) {
        return single_assign_table[dst_type_id-1][src_type_id-1]
                                    [dst_dt.is_byteswapped()][src_dt.is_byteswapped()]
                                    [errmode];
    } else {
        return NULL;
    }
}

void dnd::dtype_assign(const dtype& dst_dt, void *dst, const dtype& src_dt, const void *src, assign_error_mode errmode)
{
    if (dst_dt.extended() == NULL && src_dt.extended() == NULL) {
        assign_function_t asn = get_single_assign_function(dst_dt, src_dt, errmode);
        if (asn != NULL) {
            asn(dst, src);
            return;
        }

        throw std::runtime_error("this dtype assignment isn't yet supported");
    }

    throw std::runtime_error("this dtype assignment isn't yet supported");
}

#undef DTYPE_ASSIGN_SRC_TO_DST_ONE_CASE
#undef DTYPE_ASSIGN_SRC_TO_ANY_CASE
#undef DTYPE_ASSIGN_ANY_TO_ANY_SWITCH

// A multiple unaligned assignment function which uses one of the single assignment functions as proxy
namespace {
    class multiple_unaligned_auxiliary_data : public auxiliary_data {
    public:
        assign_function_t assign;
        int dst_itemsize, src_itemsize;

        virtual ~multiple_unaligned_auxiliary_data() {
        }
    };
}
static void assign_multiple_unaligned(void *dst, intptr_t dst_stride, const void *src, intptr_t src_stride,
                                    intptr_t count, const auxiliary_data *data)
{
    const multiple_unaligned_auxiliary_data * mgdata = static_cast<const multiple_unaligned_auxiliary_data *>(data);

    char *dst_cached = reinterpret_cast<char *>(dst);
    const char *src_cached = reinterpret_cast<const char *>(src);

    int dst_itemsize = mgdata->dst_itemsize, src_itemsize = mgdata->src_itemsize;
    // TODO: Probably want to relax the assumption of at most 8 bytes
    int64_t d;
    int64_t s;

    assign_function_t asn = mgdata->assign;

    for (intptr_t i = 0; i < count; ++i) {
        memcpy(&s, src_cached, src_itemsize);
        asn(&d, &s);
        memcpy(dst_cached, &d, dst_itemsize);
        dst_cached += dst_stride;
        src_cached += src_stride;
    }
}

// A multiple aligned assignment function which uses one of the single assignment functions as proxy
namespace {
    class multiple_aligned_auxiliary_data : public auxiliary_data {
    public:
        assign_function_t assign;

        virtual ~multiple_aligned_auxiliary_data() {
        }
    };
}
static void assign_multiple_aligned(void *dst, intptr_t dst_stride, const void *src, intptr_t src_stride,
                                    intptr_t count, const auxiliary_data *data)
{
    const multiple_aligned_auxiliary_data * mgdata = static_cast<const multiple_aligned_auxiliary_data *>(data);

    char *dst_cached = reinterpret_cast<char *>(dst);
    const char *src_cached = reinterpret_cast<const char *>(src);

    assign_function_t asn = mgdata->assign;

    for (intptr_t i = 0; i < count; ++i) {
        asn(dst_cached, src_cached);
        dst_cached += dst_stride;
        src_cached += src_stride;
    }
}


// Some specialized multiple assignment functions
template<class dst_type, class src_type>
struct multiple_assigner {
    static void assign_noexcept(void *dst, intptr_t dst_stride,
                                const void *src, intptr_t src_stride,
                                intptr_t count,
                                const auxiliary_data *)
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

    static void assign_noexcept_anystride_zerostride(void *dst, intptr_t dst_stride,
                                const void *src, intptr_t,
                                intptr_t count,
                                const auxiliary_data *)
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
                                const auxiliary_data *)
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
                                const auxiliary_data *)
    {
        const src_type *src_cached = reinterpret_cast<const src_type *>(src);
        dst_type *dst_cached = reinterpret_cast<dst_type *>(dst);

        for (intptr_t i = 0; i < count; ++i, ++dst_cached, ++src_cached) {
            *dst_cached = static_cast<dst_type>(*src_cached);
        }
    }
};

#define DTYPE_ASSIGN_SRC_TO_DST_SINGLE_CASE(dst_type, src_type, ASSIGN_FN) \
    case type_id_of<dst_type>::value: \
        return std::pair<unary_operation_t, shared_ptr<auxiliary_data> >( \
            &multiple_assigner<dst_type, src_type>::ASSIGN_FN, \
            NULL); \
        break

#define DTYPE_ASSIGN_SRC_TO_ANY_CASE(src_type, ASSIGN_FN) \
    case type_id_of<src_type>::value: \
        switch (dst_dt.type_id()) { \
            DTYPE_ASSIGN_SRC_TO_DST_SINGLE_CASE(dnd_bool, src_type, ASSIGN_FN); \
            DTYPE_ASSIGN_SRC_TO_DST_SINGLE_CASE(int8_t,   src_type, ASSIGN_FN); \
            DTYPE_ASSIGN_SRC_TO_DST_SINGLE_CASE(int16_t,  src_type, ASSIGN_FN); \
            DTYPE_ASSIGN_SRC_TO_DST_SINGLE_CASE(int32_t,  src_type, ASSIGN_FN); \
            DTYPE_ASSIGN_SRC_TO_DST_SINGLE_CASE(int64_t,  src_type, ASSIGN_FN); \
            DTYPE_ASSIGN_SRC_TO_DST_SINGLE_CASE(uint8_t,  src_type, ASSIGN_FN); \
            DTYPE_ASSIGN_SRC_TO_DST_SINGLE_CASE(uint16_t, src_type, ASSIGN_FN); \
            DTYPE_ASSIGN_SRC_TO_DST_SINGLE_CASE(uint32_t, src_type, ASSIGN_FN); \
            DTYPE_ASSIGN_SRC_TO_DST_SINGLE_CASE(uint64_t, src_type, ASSIGN_FN); \
            DTYPE_ASSIGN_SRC_TO_DST_SINGLE_CASE(float,    src_type, ASSIGN_FN); \
            DTYPE_ASSIGN_SRC_TO_DST_SINGLE_CASE(double,   src_type, ASSIGN_FN); \
        } \
        break

#define DTYPE_ASSIGN_ANY_TO_ANY_SWITCH(ASSIGN_FN) \
    switch (src_dt.type_id()) { \
        DTYPE_ASSIGN_SRC_TO_ANY_CASE(dnd_bool, ASSIGN_FN); \
        DTYPE_ASSIGN_SRC_TO_ANY_CASE(int8_t, ASSIGN_FN); \
        DTYPE_ASSIGN_SRC_TO_ANY_CASE(int16_t, ASSIGN_FN); \
        DTYPE_ASSIGN_SRC_TO_ANY_CASE(int32_t, ASSIGN_FN); \
        DTYPE_ASSIGN_SRC_TO_ANY_CASE(int64_t, ASSIGN_FN); \
        DTYPE_ASSIGN_SRC_TO_ANY_CASE(uint8_t, ASSIGN_FN); \
        DTYPE_ASSIGN_SRC_TO_ANY_CASE(uint16_t, ASSIGN_FN); \
        DTYPE_ASSIGN_SRC_TO_ANY_CASE(uint32_t, ASSIGN_FN); \
        DTYPE_ASSIGN_SRC_TO_ANY_CASE(uint64_t, ASSIGN_FN); \
        DTYPE_ASSIGN_SRC_TO_ANY_CASE(float, ASSIGN_FN); \
        DTYPE_ASSIGN_SRC_TO_ANY_CASE(double, ASSIGN_FN); \
    }

std::pair<unary_operation_t, shared_ptr<auxiliary_data> > dnd::get_dtype_strided_assign_operation(
                    const dtype& dst_dt, intptr_t dst_fixedstride, char dst_align_test,
                    const dtype& src_dt, intptr_t src_fixedstride, char src_align_test,
                    assign_error_mode errmode)
{
    bool is_aligned = dst_dt.is_data_aligned(dst_align_test) && src_dt.is_data_aligned(src_align_test);
    bool dst_byteswapped = dst_dt.is_byteswapped(), src_byteswapped = src_dt.is_byteswapped();

    // If the casting can be done losslessly, disable the error check to find faster code paths
    if (can_cast_lossless(dst_dt, src_dt)) {
        errmode = assign_error_none;
    }

    if (dst_dt.extended() == NULL && src_dt.extended() == NULL) {
        // When there's misaligned or byte-swapped data, go the slow path
        if (!is_aligned || src_byteswapped || dst_byteswapped || errmode != assign_error_none) {
            assign_function_t asn = get_single_assign_function(dst_dt, src_dt, errmode);
            if (asn != NULL) {
                std::pair<unary_operation_t, shared_ptr<auxiliary_data> > result;
                if (is_aligned) {
                    result.first = &assign_multiple_aligned;
                    multiple_aligned_auxiliary_data *auxdata = new multiple_aligned_auxiliary_data();
                    result.second.reset(auxdata);

                    auxdata->assign = asn;
                }
                else {
                    result.first = &assign_multiple_unaligned;
                    multiple_unaligned_auxiliary_data *auxdata = new multiple_unaligned_auxiliary_data();
                    result.second.reset(auxdata);

                    auxdata->assign = asn;
                    auxdata->dst_itemsize = dst_dt.itemsize();
                    auxdata->src_itemsize = src_dt.itemsize();
                }

                return std::move(result);
            }
        } else {
            if (src_fixedstride == 0) {
                if (dst_fixedstride == dst_dt.itemsize()) {
                    DTYPE_ASSIGN_ANY_TO_ANY_SWITCH(assign_noexcept_contigstride_zerostride);
                } else {
                    DTYPE_ASSIGN_ANY_TO_ANY_SWITCH(assign_noexcept_anystride_zerostride);
                }
            } else if (dst_fixedstride == dst_dt.itemsize() && src_fixedstride == src_dt.itemsize()) {
                DTYPE_ASSIGN_ANY_TO_ANY_SWITCH(assign_noexcept_contigstride_contigstride);
            } else {
                DTYPE_ASSIGN_ANY_TO_ANY_SWITCH(assign_noexcept);
            }
        }

        throw std::runtime_error("this dtype assignment isn't yet supported");
    }

    throw std::runtime_error("this dtype assignment isn't yet supported");
}

void dnd::dtype_strided_assign(const dtype& dst_dt, void *dst, intptr_t dst_stride,
                            const dtype& src_dt, const void *src, intptr_t src_stride,
                            intptr_t count, assign_error_mode errmode)
{
    std::pair<unary_operation_t, shared_ptr<auxiliary_data> > op;
    op = get_dtype_strided_assign_operation(dst_dt, dst_stride, (char)((intptr_t)dst | dst_stride),
                                            src_dt, src_stride, (char)((intptr_t)src | src_stride),
                                            errmode);
    op.first(dst, dst_stride, src, src_stride, count, op.second.get());
}

// Fixed and unknown size contiguous copy assignment functions
template<int N>
static void contig_fixedsize_copy_assign(void *dst, intptr_t, const void *src, intptr_t,
                            intptr_t count, const auxiliary_data *) {
    memcpy(dst, src, N * count);
}
namespace {
    class assign_itemsize_auxiliary_data : public auxiliary_data {
    public:
        intptr_t itemsize;

        virtual ~assign_itemsize_auxiliary_data() {
        }
    };

    template<class T>
    struct fixed_size_copy_assign_type {
        static void assign(void *dst, intptr_t dst_stride, const void *src, intptr_t src_stride,
                            intptr_t count, const auxiliary_data *) {
            T *dst_cached = reinterpret_cast<T *>(dst);
            const T *src_cached = reinterpret_cast<const T *>(src);
            dst_stride /= sizeof(T);
            src_stride /= sizeof(T);

            for (intptr_t i = 0; i < count; ++i) {
                *dst_cached = *src_cached;
                
                dst_cached += dst_stride;
                src_cached += src_stride;
            }
        }
    };

    template<int N>
    struct fixed_size_copy_assign;
    template<>
    struct fixed_size_copy_assign<1> : public fixed_size_copy_assign_type<char> {};
    template<>
    struct fixed_size_copy_assign<2> : public fixed_size_copy_assign_type<int16_t> {};
    template<>
    struct fixed_size_copy_assign<4> : public fixed_size_copy_assign_type<int32_t> {};
    template<>
    struct fixed_size_copy_assign<8> : public fixed_size_copy_assign_type<int64_t> {};
}
static void contig_copy_assign(void *dst, intptr_t, const void *src, intptr_t,
                            intptr_t count, const auxiliary_data *auxdata) {
    const assign_itemsize_auxiliary_data *data = static_cast<const assign_itemsize_auxiliary_data *>(auxdata);
    memcpy(dst, src, data->itemsize * count);
}
static void strided_copy_assign(void *dst, intptr_t dst_stride, const void *src, intptr_t src_stride,
                            intptr_t count, const auxiliary_data *auxdata) {
    char *dst_cached = reinterpret_cast<char *>(dst);
    const char *src_cached = reinterpret_cast<const char *>(src);
    const assign_itemsize_auxiliary_data *data = static_cast<const assign_itemsize_auxiliary_data *>(auxdata);
    intptr_t itemsize = data->itemsize;

    for (intptr_t i = 0; i < count; ++i) {
        memcpy(dst_cached, src_cached, itemsize);
        dst_cached += dst_stride;
        src_cached += src_stride;
    }
}


std::pair<unary_operation_t, std::shared_ptr<auxiliary_data> > get_dtype_strided_assign_operation(
                    const dtype& dt,
                    intptr_t dst_fixedstride, char dst_align_test,
                    intptr_t src_fixedstride, char src_align_test)
{
    if (!dt.is_object_type()) {
        std::pair<unary_operation_t, std::shared_ptr<auxiliary_data> > result;

        if (dst_fixedstride == dt.itemsize() && src_fixedstride == dt.itemsize()) {
            // contig -> contig uses memcpy, works with unaligned data
            switch (dt.itemsize()) {
                case 1:
                    result.first = &contig_fixedsize_copy_assign<1>;
                    break;
                case 2:
                    result.first = &contig_fixedsize_copy_assign<2>;
                    break;
                case 4:
                    result.first = &contig_fixedsize_copy_assign<4>;
                    break;
                case 8:
                    result.first = &contig_fixedsize_copy_assign<8>;
                    break;
                case 16:
                    result.first = &contig_fixedsize_copy_assign<16>;
                    break;
                default:
                    result.first = &contig_copy_assign;
                    assign_itemsize_auxiliary_data *auxdata = new assign_itemsize_auxiliary_data();
                    result.second.reset(auxdata);
                    auxdata->itemsize = dt.itemsize();
                    break;
            }
        } else {
            result.first = NULL;
            switch (dt.itemsize()) {
                case 1:
                    result.first = &fixed_size_copy_assign<1>::assign;
                    break;
                case 2:
                    if (((dst_align_test | src_align_test) & 0x1) == 0) {
                        result.first = &fixed_size_copy_assign<2>::assign;
                    }
                    break;
                case 4:
                    if (((dst_align_test | src_align_test) & 0x3) == 0) {
                        result.first = &fixed_size_copy_assign<4>::assign;
                    }
                    break;
                case 8:
                    if (((dst_align_test | src_align_test) & 0x7) == 0) {
                        result.first = &fixed_size_copy_assign<8>::assign;
                    }
                    break;
            }

            if (result.first == NULL) {
                result.first = &strided_copy_assign;
                assign_itemsize_auxiliary_data *auxdata = new assign_itemsize_auxiliary_data();
                result.second.reset(auxdata);
                auxdata->itemsize = dt.itemsize();
            }
        }

        return std::move(result);
    } else {
        throw std::runtime_error("cannot assign object dtypes yet");
    }
}
