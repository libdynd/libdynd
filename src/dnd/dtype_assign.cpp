//
// Copyright (C) 2011 Mark Wiebe (mwwiebe@gmail.com)
// All rights reserved.
//
// This is unreleased proprietary software.
//

#include <iostream> // FOR DEBUG
#include <typeinfo> // FOR DEBUG

#include <sstream>
#include <stdexcept>
#include <cstring>
#include <limits>

#include <dnd/dtype_assign.hpp>
#include <dnd/dtype_casting.hpp>

#include "single_assigner_simple.hpp"

using namespace std;
using namespace dnd;

typedef void (*assign_function_t)(void *dst, const void *src);

static assign_function_t single_assign_table[11][11][4] =
{
#define ERROR_MODE_LEVEL(dst_type, src_type) { \
        (assign_function_t)&single_assigner_simple<dst_type, src_type, assign_error_none>::assign, \
        (assign_function_t)&single_assigner_simple<dst_type, src_type, assign_error_overflow>::assign, \
        (assign_function_t)&single_assigner_simple<dst_type, src_type, assign_error_fractional>::assign, \
        (assign_function_t)&single_assigner_simple<dst_type, src_type, assign_error_inexact>::assign \
    }

#define SRC_TYPE_LEVEL(dst_type) { \
        ERROR_MODE_LEVEL(dst_type, dnd_bool), \
        ERROR_MODE_LEVEL(dst_type, int8_t), \
        ERROR_MODE_LEVEL(dst_type, int16_t), \
        ERROR_MODE_LEVEL(dst_type, int32_t), \
        ERROR_MODE_LEVEL(dst_type, int64_t), \
        ERROR_MODE_LEVEL(dst_type, uint8_t), \
        ERROR_MODE_LEVEL(dst_type, uint16_t), \
        ERROR_MODE_LEVEL(dst_type, uint32_t), \
        ERROR_MODE_LEVEL(dst_type, uint64_t), \
        ERROR_MODE_LEVEL(dst_type, float), \
        ERROR_MODE_LEVEL(dst_type, double) \
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
        return single_assign_table[dst_type_id-1][src_type_id-1][errmode];
    } else {
        return NULL;
    }
}

void dnd::dtype_assign(const dtype& dst_dt, void *dst, const dtype& src_dt, const void *src, assign_error_mode errmode)
{
    if (dst_dt.extended() == NULL && src_dt.extended() == NULL) {
        // None of the built-in scalars are more than 64-bits (currently...) so use two 64-bit
        // integers as temporary buffers for alignment and byteswapping.
        int64_t s, d;

        assign_function_t asn = get_single_assign_function(dst_dt, src_dt, errmode);
        if (asn != NULL) {
            if (src_dt.is_byteswapped()) {
                src_dt.get_byteswap_operation()(&s, src, src_dt.itemsize());
            } else {
                memcpy(&s, src, src_dt.itemsize());
            }
            asn(&d, &s);
            if (dst_dt.is_byteswapped()) {
                dst_dt.get_byteswap_operation()(dst, &d, dst_dt.itemsize());
            } else {
                memcpy(dst, &d, dst_dt.itemsize());
            }
            return;
        }

        throw std::runtime_error("this dtype assignment isn't yet supported");
    }

    throw std::runtime_error("this dtype assignment isn't yet supported");
}

static void dont_byteswap(void *dst, const void *src, intptr_t size)
{
    memcpy(dst, src, size);
}

// A multiple unaligned byteswap assignment function which uses one of the single assignment functions as proxy
namespace {
    class multiple_byteswap_unaligned_auxiliary_data : public auxiliary_data {
    public:
        assign_function_t assign;
        byteswap_operation_t src_byteswap, dst_byteswap;
        int dst_itemsize, src_itemsize;

        virtual ~multiple_byteswap_unaligned_auxiliary_data() {
        }
    };
}
static void assign_multiple_byteswap_unaligned(void *dst, intptr_t dst_stride,
                                    const void *src, intptr_t src_stride,
                                    intptr_t count, const auxiliary_data *data)
{
    const multiple_byteswap_unaligned_auxiliary_data * mgdata =
                            static_cast<const multiple_byteswap_unaligned_auxiliary_data *>(data);

    char *dst_cached = reinterpret_cast<char *>(dst);
    const char *src_cached = reinterpret_cast<const char *>(src);

    byteswap_operation_t src_byteswap = mgdata->src_byteswap, dst_byteswap = mgdata->dst_byteswap;
    int dst_itemsize = mgdata->dst_itemsize, src_itemsize = mgdata->src_itemsize;
    // TODO: Probably want to relax the assumption of at most 8 bytes
    int64_t d;
    int64_t s;

    assign_function_t asn = mgdata->assign;

    for (intptr_t i = 0; i < count; ++i) {
        src_byteswap(&s, src_cached, src_itemsize);
        asn(&d, &s);
        dst_byteswap(dst_cached, &d, dst_itemsize);
        dst_cached += dst_stride;
        src_cached += src_stride;
    }
}

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
        //DEBUG_COUT << "multiple_assigner::assign_noexcept (" << typeid(src_type).name() << " -> " << typeid(dst_type).name() << ")\n";
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
        //DEBUG_COUT << "multiple_assigner::assign_noexcept_anystride_zerostride (" << typeid(src_type).name() << " -> " << typeid(dst_type).name() << ")\n";
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
        //DEBUG_COUT << "multiple_assigner::assign_noexcept_contigstride_zerostride (" << typeid(src_type).name() << " -> " << typeid(dst_type).name() << ")\n";
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
        //DEBUG_COUT << "multiple_assigner::assign_noexcept_contigstride_contigstride (" << typeid(src_type).name() << " -> " << typeid(dst_type).name() << ")\n";
        const src_type *src_cached = reinterpret_cast<const src_type *>(src);
        dst_type *dst_cached = reinterpret_cast<dst_type *>(dst);

        for (intptr_t i = 0; i < count; ++i, ++dst_cached, ++src_cached) {
            *dst_cached = static_cast<dst_type>(*src_cached);
        }
    }
};

#define DND_XSTRINGIFY(s) #s
#define DND_STRINGIFY(s) DND_XSTRINGIFY(s)

#define DTYPE_ASSIGN_SRC_TO_DST_SINGLE_CASE(dst_type, src_type, ASSIGN_FN) \
    case type_id_of<dst_type>::value: \
        /*DEBUG_COUT << "returning " << DND_STRINGIFY(dst_type) << " " << DND_STRINGIFY(src_type) << " " << DND_STRINGIFY(ASSIGN_FN) << "\n";*/ \
        return std::pair<unary_operation_t, shared_ptr<auxiliary_data> >( \
            &multiple_assigner<dst_type, src_type>::ASSIGN_FN, \
            NULL);

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
    //DEBUG_COUT << "get_dtype_strided_assign_operation (different dtype " << dst_dt << ", " << src_dt << ", error mode " << errmode << ")\n";
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
                if (src_byteswapped || dst_byteswapped) {
                    result.first = &assign_multiple_unaligned;
                    multiple_byteswap_unaligned_auxiliary_data *auxdata =
                                            new multiple_byteswap_unaligned_auxiliary_data();
                    result.second.reset(auxdata);

                    auxdata->assign = asn;
                    auxdata->dst_itemsize = dst_dt.itemsize();
                    auxdata->src_itemsize = src_dt.itemsize();
                    if (dst_byteswapped) {
                        auxdata->dst_byteswap = dst_dt.get_byteswap_operation();
                    } else {
                        auxdata->dst_byteswap = &dont_byteswap;
                    }
                    if (src_byteswapped) {
                        auxdata->src_byteswap = src_dt.get_byteswap_operation();
                    } else {
                        auxdata->src_byteswap = &dont_byteswap;
                    }
                } else if (!is_aligned) {
                    result.first = &assign_multiple_unaligned;
                    multiple_unaligned_auxiliary_data *auxdata = new multiple_unaligned_auxiliary_data();
                    result.second.reset(auxdata);

                    auxdata->assign = asn;
                    auxdata->dst_itemsize = dst_dt.itemsize();
                    auxdata->src_itemsize = src_dt.itemsize();
                }
                else {
                    result.first = &assign_multiple_aligned;
                    multiple_aligned_auxiliary_data *auxdata = new multiple_aligned_auxiliary_data();
                    result.second.reset(auxdata);

                    auxdata->assign = asn;
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
    }

    stringstream ss;
    ss << "strided assignment from " << src_dt << " to " << dst_dt << " isn't yet supported";
    throw std::runtime_error(ss.str());
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

    template<class T>
    struct fixed_size_copy_zerostride_assign_type {
        static void assign(void *dst, intptr_t dst_stride, const void *src, intptr_t,
                            intptr_t count, const auxiliary_data *) {
            T *dst_cached = reinterpret_cast<T *>(dst);
            T s = *reinterpret_cast<const T *>(src);
            dst_stride /= sizeof(T);

            for (intptr_t i = 0; i < count; ++i) {
                *dst_cached = s;
                
                dst_cached += dst_stride;
            }
        }
    };

    template<int N>
    struct fixed_size_copy_zerostride_assign;
    template<>
    struct fixed_size_copy_zerostride_assign<1> : public fixed_size_copy_zerostride_assign_type<char> {};
    template<>
    struct fixed_size_copy_zerostride_assign<2> : public fixed_size_copy_zerostride_assign_type<int16_t> {};
    template<>
    struct fixed_size_copy_zerostride_assign<4> : public fixed_size_copy_zerostride_assign_type<int32_t> {};
    template<>
    struct fixed_size_copy_zerostride_assign<8> : public fixed_size_copy_zerostride_assign_type<int64_t> {};
}
static void contig_copy_assign(void *dst, intptr_t, const void *src, intptr_t,
                            intptr_t count, const auxiliary_data *auxdata)
{
    const assign_itemsize_auxiliary_data *data = static_cast<const assign_itemsize_auxiliary_data *>(auxdata);
    memcpy(dst, src, data->itemsize * count);
}
static void strided_copy_assign(void *dst, intptr_t dst_stride, const void *src, intptr_t src_stride,
                            intptr_t count, const auxiliary_data *auxdata)
{
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
static void fixed_size_copy_contig_zerostride_assign_memset(void *dst, intptr_t, const void *src, intptr_t,
                            intptr_t count, const auxiliary_data *)
{
    char s = *reinterpret_cast<const char *>(src);
    memset(dst, s, count);
}
static void strided_copy_zerostride_assign(void *dst, intptr_t dst_stride, const void *src, intptr_t,
                            intptr_t count, const auxiliary_data *auxdata)
{
    char *dst_cached = reinterpret_cast<char *>(dst);
    const assign_itemsize_auxiliary_data *data = static_cast<const assign_itemsize_auxiliary_data *>(auxdata);
    intptr_t itemsize = data->itemsize;

    for (intptr_t i = 0; i < count; ++i) {
        memcpy(dst_cached, src, itemsize);
        dst_cached += dst_stride;
    }
}


std::pair<unary_operation_t, std::shared_ptr<auxiliary_data> > dnd::get_dtype_strided_assign_operation(
                    const dtype& dt,
                    intptr_t dst_fixedstride, char dst_align_test,
                    intptr_t src_fixedstride, char src_align_test)
{
    //DEBUG_COUT << "get_dtype_strided_assign_operation (single dtype " << dt << ")\n";
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
        } else if (src_fixedstride == 0) {
            result.first = NULL;
            switch (dt.itemsize()) {
                case 1:
                    if (dst_fixedstride == 1) {
                        result.first = &fixed_size_copy_contig_zerostride_assign_memset;
                    } else {
                        result.first = &fixed_size_copy_zerostride_assign<1>::assign;
                    }
                    break;
                case 2:
                    if (((dst_align_test | src_align_test) & 0x1) == 0) {
                        result.first = &fixed_size_copy_zerostride_assign<2>::assign;
                    }
                    break;
                case 4:
                    if (((dst_align_test | src_align_test) & 0x3) == 0) {
                        result.first = &fixed_size_copy_zerostride_assign<4>::assign;
                    }
                    break;
                case 8:
                    if (((dst_align_test | src_align_test) & 0x7) == 0) {
                        result.first = &fixed_size_copy_zerostride_assign<8>::assign;
                    }
                    break;
            }

            if (result.first == NULL) {
                result.first = &strided_copy_zerostride_assign;
                assign_itemsize_auxiliary_data *auxdata = new assign_itemsize_auxiliary_data();
                result.second.reset(auxdata);
                auxdata->itemsize = dt.itemsize();
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
