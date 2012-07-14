//
// Copyright (C) 2011-12, Dynamic NDArray Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <dnd/config.hpp>
#include <dnd/dtype.hpp>
#include <dnd/dtype_comparisons.hpp>

#include <complex>
#include <iostream>
#include <stdint.h>

using namespace std;
using namespace dnd;

namespace {


template <typename T>
struct compare_kernel {
    static bool less_as(const char *a, const char *b, const AuxDataBase *DND_UNUSED(auxdata)) {
        return *reinterpret_cast<const T *>(a) < *reinterpret_cast<const T *>(b);
    }

    static bool less_equal_as(const char *a, const char *b, const AuxDataBase *DND_UNUSED(auxdata)) {
        return *reinterpret_cast<const T *>(a) <= *reinterpret_cast<const T *>(b);
    }

    static bool equal_as(const char *a, const char *b, const AuxDataBase *DND_UNUSED(auxdata)) {
        return *reinterpret_cast<const T *>(a) == *reinterpret_cast<const T *>(b);
    }

    static bool not_equal_as(const char *a, const char *b, const AuxDataBase *DND_UNUSED(auxdata)) {
        return *reinterpret_cast<const T *>(a) != *reinterpret_cast<const T *>(b);
    }

    static bool greater_equal_as(const char *a, const char *b, const AuxDataBase *DND_UNUSED(auxdata)) {
        return *reinterpret_cast<const T *>(a) >= *reinterpret_cast<const T *>(b);
    }

    static bool greater_as(const char *a, const char *b, const AuxDataBase *DND_UNUSED(auxdata)) {
        return *reinterpret_cast<const T *>(a) > *reinterpret_cast<const T *>(b);
    }
};

#define COMPLEX_COMPARE(OP) \
    if (reinterpret_cast<const complex<T> *>(a)->real() == reinterpret_cast<const complex<T> *>(b)->real()) { \
        return reinterpret_cast<const complex<T> *>(a)->imag() OP reinterpret_cast<const complex<T> *>(b)->imag(); \
    } \
    else { \
        return reinterpret_cast<const complex<T> *>(a)->real() OP reinterpret_cast<const complex<T> *>(b)->real(); \
    }


template <typename T>
struct compare_kernel< complex<T> > {
    static bool less_as(const char *a, const char *b, const AuxDataBase *DND_UNUSED(auxdata)) {
        COMPLEX_COMPARE(<)
    }

    static bool less_equal_as(const char *a, const char *b, const AuxDataBase *DND_UNUSED(auxdata)) {
        COMPLEX_COMPARE(<=)
    }

    static bool equal_as(const char *a, const char *b, const AuxDataBase *DND_UNUSED(auxdata)) {
        return *reinterpret_cast<const complex<T> *>(a) == *reinterpret_cast<const complex<T> *>(b);
    }

    static bool not_equal_as(const char *a, const char *b, const AuxDataBase *DND_UNUSED(auxdata)) {
        return *reinterpret_cast<const complex<T> *>(a) != *reinterpret_cast<const complex<T> *>(b);
    }

    static bool greater_equal_as(const char *a, const char *b, const AuxDataBase *DND_UNUSED(auxdata)) {
        COMPLEX_COMPARE(>=)
    }

    static bool greater_as(const char *a, const char *b, const AuxDataBase *DND_UNUSED(auxdata)) {
        COMPLEX_COMPARE(>)
    }

};

#undef COMPLEX_COMPARE

} // anonymous namespace


#define DND_BUILTIN_DTYPE_COMPARISON_TABLE_TYPE_LEVEL(type) { \
    (single_compare_operation_t)compare_kernel<type>::less_as, \
    (single_compare_operation_t)compare_kernel<type>::less_equal_as, \
    (single_compare_operation_t)compare_kernel<type>::equal_as, \
    (single_compare_operation_t)compare_kernel<type>::not_equal_as, \
    (single_compare_operation_t)compare_kernel<type>::greater_equal_as, \
    (single_compare_operation_t)compare_kernel<type>::greater_as \
    }

namespace dnd {

single_compare_operation_table_t builtin_dtype_comparisons_table[builtin_type_id_count] = {
    DND_BUILTIN_DTYPE_COMPARISON_TABLE_TYPE_LEVEL(dnd_bool), \
    DND_BUILTIN_DTYPE_COMPARISON_TABLE_TYPE_LEVEL(int8_t), \
    DND_BUILTIN_DTYPE_COMPARISON_TABLE_TYPE_LEVEL(int16_t), \
    DND_BUILTIN_DTYPE_COMPARISON_TABLE_TYPE_LEVEL(int32_t), \
    DND_BUILTIN_DTYPE_COMPARISON_TABLE_TYPE_LEVEL(int64_t), \
    DND_BUILTIN_DTYPE_COMPARISON_TABLE_TYPE_LEVEL(uint8_t), \
    DND_BUILTIN_DTYPE_COMPARISON_TABLE_TYPE_LEVEL(uint16_t), \
    DND_BUILTIN_DTYPE_COMPARISON_TABLE_TYPE_LEVEL(uint32_t), \
    DND_BUILTIN_DTYPE_COMPARISON_TABLE_TYPE_LEVEL(uint64_t), \
    DND_BUILTIN_DTYPE_COMPARISON_TABLE_TYPE_LEVEL(float), \
    DND_BUILTIN_DTYPE_COMPARISON_TABLE_TYPE_LEVEL(double), \
    DND_BUILTIN_DTYPE_COMPARISON_TABLE_TYPE_LEVEL(complex<float>), \
    DND_BUILTIN_DTYPE_COMPARISON_TABLE_TYPE_LEVEL(complex<double>) \
};

}

