//
// Copyright (C) 2011-12, Dynamic NDArray Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <dynd/config.hpp>
#include <dynd/dtype.hpp>
#include <dynd/dtype_comparisons.hpp>

#include <complex>
#include <iostream>

using namespace std;
using namespace dynd;

namespace {


template <typename T>
struct compare_kernel {
    static bool less_as(const char *a, const char *b, const AuxDataBase *DYND_UNUSED(auxdata)) {
        return *reinterpret_cast<const T *>(a) < *reinterpret_cast<const T *>(b);
    }

    static bool less_equal_as(const char *a, const char *b, const AuxDataBase *DYND_UNUSED(auxdata)) {
        return *reinterpret_cast<const T *>(a) <= *reinterpret_cast<const T *>(b);
    }

    static bool equal_as(const char *a, const char *b, const AuxDataBase *DYND_UNUSED(auxdata)) {
        return *reinterpret_cast<const T *>(a) == *reinterpret_cast<const T *>(b);
    }

    static bool not_equal_as(const char *a, const char *b, const AuxDataBase *DYND_UNUSED(auxdata)) {
        return *reinterpret_cast<const T *>(a) != *reinterpret_cast<const T *>(b);
    }

    static bool greater_equal_as(const char *a, const char *b, const AuxDataBase *DYND_UNUSED(auxdata)) {
        return *reinterpret_cast<const T *>(a) >= *reinterpret_cast<const T *>(b);
    }

    static bool greater_as(const char *a, const char *b, const AuxDataBase *DYND_UNUSED(auxdata)) {
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
    static bool less_as(const char *a, const char *b, const AuxDataBase *DYND_UNUSED(auxdata)) {
        COMPLEX_COMPARE(<)
    }

    static bool less_equal_as(const char *a, const char *b, const AuxDataBase *DYND_UNUSED(auxdata)) {
        COMPLEX_COMPARE(<=)
    }

    static bool equal_as(const char *a, const char *b, const AuxDataBase *DYND_UNUSED(auxdata)) {
        return *reinterpret_cast<const complex<T> *>(a) == *reinterpret_cast<const complex<T> *>(b);
    }

    static bool not_equal_as(const char *a, const char *b, const AuxDataBase *DYND_UNUSED(auxdata)) {
        return *reinterpret_cast<const complex<T> *>(a) != *reinterpret_cast<const complex<T> *>(b);
    }

    static bool greater_equal_as(const char *a, const char *b, const AuxDataBase *DYND_UNUSED(auxdata)) {
        COMPLEX_COMPARE(>=)
    }

    static bool greater_as(const char *a, const char *b, const AuxDataBase *DYND_UNUSED(auxdata)) {
        COMPLEX_COMPARE(>)
    }

};

#undef COMPLEX_COMPARE

} // anonymous namespace


#define DYND_BUILTIN_DTYPE_COMPARISON_TABLE_TYPE_LEVEL(type) { \
    (single_compare_operation_t)compare_kernel<type>::less_as, \
    (single_compare_operation_t)compare_kernel<type>::less_equal_as, \
    (single_compare_operation_t)compare_kernel<type>::equal_as, \
    (single_compare_operation_t)compare_kernel<type>::not_equal_as, \
    (single_compare_operation_t)compare_kernel<type>::greater_equal_as, \
    (single_compare_operation_t)compare_kernel<type>::greater_as \
    }

namespace dynd {

single_compare_operation_table_t builtin_dtype_comparisons_table[builtin_type_id_count] = {
    DYND_BUILTIN_DTYPE_COMPARISON_TABLE_TYPE_LEVEL(dynd_bool), \
    DYND_BUILTIN_DTYPE_COMPARISON_TABLE_TYPE_LEVEL(int8_t), \
    DYND_BUILTIN_DTYPE_COMPARISON_TABLE_TYPE_LEVEL(int16_t), \
    DYND_BUILTIN_DTYPE_COMPARISON_TABLE_TYPE_LEVEL(int32_t), \
    DYND_BUILTIN_DTYPE_COMPARISON_TABLE_TYPE_LEVEL(int64_t), \
    DYND_BUILTIN_DTYPE_COMPARISON_TABLE_TYPE_LEVEL(uint8_t), \
    DYND_BUILTIN_DTYPE_COMPARISON_TABLE_TYPE_LEVEL(uint16_t), \
    DYND_BUILTIN_DTYPE_COMPARISON_TABLE_TYPE_LEVEL(uint32_t), \
    DYND_BUILTIN_DTYPE_COMPARISON_TABLE_TYPE_LEVEL(uint64_t), \
    DYND_BUILTIN_DTYPE_COMPARISON_TABLE_TYPE_LEVEL(float), \
    DYND_BUILTIN_DTYPE_COMPARISON_TABLE_TYPE_LEVEL(double), \
    DYND_BUILTIN_DTYPE_COMPARISON_TABLE_TYPE_LEVEL(complex<float>), \
    DYND_BUILTIN_DTYPE_COMPARISON_TABLE_TYPE_LEVEL(complex<double>) \
};

}

