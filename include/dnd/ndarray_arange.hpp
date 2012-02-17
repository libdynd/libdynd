//
// Copyright (C) 2011 Mark Wiebe (mwwiebe@gmail.com)
// All rights reserved.
//
// This is unreleased proprietary software.
//
#ifndef _DND__NDARRAY_ARANGE_HPP_
#define _DND__NDARRAY_ARANGE_HPP_

#include <dnd/ndarray.hpp>

namespace dnd {

/**
 * General version of arange, with raw pointers to the values. Returns
 * a one-dimensional array with the values {beginval, beginval + stepval, ...,
 * beginval + (k-1) * stepval} where the next value in the sequence would hit
 * or cross endval.
 */
ndarray arange(const dtype& dt, const void *beginval, const void *endval, const void *stepval);

/**
 * Version of arange templated for C++ scalar types.
 */
template<class T>
typename enable_if<is_dtype_scalar<T>::value, ndarray>::type arange(T beginval, T endval,
                                                                    T stepval = T(1)) {
    return arange(make_dtype<T>(), &beginval, &endval, &stepval);
}

/**
 * Version of arange templated for C++ scalar types, with just the end parameter.
 */
template<class T>
typename enable_if<is_dtype_scalar<T>::value, ndarray>::type arange(T endval) {
    T beginval = T(0), stepval = T(1);
    return arange(make_dtype<T>(), &beginval, &endval, &stepval);
}

/**
 * Version of arange based on an irange object.
 */
inline ndarray arange(const irange& i) {
    return arange(make_dtype<intptr_t>(), &i.start(), &i.finish(), &i.step());
}

/**
 * Creates a one-dimensional array of 'count' values, evenly spaced
 * from 'startval' to 'stopval', including both values.
 *
 * The only built-in types supported are float and double.
 */
ndarray linspace(const dtype& dt, const void *startval, const void *stopval, intptr_t count);

inline ndarray linspace(float start, float stop, intptr_t count = 50) {
    return linspace(dtype(float32_type_id), &start, &stop, count);
}

inline ndarray linspace(double start, double stop, intptr_t count = 50) {
    return linspace(dtype(float64_type_id), &start, &stop, count);
}

template <class T>
typename enable_if<dtype_kind_of<T>::value == int_kind ||
                dtype_kind_of<T>::value == uint_kind, ndarray>::type linspace(
                                                    T start, T stop, intptr_t count = 50) {
    return linspace((double)start, (double)stop, count);
}

} // namespace dnd

#endif // _DND__NDARRAY_ARANGE_HPP_
