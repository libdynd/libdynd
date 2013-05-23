//
// Copyright (C) 2011-13 Mark Wiebe, DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#ifndef _DYND__NDOBJECT_ARANGE_HPP_
#define _DYND__NDOBJECT_ARANGE_HPP_

#include <dynd/ndobject.hpp>

namespace dynd {

/**
 * General version of arange, with raw pointers to the values. Returns
 * a one-dimensional array with the values {beginval, beginval + stepval, ...,
 * beginval + (k-1) * stepval} where the next value in the sequence would hit
 * or cross endval.
 */
ndobject arange(const dtype& scalar_dtype, const void *beginval, const void *endval, const void *stepval);

/**
 * Version of arange templated for C++ scalar types.
 */
template<class T>
typename enable_if<is_dtype_scalar<T>::value, ndobject>::type arange(T beginval, T endval,
                                                                    T stepval = T(1)) {
    return arange(make_dtype<T>(), &beginval, &endval, &stepval);
}

/**
 * Version of arange templated for C++ scalar types, with just the end parameter.
 */
template<class T>
typename enable_if<is_dtype_scalar<T>::value, ndobject>::type arange(T endval) {
    T beginval = T(0), stepval = T(1);
    return arange(make_dtype<T>(), &beginval, &endval, &stepval);
}

/**
 * Version of arange based on an irange object.
 */
inline ndobject arange(const irange& i) {
    return arange(make_dtype<intptr_t>(), &i.start(), &i.finish(), &i.step());
}

/**
 * Most general linspace function, creates an array of length 'count', linearly
 * interpolating from the value 'start' to the value 'stop'.
 *
 * \param start  The value placed at index 0 of the result.
 * \param stop  The value placed at index count-1 of the result.
 * \param count  The size of the result's first dimension.
 * \param dt  The required dtype of the output.
 */
ndobject linspace(const ndobject& start, const ndobject& stop, intptr_t count, const dtype& dt);

/**
 * Most general linspace function, creates an array of length 'count', linearly
 * interpolating from the value 'start' to the value 'stop'. This version
 * figures out the dtype from that of 'start' and 'stop'.
 *
 * \param start  The value placed at index 0 of the result.
 * \param stop  The value placed at index count-1 of the result.
 * \param count  The size of the result's first dimension.
 */
ndobject linspace(const ndobject& start, const ndobject& stop, intptr_t count);

/**
 * Creates a one-dimensional array of 'count' values, evenly spaced
 * from 'startval' to 'stopval', including both values.
 *
 * The only built-in types supported are float and double.
 */
ndobject linspace(const dtype& dt, const void *startval, const void *stopval, intptr_t count);

inline ndobject linspace(float start, float stop, intptr_t count = 50) {
    return linspace(dtype(float32_type_id), &start, &stop, count);
}

inline ndobject linspace(double start, double stop, intptr_t count = 50) {
    return linspace(dtype(float64_type_id), &start, &stop, count);
}

template <class T>
typename enable_if<dtype_kind_of<T>::value == int_kind ||
                dtype_kind_of<T>::value == uint_kind, ndobject>::type linspace(
                                                    T start, T stop, intptr_t count = 50) {
    return linspace((double)start, (double)stop, count);
}

} // namespace dynd

#endif // _DYND__NDOBJECT_ARANGE_HPP_
