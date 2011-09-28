//
// Copyright (C) 2011 Mark Wiebe (mwwiebe@gmail.com)
// All rights reserved.
//
// This is unreleased proprietary software.
//
#ifndef _DND__NDARRAY_ARANGE_HPP_
#define _DND__NDARRAY_ARANGE_HPP_

#include <boost/utility/enable_if.hpp>

#include <dnd/ndarray.hpp>

namespace dnd {

/**
 * General version of arange, with raw pointers to the values.
 */
ndarray arange(const dtype& dt, const void *beginval, const void *endval, const void *stepval);

/**
 * Version of arange templated for C++ scalar types.
 */
template<class T>
typename boost::enable_if<is_dtype_scalar<T>, ndarray>::type arange(T beginval, T endval,
                                                                    T stepval = T(1)) {
    return arange(make_dtype<T>(), &beginval, &endval, &stepval);
}

/**
 * Version of arange templated for C++ scalar types, with just the end parameter.
 */
template<class T>
typename boost::enable_if<is_dtype_scalar<T>, ndarray>::type arange(T endval) {
    T beginval = T(0), stepval = T(1);
    return arange(make_dtype<T>(), &beginval, &endval, &stepval);
}

} // namespace dnd

#endif // _DND__NDARRAY_ARANGE_HPP_
