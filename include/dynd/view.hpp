//
// Copyright (C) 2011-14 Mark Wiebe, DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#ifndef _DYND__VIEW_HPP_
#define _DYND__VIEW_HPP_

#include <dynd/array.hpp>

namespace dynd { namespace nd {

/**
 * If possible returns an array which is a view of 'arr'.
 * When the types match exactly, returns 'arr' itself.
 *
 * Raises a type_error if the view is not possible.
 *
 * \param arr  The array to view as another type.
 * \param tp  The type to view the array as.
 */
array view(const array& arr, const ndt::type& tp);

}} // namespace dynd::nd

#endif // _DYND__VIEW_HPP_
