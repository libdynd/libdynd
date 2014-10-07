//
// Copyright (C) 2011-14 Mark Wiebe, DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#ifndef DYND__TYPES_SUBSTITUTE_SHAPE_HPP
#define DYND__TYPES_SUBSTITUTE_SHAPE_HPP

#include <map>

#include <dynd/type.hpp>
#include <dynd/string.hpp>

namespace dynd { namespace ndt {

/**
 * Substitutes a shape into a pattern type.
 *
 * For example, can combine a type "strided * strided * int32" with
 * a shape (3, 6) to produce the type "3 * 6 * int32".
 *
 * \param pattern  A symbolic type within which to substitute the shape.
 * \param ndim  Number of dimensions in the shape.
 * \param shape  The dimensions to substitute.
 */
ndt::type substitute_shape(const ndt::type &pattern, intptr_t ndim,
                           const intptr_t *shape);

}} // namespace dynd::ndt

#endif // DYND__TYPES_SUBSTITUTE_SHAPE_HPP
