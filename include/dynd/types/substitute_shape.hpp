//
// Copyright (C) 2011-15 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#pragma once

#include <map>

#include <dynd/type.hpp>
#include <dynd/string.hpp>

namespace dynd { namespace ndt {

/**
 * Substitutes a shape into a pattern type.
 *
 * For example, can combine a type "Fixed * Fixed * int32" with
 * a shape (3, 6) to produce the type "3 * 6 * int32".
 *
 * \param pattern  A symbolic type within which to substitute the shape.
 * \param ndim  Number of dimensions in the shape.
 * \param shape  The dimensions to substitute.
 */
DYND_API ndt::type substitute_shape(const ndt::type &pattern, intptr_t ndim,
                           const intptr_t *shape);

}} // namespace dynd::ndt
