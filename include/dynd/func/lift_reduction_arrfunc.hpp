//
// Copyright (C) 2011-15 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#pragma once

#include <dynd/config.hpp>
#include <dynd/array.hpp>
#include <dynd/func/arrfunc.hpp>

namespace dynd {

/**
 * Lifts the provided arrfunc, broadcasting it as necessary to execute
 * across the additional dimensions in the ``lifted_types`` array.
 *
 * \param elwise_reduction  The arrfunc to be lifted. This must
 *                          be a unary operation, which modifies the output
 *                          in place.
 * \param lifted_arr_type  The type the input should be lifted to.
 * \param dst_initialization  Either a NULL nd::array, or a arrfunc
 *                            which initializes an accumulator value from an
 *                            input value. If it is NULL, either the value in
 *                            `reduction_identity` is used, or a copy operation
 *                            is used if that is NULL.
 * \param keepdims  If true, the output type should keep reduced dimensions as
 *                  size one, otherwise they are eliminated.
 * \param reduction_ndim  The number of dimensions being lifted. This should
 *                        be equal to the number of dimensions added in
 *                        `lifted_types` over what is in `elwise_reduction`.
 * \param reduction_dimflags  An array of length `reduction_ndim`, indicating
 *                            for each dimension whether it is being reduced.
 * \param associative  Indicate whether the operation the reduction is derived
 *                     from is associative.
 * \param commutative  Indicate whether the operation the reduction is derived
 *                     from is commutative.
 * \param right_associative  If true, the reduction associates to the right instead of
 *                           the left. This is relevant if 'associative' and/or 'commutative'
 *                           are false, and in that case forces the reduction to happen
 *                           from right to left instead of left to right.
 * \param reduction_identity  If not a NULL nd::array, this is the identity
 *                            value for the accumulator.
 */
nd::arrfunc lift_reduction_arrfunc(const nd::arrfunc &elwise_reduction,
                                   const ndt::type &lifted_arr_type,
                                   const nd::arrfunc &dst_initialization,
                                   bool keepdims, intptr_t reduction_ndim,
                                   const bool *reduction_dimflags,
                                   bool associative, bool commutative,
                                   bool right_associative,
                                   const nd::array &reduction_identity);

} // namespace dynd
