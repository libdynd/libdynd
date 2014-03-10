//
// Copyright (C) 2011-14 Mark Wiebe, DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#ifndef _DYND__LIFT_REDUCTION_CKERNEL_DEFERRED_HPP_
#define _DYND__LIFT_REDUCTION_CKERNEL_DEFERRED_HPP_

#include <dynd/config.hpp>
#include <dynd/array.hpp>
#include <dynd/kernels/ckernel_deferred.hpp>

namespace dynd {

/**
 * Lifts the provided ckernel, broadcasting it as necessary to execute
 * across the additional dimensions in the ``lifted_types`` array.
 *
 * \param out_ckd  The output ckernel_deferred which is filled.
 * \param elwise_reduction  The ckernel_deferred to be lifted. This must
 *                          be a unary operation, which modifies the output
 *                          in place.
 * \param dst_initialization  Either a NULL nd::array, or a ckernel_deferred
 *                            which initializes an accumulator value from an
 *                            input value. If it is NULL, either the value in
 *                            `reduction_identity` is used, or a copy operation
 *                            is used if that is NULL.
 * \param lifted_types  The types to lift the ckernel to. The output ckernel
 *                      is for these types.
 * \param reduction_ndim  The number of dimensions being lifted. This should
 *                        be equal to the number of dimensions added in
 *                        `lifted_types` over what is in `elwise_reduction`.
 * \param reduction_dimflags  An array of length `reduction_ndim`, indicating
 *                            for each dimension whether it is being reduced.
 * \param associative  Indicate whether the operation the reduction is derived
 *                     from is associative.
 * \param commutative  Indicate whether the operation the reduction is derived
 *                     from is commutative.
 * \param reduction_identity  If not a NULL nd::array, this is the identity
 *                            value for the accumulator.
 */
void lift_reduction_ckernel_deferred(ckernel_deferred *out_ckd,
                const nd::array& elwise_reduction,
                const nd::array& dst_initialization,
                const std::vector<ndt::type>& lifted_types,
                intptr_t reduction_ndim,
                const bool *reduction_dimflags,
                bool associative,
                bool commutative,
                const nd::array& reduction_identity);

} // namespace dynd

#endif // _DYND__LIFT_REDUCTION_CKERNEL_DEFERRED_HPP_
