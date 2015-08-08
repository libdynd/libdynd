//
// Copyright (C) 2011-15 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#pragma once

#include <dynd/config.hpp>
#include <dynd/array.hpp>
#include <dynd/func/callable.hpp>

namespace dynd {
namespace nd {

  /**
   * Lifts the provided reduction ckernel, broadcasting against the output
   *
   * This lifts a unary_operation ckernel, which accumulates an individual value
   * or a strided run of values with the possibility of a 0-stride in the
   *output.
   *
   * \param elwise_reduction  The callable being lifted
   * \param dst_initialization  A callable for initializing the
   *                            accumulator values from the source data.
   *                            If this is NULL, an assignment
   *                            kernel is used here.
   * \param ckb  The ckernel_builder into which to place the ckernel.
   * \param ckb_offset  Where within the ckernel_builder to place the ckernel.
   * \param dst_tp  The destination type to lift to.
   * \param dst_arrmeta  The destination arrmeta to lift to.
   * \param src_tp  The source type to lift to.
   * \param src_arrmeta  The source arrmeta to lift to.
   * \param reduction_ndim  The number of dimensions being reduced.
   * \param reduction_dimflags  Boolean flags indicating which dimensions to
   *                            reduce. This can typically be derived from an
   *                            "axis=" parameter.
   * \param associative  Whether we can assume the reduction kernel is
   *                     associative.
   * \param commutative  Whether we can assume the reduction kernel is
   *                     commutative.
   * \param right_associative  If true, the reduction is to be evaluated right
   *to
   *                           left instead of left to right.
   * \param reduction_identity  Either a NULL array if there is no identity, or
   *                            a value that the output can be initialized to at
   *                            the start.
   * \param kernreq  Either dynd::kernel_request_single or
   *                 dynd::kernel_request_strided,
   *                 as required by the caller.
   * \param ectx  The evaluation context to use.
   */
  size_t make_lifted_reduction_ckernel(
      const callable_type_data *elwise_reduction,
      const ndt::callable_type *elwise_reduction_tp,
      const callable_type_data *dst_initialization,
      const ndt::callable_type *dst_initialization_tp, void *ckb,
      intptr_t ckb_offset, const ndt::type &dst_tp, const char *dst_arrmeta,
      const ndt::type &src_tp, const char *src_arrmeta, intptr_t reduction_ndim,
      const bool *reduction_dimflags, bool associative, bool commutative,
      bool right_associative, const nd::array &reduction_identity,
      dynd::kernel_request_t kernreq,
      const eval::eval_context *ectx = &eval::default_eval_context);

  namespace functional {

    /**
     * Lifts the provided callable, broadcasting it as necessary to execute
     * across the additional dimensions in the ``lifted_types`` array.
     *
     * \param elwise_reduction  The callable to be lifted. This must
     *                          be a unary operation, which modifies the output
     *                          in place.
     * \param lifted_arr_type  The type the input should be lifted to.
     * \param dst_initialization  Either a NULL nd::array, or a callable
     *                            which initializes an accumulator value from an
     *                            input value. If it is NULL, either the value
     *in
     *                            `reduction_identity` is used, or a copy
     *operation
     *                            is used if that is NULL.
     * \param keepdims  If true, the output type should keep reduced dimensions
     *as
     *                  size one, otherwise they are eliminated.
     * \param reduction_ndim  The number of dimensions being lifted. This should
     *                        be equal to the number of dimensions added in
     *                        `lifted_types` over what is in `elwise_reduction`.
     * \param reduction_dimflags  An array of length `reduction_ndim`,
     *indicating
     *                            for each dimension whether it is being
     *reduced.
     * \param associative  Indicate whether the operation the reduction is
     *derived
     *                     from is associative.
     * \param commutative  Indicate whether the operation the reduction is
     *derived
     *                     from is commutative.
     * \param right_associative  If true, the reduction associates to the right
     *instead of
     *                           the left. This is relevant if 'associative'
     *and/or
     *'commutative'
     *                           are false, and in that case forces the
     *reduction
     *to
     *happen
     *                           from right to left instead of left to right.
     * \param reduction_identity  If not a NULL nd::array, this is the identity
     *                            value for the accumulator.
     */
    callable reduction(const callable &elwise_reduction,
                       const ndt::type &lifted_arr_type,
                       const callable &dst_initialization, bool keepdims,
                       intptr_t reduction_ndim, const bool *reduction_dimflags,
                       bool associative, bool commutative,
                       bool right_associative, const array &reduction_identity);

  } // namespace dynd::nd::functional
} // namespace dynd::nd
} // namespace dynd
