//
// Copyright (C) 2011-15 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#pragma once

#include <dynd/callable.hpp>
#include <dynd/callables/call_callable.hpp>
#include <dynd/callables/forward_na_callable.hpp>

namespace dynd {
namespace nd {
  namespace functional {

    DYND_API callable adapt(const ndt::type &value_tp, const callable &forward);

    template <callable &Callable>
    callable call(const ndt::type &tp)
    {
      return make_callable<call_callable<Callable>>(tp);
    }

    /**
     * Returns an callable which composes the two callables together.
     * The buffer used to connect them is made out of the provided ``buf_tp``.
     */
    DYND_API callable compose(const callable &first, const callable &second, const ndt::type &buf_tp = ndt::type());

    /**
     * Makes a ckernel that ignores the src values, and writes
     * constant values to the output.
     */
    DYND_API callable constant(const array &val);

    /**
     * Adds an adapter ckernel which wraps a child binary expr ckernel
     * as a unary reduction ckernel. The three types of the binary
     * expr kernel must all be equal.
     *
     * \param ckb  The ckernel_builder into which the kernel adapter is placed.
     * \param ckb_offset  The offset within the ckernel_builder at which to
     *place the adapter.
     * \param right_associative  If true, the reduction is to be evaluated right
     *to left,
     *                           (x0 * (x1 * (x2 * x3))), if false, the
     *reduction is to be
     *                           evaluted left to right (((x0 * x1) * x2) * x3).
     * \param kernreq  The type of kernel to produce (single or strided).
     *
     * \returns  The ckb_offset where the child ckernel should be placed.
     */
    DYND_API callable left_compound(const callable &child);

    DYND_API callable right_compound(const callable &child);

    /**
     * Lifts the provided ckernel, broadcasting it as necessary to execute
     * across the additional dimensions in the ``lifted_types`` array.
     *
     * \param child  The callable being lifted
     */
    DYND_API callable elwise(const callable &child);

    DYND_API callable elwise(const ndt::type &self_tp, const callable &child);

    DYND_API ndt::type elwise_make_type(const ndt::callable_type *child_tp);

    template <int... I>
    callable forward_na(const callable &child)
    {
      ndt::type tp = ndt::callable_type::make(ndt::make_type<ndt::option_type>(child.get_ret_type()),
                                              {ndt::type("Any"), ndt::type("Any")});
      return make_callable<forward_na_callable<I...>>(tp, child);
    }

    DYND_API callable outer(const callable &child);

    DYND_API ndt::type outer_make_type(const ndt::callable_type *child_tp);

    /**
     * Create an callable which applies a given window_op in a
     * rolling window fashion.
     *
     * \param neighborhood_op  An callable object which transforms a
     *neighborhood
     *into
     *                         a single output value. Signature
     *                         '(Fixed * Fixed * NH, Fixed * Fixed * MSK) ->
     *OUT',
     */
    DYND_API callable neighborhood(const callable &child, const callable &boundary_child = callable());

    /**
     * Lifts the provided callable, broadcasting it as necessary to execute
     * across the additional dimensions in the ``lifted_types`` array.
     */
    DYND_API callable reduction(const callable &child);

    DYND_API callable reduction(const callable &child,
                                const std::initializer_list<std::pair<const char *, array>> &kwds);

  } // namespace dynd::nd::functional
} // namespace dynd::nd
} // namespace dynd
