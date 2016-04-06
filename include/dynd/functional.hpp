//
// Copyright (C) 2011-16 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#pragma once

#include <dynd/callable.hpp>
#include <dynd/callables/apply_function_callable.hpp>
#include <dynd/callables/apply_member_function_callable.hpp>
#include <dynd/callables/call_callable.hpp>
#include <dynd/callables/construct_then_apply_callable_callable.hpp>
#include <dynd/callables/forward_na_callable.hpp>

namespace dynd {
namespace nd {
  namespace functional {

    DYND_API callable adapt(const ndt::type &value_tp, const callable &forward);

    /**
     * Makes a callable out of function ``func``, using the provided keyword
     * parameter names. This function takes ``func`` as a template
     * parameter, so can call it efficiently.
     */
    template <typename func_type, func_type func, typename... T>
    callable apply(T &&... names) {
      return make_callable<apply_function_callable<func_type, func, arity_of<func_type>::value - sizeof...(T)>>(
          std::forward<T>(names)...);
    }

    /**
     * Makes a callable out of the function object ``func``, using the provided
     * keyword parameter names. This version makes a copy of provided ``func``
     * object.
     */
    template <typename func_type, typename... T>
    typename std::enable_if<!is_function_pointer<func_type>::value, callable>::type apply(func_type func,
                                                                                          T &&... names) {
      static_assert(all_char_string_params<T...>::value, "All the names must be strings");
      return make_callable<apply_callable_callable<func_type, arity_of<func_type>::value - sizeof...(T)>>(
          func, std::forward<T>(names)...);
    }

    template <typename func_type, typename... T>
    callable apply(func_type *func, T &&... names) {
      return make_callable<apply_callable_callable<func_type *, arity_of<func_type>::value - sizeof...(T)>>(
          func, std::forward<T>(names)...);
    }

    /**
     * Makes a callable out of the provided function object type, which
     * constructs and calls the function object on demand.
     */
    template <typename func_type, typename... KwdTypes, typename... T>
    callable apply(T &&... names) {
      return make_callable<construct_then_apply_callable_callable<func_type, KwdTypes...>>(std::forward<T>(names)...);
    }

    template <typename T, typename R, typename... A, typename... S>
    callable apply(T *obj, R (T::*mem_func)(A...), S &&... names) {
      return make_callable<apply_member_function_callable<T *, R (T::*)(A...), sizeof...(A) - sizeof...(S)>>(
          obj, mem_func, std::forward<S>(names)...);
    }

    template <callable &Callable>
    callable call(const ndt::type &tp) {
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

    DYND_API callable cyclic_elwise(const ndt::type &tp);

    DYND_API callable elwise(const ndt::type &self_tp, const callable &child);

    DYND_API ndt::type elwise_make_type(const ndt::callable_type *child_tp);

    template <int... I>
    callable forward_na(const callable &child) {
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
