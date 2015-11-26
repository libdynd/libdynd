//
// Copyright (C) 2011-15 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#pragma once

#include <dynd/func/callable.hpp>
#include <dynd/kernels/apply.hpp>

namespace dynd {
namespace nd {
  namespace functional {
    /**
     * Makes an callable out of function ``func``, using the provided keyword
     * parameter names. This function takes ``func`` as a template
     * parameter, so can call it efficiently.
     */
    template <kernel_request_t kernreq, typename func_type, func_type func, typename... T>
    callable apply(T &&... names)
    {
      typedef as_apply_function_ck<func_type, func, arity_of<func_type>::value - sizeof...(T)> CKT;

      ndt::type self_tp = ndt::type::make<typename funcproto_of<func_type>::type>(std::forward<T>(names)...);

      return callable::make<CKT>(self_tp);
    }

    template <typename func_type, func_type func, typename... T>
    callable apply(T &&... names)
    {
      return apply<kernel_request_host, func_type, func>(std::forward<T>(names)...);
    }

    /**
     * Makes an callable out of the function object ``func``, using the provided
     * keyword parameter names. This version makes a copy of provided ``func``
     * object.
     */
    template <kernel_request_t kernreq, typename func_type, typename... T>
    typename std::enable_if<!is_function_pointer<func_type>::value, callable>::type apply(func_type func, T &&... names)
    {
      typedef as_apply_callable_ck<func_type, arity_of<func_type>::value - sizeof...(T)> ck_type;

      ndt::type self_tp = ndt::type::make<typename funcproto_of<func_type>::type>(std::forward<T>(names)...);

      return callable::make<ck_type>(self_tp, func);
    }

    template <typename func_type, typename... T>
    typename std::enable_if<!is_function_pointer<func_type>::value, callable>::type apply(func_type func, T &&... names)
    {
      static_assert(all_char_string_params<T...>::value, "All the names must be strings");
      return apply<kernel_request_host>(func, std::forward<T>(names)...);
    }

    template <kernel_request_t kernreq, typename func_type, typename... T>
    callable apply(func_type *func, T &&... names)
    {
      typedef as_apply_callable_ck<func_type *, arity_of<func_type>::value - sizeof...(T)> ck_type;

      ndt::type self_tp = ndt::type::make<typename funcproto_of<func_type>::type>(std::forward<T>(names)...);

      return callable::make<ck_type>(self_tp, func);
    }

    template <typename func_type, typename... T>
    callable apply(func_type *func, T &&... names)
    {
      return apply<kernel_request_host>(func, std::forward<T>(names)...);
    }

    template <kernel_request_t kernreq, typename T, typename R, typename... A, typename... S>
    callable apply(T *obj, R (T::*mem_func)(A...), S &&... names)
    {
      typedef as_apply_member_function_ck<T *, R (T::*)(A...), sizeof...(A) - sizeof...(S)> ck_type;

      ndt::type self_tp = ndt::type::make<typename funcproto_of<R (T::*)(A...)>::type>(std::forward<S>(names)...);

      return callable::make<ck_type>(self_tp, typename ck_type::data_type(obj, mem_func));
    }

    template <typename O, typename R, typename... A, typename... T>
    callable apply(O *obj, R (O::*mem_func)(A...), T &&... names)
    {
      return apply<kernel_request_host>(obj, mem_func, std::forward<T>(names)...);
    }

    /**
     * Makes an callable out of the provided function object type, specialized
     * for a memory_type such as cuda_device based on the ``kernreq``.
     */
    template <kernel_request_t kernreq, typename func_type, typename... K, typename... T>
    callable apply(T &&... names)
    {
      typedef as_construct_then_apply_callable_ck<func_type, K...> ck_type;

      ndt::type self_tp = ndt::type::make<typename funcproto_of<func_type, K...>::type>(std::forward<T>(names)...);

      return callable::make<ck_type>(self_tp);
    }

    /**
     * Makes an callable out of the provided function object type, which
     * constructs and calls the function object on demand.
     */
    template <typename func_type, typename... K, typename... T>
    callable apply(T &&... names)
    {
      return apply<kernel_request_host, func_type, K...>(std::forward<T>(names)...);
    }
  } // namespace dynd::nd::functional
} // namespace dynd::nd
} // namespace dynd
