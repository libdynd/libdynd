//
// Copyright (C) 2011-14 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#pragma once

#include <dynd/func/arrfunc.hpp>
#include <dynd/kernels/apply_kernels.hpp>

namespace dynd {
namespace nd {

  template <typename func_type, func_type func, typename... T>
  arrfunc make_apply_arrfunc(T &&... names)
  {
    typedef funcproto_for<func_type> funcproto_type;
    typedef kernels::apply_function_ck<
        func_type, func,
        kernels::args_of<funcproto_type>::type::size - sizeof...(T)> ck_type;

    return make_arrfunc(
        ndt::make_funcproto<funcproto_type>(std::forward<T>(names)...),
        &ck_type::instantiate);
  }

  template <typename R, typename... A, typename... T>
  arrfunc make_apply_arrfunc(R (*func)(A...), T &&... names)
  {
    typedef kernels::apply_callable_ck<R (*)(A...), sizeof...(A) - sizeof...(T)>
        ck_type;

    return make_arrfunc(ndt::make_funcproto<R(A...)>(names...),
                        &ck_type::instantiate, std::forward<R (*)(A...)>(func));
  }

  template <bool copy, typename func_type, typename... T>
  arrfunc make_apply_arrfunc(const func_type &func, T &&... names)
  {
    typedef funcproto_for<func_type> funcproto_type;
    typedef kernels::apply_callable_ck<
        func_type, kernels::args_of<funcproto_type>::type::size - sizeof...(T)>
        ck_type;

    return make_arrfunc(
        ndt::make_funcproto<funcproto_type>(std::forward<T>(names)...),
        &ck_type::instantiate, func);
  }

  template <typename func_type, typename... T>
  arrfunc make_apply_arrfunc(const func_type &func, T &&... names)
  {
    return make_apply_arrfunc<true>(func, std::forward<T>(names)...);
  }

  template <kernel_request_t kernreq, typename func_type, typename... K,
            typename... T>
  arrfunc make_apply_arrfunc(T &&... names)
  {
    typedef kernels::construct_then_apply_callable_ck<
        typename ckernel_builder_for<kernreq>::type, func_type,
        K...> ck_type;

    return make_arrfunc(
        ndt::make_funcproto<kernreq, funcproto_for<func_type, K...>>(
            std::forward<T>(names)...),
        &ck_type::instantiate);
  }

  template <typename func_type, typename... K, typename... T>
  arrfunc make_apply_arrfunc(T &&... names)
  {
    return make_apply_arrfunc<kernel_request_host, func_type, K...>(
        std::forward<T>(names)...);
  }
}
} // namespace dynd::nd
