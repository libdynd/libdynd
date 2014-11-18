//
// Copyright (C) 2011-14 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#pragma once

#include <dynd/func/arrfunc.hpp>
#include <dynd/kernels/apply_kernels.hpp>
#include <dynd/types/arrfunc_type.hpp>

namespace dynd { namespace nd { namespace detail {

template <typename func_type>
struct apply_arrfunc_factory;

template <typename R, typename... A>
struct apply_arrfunc_factory<R (*)(A...)>
{
  template <R (*func)(A...), typename... T>
  static nd::arrfunc make(T... names)
  {
    typedef typename to<sizeof...(A) - sizeof...(T), A...>::type args;
    typedef typename from<sizeof...(A) - sizeof...(T), A...>::type kwds;

    nd::array af = nd::empty(ndt::make_funcproto<R (A...)>(names...));
    arrfunc_type_data *out_af = reinterpret_cast<arrfunc_type_data *>(af.get_readwrite_originptr());
    out_af->instantiate = &kernels::apply_ck<R (*)(A...), func, R,
      args, make_index_sequence<args::size>, kwds, make_index_sequence<kwds::size> >::instantiate;
    af.flag_as_immutable();

    return af;
  }

  typedef R (funcproto_type)(A...);
  typedef funcproto_type *func_type;

  template <typename... T>
  static nd::arrfunc make(func_type func, T... names)
  {
    typedef typename to<sizeof...(A) - sizeof...(T), A...>::type args;
    typedef typename from<sizeof...(A) - sizeof...(T), A...>::type kwds;

    nd::array af = nd::empty(ndt::make_funcproto<R (A...)>(names...));
    arrfunc_type_data *out_af = reinterpret_cast<arrfunc_type_data *>(af.get_readwrite_originptr());
    out_af->instantiate = &kernels::apply_callable_ck<func_type, R,
      args, make_index_sequence<args::size>, kwds, make_index_sequence<kwds::size> >::instantiate;
    *out_af->get_data_as<func_type>() = func;
    af.flag_as_immutable();

    return af;
  }
};

template <typename func_type>
struct apply_arrfunc_factory
{
  template <typename... K, typename R, typename... A, typename... T>
  static nd::arrfunc make(R (func_type::*)(A...) const, T &&... names)
  {
    typedef type_sequence<A...> args;
    typedef type_sequence<K...> kwds;

    nd::array af = nd::empty(ndt::make_funcproto<R (A..., K...)>(std::forward<T>(names)...));
    arrfunc_type_data *out_af = reinterpret_cast<arrfunc_type_data *>(af.get_readwrite_originptr());
    out_af->instantiate = &kernels::construct_and_apply_callable_ck<func_type, R,
      args, make_index_sequence<args::size>, kwds, make_index_sequence<kwds::size> >::instantiate;
    af.flag_as_immutable();
    return af;
  }

  template <typename... K, typename... T>
  static nd::arrfunc make(T &&... names)
  {
    return make<K...>(&func_type::operator (), std::forward<T>(names)...);
  }

  template <typename... T>
  static nd::arrfunc make(const func_type &func, T &&... names)
  {
    return make(func, &func_type::operator (), std::forward<T>(names)...);
  }

  template <typename R, typename... A, typename... T>
  static nd::arrfunc make(const func_type &func, R (func_type::*)(A...) const, T &&... names)
  {
    typedef typename to<sizeof...(A) - sizeof...(T), A...>::type args;
    typedef typename from<sizeof...(A) - sizeof...(T), A...>::type kwds;

    nd::array af = nd::empty(ndt::make_funcproto<R (A...)>(std::forward<T>(names)...));
    arrfunc_type_data *out_af = reinterpret_cast<arrfunc_type_data *>(af.get_readwrite_originptr());
    *out_af->get_data_as<func_type>() = func;
    out_af->instantiate = &kernels::apply_callable_ck<func_type, R,
      args, make_index_sequence<args::size>, kwds, make_index_sequence<kwds::size> >::instantiate;
    af.flag_as_immutable();
    return af;
  }
};

} // detail

template <typename func_type, func_type func, typename... T>
nd::arrfunc make_apply_arrfunc(T &&... names)
{
  return detail::apply_arrfunc_factory<func_type>::template make<func>(std::forward<T>(names)...);
}

template <typename func_type, typename... T>
typename std::enable_if<is_function_pointer<func_type>::value, nd::arrfunc>::type 
  make_apply_arrfunc(func_type func, T &&... names)
{
  return detail::apply_arrfunc_factory<func_type>::make(func, std::forward<T>(names)...);
}

template <typename func_type, typename... T>
typename std::enable_if<std::is_function<func_type>::value, nd::arrfunc>::type
  make_apply_arrfunc(func_type func, T &&... names)
{
  return make_apply_arrfunc(&func, std::forward<T>(names)...);
}

template <bool copy = true, typename func_type, typename... T>
typename std::enable_if<!std::is_function<func_type>::value && !is_function_pointer<func_type>::value,
  nd::arrfunc>::type make_apply_arrfunc(const func_type &func, T &&... names)
{
  return detail::apply_arrfunc_factory<func_type>::make(func, std::forward<T>(names)...);
}

template <typename func_type, typename... K, typename... T>
nd::arrfunc make_apply_arrfunc(T &&... names)
{
  return detail::apply_arrfunc_factory<func_type>::template make<K...>(std::forward<T>(names)...);
}

}} // namespace dynd::nd
