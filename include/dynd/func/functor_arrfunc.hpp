//
// Copyright (C) 2011-14 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#pragma once

#include <dynd/func/arrfunc.hpp>
#include <dynd/kernels/functor_kernels.hpp>
#include <dynd/types/arrfunc_type.hpp>

namespace dynd { namespace nd { namespace detail {

template <typename func_type>
struct apply_arrfunc_factory;

template <typename R, typename... A>
struct apply_arrfunc_factory<R (A...)>
{
  typedef typename to<sizeof...(A) - 0, A...>::type args;
  typedef typename from<sizeof...(A) - 0, A...>::type kwds;

  template <R (func)(A...)>
  static nd::arrfunc make()
  {
    nd::array af = nd::empty(ndt::make_funcproto<R (A...)>());
    arrfunc_type_data *out_af = reinterpret_cast<arrfunc_type_data *>(af.get_readwrite_originptr());
    out_af->instantiate = &kernels::apply_ck<R (*)(A...), func, R,
      args, make_index_sequence<args::size>, kwds, make_index_sequence<kwds::size> >::instantiate;
    af.flag_as_immutable();

    return af;
  }

  static nd::arrfunc make(R (func)(A...))
  {
    return apply_arrfunc_factory<R (*)(A...)>::make(func);
  }
};

template <typename R, typename... A>
struct apply_arrfunc_factory<R (*)(A...)>
{
  typedef typename to<sizeof...(A) - 0, A...>::type args;
  typedef typename from<sizeof...(A) - 0, A...>::type kwds;

  template <R (*func)(A...)>
  static nd::arrfunc make()
  {
    nd::array af = nd::empty(ndt::make_funcproto<R (A...)>());
    arrfunc_type_data *out_af = reinterpret_cast<arrfunc_type_data *>(af.get_readwrite_originptr());
    out_af->instantiate = &kernels::apply_ck<R (*)(A...), func, R,
      args, make_index_sequence<args::size>, kwds, make_index_sequence<kwds::size> >::instantiate;
    af.flag_as_immutable();

    return af;
  }

  typedef R (funcproto_type)(A...);
  typedef funcproto_type *func_type;

  static nd::arrfunc make(func_type func)
  {
    nd::array af = nd::empty(ndt::make_funcproto<funcproto_type>());
    arrfunc_type_data *out_af = reinterpret_cast<arrfunc_type_data *>(af.get_readwrite_originptr());
    out_af->instantiate = &kernels::apply_callable_ck<func_type, R,
      type_sequence<A...>, make_index_sequence<sizeof...(A)>, type_sequence<>, make_index_sequence<0> >::instantiate;
    *out_af->get_data_as<func_type>() = func;
    af.flag_as_immutable();

    return af;
  }
};

template <typename func_type>
struct apply_arrfunc_factory
{
  template <typename R, typename... A>
  static nd::arrfunc make(R (func_type::*)(A...) const)
  {
    typedef R (funcproto_type)(A...);

    nd::array af = nd::empty(ndt::make_funcproto<funcproto_type>());
    arrfunc_type_data *out_af = reinterpret_cast<arrfunc_type_data *>(af.get_readwrite_originptr());
    out_af->instantiate = &kernels::construct_and_apply_callable_ck<func_type, R,
      type_sequence<A...>, make_index_sequence<sizeof...(A)>, type_sequence<>, make_index_sequence<0> >::instantiate;
    af.flag_as_immutable();
    return af;
  }

  template <typename... K>
  static nd::arrfunc make()
  {
    return make(&func_type::operator ());
  }

  static nd::arrfunc make(const func_type &func, bool copy)
  {
    return make(func, &func_type::operator (), copy);
  }

  template <typename R, typename... A>
  static nd::arrfunc make(const func_type &func, R (func_type::*)(A...) const, bool DYND_UNUSED(copy))
  {
    typedef R (funcproto_type)(A...);

    nd::array af = nd::empty(ndt::make_funcproto<funcproto_type>());
    arrfunc_type_data *out_af = reinterpret_cast<arrfunc_type_data *>(af.get_readwrite_originptr());
    *out_af->get_data_as<func_type>() = func;
    out_af->instantiate = &kernels::apply_callable_ck<func_type, R,
      type_sequence<A...>, make_index_sequence<sizeof...(A)>, type_sequence<>, make_index_sequence<0> >::instantiate;
    af.flag_as_immutable();
    return af;
  }
};

} // detail

template <typename func_type, func_type func>
nd::arrfunc make_apply_arrfunc()
{
  return detail::apply_arrfunc_factory<func_type>::template make<func>();
}

template <typename func_type>
typename std::enable_if<std::is_function<func_type>::value || is_function_pointer<func_type>::value,
  nd::arrfunc>::type make_apply_arrfunc(const func_type &func)
{
  return detail::apply_arrfunc_factory<func_type>::make(func);
}

template <typename func_type>
typename std::enable_if<!std::is_function<func_type>::value && !is_function_pointer<func_type>::value,
  nd::arrfunc>::type make_apply_arrfunc(const func_type &func, bool copy = true)
{
  return detail::apply_arrfunc_factory<func_type>::make(func, copy);    
}

template <typename func_type, typename... K>
nd::arrfunc make_apply_arrfunc()
{
  return detail::apply_arrfunc_factory<func_type>::template make<K...>();
}

}} // namespace dynd::nd
