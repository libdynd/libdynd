//
// Copyright (C) 2011-14 Mark Wiebe, Irwin Zaid, DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#ifndef DYND__FUNC_FUNCTOR_ARRFUNC_HPP
#define DYND__FUNC_FUNCTOR_ARRFUNC_HPP

#include <dynd/buffer.hpp>
#include <dynd/kernels/functor_kernels.hpp>
#include <dynd/types/funcproto_type.hpp>
#include <dynd/types/cfixed_dim_type.hpp>
#include <dynd/func/arrfunc.hpp>
#include <dynd/func/make_callable.hpp>
#include <dynd/pp/arrfunc_util.hpp>
#include <iostream>

template<typename testType>
struct is_function_pointer
{
    static const bool value =
        std::tr1::is_pointer<testType>::value ?
        std::tr1::is_function<typename std::tr1::remove_pointer<testType>::type>::value :
        false;
};

namespace dynd { namespace nd { namespace detail {

template <typename func_type, bool func_pointer>
struct functor_arrfunc_from;

template <typename func_type>
struct functor_arrfunc_from<func_type, true> {
  static void make(func_type func, arrfunc_type_data *out_af) {
    out_af->func_proto = ndt::make_funcproto<func_type>();
    *out_af->get_data_as<func_type>() = func;
    out_af->instantiate = &dynd::nd::functor_ckernel<func_type>::instantiate;
    out_af->free_func = NULL;
  }
};

template <typename T>
struct functor_arrfunc_from<T, false> {
  template <typename func_type>
  static void make(T func, func_type, arrfunc_type_data *out_af) {
    typedef typename func_like<func_type>::type funcproto_type;
    out_af->func_proto = ndt::make_funcproto<funcproto_type>();
    *out_af->get_data_as<T>() = func;
    out_af->instantiate = &dynd::nd::functor_ckernel<T, funcproto_type>::instantiate;
    out_af->free_func = NULL;
  }

  static void make(T func, arrfunc_type_data *out_af) {
    make(func, &T::operator(), out_af);
  }
};

} // namespace detail

template <typename func_type>
void make_functor_arrfunc(func_type func, arrfunc_type_data *out_af) {
  detail::functor_arrfunc_from<func_type, is_function_pointer<func_type>::value>::make(func, out_af);
}

template <typename T, typename func_type>
void make_functor_arrfunc(T DYND_UNUSED(obj), func_type func, arrfunc_type_data *out_af) {
  out_af->func_proto = ndt::make_funcproto<func_type>();
  *out_af->get_data_as<func_type>() = func;
  out_af->instantiate = &dynd::nd::functor_ckernel<func_type>::instantiate;
  out_af->free_func = NULL;
}

template <typename func_type>
nd::arrfunc make_functor_arrfunc(func_type func) {
  nd::array af = nd::empty(ndt::make_arrfunc());
  make_functor_arrfunc(func, reinterpret_cast<arrfunc_type_data *>(af.get_readwrite_originptr()));
  af.flag_as_immutable();
  return af;
}

template <typename T, typename func_type>
nd::arrfunc make_functor_arrfunc(T obj, func_type func) {
  nd::array af = nd::empty(ndt::make_arrfunc());
  make_functor_arrfunc(obj, func, reinterpret_cast<arrfunc_type_data *>(af.get_readwrite_originptr()));
  af.flag_as_immutable();
  return af;
}

}} // namespace dynd::nd

#endif // DYND__FUNC_FUNCTOR_ARRFUNC_HPP
