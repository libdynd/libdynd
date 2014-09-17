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
#include <dynd/pp/meta.hpp>
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

template <typename mem_func_type>
class mem_func_wrapper;

#define MEM_FUNC_WRAPPER(N) \
  template <typename T, typename R, DYND_PP_JOIN_MAP_1(DYND_PP_META_TYPENAME, (,), DYND_PP_META_NAME_RANGE(A, N))> \
  class mem_func_wrapper<R (T::*)DYND_PP_META_NAME_RANGE(A, N)> { \
    typedef R (T::*mem_func_type)DYND_PP_META_NAME_RANGE(A, N); \
\
    T m_obj; \
    mem_func_type m_mem_func; \
\
  public: \
    mem_func_wrapper(T obj, mem_func_type mem_func) : m_obj(obj), m_mem_func(mem_func) { \
    } \
\
    R operator() DYND_PP_ELWISE_1(DYND_PP_META_DECL, DYND_PP_META_NAME_RANGE(A, N), DYND_PP_META_NAME_RANGE(a, N)) { \
      return (m_obj.*m_mem_func)DYND_PP_META_NAME_RANGE(a, N); \
    } \
  };

DYND_PP_JOIN_MAP(MEM_FUNC_WRAPPER, (), DYND_PP_RANGE(1, DYND_PP_INC(DYND_SRC_MAX)))

#undef MEM_FUNC_WRAPPER

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

template <typename T, typename mem_func_type>
void make_functor_arrfunc(T obj, mem_func_type mem_func, arrfunc_type_data *out_af) {
  make_functor_arrfunc(detail::mem_func_wrapper<mem_func_type>(obj, mem_func), out_af);
}

template <typename func_type>
nd::arrfunc make_functor_arrfunc(func_type func) {
  nd::array af = nd::empty(ndt::make_arrfunc());
  make_functor_arrfunc(func, reinterpret_cast<arrfunc_type_data *>(af.get_readwrite_originptr()));
  af.flag_as_immutable();
  return af;
}

template <typename T, typename mem_func_type>
nd::arrfunc make_functor_arrfunc(T obj, mem_func_type mem_func) {
  return make_functor_arrfunc(detail::mem_func_wrapper<mem_func_type>(obj, mem_func));
}

}} // namespace dynd::nd

#endif // DYND__FUNC_FUNCTOR_ARRFUNC_HPP
