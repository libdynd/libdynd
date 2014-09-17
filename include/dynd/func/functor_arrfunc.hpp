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

namespace dynd { namespace nd { namespace detail {
  /**
   * Metaprogram to tell whether an argument is const or by value.
   */
  template<typename T>
  struct is_suitable_input {
    // TODO: Reenable - error was triggering when not expected
    enum { value = true }; //is_const<T>::value || !is_reference<T>::value };
  };
}} // namespace nd::detail

} // namespace dynd

namespace dynd { namespace nd { namespace detail {

  template <typename func_type, bool callable>
  struct functor_arrfunc_from;

  template <typename func_type>
  struct functor_arrfunc_from<func_type, false> {
    static void make(func_type func, arrfunc_type_data *out_af) {
      out_af->func_proto = ndt::make_funcproto<func_type>();
      *out_af->get_data_as<func_type>() = func;
      out_af->instantiate = &dynd::nd::functor_ckernel<func_type, func_type>::instantiate;
      out_af->free_func = NULL;
    }
  };

  template <typename Functor>
  struct functor_arrfunc_from<Functor, true> {
  template <typename func_type>
  inline static void make_tagged(Functor func, arrfunc_type_data *out_af, func_type)
  {
    out_af->func_proto = ndt::make_funcproto<func_type>();
    *out_af->get_data_as<Functor>() = func;
    out_af->instantiate = &dynd::nd::functor_ckernel<Functor, typename std::decay<typename func_like<func_type>::type>::type>::instantiate;
    out_af->free_func = NULL;
  }

  static void make(Functor func, arrfunc_type_data *out_af)
  {
    make_tagged(func, out_af, &Functor::operator());
  }
};

} // namespace detail

template <class O>
void make_functor_arrfunc(O obj, arrfunc_type_data *out_af)
{
  detail::functor_arrfunc_from<O, true>::make(obj, out_af);
}

template <class O>
nd::arrfunc make_functor_arrfunc(O obj)
{
  nd::array af = nd::empty(ndt::make_arrfunc());
  make_functor_arrfunc(obj, reinterpret_cast<arrfunc_type_data *>(af.get_readwrite_originptr()));
  af.flag_as_immutable();
  return af;
}

#define MAKE_FUNCTOR_ARRFUNC(N) _MAKE_FUNCTOR_ARRFUNC(N, DYND_PP_META_NAME_RANGE(A, N))
#define _MAKE_FUNCTOR_ARRFUNC(N, ARG_TYPENAMES) \
  template <typename R, DYND_PP_JOIN_MAP_1(DYND_PP_META_TYPENAME, (,), ARG_TYPENAMES)> \
  void make_functor_arrfunc(R (*func) ARG_TYPENAMES, arrfunc_type_data *out_af) \
  { \
    typedef R (*func_type) ARG_TYPENAMES; \
    detail::functor_arrfunc_from<func_type, false>::make(func, out_af); \
  } \
\
  template <typename R, DYND_PP_JOIN_MAP_1(DYND_PP_META_TYPENAME, (,), ARG_TYPENAMES)> \
  nd::arrfunc make_functor_arrfunc(R (*func) ARG_TYPENAMES) \
  { \
    nd::array af = nd::empty(ndt::make_arrfunc()); \
    make_functor_arrfunc(func, reinterpret_cast<arrfunc_type_data *>(af.get_readwrite_originptr())); \
    af.flag_as_immutable(); \
    return af; \
  }

DYND_PP_JOIN_MAP(MAKE_FUNCTOR_ARRFUNC, (), DYND_PP_RANGE(1, DYND_PP_INC(DYND_ARG_MAX)))

#undef _MAKE_FUNCTOR_ARRFUNC
#undef MAKE_FUNCTOR_ARRFUNC

#define MAKE_FUNCTOR_ARRFUNC(N) _MAKE_FUNCTOR_ARRFUNC(N, DYND_PP_META_NAME_RANGE(A, N))
#define _MAKE_FUNCTOR_ARRFUNC(N, ARG_TYPENAMES) \
  template <typename O, typename R, DYND_PP_JOIN_MAP_1(DYND_PP_META_TYPENAME, (,), ARG_TYPENAMES)> \
  void make_functor_arrfunc(const O &DYND_UNUSED(obj), R (O::*func) ARG_TYPENAMES, arrfunc_type_data *out_af) \
  { \
    typedef R (*func_type) ARG_TYPENAMES; \
    detail::functor_arrfunc_from<func_type, false>::make(func, out_af); \
  } \
\
  template <class O, typename R, DYND_PP_JOIN_MAP_1(DYND_PP_META_TYPENAME, (,), ARG_TYPENAMES)> \
  nd::arrfunc make_functor_arrfunc(const O obj, R (O::*func) ARG_TYPENAMES) \
  { \
    nd::array af = nd::empty(ndt::make_arrfunc()); \
    make_functor_arrfunc(obj, func, reinterpret_cast<arrfunc_type_data *>(af.get_readwrite_originptr())); \
    af.flag_as_immutable(); \
    return af; \
  }

DYND_PP_JOIN_MAP(MAKE_FUNCTOR_ARRFUNC, (), DYND_PP_RANGE(1, DYND_PP_INC(DYND_ARG_MAX)))

#undef _MAKE_FUNCTOR_ARRFUNC
#undef MAKE_FUNCTOR_ARRFUNC

}} // namespace dynd::nd

#endif // DYND__FUNC_FUNCTOR_ARRFUNC_HPP
