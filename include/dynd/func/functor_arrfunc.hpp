//
// Copyright (C) 2011-14 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#pragma once

#include <dynd/func/arrfunc.hpp>
#include <dynd/func/funcproto.hpp>
#include <dynd/kernels/functor_kernels.hpp>
#include <dynd/types/arrfunc_type.hpp>

namespace dynd { namespace nd { namespace detail {

template <typename mem_func_type, bool copy>
class mem_func_wrapper;

template <typename T, typename R, typename... A>
class mem_func_wrapper<R (T::*)(A...) const, true>
{
  typedef R (T::*mem_func_type)(A...) const;

  T m_obj;
  mem_func_type m_mem_func;

public:
  mem_func_wrapper(const T &obj, mem_func_type mem_func)
    : m_obj(obj), m_mem_func(mem_func)
  {
  }

  R operator ()(A... a) const
  {
    return (m_obj.*m_mem_func)(a...);
  }
};

template <typename T, typename R, typename... A>
class mem_func_wrapper<R (T::*)(A...) const, false>
{
  typedef R (T::*mem_func_type)(A...) const;

  const T *m_obj;
  mem_func_type m_mem_func;

public:
  mem_func_wrapper(const T &obj, mem_func_type mem_func)
    : m_obj(&obj), m_mem_func(mem_func)
  {
  }

  R operator ()(A... a) const
  {
    return (m_obj->*m_mem_func)(a...);
  }
};

template <int aux_param_count, typename func_type, bool copy, bool func_or_func_pointer>
struct functor_arrfunc_from;

template <int aux_param_count, typename arrfunc_type, bool copy>
struct functor_arrfunc_from<aux_param_count, arrfunc_type *, copy, true> {
  typedef arrfunc_type *func_type;

  static nd::arrfunc make(const func_type &func)
  {
    nd::array af =
        nd::empty(ndt::make_funcproto<arrfunc_type>(aux_param_count));
    arrfunc_type_data *out_af =
        reinterpret_cast<arrfunc_type_data *>(af.get_readwrite_originptr());
    *out_af->get_data_as<func_type>() = func;
    out_af->instantiate = &functor_ck<func_type, arrfunc_type, aux_param_count,
                                      false>::instantiate;
    out_af->free_func = NULL;
    af.flag_as_immutable();
    return af;
  }
};

template <int aux_param_count, typename arrfunc_type, bool copy>
struct functor_arrfunc_from<aux_param_count, arrfunc_type, copy, true> {
  static nd::arrfunc make(arrfunc_type &func)
  {
    return functor_arrfunc_from<aux_param_count, arrfunc_type *, copy,
                                true>::make(&func);
  }
};

template <int aux_param_count, typename obj_type>
struct functor_arrfunc_from<aux_param_count, obj_type, true, false> {
  static nd::arrfunc make(const obj_type &obj)
  {
    return make(obj, &obj_type::operator());
  }

  template <typename func_type>
  static nd::arrfunc make(const obj_type &obj, func_type)
  {
    typedef typename funcproto_from<func_type>::type arrfunc_type;

    nd::array af =
        nd::empty(ndt::make_funcproto<arrfunc_type>(aux_param_count));
    arrfunc_type_data *out_af =
        reinterpret_cast<arrfunc_type_data *>(af.get_readwrite_originptr());
    *out_af->get_data_as<obj_type>() = obj;
    out_af->instantiate = &functor_ck<obj_type, arrfunc_type, aux_param_count,
                                      false>::instantiate;
    out_af->free_func = NULL;
    af.flag_as_immutable();
    return af;
  }
};

template <int aux_param_count, typename obj_type>
struct functor_arrfunc_from<aux_param_count, obj_type, false, false> {
  static nd::arrfunc make(const obj_type &obj)
  {
    return make(obj, &obj_type::operator());
  }

  template <typename func_type>
  static nd::arrfunc make(const obj_type &obj, func_type)
  {
    typedef typename funcproto_from<func_type>::type arrfunc_type;
    typedef func_wrapper<obj_type, arrfunc_type> wrapper_type;

    nd::array af =
        nd::empty(ndt::make_funcproto<arrfunc_type>(aux_param_count));
    arrfunc_type_data *out_af =
        reinterpret_cast<arrfunc_type_data *>(af.get_readwrite_originptr());
    *out_af->get_data_as<wrapper_type>() = wrapper_type(obj);
    out_af->instantiate = &functor_ck<wrapper_type, arrfunc_type,
                                      aux_param_count, false>::instantiate;
    out_af->free_func = NULL;
    af.flag_as_immutable();
    return af;
  }
};

template <typename arrfunc_type>
struct functor_arrfunc_factory;

#define MAKE(NSRC, NARG)                                                       \
  template <DYND_PP_JOIN_MAP_2(                                                \
      DYND_PP_META_TYPENAME, (, ),                                             \
      DYND_PP_APPEND(func_type, DYND_PP_META_NAME_RANGE(A, NSRC, NARG)))>      \
  static nd::arrfunc make()                                                    \
  {                                                                            \
    typedef R(arrfunc_type) DYND_PP_META_NAME_RANGE(A, NARG);                  \
                                                                               \
    nd::array af =                                                             \
        nd::empty(ndt::make_funcproto<arrfunc_type>(DYND_PP_SUB(NARG, NSRC))); \
    arrfunc_type_data *out_af =                                                \
        reinterpret_cast<arrfunc_type_data *>(af.get_readwrite_originptr());   \
    out_af->instantiate =                                                      \
        &functor_ck<func_type, arrfunc_type, DYND_PP_SUB(NARG, NSRC),          \
                    true>::instantiate;                                        \
    af.flag_as_immutable();                                                    \
    return af;                                                                 \
  }

#define FUNCTOR_ARRFUNC_FACTORY(NSRC)                                          \
  template <DYND_PP_JOIN_MAP_1(                                                \
      DYND_PP_META_TYPENAME, (, ),                                             \
      DYND_PP_PREPEND(R, DYND_PP_META_NAME_RANGE(A, NSRC)))>                   \
  struct functor_arrfunc_factory<R DYND_PP_META_NAME_RANGE(A, NSRC)> {         \
    DYND_PP_JOIN_ELWISE_1(                                                     \
        MAKE, (),                                                              \
        DYND_PP_REPEAT_1(NSRC, DYND_PP_INC(DYND_PP_SUB(DYND_ARG_MAX, NSRC))),  \
        DYND_PP_RANGE(NSRC, DYND_PP_INC(DYND_ARG_MAX)))                        \
  };

DYND_PP_JOIN_MAP(FUNCTOR_ARRFUNC_FACTORY, (),
                 DYND_PP_RANGE(DYND_PP_INC(DYND_ARG_MAX)))

#undef FUNCTOR_ARRFUNC_FACTORY

#undef MAKE

#define FUNCTOR_ARRFUNC_FACTORY_DISPATCHER_MAKE(NARG)                          \
  template <DYND_PP_JOIN_MAP_1(                                                \
      DYND_PP_META_TYPENAME, (, ),                                             \
      DYND_PP_CHAIN(DYND_PP_META_NAME_RANGE(A, NARG), (obj_type, func_type)))> \
  static nd::arrfunc make(func_type)                                           \
  {                                                                            \
    typedef typename funcproto_from<func_type>::type arrfunc_type;             \
                                                                               \
    return functor_arrfunc_factory<arrfunc_type>::template make<               \
        DYND_PP_JOIN_1((, ), DYND_PP_APPEND(obj_type, DYND_PP_META_NAME_RANGE( \
                                                          A, NARG)))>();       \
  }                                                                            \
                                                                               \
  template <DYND_PP_JOIN_MAP_1(                                                \
      DYND_PP_META_TYPENAME, (, ),                                             \
      DYND_PP_APPEND(obj_type, DYND_PP_META_NAME_RANGE(A, NARG)))>             \
  static nd::arrfunc make()                                                    \
  {                                                                            \
    return make<DYND_PP_JOIN_1(                                                \
        (, ), DYND_PP_APPEND(obj_type, DYND_PP_META_NAME_RANGE(A, NARG)))>(    \
        &obj_type::operator());                                                \
  }

struct functor_arrfunc_factory_dispatcher {
  DYND_PP_JOIN_MAP(FUNCTOR_ARRFUNC_FACTORY_DISPATCHER_MAKE, (),
                   DYND_PP_RANGE(DYND_PP_INC(DYND_ARG_MAX)))
};

#undef FUNCTOR_ARRFUNC_FACTORY_DISPATCHER

} // namespace detail

template <int aux_param_count, typename func_type>
nd::arrfunc make_functor_arrfunc(const func_type &func, bool copy = true)
{
  if (copy) {
    return detail::functor_arrfunc_from < aux_param_count, func_type, true,
           std::is_function<func_type>::value ||
               is_function_pointer<func_type>::value > ::make(func);
  }
  else {
    return detail::functor_arrfunc_from < aux_param_count, func_type, false,
           std::is_function<func_type>::value ||
               is_function_pointer<func_type>::value > ::make(func);
  }
}

namespace detail
{

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
    out_af->instantiate = &kernels::apply_ck<R (A...), func, R,
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



template <typename func_type>
nd::arrfunc make_functor_arrfunc(const func_type &func, bool copy = true)
{
  return make_functor_arrfunc<0>(func, copy);
}

/*
template <typename obj_type, typename mem_func_type>
nd::arrfunc make_functor_arrfunc(const obj_type &obj, mem_func_type mem_func,
                                 bool copy = true)
{
  if (copy) {
    typedef detail::mem_func_wrapper<mem_func_type, true> wrapper_type;
    return detail::functor_arrfunc_from<0, wrapper_type, true, false>::make(
        wrapper_type(obj, mem_func));
  }
  else {
    typedef detail::mem_func_wrapper<mem_func_type, false> wrapper_type;
    return detail::functor_arrfunc_from<0, wrapper_type, true, false>::make(
        wrapper_type(obj, mem_func));
  }
}
*/

}} // namespace dynd::nd
