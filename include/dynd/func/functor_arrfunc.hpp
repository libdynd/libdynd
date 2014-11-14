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

template <typename func_type, typename arrfunc_type>
class func_wrapper;

#define FUNC_WRAPPER(N) \
  template <typename func_type, typename R, DYND_PP_JOIN_MAP_1(DYND_PP_META_TYPENAME, (,), DYND_PP_META_NAME_RANGE(A, N))> \
  class func_wrapper<func_type, R DYND_PP_META_NAME_RANGE(A, N)> { \
    const func_type *m_func; \
\
  public: \
    func_wrapper(const func_type &func) : m_func(&func) { \
    } \
\
    R operator() DYND_PP_ELWISE_1(DYND_PP_META_DECL, DYND_PP_META_NAME_RANGE(A, N), DYND_PP_META_NAME_RANGE(a, N)) const { \
        return (*m_func) DYND_PP_META_NAME_RANGE(a, N); \
    } \
  };

DYND_PP_JOIN_MAP(FUNC_WRAPPER, (), DYND_PP_RANGE(1, DYND_PP_INC(DYND_ARG_MAX)))

#undef FUNC_WRAPPER

template <typename mem_func_type, bool copy>
class mem_func_wrapper;

#define MEM_FUNC_WRAPPER(N)                                                    \
  template <typename T, typename R,                                            \
            DYND_PP_JOIN_MAP_1(DYND_PP_META_TYPENAME, (, ),                    \
                               DYND_PP_META_NAME_RANGE(A, N))>                 \
  class mem_func_wrapper<R(T::*) DYND_PP_META_NAME_RANGE(A, N) const, true> {  \
    typedef R(T::*mem_func_type) DYND_PP_META_NAME_RANGE(A, N) const;          \
                                                                               \
    T m_obj;                                                                   \
    mem_func_type m_mem_func;                                                  \
                                                                               \
  public:                                                                      \
    mem_func_wrapper(const T &obj, mem_func_type mem_func)                     \
        : m_obj(obj), m_mem_func(mem_func)                                     \
    {                                                                          \
    }                                                                          \
                                                                               \
    R operator()                                                               \
        DYND_PP_ELWISE_1(DYND_PP_META_DECL, DYND_PP_META_NAME_RANGE(A, N),     \
                         DYND_PP_META_NAME_RANGE(a, N)) const                  \
    {                                                                          \
      return (m_obj.*m_mem_func)DYND_PP_META_NAME_RANGE(a, N);                 \
    }                                                                          \
  };                                                                           \
                                                                               \
  template <typename T, typename R,                                            \
            DYND_PP_JOIN_MAP_1(DYND_PP_META_TYPENAME, (, ),                    \
                               DYND_PP_META_NAME_RANGE(A, N))>                 \
  class mem_func_wrapper<R(T::*) DYND_PP_META_NAME_RANGE(A, N) const, false> { \
    typedef R(T::*mem_func_type) DYND_PP_META_NAME_RANGE(A, N) const;          \
                                                                               \
    const T *m_obj;                                                            \
    mem_func_type m_mem_func;                                                  \
                                                                               \
  public:                                                                      \
    mem_func_wrapper(const T &obj, mem_func_type mem_func)                     \
        : m_obj(&obj), m_mem_func(mem_func)                                    \
    {                                                                          \
    }                                                                          \
                                                                               \
    R operator()                                                               \
        DYND_PP_ELWISE_1(DYND_PP_META_DECL, DYND_PP_META_NAME_RANGE(A, N),     \
                         DYND_PP_META_NAME_RANGE(a, N)) const                  \
    {                                                                          \
      return (m_obj->*m_mem_func)DYND_PP_META_NAME_RANGE(a, N);                \
    }                                                                          \
  };

DYND_PP_JOIN_MAP(MEM_FUNC_WRAPPER, (), DYND_PP_RANGE(1, DYND_PP_INC(DYND_ARG_MAX)))

#undef MEM_FUNC_WRAPPER

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

template <typename func_type>
nd::arrfunc make_functor_arrfunc(const func_type &func, bool copy = true)
{
  return make_functor_arrfunc<0>(func, copy);
}

#define MAKE_FUNCTOR_ARRFUNC(NAUX)                                             \
  template <DYND_PP_JOIN_MAP_1(                                                \
      DYND_PP_META_TYPENAME, (, ),                                             \
      DYND_PP_APPEND(func_type, DYND_PP_META_NAME_RANGE(A, NAUX)))>            \
  nd::arrfunc make_functor_arrfunc()                                           \
  {                                                                            \
    return detail::functor_arrfunc_factory_dispatcher::template make<          \
        DYND_PP_JOIN((, ), DYND_PP_APPEND(func_type, DYND_PP_META_NAME_RANGE(  \
                                                         A, NAUX)))>();        \
  }

DYND_PP_JOIN_MAP(MAKE_FUNCTOR_ARRFUNC, (),
                 DYND_PP_RANGE(DYND_PP_INC(DYND_ARG_MAX)))

#undef MAKE_FUNCTOR_ARRFUNC

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

}} // namespace dynd::nd
