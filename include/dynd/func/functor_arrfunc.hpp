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
#include <tr1/type_traits>

namespace dynd { namespace nd { namespace detail {

template <typename func_type, typename funcproto_type>
class func_wrapper;

#define FUNC_WRAPPER(N) \
  template <typename func_type, typename R, DYND_PP_JOIN_MAP_1(DYND_PP_META_TYPENAME, (,), DYND_PP_META_NAME_RANGE(A, N))> \
  class func_wrapper<func_type, R DYND_PP_META_NAME_RANGE(A, N)> { \
    func_type *m_func; \
\
  public: \
    func_wrapper() : m_func(NULL) { \
    } \
\
    func_wrapper(const func_wrapper &other) { \
        m_func = reinterpret_cast<func_type *>(malloc(sizeof(func_type))); \
        DYND_MEMCPY(m_func, other.m_func, sizeof(func_type)); \
    } \
\
    func_wrapper(func_type func) { \
        m_func = reinterpret_cast<func_type *>(malloc(sizeof(func_type))); \
        DYND_MEMCPY(m_func, &func, sizeof(func_type)); \
    } \
\
    ~func_wrapper() { \
        if (m_func != NULL) { \
            free(m_func); \
        } \
    } \
\
    R operator() DYND_PP_ELWISE_1(DYND_PP_META_DECL, DYND_PP_META_NAME_RANGE(A, N), DYND_PP_META_NAME_RANGE(a, N)) { \
        return (*m_func) DYND_PP_META_NAME_RANGE(a, N); \
    } \
\
    func_wrapper &operator =(const func_wrapper &other) { \
        if(this == &other) { \
            return *this; \
        } \
\
        m_func = reinterpret_cast<func_type *>(malloc(sizeof(func_type))); \
        DYND_MEMCPY(m_func, other.m_func, sizeof(func_type)); \
        return *this; \
    } \
  };

DYND_PP_JOIN_MAP(FUNC_WRAPPER, (), DYND_PP_RANGE(1, DYND_PP_INC(DYND_SRC_MAX)))

#undef FUNC_WRAPPER

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
    mem_func_wrapper() : m_obj(NULL), m_mem_func(NULL) { \
    } \
\
    mem_func_wrapper(const T &obj, mem_func_type mem_func) : m_obj(obj), m_mem_func(mem_func) { \
    } \
\
    R operator() DYND_PP_ELWISE_1(DYND_PP_META_DECL, DYND_PP_META_NAME_RANGE(A, N), DYND_PP_META_NAME_RANGE(a, N)) { \
      return (m_obj.*m_mem_func)DYND_PP_META_NAME_RANGE(a, N); \
    } \
  };

DYND_PP_JOIN_MAP(MEM_FUNC_WRAPPER, (), DYND_PP_RANGE(1, DYND_PP_INC(DYND_SRC_MAX)))

#undef MEM_FUNC_WRAPPER

template <typename func_type, bool func_or_func_pointer>
struct functor_arrfunc_from;

template <typename funcproto_type>
struct functor_arrfunc_from<funcproto_type *, true> {
    typedef funcproto_type *func_type;

    static void make(func_type func, arrfunc_type_data *out_af) {
        out_af->func_proto = ndt::make_funcproto<funcproto_type>();
        *out_af->get_data_as<func_type>() = func;
        out_af->instantiate = &nd::functor_ck<func_type, funcproto_type>::instantiate;
        out_af->free_func = NULL;
    }
};

template <typename funcproto_type>
struct functor_arrfunc_from<funcproto_type, true> {
    static void make(funcproto_type func, arrfunc_type_data *out_af) {
        functor_arrfunc_from<funcproto_type *, true>::make(&func, out_af);
    }
};

template <typename obj_type>
struct functor_arrfunc_from<obj_type, false> {
    static void make(obj_type obj, arrfunc_type_data *out_af) {
        make(obj, &obj_type::operator(), out_af);
    }

    template <typename func_type>
    static void make(obj_type obj, func_type, arrfunc_type_data *out_af) {
        typedef typename func_like<func_type>::type funcproto_type;
        typedef func_wrapper<obj_type, funcproto_type> wrapper_type;

        out_af->func_proto = ndt::make_funcproto<funcproto_type>();
        *out_af->get_data_as<wrapper_type>() = wrapper_type(obj);
        out_af->instantiate = &nd::functor_ck<wrapper_type, funcproto_type>::instantiate;
        out_af->free_func = NULL;
    }
};

} // namespace detail

template <typename func_type>
void make_functor_arrfunc(func_type func, arrfunc_type_data *out_af) {
    detail::functor_arrfunc_from<func_type,
        std::tr1::is_function<func_type>::value || is_function_pointer<func_type>::value>::make(func, out_af);
}

template <typename obj_type, typename mem_func_type>
void make_functor_arrfunc(obj_type obj, mem_func_type mem_func, arrfunc_type_data *out_af) {
    make_functor_arrfunc(detail::mem_func_wrapper<mem_func_type>(obj, mem_func), out_af);
}

template <typename func_type>
arrfunc make_functor_arrfunc(func_type func) {
    array af = empty(ndt::make_arrfunc());
    make_functor_arrfunc(func, reinterpret_cast<arrfunc_type_data *>(af.get_readwrite_originptr()));
    af.flag_as_immutable();
    return af;
}

template <typename obj_type, typename mem_func_type>
arrfunc make_functor_arrfunc(obj_type obj, mem_func_type mem_func) {
    return make_functor_arrfunc(detail::mem_func_wrapper<mem_func_type>(obj, mem_func));
}

}} // namespace dynd::nd

#endif // DYND__FUNC_FUNCTOR_ARRFUNC_HPP
