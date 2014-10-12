//
// Copyright (C) 2011-14 Mark Wiebe, Irwin Zaid, DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#ifndef DYND__FUNC_FUNCTOR_ARRFUNC_HPP
#define DYND__FUNC_FUNCTOR_ARRFUNC_HPP

#include <dynd/func/arrfunc.hpp>
#include <dynd/kernels/functor_kernels.hpp>
#include <dynd/types/funcproto_type.hpp>

namespace dynd { namespace nd { namespace detail {

template <typename func_type, typename funcproto_type>
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

#define MEM_FUNC_WRAPPER(N) \
  template <typename T, typename R, DYND_PP_JOIN_MAP_1(DYND_PP_META_TYPENAME, (,), DYND_PP_META_NAME_RANGE(A, N))> \
  class mem_func_wrapper<R (T::*) DYND_PP_META_NAME_RANGE(A, N) const, true> { \
    typedef R (T::*mem_func_type) DYND_PP_META_NAME_RANGE(A, N) const; \
\
    T m_obj; \
    mem_func_type m_mem_func; \
\
  public: \
    mem_func_wrapper(const T &obj, mem_func_type mem_func) : m_obj(obj), m_mem_func(mem_func) { \
    } \
\
    R operator() DYND_PP_ELWISE_1(DYND_PP_META_DECL, DYND_PP_META_NAME_RANGE(A, N), DYND_PP_META_NAME_RANGE(a, N)) const { \
      return (m_obj.*m_mem_func)DYND_PP_META_NAME_RANGE(a, N); \
    } \
  }; \
\
  template <typename T, typename R, DYND_PP_JOIN_MAP_1(DYND_PP_META_TYPENAME, (,), DYND_PP_META_NAME_RANGE(A, N))> \
  class mem_func_wrapper<R (T::*) DYND_PP_META_NAME_RANGE(A, N) const, false> { \
    typedef R (T::*mem_func_type) DYND_PP_META_NAME_RANGE(A, N) const; \
\
    const T *m_obj; \
    mem_func_type m_mem_func; \
\
  public: \
    mem_func_wrapper(const T &obj, mem_func_type mem_func) : m_obj(&obj), m_mem_func(mem_func) { \
    } \
\
    R operator() DYND_PP_ELWISE_1(DYND_PP_META_DECL, DYND_PP_META_NAME_RANGE(A, N), DYND_PP_META_NAME_RANGE(a, N)) const { \
      return (m_obj->*m_mem_func)DYND_PP_META_NAME_RANGE(a, N); \
    } \
  };

DYND_PP_JOIN_MAP(MEM_FUNC_WRAPPER, (), DYND_PP_RANGE(1, DYND_PP_INC(DYND_ARG_MAX)))

#undef MEM_FUNC_WRAPPER

template <typename func_type, bool copy, bool func_or_func_pointer>
struct functor_arrfunc_from;

template <typename funcproto_type, bool copy>
struct functor_arrfunc_from<funcproto_type *, copy, true> {
    typedef funcproto_type *func_type;

    static void make(const func_type &func, arrfunc_type_data *out_af) {
        out_af->func_proto = ndt::make_funcproto<funcproto_type>();
        *out_af->get_data_as<func_type>() = func;
        out_af->instantiate = &nd::functor_ck<func_type, funcproto_type>::instantiate;
        out_af->free_func = NULL;
    }
};

template <typename funcproto_type, bool copy>
struct functor_arrfunc_from<funcproto_type, copy, true> {
    static void make(funcproto_type &func, arrfunc_type_data *out_af) {
        functor_arrfunc_from<funcproto_type *, copy, true>::make(&func, out_af);
    }
};

template <typename obj_type>
struct functor_arrfunc_from<obj_type, true, false> {
    static void make(const obj_type &obj, arrfunc_type_data *out_af) {
        make(obj, &obj_type::operator(), out_af);
    }

    template <typename func_type>
    static void make(const obj_type &obj, func_type, arrfunc_type_data *out_af) {
        typedef typename funcproto_from<func_type>::type funcproto_type;

        out_af->func_proto = ndt::make_funcproto<funcproto_type>();
        *out_af->get_data_as<obj_type>() = obj;
        out_af->instantiate = &nd::functor_ck<obj_type, funcproto_type>::instantiate;
        out_af->free_func = NULL;
    }
};

template <typename obj_type>
struct functor_arrfunc_from<obj_type, false, false> {
    static void make(const obj_type &obj, arrfunc_type_data *out_af) {
        make(obj, &obj_type::operator(), out_af);
    }

    template <typename func_type>
    static void make(const obj_type &obj, func_type, arrfunc_type_data *out_af) {
        typedef typename funcproto_from<func_type>::type funcproto_type;
        typedef func_wrapper<obj_type, funcproto_type> wrapper_type;

        out_af->func_proto = ndt::make_funcproto<funcproto_type>();
        *out_af->get_data_as<wrapper_type>() = wrapper_type(obj);
        out_af->instantiate = &nd::functor_ck<wrapper_type, funcproto_type>::instantiate;
        out_af->free_func = NULL;
    }
};

} // namespace detail

template <typename func_type>
void make_functor_arrfunc(arrfunc_type_data *out_af, const func_type &func, bool copy) {
    if (copy) {
        detail::functor_arrfunc_from<func_type, true,
            std::is_function<func_type>::value || is_function_pointer<func_type>::value>::make(func, out_af);
    } else {
        detail::functor_arrfunc_from<func_type, false,
            std::is_function<func_type>::value || is_function_pointer<func_type>::value>::make(func, out_af);
    }
}

template <typename func_type>
arrfunc make_functor_arrfunc(const func_type &func, bool copy = true) {
    array af = empty(ndt::make_arrfunc());
    make_functor_arrfunc(reinterpret_cast<arrfunc_type_data *>(af.get_readwrite_originptr()),
        func, copy);
    af.flag_as_immutable();
    return af;
}

template <typename obj_type, typename mem_func_type>
void make_functor_arrfunc(arrfunc_type_data *out_af, const obj_type &obj, mem_func_type mem_func, bool copy) {
    if (copy) {
        typedef detail::mem_func_wrapper<mem_func_type, true> wrapper_type;
        detail::functor_arrfunc_from<wrapper_type, true, false>::make(wrapper_type(obj, mem_func), out_af);
    } else {
        typedef detail::mem_func_wrapper<mem_func_type, false> wrapper_type;
        detail::functor_arrfunc_from<wrapper_type, false, false>::make(wrapper_type(obj, mem_func), out_af);
    }
}

template <typename obj_type, typename mem_func_type>
arrfunc make_functor_arrfunc(const obj_type &obj, mem_func_type mem_func, bool copy = true) {
    array af = empty(ndt::make_arrfunc());
    make_functor_arrfunc(reinterpret_cast<arrfunc_type_data *>(af.get_readwrite_originptr()),
        obj, mem_func, copy);
    af.flag_as_immutable();
    return af;
}

}} // namespace dynd::nd

#endif // DYND__FUNC_FUNCTOR_ARRFUNC_HPP
