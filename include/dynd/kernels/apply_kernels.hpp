//
// Copyright (C) 2011-14 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#pragma once

#include <dynd/strided_vals.hpp>
#include <dynd/kernels/cuda_kernels.hpp>
#include <dynd/kernels/ckernel_common_functions.hpp>
#include <dynd/kernels/expr_kernels.hpp>
#include <dynd/types/arrfunc_type.hpp>

namespace dynd {
/*
template <typename func_type, typename arrfunc_type>
class func_wrapper;

template <typename func_type, typename R, typename... A>
class func_wrapper<func_type, R (A...)> {
  const func_type *m_func;

public:
  func_wrapper(const func_type &func) : m_func(&func) {
  }

  R operator()(A... a) {
    return (*m_func)(a...);
  }
};

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
*/

namespace kernels { namespace detail {

template <typename A, size_t I>
struct arg {
  typedef typename std::remove_cv<typename std::remove_reference<A>::type>::type D;

  arg(const ndt::type &DYND_UNUSED(tp), const char *DYND_UNUSED(arrmeta),
    const nd::array &DYND_UNUSED(kwds))
  {
  }

  DYND_CUDA_HOST_DEVICE D &get(char *data) {
    return *reinterpret_cast<D *>(data);
  }
};

template <typename T, int N, size_t I>
struct arg<const nd::strided_vals<T, N> &, I> {
  nd::strided_vals<T, N> m_vals;

  arg(const ndt::type &DYND_UNUSED(tp), const char *arrmeta,
            const nd::array &kwds)
  {
    m_vals.set_data(
        NULL, reinterpret_cast<const size_stride_t *>(arrmeta),
        reinterpret_cast<start_stop_t *>(kwds.p("start_stop").as<intptr_t>()));

    ndt::type dt = kwds.get_dtype();
    // TODO: Remove all try/catch(...) in the code
    try {
      const nd::array &mask = kwds.p("mask").f("dereference");
      m_vals.set_mask(
          mask.get_readonly_originptr(),
          reinterpret_cast<const size_stride_t *>(mask.get_arrmeta()));
    }
    catch (...) {
      m_vals.set_mask(NULL);
    }
  }

  nd::strided_vals<T, N> &get(char *data)
  {
    m_vals.set_data(data);
    return m_vals;
  }
};

template <typename T, size_t I>
class kwd {
  T m_val;

public:
  kwd(T val) : m_val(val) {
  }

  DYND_CUDA_HOST_DEVICE T get() {
    return m_val;
  }
};

// apply_function_ck
// apply_callable_ck
// construct_then_apply_callable_ck

template <kernel_request_t kernreq, typename func_type, func_type func, typename R, typename A, typename I, typename K, typename J>
struct apply_function_ck;

template <typename func_type, func_type func, typename R, typename... A, size_t... I, typename... K, size_t... J>
struct apply_function_ck<kernel_request_host, func_type, func, R, type_sequence<A...>, index_sequence<I...>, type_sequence<K...>, index_sequence<J...> >
  : kernels::expr_ck<apply_function_ck<kernel_request_host, func_type, func, R, type_sequence<A...>, index_sequence<I...>, type_sequence<K...>, index_sequence<J...> >, sizeof...(A)>,
    detail::arg<A, I>..., detail::kwd<K, J>...
{
  typedef apply_function_ck<kernel_request_host, func_type, func, R, type_sequence<A...>, index_sequence<I...>, type_sequence<K...>, index_sequence<J...> > self_type;

  DYND_CUDA_HOST_DEVICE apply_function_ck(detail::arg<A, I>... args, const detail::kwd<K, J> &... kwds)
    : detail::arg<A, I>(args)..., detail::kwd<K, J>(kwds)...
  {
  }

  DYND_CUDA_HOST_DEVICE void single(char *dst, char **DYND_CONDITIONAL_UNUSED(src))
  {
    *reinterpret_cast<R *>(dst) = func(detail::arg<A, I>::get(src[I])..., detail::kwd<K, J>::get()...);
  }

  static intptr_t instantiate(const arrfunc_type_data *DYND_UNUSED(af_self), const arrfunc_type *af_tp,
    ckernel_builder *ckb, intptr_t ckb_offset,
    const ndt::type &dst_tp, const char *DYND_UNUSED(dst_arrmeta),
    const ndt::type *src_tp, const char *const *DYND_CONDITIONAL_UNUSED(src_arrmeta),
    kernel_request_t kernreq, const eval::eval_context *DYND_UNUSED(ectx),
    const nd::array &DYND_UNUSED(args), const nd::array &kwds)
  {
    for (intptr_t i = 0; i < af_tp->get_npos(); ++i) {
      if (src_tp[i] != af_tp->get_arg_type(i)) {
        std::stringstream ss;
        ss << "Provided types " << ndt::make_funcproto(sizeof...(A), src_tp, dst_tp)
           << " do not match the arrfunc proto " << af_tp;
        throw type_error(ss.str());
      }
    }
    if (dst_tp != af_tp->get_return_type()) {
      std::stringstream ss;
      ss << "Provided types " << ndt::make_funcproto(sizeof...(A), src_tp, dst_tp)
         << " do not match the arrfunc proto " << af_tp;
      throw type_error(ss.str());
    }

    self_type::create(ckb, kernreq, ckb_offset,
      arg<A, I>(src_tp[I], src_arrmeta[I], kwds)..., kwd<K, J>(kwds.at(J).as<K>())...);
    return ckb_offset;
  }
};

template <typename func_type, typename R, typename A, typename I, typename K, typename J>
struct apply_callable_ck;

template <typename func_type, typename R, typename... A, size_t... I, typename... K, size_t... J>
struct apply_callable_ck<func_type, R, type_sequence<A...>, index_sequence<I...>, type_sequence<K...>, index_sequence<J...> >
  : kernels::expr_ck<apply_callable_ck<func_type, R, type_sequence<A...>, index_sequence<I...>, type_sequence<K...>, index_sequence<J...> >,
    sizeof...(A)>, detail::arg<A, I>..., detail::kwd<K, J>...
{
    typedef apply_callable_ck<func_type, R, type_sequence<A...>, index_sequence<I...>, type_sequence<K...>, index_sequence<J...> > self_type;

    func_type func;

    apply_callable_ck(const func_type &func, detail::arg<A, I>... args, detail::kwd<K, J>... kwds)
      : detail::arg<A, I>(args)..., detail::kwd<K, J>(kwds)..., func(func) {
    }

    void single(char *dst, char **DYND_CONDITIONAL_UNUSED(src)) {
      *reinterpret_cast<R *>(dst) = func(detail::arg<A, I>::get(src[I])..., detail::kwd<K, J>::get()...);
    }

    static intptr_t instantiate(const arrfunc_type_data *af_self, const arrfunc_type *af_tp,
      dynd::ckernel_builder *ckb, intptr_t ckb_offset,
      const ndt::type &dst_tp, const char *DYND_UNUSED(dst_arrmeta),
      const ndt::type *src_tp, const char *const *DYND_CONDITIONAL_UNUSED(src_arrmeta),
      kernel_request_t kernreq, const eval::eval_context *DYND_UNUSED(ectx),
      const nd::array &DYND_UNUSED(args), const nd::array &kwds)
    {
      for (intptr_t i = 0; i < af_tp->get_npos(); ++i) {
        if (src_tp[i] != af_tp->get_arg_type(i)) {
          std::stringstream ss;
          ss << "Provided types " << ndt::make_funcproto(sizeof...(A), src_tp, dst_tp)
             << " do not match the arrfunc proto " << af_tp;
          throw type_error(ss.str());
        }
      }
      if (dst_tp != af_tp->get_return_type()) {
        std::stringstream ss;
        ss << "Provided types " << ndt::make_funcproto(sizeof...(A), src_tp, dst_tp)
           << " do not match the arrfunc proto " << af_tp;
        throw type_error(ss.str());
      }

      self_type::create(ckb, kernreq, ckb_offset, *af_self->get_data_as<func_type>(),
        detail::arg<A, I>(src_tp[I], src_arrmeta[I], kwds)..., detail::kwd<K, J>(kwds.at(J).as<K>())...);
      return ckb_offset;
    }
};

template <kernel_request_t kernreq, typename func_type, typename R, typename A,
          typename I, typename K, typename J>
struct construct_then_apply_callable_ck;

template <typename func_type, typename R, typename... A, size_t... I,
          typename... K, size_t... J>
struct construct_then_apply_callable_ck<
    kernel_request_host, func_type, R, type_sequence<A...>,
    index_sequence<I...>, type_sequence<K...>, index_sequence<J...>>
    : kernels::expr_ck<
          construct_then_apply_callable_ck<
              kernel_request_host, func_type, R, type_sequence<A...>,
              index_sequence<I...>, type_sequence<K...>, index_sequence<J...>>,
          sizeof...(A)>,
      detail::arg<A, I>... {
  typedef construct_then_apply_callable_ck<
      kernel_request_host, func_type, R, type_sequence<A...>,
      index_sequence<I...>, type_sequence<K...>,
      index_sequence<J...>> self_type;

  func_type func;

  construct_then_apply_callable_ck(detail::arg<A, I>... args,
                                   detail::kwd<K, J>... kwds)
      : detail::arg<A, I>(args)..., func(kwds.get()...)
  {
  }

  void single(char *dst, char **DYND_CONDITIONAL_UNUSED(src))
  {
    *reinterpret_cast<R *>(dst) = func(detail::arg<A, I>::get(src[I])...);
  }

  static intptr_t instantiate(
      const arrfunc_type_data *DYND_UNUSED(af_self), const arrfunc_type *af_tp,
      dynd::ckernel_builder *ckb, intptr_t ckb_offset, const ndt::type &dst_tp,
      const char *DYND_UNUSED(dst_arrmeta), const ndt::type *src_tp,
      const char *const *DYND_CONDITIONAL_UNUSED(src_arrmeta),
      kernel_request_t kernreq, const eval::eval_context *DYND_UNUSED(ectx),
      const nd::array &DYND_UNUSED(args), const nd::array &kwds)
  {
    for (intptr_t i = 0; i < af_tp->get_npos(); ++i) {
      if (src_tp[i] != af_tp->get_arg_type(i)) {
        std::stringstream ss;
        ss << "Provided types "
           << ndt::make_funcproto(sizeof...(A), src_tp, dst_tp)
           << " do not match the arrfunc proto " << af_tp;
        throw type_error(ss.str());
      }
    }
    if (dst_tp != af_tp->get_return_type()) {
      std::stringstream ss;
      ss << "Provided types "
         << ndt::make_funcproto(sizeof...(A), src_tp, dst_tp)
         << " do not match the arrfunc proto " << af_tp;
      throw type_error(ss.str());
    }

    self_type::create(ckb, kernreq, ckb_offset,
                      detail::arg<A, I>(src_tp[I], src_arrmeta[I], kwds)...,
                      detail::kwd<K, J>(kwds.at(J).as<K>())...);
    return ckb_offset;
  }
};

#ifdef __CUDACC__

template <typename func_type, typename R, typename... A, size_t... I,
          typename... K, size_t... J>
struct construct_then_apply_callable_ck<
    kernel_request_cuda_device, func_type, R, type_sequence<A...>,
    index_sequence<I...>, type_sequence<K...>, index_sequence<J...>>
    : kernels::expr_ck<
          construct_then_apply_callable_ck<
              kernel_request_cuda_device, func_type, R, type_sequence<A...>,
              index_sequence<I...>, type_sequence<K...>, index_sequence<J...>>,
          sizeof...(A)>,
      detail::arg<A, I>... {
  typedef construct_then_apply_callable_ck<
      kernel_request_cuda_device, func_type, R, type_sequence<A...>,
      index_sequence<I...>, type_sequence<K...>,
      index_sequence<J...>> self_type;

  func_type func;

  __device__ construct_then_apply_callable_ck(detail::arg<A, I>... args,
                                              detail::kwd<K, J>... kwds)
      : detail::arg<A, I>(args)..., func(kwds.get()...)
  {
  }

  __device__ void single(char *dst, char **DYND_CONDITIONAL_UNUSED(src))
  {
    *reinterpret_cast<R *>(dst) = func(detail::arg<A, I>::get(src[I])...);
  }

  static intptr_t instantiate(
      const arrfunc_type_data *DYND_UNUSED(af_self), const arrfunc_type *af_tp,
      dynd::ckernel_builder *ckb, intptr_t ckb_offset, const ndt::type &dst_tp,
      const char *DYND_UNUSED(dst_arrmeta), const ndt::type *src_tp,
      const char *const *DYND_UNUSED(src_arrmeta),
      kernel_request_t kernreq, const eval::eval_context *DYND_UNUSED(ectx),
      const nd::array &DYND_UNUSED(args), const nd::array &DYND_UNUSED(kwds))
  {
    for (intptr_t i = 0; i < af_tp->get_npos(); ++i) {
      if (src_tp[i] != af_tp->get_arg_type(i)) {
        std::stringstream ss;
        ss << "Provided types "
           << ndt::make_funcproto(sizeof...(A), src_tp, dst_tp)
           << " do not match the arrfunc proto " << af_tp;
        throw type_error(ss.str());
      }
    }
    if (dst_tp != af_tp->get_return_type()) {
      std::stringstream ss;
      ss << "Provided types "
         << ndt::make_funcproto(sizeof...(A), src_tp, dst_tp)
         << " do not match the arrfunc proto " << af_tp;
      throw type_error(ss.str());
    }

    if ((kernreq & kernel_request_cuda_device) == false) {
      cuda_parallel_ck::create(ckb, kernreq, ckb_offset, 1, 1);
    //  cuda_parallel_ck *self = ckb->get_at<cuda_parallel_ck>(ckb_offset);
  //    cuda_device_ckernel_builder = self->get_ckb();
//      cuda_parallel_ck::create(
    }

 //   self_type::create(ckb, kernreq, ckb_offset,
   //                   detail::arg<A, I>(src_tp[I], src_arrmeta[I], kwds)...,
     //                 detail::kwd<K, J>(kwds.at(J).as<K>())...);
    return ckb_offset;
  }
};

#endif

} // detail

template <typename func_type, func_type func, int N, typename R, typename... A>
using apply_function_ck = detail::apply_function_ck<kernel_request_host, func_type, func, R,
  typename to<N, A...>::type, typename make_index_sequence<N>::type,
  typename from<N, A...>::type, typename make_index_sequence<sizeof...(A) - N>::type>;

template <typename func_type, int N, typename R, typename... A>
using apply_callable_ck = detail::apply_callable_ck<func_type, R,
  typename to<N, A...>::type, typename make_index_sequence<N>::type,
  typename from<N, A...>::type, typename make_index_sequence<sizeof...(A) - N>::type>;

template <kernel_request_t kernreq, typename func_type, typename R, typename A, typename K>
using construct_then_apply_callable_ck = detail::construct_then_apply_callable_ck<kernreq, func_type,
  R, A, typename make_index_sequence<A::size>::type, K, typename make_index_sequence<K::size>::type>;

}} // namespace dynd::kernels
