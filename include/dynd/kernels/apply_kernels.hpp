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
namespace detail {

  template <typename func_type, typename... B>
  struct funcproto {
    typedef typename funcproto<decltype(&func_type::operator()), B...>::type
        type;
  };

  template <typename R, typename... A, typename... B>
  struct funcproto<R(A...), B...> {
    typedef R(type)(A..., B...);
  };

  template <typename R, typename... A, typename... B>
  struct funcproto<R (*)(A...), B...> {
    typedef typename funcproto<R(A...), B...>::type type;
  };

  template <typename T, typename R, typename... A, typename... B>
  struct funcproto<R (T::*)(A...), B...> {
    typedef typename funcproto<R(A...), B...>::type type;
  };

  template <typename T, typename R, typename... A, typename... B>
  struct funcproto<R (T::*)(A...) const, B...> {
    typedef typename funcproto<R(A...), B...>::type type;
  };
}

template <typename func_type, typename... B>
using funcproto_for = typename detail::funcproto<func_type, B...>::type;

template <typename funcproto_type>
struct return_of;

template <typename R, typename... A>
struct return_of<R(A...)> {
  typedef R type;
};

namespace kernels {
  namespace detail {

    template <typename A, size_t I>
    struct arg {
      typedef typename std::remove_cv<
          typename std::remove_reference<A>::type>::type D;

      arg(const ndt::type &DYND_UNUSED(tp), const char *DYND_UNUSED(arrmeta),
          const nd::array &DYND_UNUSED(kwds))
      {
      }

      DYND_CUDA_HOST_DEVICE D &get(char *data)
      {
        return *reinterpret_cast<D *>(data);
      }
    };

    template <typename T, int N, size_t I>
    struct arg<const nd::strided_vals<T, N> &, I> {
      nd::strided_vals<T, N> m_vals;

      arg(const ndt::type &DYND_UNUSED(tp), const char *arrmeta,
          const nd::array &kwds)
      {
        m_vals.set_data(NULL, reinterpret_cast<const size_stride_t *>(arrmeta),
                        reinterpret_cast<start_stop_t *>(
                            kwds.p("start_stop").as<intptr_t>()));

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

    template <typename A, typename I>
    class args;

    template <typename... A, size_t... I>
    class args<type_sequence<A...>, index_sequence<I...>>
        : public arg<A, I>... {
    public:
      args(const ndt::type *DYND_CONDITIONAL_UNUSED(src_tp),
           const char *const *DYND_CONDITIONAL_UNUSED(src_arrmeta),
           const nd::array &kwds2)
          : arg<A, I>(src_tp[I], src_arrmeta[I], kwds2)...
      {
      }
    };

    template <typename T, size_t I>
    class kwd {
      T m_val;

    public:
      kwd(T val) : m_val(val) {}

      DYND_CUDA_HOST_DEVICE T get() { return m_val; }
    };

    template <typename K, typename J>
    class kwds;

    template <typename... K, size_t... J>
    class kwds<type_sequence<K...>, index_sequence<J...>>
        : public kwd<K, J>... {
    public:
      kwds(const nd::array &kwds) : kwd<K, J>(kwds.at(J).as<K>())... {}
    };

    template <kernel_request_t kernreq, typename func_type, typename K,
              typename J>
    class func;

    template <typename func_type, typename... K, size_t... J>
    class func<kernel_request_host, func_type, type_sequence<K...>,
               index_sequence<J...>> {
    public:
      func_type m_func;

      func(detail::kwds<type_sequence<K...>, index_sequence<J...>>
               DYND_CONDITIONAL_UNUSED(kwds))
          : m_func(static_cast<detail::kwd<K, J> *>(&kwds)->get()...)
      {
      }

      const func_type &get() { return m_func; }
    };

#ifdef __CUDACC__

    template <typename func_type, typename... K, size_t... J>
    class func<kernel_request_cuda_device, func_type, type_sequence<K...>,
               index_sequence<J...>> {
    public:
      func_type m_func;

      __device__ func(detail::kwds<type_sequence<K...>, index_sequence<J...>>
                          DYND_CONDITIONAL_UNUSED(kwds))
          : m_func(static_cast<detail::kwd<K, J> *>(&kwds)->get()...)
      {
      }

      __device__ const func_type &get() { return m_func; }
    };

#endif

    template <kernel_request_t kernreq>
    struct apply;

#define APPLY(CKBT, ...)                                                       \
  template <>                                                                  \
  struct apply<CKBT> {                                                         \
                                                                               \
    template <typename func_type, typename... A, size_t... I, typename... K,   \
              size_t... J>                                                     \
    __VA_ARGS__ static void                                                    \
    single(func<CKBT, func_type, type_sequence<K...>, index_sequence<J...>> *  \
               func,                                                           \
           args<type_sequence<A...>, index_sequence<I...>> *                   \
               DYND_CONDITIONAL_UNUSED(args),                                  \
           char *dst, char **DYND_CONDITIONAL_UNUSED(src))                     \
    {                                                                          \
      typedef typename return_of<funcproto_for<func_type>>::type R;            \
                                                                               \
      *reinterpret_cast<R *>(dst) =                                            \
          func->get()(static_cast<arg<A, I> *>(args)->get(src[I])...);         \
    }                                                                          \
  };

    APPLY(kernel_request_host)
#ifdef __CUDACC__
    APPLY(kernel_request_cuda_device, __device__)
#endif
  }

  template <typename funcproto_type>
  struct args_of;

  template <typename R, typename... A>
  struct args_of<R(A...)> {
    typedef dynd::type_sequence<A...> type;
  };

  template <typename A>
  using args = detail::args<A, typename make_index_sequence<A::size>::type>;

  template <typename func_type,
            int N = args_of<funcproto_for<func_type>>::type::size>
  using args_for = args<
      typename to<N, typename args_of<funcproto_for<func_type>>::type>::type>;

  template <typename func_type, int N>
  struct xkwds {
    typedef typename xkwds<funcproto_for<func_type>, N>::type type;
  };

  template <typename R, typename... A, int N>
  struct xkwds<R(A...), N> {
    typedef detail::kwds<typename from<N, A...>::type,
                         typename make_index_sequence<sizeof...(A)-N>::type>
        type;
  };

  template <typename K>
  using kwds = detail::kwds<K, typename make_index_sequence<K::size>::type>;

  template <typename func_type, int N>
  using kwds_for = typename xkwds<func_type, N>::type;

  template <kernel_request_t kernreq, typename func_type, typename... K>
  using func_for =
      detail::func<kernreq, func_type, type_sequence<K...>,
                   typename make_index_sequence<sizeof...(K)>::type>;

  template <typename func_type>
  struct arity {
    static const intptr_t value = args_of<funcproto_for<func_type>>::type::size;
  };

  template <kernel_request_t kernreq, typename func_type, func_type func, int N>
  class apply_function_ck;

#define APPLY_FUNCTION_CK(KERNREQ, ...)                                        \
  template <typename func_type, func_type func, int N>                         \
  class apply_function_ck<KERNREQ, func_type, func, N>                         \
      : public expr_ck<apply_function_ck<KERNREQ, func_type, func, N>,         \
                       KERNREQ, N>,                                            \
        args_for<func_type, N>,                                                \
        kwds_for<func_type, N> {                                               \
    typedef apply_function_ck<KERNREQ, func_type, func, N> self_type;          \
                                                                               \
  public:                                                                      \
    __VA_ARGS__ apply_function_ck(args_for<func_type, N> args,                 \
                                  kwds_for<func_type, N> kwds)                 \
        : args_for<func_type, N>(args), kwds_for<func_type, N>(kwds)           \
    {                                                                          \
    }                                                                          \
                                                                               \
    __VA_ARGS__ void single(char *dst, char **src)                             \
    {                                                                          \
      single(dst, src, this, this);                                            \
    }                                                                          \
                                                                               \
    template <typename... A, size_t... I, typename... K, size_t... J>          \
    static void                                                                \
    single(char *dst, char **DYND_CONDITIONAL_UNUSED(src),                     \
           detail::args<type_sequence<A...>, index_sequence<I...>> *           \
               DYND_CONDITIONAL_UNUSED(args),                                  \
           detail::kwds<type_sequence<K...>, index_sequence<J...>> *           \
               DYND_CONDITIONAL_UNUSED(kwds))                                  \
    {                                                                          \
      typedef typename return_of<                                              \
          typename std::remove_pointer<func_type>::type>::type R;              \
                                                                               \
      *reinterpret_cast<R *>(dst) =                                            \
          func(static_cast<detail::arg<A, I> *>(args)->get(src[I])...,         \
               static_cast<detail::kwd<K, J> *>(kwds)->get()...);              \
    }                                                                          \
                                                                               \
    static intptr_t                                                            \
    instantiate(const arrfunc_type_data *DYND_UNUSED(af_self),                 \
                const arrfunc_type *DYND_UNUSED(af_tp), void *ckb,             \
                intptr_t ckb_offset, const ndt::type &DYND_UNUSED(dst_tp),     \
                const char *DYND_UNUSED(dst_arrmeta), const ndt::type *src_tp, \
                const char *const *src_arrmeta, kernel_request_t kernreq,      \
                const eval::eval_context *DYND_UNUSED(ectx),                   \
                const nd::array &DYND_UNUSED(a), const nd::array &kwds)        \
    {                                                                          \
      self_type::create(ckb, kernreq, ckb_offset,                              \
                        args_for<func_type, N>(src_tp, src_arrmeta, kwds),     \
                        kwds_for<func_type, N>(kwds));                         \
      return ckb_offset;                                                       \
    }                                                                          \
  }

  APPLY_FUNCTION_CK(kernel_request_host);

#undef APPLY_FUNCTION_CK

  template <kernel_request_t kernreq, typename func_type, int N>
  class apply_callable_ck;

#define APPLY_CALLABLE_CK(KERNREQ, ...)                                        \
  template <typename func_type, int N>                                         \
  class apply_callable_ck<KERNREQ, func_type, N>                               \
      : public expr_ck<apply_callable_ck<KERNREQ, func_type, N>, KERNREQ, N>,  \
        args_for<func_type, N>,                                                \
        kwds_for<func_type, N> {                                               \
    typedef apply_callable_ck<KERNREQ, func_type, N> self_type;                \
                                                                               \
    func_type func;                                                            \
                                                                               \
  public:                                                                      \
    __VA_ARGS__ apply_callable_ck(const func_type &func,                       \
                                  args_for<func_type, N> args,                 \
                                  kwds_for<func_type, N> kwds)                 \
        : args_for<func_type, N>(args), kwds_for<func_type, N>(kwds),          \
          func(func)                                                           \
    {                                                                          \
    }                                                                          \
                                                                               \
    template <typename... A, size_t... I, typename... K, size_t... J>          \
    __VA_ARGS__ static void                                                    \
    single(char *dst, char **DYND_CONDITIONAL_UNUSED(src),                     \
           const func_type &func,                                              \
           detail::args<type_sequence<A...>, index_sequence<I...>> *           \
               DYND_CONDITIONAL_UNUSED(args),                                  \
           detail::kwds<type_sequence<K...>, index_sequence<J...>> *           \
               DYND_CONDITIONAL_UNUSED(kwds))                                  \
    {                                                                          \
      typedef typename return_of<funcproto_for<func_type>>::type RR;           \
                                                                               \
      *reinterpret_cast<RR *>(dst) =                                           \
          func(static_cast<detail::arg<A, I> *>(args)->get(src[I])...,         \
               static_cast<detail::kwd<K, J> *>(kwds)->get()...);              \
    }                                                                          \
                                                                               \
    __VA_ARGS__ void single(char *dst, char **src)                             \
    {                                                                          \
      single(dst, src, func, this, this);                                      \
    }                                                                          \
                                                                               \
    static intptr_t                                                            \
    instantiate(const arrfunc_type_data *af_self,                              \
                const arrfunc_type *DYND_UNUSED(af_tp), void *ckb,             \
                intptr_t ckb_offset, const ndt::type &DYND_UNUSED(dst_tp),     \
                const char *DYND_UNUSED(dst_arrmeta), const ndt::type *src_tp, \
                const char *const *src_arrmeta, kernel_request_t kernreq,      \
                const eval::eval_context *DYND_UNUSED(ectx),                   \
                const nd::array &DYND_UNUSED(args), const nd::array &kwds)     \
    {                                                                          \
      self_type::create(ckb, kernreq, ckb_offset,                              \
                        *af_self->get_data_as<func_type>(),                    \
                        args_for<func_type, N>(src_tp, src_arrmeta, kwds),     \
                        kwds_for<func_type, N>(kwds));                         \
      return ckb_offset;                                                       \
    }                                                                          \
  }

  APPLY_CALLABLE_CK(kernel_request_host);

#undef APPLY_CALLABLE_CK

  template <kernel_request_t kernreq, typename func_type, typename... K>
  struct construct_then_apply_callable_ck;

#define CONSTRUCT_THEN_APPLY_CALLABLE_CK(KERNREQ, ...)                         \
  template <typename func_type, typename... K>                                 \
  struct construct_then_apply_callable_ck<KERNREQ, func_type, K...>            \
      : public expr_ck<                                                        \
            construct_then_apply_callable_ck<KERNREQ, func_type, K...>,        \
            KERNREQ, arity<func_type>::value>,                                 \
        public args_for<func_type>,                                            \
        public func_for<KERNREQ, func_type, K...> {                            \
    typedef construct_then_apply_callable_ck<KERNREQ, func_type, K...>         \
        self_type;                                                             \
                                                                               \
    __VA_ARGS__ construct_then_apply_callable_ck(args_for<func_type> a,        \
                                                 kwds<type_sequence<K...>> k)  \
        : args_for<func_type>(a), func_for<KERNREQ, func_type, K...>(k)        \
    {                                                                          \
    }                                                                          \
                                                                               \
    __VA_ARGS__ void single(char *dst, char **src)                             \
    {                                                                          \
      detail::apply<KERNREQ>::single(this, this, dst, src);                    \
    }                                                                          \
                                                                               \
    static intptr_t                                                            \
    instantiate(const arrfunc_type_data *af_self, const arrfunc_type *af_tp,   \
                void *ckb, intptr_t ckb_offset, const ndt::type &dst_tp,       \
                const char *dst_arrmeta, const ndt::type *src_tp,              \
                const char *const *src_arrmeta, kernel_request_t kernreq,      \
                const eval::eval_context *ectx, const nd::array &args,         \
                const nd::array &kwds);                                        \
  };

  CONSTRUCT_THEN_APPLY_CALLABLE_CK(kernel_request_host)

  template <typename func_type, typename... K>
  intptr_t
  construct_then_apply_callable_ck<kernel_request_host, func_type, K...>::
      instantiate(const arrfunc_type_data *DYND_UNUSED(af_self),
                  const arrfunc_type *DYND_UNUSED(af_tp), void *ckb,
                  intptr_t ckb_offset, const ndt::type &DYND_UNUSED(dst_tp),
                  const char *DYND_UNUSED(dst_arrmeta), const ndt::type *src_tp,
                  const char *const *src_arrmeta, kernel_request_t kernreq,
                  const eval::eval_context *DYND_UNUSED(ectx),
                  const nd::array &DYND_UNUSED(args), const nd::array &kwds_)
  {
    self_type::create(ckb, kernreq, ckb_offset,
                      kernels::args_for<func_type>(src_tp, src_arrmeta, kwds_),
                      kernels::kwds<type_sequence<K...>>(kwds_));
    return ckb_offset;
  }

#ifdef __CUDACC__

  CONSTRUCT_THEN_APPLY_CALLABLE_CK(kernel_request_cuda_device, __device__)

  template <typename func_type, typename... K>
  intptr_t construct_then_apply_callable_ck<
      kernel_request_cuda_device, func_type,
      K...>::instantiate(const arrfunc_type_data *DYND_UNUSED(af_self),
                         const arrfunc_type *DYND_UNUSED(af_tp), void *ckb,
                         intptr_t ckb_offset,
                         const ndt::type &DYND_UNUSED(dst_tp),
                         const char *DYND_UNUSED(dst_arrmeta),
                         const ndt::type *src_tp,
                         const char *const *src_arrmeta,
                         kernel_request_t kernreq,
                         const eval::eval_context *DYND_UNUSED(ectx),
                         const nd::array &DYND_UNUSED(args),
                         const nd::array &kwds_)
  {
    if ((kernreq & kernel_request_cuda_device) == false) {
      typedef cuda_parallel_ck<arity<func_type>::value> self_type;
      self_type *self = self_type::create(ckb, kernreq, ckb_offset, 1, 1);
      ckb = self->get_ckb();
      ckb_offset = 0;
    }

    self_type::create(ckb, kernreq, ckb_offset,
                      args_for<func_type>(src_tp, src_arrmeta, kwds_),
                      kwds<type_sequence<K...>>(kwds_));
    return ckb_offset;
  }

#endif

#undef CONSTRUCT_THEN_APPLY_CALLABLE_CK
}
} // namespace dynd::kernels
