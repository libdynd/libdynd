//
// Copyright (C) 2011-15 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#pragma once

#include <dynd/strided_vals.hpp>
#include <dynd/kernels/cuda_kernels.hpp>
#include <dynd/kernels/expr_kernels.hpp>

template <typename T>
class value_wrapper {
  T m_value;

public:
  value_wrapper(const T &value) : m_value(value) {}

  DYND_CUDA_HOST_DEVICE operator T() const { return m_value; }
};

template <typename T>
value_wrapper<T> make_value_wrapper(const T &value)
{
  return value_wrapper<T>(value);
}

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
struct funcproto_of {
  typedef typename detail::funcproto<func_type, B...>::type type;
};

template <typename func_type>
struct return_of {
  typedef typename return_of<typename funcproto_of<func_type>::type>::type type;
};

template <typename R, typename... A>
struct return_of<R(A...)> {
  typedef R type;
};

template <typename func_type>
struct args_of {
  typedef typename args_of<typename funcproto_of<func_type>::type>::type type;
};

template <typename R, typename... A>
struct args_of<R(A...)> {
  typedef type_sequence<A...> type;
};

template <typename func_type>
struct arity_of {
  static const size_t value =
      arity_of<typename funcproto_of<func_type>::type>::value;
};

template <typename R, typename... A>
struct arity_of<R(A...)> {
  static const size_t value = sizeof...(A);
};

namespace kernels {

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

  template <typename A, typename I = make_index_sequence<A::size>>
  struct args;

  template <typename... A, size_t... I>
  struct args<type_sequence<A...>, index_sequence<I...>> : arg<A, I>... {
    args(const ndt::type *DYND_IGNORE_UNUSED(src_tp),
         const char *const *DYND_IGNORE_UNUSED(src_arrmeta),
         const nd::array &kwds)
        : arg<A, I>(src_tp[I], src_arrmeta[I], kwds)...
    {
    }
  };

  template <typename func_type,
            int Nsrc =
                args_of<typename funcproto_of<func_type>::type>::type::size>
  using args_for = args<typename to<
      typename args_of<typename funcproto_of<func_type>::type>::type,
      Nsrc>::type>;

  template <typename T, size_t I>
  struct kwd {
    T m_val;

    kwd(nd::array val)
    //        : m_val(val.as<T>())
    {
      if (val.get_type().get_type_id() == pointer_type_id) {
        m_val = val.f("dereference").as<T>();
      } else {
        m_val = val.as<T>();
      }
    }

    DYND_CUDA_HOST_DEVICE T get() { return m_val; }
  };

  template <typename K, typename J = make_index_sequence<K::size>>
  struct kwds;

  template <>
  struct kwds<type_sequence<>, index_sequence<>> {
    kwds(const nd::array &DYND_UNUSED(kwds)) {}
  };

  template <typename... K, size_t... J>
  struct kwds<type_sequence<K...>, index_sequence<J...>> : kwd<K, J>... {
    kwds(const nd::array &kwds) : kwd<K, J>(kwds.at(J))... {}
  };

  template <typename func_type, int Nsrc>
  using kwds_for = kwds<typename from<
      typename args_of<typename funcproto_of<func_type>::type>::type,
      Nsrc>::type>;

  template <kernel_request_t kernreq, typename func_type, func_type func,
            int Nsrc>
  struct apply_function_ck;

#define APPLY_FUNCTION_CK(KERNREQ, ...)                                        \
  template <typename func_type, func_type func, int Nsrc>                      \
  struct apply_function_ck<KERNREQ, func_type, func, Nsrc>                     \
      : expr_ck<apply_function_ck<KERNREQ, func_type, func, Nsrc>, KERNREQ,    \
                Nsrc>,                                                         \
        args_for<func_type, Nsrc>,                                             \
        kwds_for<func_type, Nsrc> {                                            \
    typedef apply_function_ck<KERNREQ, func_type, func, Nsrc> self_type;       \
                                                                               \
    __VA_ARGS__ apply_function_ck(args_for<func_type, Nsrc> args,              \
                                  kwds_for<func_type, Nsrc> kwds)              \
        : args_for<func_type, Nsrc>(args), kwds_for<func_type, Nsrc>(kwds)     \
    {                                                                          \
    }                                                                          \
                                                                               \
    __VA_ARGS__ void single(char *dst, char *const *src)                       \
    {                                                                          \
      single<typename return_of<func_type>::type>(this, this, dst, src);       \
    }                                                                          \
                                                                               \
    static intptr_t                                                            \
    instantiate(const arrfunc_type_data *DYND_UNUSED(self),                    \
                const arrfunc_type *DYND_UNUSED(self_tp), void *ckb,           \
                intptr_t ckb_offset, const ndt::type &DYND_UNUSED(dst_tp),     \
                const char *DYND_UNUSED(dst_arrmeta), const ndt::type *src_tp, \
                const char *const *src_arrmeta, kernel_request_t kernreq,      \
                const eval::eval_context *DYND_UNUSED(ectx),                   \
                const nd::array &kwds,                                         \
                const std::map<nd::string, ndt::type> &DYND_UNUSED(tp_vars))   \
    {                                                                          \
      self_type::create(ckb, kernreq, ckb_offset,                              \
                        args_for<func_type, Nsrc>(src_tp, src_arrmeta, kwds),  \
                        kwds_for<func_type, Nsrc>(kwds));                      \
      return ckb_offset;                                                       \
    }                                                                          \
                                                                               \
  private:                                                                     \
    template <typename R, typename... A, size_t... I, typename... K,           \
              size_t... J>                                                     \
    __VA_ARGS__ typename std::enable_if<std::is_same<R, void>::value,          \
                                        void>::type                            \
    single(args<type_sequence<A...>, index_sequence<I...>> *                   \
               DYND_IGNORE_UNUSED(args),                                       \
           kwds<type_sequence<K...>, index_sequence<J...>> *                   \
               DYND_IGNORE_UNUSED(kwds),                                       \
           char *DYND_UNUSED(dst), char *const *DYND_IGNORE_UNUSED(src))       \
    {                                                                          \
      func(static_cast<arg<A, I> *>(args)->get(src[I])...,                     \
           static_cast<kwd<K, J> *>(kwds)->get()...);                          \
    }                                                                          \
                                                                               \
    template <typename R, typename... A, size_t... I, typename... K,           \
              size_t... J>                                                     \
    __VA_ARGS__ typename std::enable_if<!std::is_same<R, void>::value,         \
                                        void>::type                            \
    single(args<type_sequence<A...>, index_sequence<I...>> *                   \
               DYND_IGNORE_UNUSED(args),                                       \
           kwds<type_sequence<K...>, index_sequence<J...>> *                   \
               DYND_IGNORE_UNUSED(kwds),                                       \
           char *dst, char *const *DYND_IGNORE_UNUSED(src))                    \
    {                                                                          \
      *reinterpret_cast<R *>(dst) =                                            \
          func(static_cast<arg<A, I> *>(args)->get(src[I])...,                 \
               static_cast<kwd<K, J> *>(kwds)->get()...);                      \
    }                                                                          \
  }

  APPLY_FUNCTION_CK(kernel_request_host);

#undef APPLY_FUNCTION_CK

  template <kernel_request_t kernreq, typename T, typename mem_func_type, int N>
  struct apply_member_function_ck;

#define APPLY_MEMBER_FUNCTION_CK(KERNREQ, ...)                                 \
  template <typename T, typename mem_func_type, int Nsrc>                      \
  struct apply_member_function_ck<KERNREQ, T *, mem_func_type, Nsrc>           \
      : expr_ck<apply_member_function_ck<KERNREQ, T *, mem_func_type, Nsrc>,   \
                KERNREQ, Nsrc>,                                                \
        args_for<mem_func_type, Nsrc>,                                         \
        kwds_for<mem_func_type, Nsrc> {                                        \
    typedef apply_member_function_ck self_type;                                \
    typedef std::pair<T *, mem_func_type> data_type;                           \
                                                                               \
    T *obj;                                                                    \
    mem_func_type mem_func;                                                    \
                                                                               \
    __VA_ARGS__ apply_member_function_ck(T *obj, mem_func_type mem_func,       \
                                         args_for<mem_func_type, Nsrc> args,   \
                                         kwds_for<mem_func_type, Nsrc> kwds)   \
        : args_for<mem_func_type, Nsrc>(args),                                 \
          kwds_for<mem_func_type, Nsrc>(kwds), obj(obj), mem_func(mem_func)    \
    {                                                                          \
    }                                                                          \
                                                                               \
    __VA_ARGS__ void single(char *dst, char *const *src)                       \
    {                                                                          \
      single<typename return_of<mem_func_type>::type>(this, this, dst, src);   \
    }                                                                          \
                                                                               \
    static intptr_t                                                            \
    instantiate(const arrfunc_type_data *self, const arrfunc_type *self_tp,    \
                void *ckb, intptr_t ckb_offset, const ndt::type &dst_tp,       \
                const char *dst_arrmeta, const ndt::type *src_tp,              \
                const char *const *src_arrmeta, kernel_request_t kernreq,      \
                const eval::eval_context *ectx, const nd::array &kwds,         \
                const std::map<nd::string, ndt::type> &tp_vars);               \
                                                                               \
  private:                                                                     \
    template <typename R, typename... A, size_t... I, typename... K,           \
              size_t... J>                                                     \
    __VA_ARGS__ typename std::enable_if<std::is_same<R, void>::value,          \
                                        void>::type                            \
    single(args<type_sequence<A...>, index_sequence<I...>> *                   \
               DYND_IGNORE_UNUSED(args),                                       \
           kwds<type_sequence<K...>, index_sequence<J...>> *                   \
               DYND_IGNORE_UNUSED(kwds),                                       \
           char *DYND_UNUSED(dst), char *const *DYND_IGNORE_UNUSED(src))       \
    {                                                                          \
      (obj->*mem_func)(static_cast<arg<A, I> *>(args)->get(src[I])...,         \
                       static_cast<kwd<K, J> *>(kwds)->get()...);              \
    }                                                                          \
                                                                               \
    template <typename R, typename... A, size_t... I, typename... K,           \
              size_t... J>                                                     \
    __VA_ARGS__ typename std::enable_if<!std::is_same<R, void>::value,         \
                                        void>::type                            \
    single(args<type_sequence<A...>, index_sequence<I...>> *                   \
               DYND_IGNORE_UNUSED(args),                                       \
           kwds<type_sequence<K...>, index_sequence<J...>> *                   \
               DYND_IGNORE_UNUSED(kwds),                                       \
           char *dst, char *const *DYND_IGNORE_UNUSED(src))                    \
    {                                                                          \
      *reinterpret_cast<R *>(dst) =                                            \
          (obj->*mem_func)(static_cast<arg<A, I> *>(args)->get(src[I])...,     \
                           static_cast<kwd<K, J> *>(kwds)->get()...);          \
    }                                                                          \
  }

  APPLY_MEMBER_FUNCTION_CK(kernel_request_host);

  template <typename T, typename mem_func_type, int Nsrc>
  intptr_t
  apply_member_function_ck<kernel_request_host, T *, mem_func_type, Nsrc>::
      instantiate(const arrfunc_type_data *self,
                  const arrfunc_type *DYND_UNUSED(self_tp), void *ckb,
                  intptr_t ckb_offset, const ndt::type &DYND_UNUSED(dst_tp),
                  const char *DYND_UNUSED(dst_arrmeta), const ndt::type *src_tp,
                  const char *const *src_arrmeta, kernel_request_t kernreq,
                  const eval::eval_context *DYND_UNUSED(ectx),
                  const nd::array &kwds,
                  const std::map<nd::string, ndt::type> &DYND_UNUSED(tp_vars))
  {
    self_type::create(
        ckb, kernreq, ckb_offset, self->get_data_as<data_type>()->first,
        make_value_wrapper(self->get_data_as<data_type>()->second),
        args_for<mem_func_type, Nsrc>(src_tp, src_arrmeta, kwds),
        kwds_for<mem_func_type, Nsrc>(kwds));
    return ckb_offset;
  }

#undef APPLY_MEMBER_FUNCTION_CK

  template <kernel_request_t kernreq, typename func_type, int Nsrc>
  struct apply_callable_ck;

#define APPLY_CALLABLE_CK(KERNREQ, ...)                                        \
  template <typename func_type, int Nsrc>                                      \
  struct apply_callable_ck<KERNREQ, func_type, Nsrc>                           \
      : expr_ck<apply_callable_ck<KERNREQ, func_type, Nsrc>, KERNREQ, Nsrc>,   \
        args_for<func_type, Nsrc>,                                             \
        kwds_for<func_type, Nsrc> {                                            \
    typedef apply_callable_ck<KERNREQ, func_type, Nsrc> self_type;             \
                                                                               \
    func_type func;                                                            \
                                                                               \
    __VA_ARGS__ apply_callable_ck(func_type func,                              \
                                  args_for<func_type, Nsrc> args,              \
                                  kwds_for<func_type, Nsrc> kwds)              \
        : args_for<func_type, Nsrc>(args), kwds_for<func_type, Nsrc>(kwds),    \
          func(func)                                                           \
    {                                                                          \
    }                                                                          \
                                                                               \
    __VA_ARGS__ void single(char *dst, char *const *src)                       \
    {                                                                          \
      single<typename return_of<func_type>::type>(this, this, dst, src);       \
    }                                                                          \
                                                                               \
    static intptr_t instantiate_without_wrapper(                               \
        const arrfunc_type_data *self,                                         \
        const arrfunc_type *DYND_UNUSED(self_tp), void *ckb,                   \
        intptr_t ckb_offset, const ndt::type &DYND_UNUSED(dst_tp),             \
        const char *DYND_UNUSED(dst_arrmeta), const ndt::type *src_tp,         \
        const char *const *src_arrmeta, kernel_request_t kernreq,              \
        const eval::eval_context *DYND_UNUSED(ectx), const nd::array &kwds,    \
        const std::map<nd::string, ndt::type> &DYND_UNUSED(tp_vars))           \
    {                                                                          \
      self_type::create(ckb, kernreq, ckb_offset,                              \
                        make_value_wrapper(*self->get_data_as<func_type>()),   \
                        args_for<func_type, Nsrc>(src_tp, src_arrmeta, kwds),  \
                        kwds_for<func_type, Nsrc>(kwds));                      \
      return ckb_offset;                                                       \
    }                                                                          \
                                                                               \
    static intptr_t                                                            \
    instantiate(const arrfunc_type_data *self, const arrfunc_type *self_tp,    \
                void *ckb, intptr_t ckb_offset, const ndt::type &dst_tp,       \
                const char *dst_arrmeta, const ndt::type *src_tp,              \
                const char *const *src_arrmeta, kernel_request_t kernreq,      \
                const eval::eval_context *ectx, const nd::array &kwds,         \
                const std::map<nd::string, ndt::type> &tp_vars);               \
                                                                               \
  private:                                                                     \
    template <typename R, typename... A, size_t... I, typename... K,           \
              size_t... J>                                                     \
    __VA_ARGS__ typename std::enable_if<std::is_same<R, void>::value,          \
                                        void>::type                            \
    single(args<type_sequence<A...>, index_sequence<I...>> *                   \
               DYND_IGNORE_UNUSED(args),                                       \
           kwds<type_sequence<K...>, index_sequence<J...>> *                   \
               DYND_IGNORE_UNUSED(kwds),                                       \
           char *DYND_UNUSED(dst), char *const *DYND_IGNORE_UNUSED(src))       \
    {                                                                          \
      func(static_cast<arg<A, I> *>(args)->get(src[I])...,                     \
           static_cast<kwd<K, J> *>(kwds)->get()...);                          \
    }                                                                          \
                                                                               \
    template <typename R, typename... A, size_t... I, typename... K,           \
              size_t... J>                                                     \
    __VA_ARGS__ typename std::enable_if<!std::is_same<R, void>::value,         \
                                        void>::type                            \
    single(args<type_sequence<A...>, index_sequence<I...>> *                   \
               DYND_IGNORE_UNUSED(args),                                       \
           kwds<type_sequence<K...>, index_sequence<J...>> *                   \
               DYND_IGNORE_UNUSED(kwds),                                       \
           char *dst, char *const *DYND_IGNORE_UNUSED(src))                    \
    {                                                                          \
      *reinterpret_cast<R *>(dst) =                                            \
          func(static_cast<arg<A, I> *>(args)->get(src[I])...,                 \
               static_cast<kwd<K, J> *>(kwds)->get()...);                      \
    }                                                                          \
  }

  APPLY_CALLABLE_CK(kernel_request_host);

  template <typename func_type, int Nsrc>
  intptr_t apply_callable_ck<kernel_request_host, func_type, Nsrc>::instantiate(
      const arrfunc_type_data *self, const arrfunc_type *self_tp, void *ckb,
      intptr_t ckb_offset, const ndt::type &dst_tp, const char *dst_arrmeta,
      const ndt::type *src_tp, const char *const *src_arrmeta,
      kernel_request_t kernreq, const eval::eval_context *ectx,
      const nd::array &kwds, const std::map<nd::string, ndt::type> &tp_vars)
  {
    return instantiate_without_wrapper(self, self_tp, ckb, ckb_offset, dst_tp,
                                       dst_arrmeta, src_tp, src_arrmeta,
                                       kernreq, ectx, kwds, tp_vars);
  }

#ifdef __CUDACC__

  APPLY_CALLABLE_CK(kernel_request_cuda_device, __device__);

  template <typename func_type, int Nsrc>
  intptr_t
  apply_callable_ck<kernel_request_cuda_device, func_type, Nsrc>::instantiate(
      const arrfunc_type_data *self, const arrfunc_type *self_tp, void *ckb,
      intptr_t ckb_offset, const ndt::type &dst_tp, const char *dst_arrmeta,
      const ndt::type *src_tp, const char *const *src_arrmeta,
      kernel_request_t kernreq, const eval::eval_context *ectx,
      const nd::array &kwds, const std::map<nd::string, ndt::type> &tp_vars)
  {
    return cuda_launch_ck<Nsrc>::template instantiate<
        &instantiate_without_wrapper>(self, self_tp, ckb, ckb_offset, dst_tp,
                                      dst_arrmeta, src_tp, src_arrmeta, kernreq,
                                      ectx, kwds, tp_vars);
  }

  APPLY_CALLABLE_CK(kernel_request_cuda_host_device, __host__ __device__);

  template <typename func_type, int Nsrc>
  intptr_t apply_callable_ck<kernel_request_cuda_host_device, func_type, Nsrc>::
      instantiate(const arrfunc_type_data *self, const arrfunc_type *self_tp,
                  void *ckb, intptr_t ckb_offset, const ndt::type &dst_tp,
                  const char *dst_arrmeta, const ndt::type *src_tp,
                  const char *const *src_arrmeta, kernel_request_t kernreq,
                  const eval::eval_context *ectx, const nd::array &kwds,
                  const std::map<nd::string, ndt::type> &tp_vars)
  {
    return cuda_launch_ck<Nsrc>::template instantiate<
        &instantiate_without_wrapper>(self, self_tp, ckb, ckb_offset, dst_tp,
                                      dst_arrmeta, src_tp, src_arrmeta, kernreq,
                                      ectx, kwds, tp_vars);
  }

#endif

#undef APPLY_CALLABLE_CK

#define APPLY_CALLABLE_CK(KERNREQ, ...)                                        \
  template <typename func_type, int Nsrc>                                      \
  struct apply_callable_ck<KERNREQ, func_type *, Nsrc>                         \
      : expr_ck<apply_callable_ck<KERNREQ, func_type *, Nsrc>, KERNREQ, Nsrc>, \
        args_for<func_type, Nsrc>,                                             \
        kwds_for<func_type, Nsrc> {                                            \
    typedef apply_callable_ck self_type;                                       \
                                                                               \
    func_type *func;                                                           \
                                                                               \
    __VA_ARGS__ apply_callable_ck(func_type *func,                             \
                                  args_for<func_type, Nsrc> args,              \
                                  kwds_for<func_type, Nsrc> kwds)              \
        : args_for<func_type, Nsrc>(args), kwds_for<func_type, Nsrc>(kwds),    \
          func(func)                                                           \
    {                                                                          \
    }                                                                          \
                                                                               \
    __VA_ARGS__ void single(char *dst, char *const *src)                       \
    {                                                                          \
      single<typename return_of<func_type>::type>(this, this, dst, src);       \
    }                                                                          \
                                                                               \
    static intptr_t instantiate_without_wrapper(                               \
        const arrfunc_type_data *self,                                         \
        const arrfunc_type *DYND_UNUSED(self_tp), void *ckb,                   \
        intptr_t ckb_offset, const ndt::type &DYND_UNUSED(dst_tp),             \
        const char *DYND_UNUSED(dst_arrmeta), const ndt::type *src_tp,         \
        const char *const *src_arrmeta, kernel_request_t kernreq,              \
        const eval::eval_context *DYND_UNUSED(ectx), const nd::array &kwds,    \
        const std::map<nd::string, ndt::type> &DYND_UNUSED(tp_vars))           \
    {                                                                          \
      self_type::create(ckb, kernreq, ckb_offset,                              \
                        make_value_wrapper(*self->get_data_as<func_type *>()), \
                        args_for<func_type, Nsrc>(src_tp, src_arrmeta, kwds),  \
                        kwds_for<func_type, Nsrc>(kwds));                      \
      return ckb_offset;                                                       \
    }                                                                          \
                                                                               \
    static intptr_t                                                            \
    instantiate(const arrfunc_type_data *self, const arrfunc_type *self_tp,    \
                void *ckb, intptr_t ckb_offset, const ndt::type &dst_tp,       \
                const char *dst_arrmeta, const ndt::type *src_tp,              \
                const char *const *src_arrmeta, kernel_request_t kernreq,      \
                const eval::eval_context *ectx, const nd::array &kwds,         \
                const std::map<nd::string, ndt::type> &tp_vars);               \
                                                                               \
  private:                                                                     \
    template <typename R, typename... A, size_t... I, typename... K,           \
              size_t... J>                                                     \
    __VA_ARGS__ typename std::enable_if<std::is_same<R, void>::value,          \
                                        void>::type                            \
    single(args<type_sequence<A...>, index_sequence<I...>> *                   \
               DYND_IGNORE_UNUSED(args),                                       \
           kwds<type_sequence<K...>, index_sequence<J...>> *                   \
               DYND_IGNORE_UNUSED(kwds),                                       \
           char *DYND_UNUSED(dst), char *const *DYND_IGNORE_UNUSED(src))       \
    {                                                                          \
      (*func)(static_cast<arg<A, I> *>(args)->get(src[I])...,                  \
              static_cast<kwd<K, J> *>(kwds)->get()...);                       \
    }                                                                          \
                                                                               \
    template <typename R, typename... A, size_t... I, typename... K,           \
              size_t... J>                                                     \
    __VA_ARGS__ typename std::enable_if<!std::is_same<R, void>::value,         \
                                        void>::type                            \
    single(args<type_sequence<A...>, index_sequence<I...>> *                   \
               DYND_IGNORE_UNUSED(args),                                       \
           kwds<type_sequence<K...>, index_sequence<J...>> *                   \
               DYND_IGNORE_UNUSED(kwds),                                       \
           char *dst, char *const *DYND_IGNORE_UNUSED(src))                    \
    {                                                                          \
      *reinterpret_cast<R *>(dst) =                                            \
          (*func)(static_cast<arg<A, I> *>(args)->get(src[I])...,              \
                  static_cast<kwd<K, J> *>(kwds)->get()...);                   \
    }                                                                          \
  }

  APPLY_CALLABLE_CK(kernel_request_host);

  template <typename func_type, int Nsrc>
  intptr_t
  apply_callable_ck<kernel_request_host, func_type *, Nsrc>::instantiate(
      const arrfunc_type_data *self, const arrfunc_type *self_tp, void *ckb,
      intptr_t ckb_offset, const ndt::type &dst_tp, const char *dst_arrmeta,
      const ndt::type *src_tp, const char *const *src_arrmeta,
      kernel_request_t kernreq, const eval::eval_context *ectx,
      const nd::array &kwds, const std::map<nd::string, ndt::type> &tp_vars)
  {
    return instantiate_without_wrapper(self, self_tp, ckb, ckb_offset, dst_tp,
                                       dst_arrmeta, src_tp, src_arrmeta,
                                       kernreq, ectx, kwds, tp_vars);
  }

#undef APPLY_CALLABLE_CK

  template <kernel_request_t kernreq, typename func_type, typename... K>
  struct construct_then_apply_callable_ck;

#define CONSTRUCT_THEN_APPLY_CALLABLE_CK(KERNREQ, ...)                         \
  template <typename func_type, typename... K>                                 \
  struct construct_then_apply_callable_ck<KERNREQ, func_type, K...>            \
      : expr_ck<construct_then_apply_callable_ck<KERNREQ, func_type, K...>,    \
                KERNREQ, arity_of<func_type>::value>,                          \
        args_for<func_type> {                                                  \
    typedef construct_then_apply_callable_ck<KERNREQ, func_type, K...>         \
        self_type;                                                             \
                                                                               \
    func_type func;                                                            \
                                                                               \
    template <size_t... J>                                                     \
    __VA_ARGS__ construct_then_apply_callable_ck(                              \
        args_for<func_type> args,                                              \
        kwds<type_sequence<K...>, index_sequence<J...>> DYND_IGNORE_UNUSED(    \
            kwds))                                                             \
        : args_for<func_type>(args), func(kwds.kwd<K, J>::get()...)            \
    {                                                                          \
    }                                                                          \
                                                                               \
    __VA_ARGS__ void single(char *dst, char *const *src)                       \
    {                                                                          \
      single<typename return_of<func_type>::type>(this, dst, src);             \
    }                                                                          \
                                                                               \
    static intptr_t instantiate_without_wrapper(                               \
        const arrfunc_type_data *DYND_UNUSED(self),                            \
        const arrfunc_type *DYND_UNUSED(self_tp), void *ckb,                   \
        intptr_t ckb_offset, const ndt::type &DYND_UNUSED(dst_tp),             \
        const char *DYND_UNUSED(dst_arrmeta), const ndt::type *src_tp,         \
        const char *const *src_arrmeta, kernel_request_t kernreq,              \
        const eval::eval_context *DYND_UNUSED(ectx), const nd::array &kwds,    \
        const std::map<nd::string, ndt::type> &DYND_UNUSED(tp_vars))           \
    {                                                                          \
      self_type::create(ckb, kernreq, ckb_offset,                              \
                        args_for<func_type>(src_tp, src_arrmeta, kwds),        \
                        kernels::kwds<type_sequence<K...>>(kwds));             \
      return ckb_offset;                                                       \
    }                                                                          \
                                                                               \
    static intptr_t                                                            \
    instantiate(const arrfunc_type_data *self, const arrfunc_type *self_tp,    \
                void *ckb, intptr_t ckb_offset, const ndt::type &dst_tp,       \
                const char *dst_arrmeta, const ndt::type *src_tp,              \
                const char *const *src_arrmeta, kernel_request_t kernreq,      \
                const eval::eval_context *ectx, const nd::array &kwds,         \
                const std::map<nd::string, ndt::type> &tp_vars);               \
                                                                               \
  private:                                                                     \
    template <typename R, typename... A, size_t... I>                          \
    __VA_ARGS__ typename std::enable_if<std::is_same<R, void>::value,          \
                                        void>::type                            \
    single(args<type_sequence<A...>, index_sequence<I...>> *                   \
               DYND_IGNORE_UNUSED(args),                                       \
           char *DYND_UNUSED(dst), char *const *DYND_IGNORE_UNUSED(src))       \
    {                                                                          \
      func(static_cast<arg<A, I> *>(args)->get(src[I])...);                    \
    }                                                                          \
                                                                               \
    template <typename R, typename... A, size_t... I>                          \
    __VA_ARGS__ typename std::enable_if<!std::is_same<R, void>::value,         \
                                        void>::type                            \
    single(args<type_sequence<A...>, index_sequence<I...>> *                   \
               DYND_IGNORE_UNUSED(args),                                       \
           char *dst, char *const *DYND_IGNORE_UNUSED(src))                    \
    {                                                                          \
      *reinterpret_cast<R *>(dst) =                                            \
          func(static_cast<arg<A, I> *>(args)->get(src[I])...);                \
    }                                                                          \
  };

  CONSTRUCT_THEN_APPLY_CALLABLE_CK(kernel_request_host)

  template <typename func_type, typename... K>
  intptr_t construct_then_apply_callable_ck<
      kernel_request_host, func_type,
      K...>::instantiate(const arrfunc_type_data *self,
                         const arrfunc_type *self_tp, void *ckb,
                         intptr_t ckb_offset, const ndt::type &dst_tp,
                         const char *dst_arrmeta, const ndt::type *src_tp,
                         const char *const *src_arrmeta,
                         kernel_request_t kernreq,
                         const eval::eval_context *ectx, const nd::array &kwds,
                         const std::map<nd::string, ndt::type> &tp_vars)
  {
    return instantiate_without_wrapper(self, self_tp, ckb, ckb_offset, dst_tp,
                                       dst_arrmeta, src_tp, src_arrmeta,
                                       kernreq, ectx, kwds, tp_vars);
  }

#ifdef __CUDACC__

  CONSTRUCT_THEN_APPLY_CALLABLE_CK(kernel_request_cuda_device, __device__)

  template <typename func_type, typename... K>
  intptr_t construct_then_apply_callable_ck<
      kernel_request_cuda_device, func_type,
      K...>::instantiate(const arrfunc_type_data *self,
                         const arrfunc_type *self_tp, void *ckb,
                         intptr_t ckb_offset, const ndt::type &dst_tp,
                         const char *dst_arrmeta, const ndt::type *src_tp,
                         const char *const *src_arrmeta,
                         kernel_request_t kernreq,
                         const eval::eval_context *ectx, const nd::array &kwds,
                         const std::map<nd::string, ndt::type> &tp_vars)
  {
    return cuda_launch_ck<arity_of<func_type>::value>::template instantiate<
        &instantiate_without_wrapper>(self, self_tp, ckb, ckb_offset, dst_tp,
                                      dst_arrmeta, src_tp, src_arrmeta, kernreq,
                                      ectx, kwds, tp_vars);
  }

#endif

#undef CONSTRUCT_THEN_APPLY_CALLABLE_CK

} // namespace dynd::kernels
} // namespace dynd