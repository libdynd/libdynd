//
// Copyright (C) 2011-15 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#pragma once

#include <dynd/strided_vals.hpp>
#include <dynd/kernels/cuda_launch.hpp>
#include <dynd/kernels/base_kernel.hpp>
#include <dynd/gfunc/call_gcallable.hpp>

namespace dynd {
namespace nd {
  namespace functional {

    template <typename A, size_t I>
    struct apply_arg {
      typedef typename std::remove_cv<typename std::remove_reference<A>::type>::type D;

      apply_arg(const ndt::type &DYND_UNUSED(tp), const char *DYND_UNUSED(arrmeta), const nd::array *DYND_UNUSED(kwds))
      {
      }

      DYND_CUDA_HOST_DEVICE D &get(char *data)
      {
        return *reinterpret_cast<D *>(data);
      }
    };

    template <typename T, int N, size_t I>
    struct apply_arg<const nd::strided_vals<T, N> &, I> {
      nd::strided_vals<T, N> m_vals;

      apply_arg(const ndt::type &DYND_UNUSED(tp), const char *arrmeta, const nd::array *kwds)
      {
        m_vals.set_data(NULL, reinterpret_cast<const size_stride_t *>(arrmeta),
                        reinterpret_cast<start_stop_t *>(kwds[3].as<intptr_t>()));

        const nd::array &mask = kwds[2];
        if (mask.is_missing()) {
          m_vals.set_mask(NULL);
        } else {
          m_vals.set_mask(mask.cdata(), reinterpret_cast<const size_stride_t *>(mask.get()->metadata()));
        }
      }

      nd::strided_vals<T, N> &get(char *data)
      {
        m_vals.set_data(data);
        return m_vals;
      }
    };

    template <typename ElementType, size_t I>
    struct apply_arg<fixed_dim<ElementType>, I> {
      fixed_dim<ElementType> value;

      apply_arg(const ndt::type &DYND_UNUSED(tp), const char *arrmeta, const nd::array *DYND_UNUSED(kwds))
          : value(arrmeta, NULL)
      {
      }

      fixed_dim<ElementType> &get(char *data)
      {
        value.set_data(data);
        return value;
      }
    };

    template <typename func_type, int N = args_of<typename funcproto_of<func_type>::type>::type::size>
    using as_apply_arg_sequence = typename to<typename args_of<typename funcproto_of<func_type>::type>::type, N>::type;

    template <typename A, typename I = make_index_sequence<A::size>>
    struct apply_args;

    template <typename... A, size_t... I>
    struct apply_args<type_sequence<A...>, index_sequence<I...>> : apply_arg<A, I>... {
      apply_args(const ndt::type *DYND_IGNORE_UNUSED(src_tp), const char *const *DYND_IGNORE_UNUSED(src_arrmeta),
                 const nd::array *DYND_IGNORE_UNUSED(kwds))
          : apply_arg<A, I>(src_tp[I], src_arrmeta[I], kwds)...
      {
      }
    };

    template <typename T, size_t I>
    struct apply_kwd {
      T m_val;

      apply_kwd(nd::array val)
      //        : m_val(val.as<T>())
      {
        if (val.get_type().get_type_id() == pointer_type_id) {
          m_val = val.f("dereference").as<T>();
        } else {
          m_val = val.as<T>();
        }
      }

      DYND_CUDA_HOST_DEVICE T get()
      {
        return m_val;
      }
    };

    template <typename K, typename J = make_index_sequence<K::size>>
    struct apply_kwds;

    template <>
    struct apply_kwds<type_sequence<>, index_sequence<>> {
      apply_kwds(intptr_t DYND_UNUSED(nkwd), const nd::array *DYND_UNUSED(kwds))
      {
      }
    };

    template <typename... K, size_t... J>
    struct apply_kwds<type_sequence<K...>, index_sequence<J...>> : apply_kwd<K, J>... {
      apply_kwds(intptr_t DYND_UNUSED(nkwd), const nd::array *kwds) : apply_kwd<K, J>(kwds[J])...
      {
      }
    };

    template <typename func_type, int N>
    using as_apply_kwd_sequence =
        typename from<typename args_of<typename funcproto_of<func_type>::type>::type, N>::type;

    template <typename func_type, func_type func, typename R, typename A, typename I, typename K, typename J>
    struct apply_function_ck;

#define APPLY_FUNCTION_CK(...)                                                                                         \
  template <typename func_type, func_type func, typename R, typename... A, size_t... I, typename... K, size_t... J>    \
  struct apply_function_ck<                                                                                            \
      func_type, func, R, type_sequence<A...>, index_sequence<I...>, type_sequence<K...>,                              \
      index_sequence<J...>> : base_kernel<apply_function_ck<func_type, func, R, type_sequence<A...>,                   \
                                                            index_sequence<I...>, type_sequence<K...>,                 \
                                                            index_sequence<J...>>,                                     \
                                          sizeof...(A)>,                                                               \
                              apply_args<type_sequence<A...>, index_sequence<I...>>,                                   \
                              apply_kwds<type_sequence<K...>, index_sequence<J...>> {                                  \
    typedef apply_function_ck self_type;                                                                               \
    typedef apply_args<type_sequence<A...>, index_sequence<I...>> args_type;                                           \
    typedef apply_kwds<type_sequence<K...>, index_sequence<J...>> kwds_type;                                           \
                                                                                                                       \
    __VA_ARGS__ apply_function_ck(args_type args, kwds_type kwds) : args_type(args), kwds_type(kwds)                   \
    {                                                                                                                  \
    }                                                                                                                  \
                                                                                                                       \
    __VA_ARGS__ void single(char *dst, char *const *DYND_IGNORE_UNUSED(src))                                           \
    {                                                                                                                  \
      *reinterpret_cast<R *>(dst) = func(apply_arg<A, I>::get(src[I])..., apply_kwd<K, J>::get()...);                  \
    }                                                                                                                  \
                                                                                                                       \
    __VA_ARGS__ void strided(char *dst, intptr_t dst_stride, char *const *DYND_IGNORE_UNUSED(src_copy),                \
                             const intptr_t *DYND_IGNORE_UNUSED(src_stride), size_t count)                             \
    {                                                                                                                  \
      dynd::detail::array_wrapper<char *, sizeof...(A)> src;                                                           \
                                                                                                                       \
      dst += DYND_THREAD_ID(0) * dst_stride;                                                                           \
      for (size_t j = 0; j != sizeof...(A); ++j) {                                                                     \
        src[j] = src_copy[j] + DYND_THREAD_ID(0) * src_stride[j];                                                      \
      }                                                                                                                \
                                                                                                                       \
      for (std::ptrdiff_t i = DYND_THREAD_ID(0); i < (std::ptrdiff_t)count; i += DYND_THREAD_COUNT(0)) {               \
        *reinterpret_cast<R *>(dst) = func(apply_arg<A, I>::get(src[I])..., apply_kwd<K, J>::get()...);                \
        dst += DYND_THREAD_COUNT(0) * dst_stride;                                                                      \
        for (size_t j = 0; j != sizeof...(A); ++j) {                                                                   \
          src[j] += DYND_THREAD_COUNT(0) * src_stride[j];                                                              \
        }                                                                                                              \
      }                                                                                                                \
    }                                                                                                                  \
                                                                                                                       \
    static intptr_t instantiate(char *DYND_UNUSED(static_data), char *DYND_UNUSED(data), void *ckb,                    \
                                intptr_t ckb_offset, const ndt::type &DYND_UNUSED(dst_tp),                             \
                                const char *DYND_UNUSED(dst_arrmeta), intptr_t DYND_UNUSED(nsrc),                      \
                                const ndt::type *src_tp, const char *const *src_arrmeta, kernel_request_t kernreq,     \
                                const eval::eval_context *DYND_UNUSED(ectx), intptr_t nkwd, const nd::array *kwds,     \
                                const std::map<std::string, ndt::type> &DYND_UNUSED(tp_vars))                          \
    {                                                                                                                  \
      self_type::make(ckb, kernreq, ckb_offset, args_type(src_tp, src_arrmeta, kwds), kwds_type(nkwd, kwds));          \
      return ckb_offset;                                                                                               \
    }                                                                                                                  \
  };                                                                                                                   \
                                                                                                                       \
  template <typename func_type, func_type func, typename... A, size_t... I, typename... K, size_t... J>                \
  struct apply_function_ck<                                                                                            \
      func_type, func, void, type_sequence<A...>, index_sequence<I...>, type_sequence<K...>,                           \
      index_sequence<J...>> : base_kernel<apply_function_ck<func_type, func, void, type_sequence<A...>,                \
                                                            index_sequence<I...>, type_sequence<K...>,                 \
                                                            index_sequence<J...>>,                                     \
                                          sizeof...(A)>,                                                               \
                              apply_args<type_sequence<A...>, index_sequence<I...>>,                                   \
                              apply_kwds<type_sequence<K...>, index_sequence<J...>> {                                  \
    typedef apply_function_ck self_type;                                                                               \
    typedef apply_args<type_sequence<A...>, index_sequence<I...>> args_type;                                           \
    typedef apply_kwds<type_sequence<K...>, index_sequence<J...>> kwds_type;                                           \
                                                                                                                       \
    __VA_ARGS__ apply_function_ck(args_type args, kwds_type kwds) : args_type(args), kwds_type(kwds)                   \
    {                                                                                                                  \
    }                                                                                                                  \
                                                                                                                       \
    __VA_ARGS__ void single(char *DYND_UNUSED(dst), char *const *DYND_IGNORE_UNUSED(src))                              \
    {                                                                                                                  \
      func(apply_arg<A, I>::get(src[I])..., apply_kwd<K, J>::get()...);                                                \
    }                                                                                                                  \
                                                                                                                       \
    __VA_ARGS__ void strided(char *DYND_UNUSED(dst), intptr_t DYND_UNUSED(dst_stride),                                 \
                             char *const *DYND_IGNORE_UNUSED(src_copy),                                                \
                             const intptr_t *DYND_IGNORE_UNUSED(src_stride), size_t count)                             \
    {                                                                                                                  \
      dynd::detail::array_wrapper<char *, sizeof...(A)> src;                                                           \
                                                                                                                       \
      for (size_t j = 0; j != sizeof...(A); ++j) {                                                                     \
        src[j] = src_copy[j] + DYND_THREAD_ID(0) * src_stride[j];                                                      \
      }                                                                                                                \
                                                                                                                       \
      for (std::ptrdiff_t i = DYND_THREAD_ID(0); i < (std::ptrdiff_t)count; i += DYND_THREAD_COUNT(0)) {               \
        func(apply_arg<A, I>::get(src[I])..., apply_kwd<K, J>::get()...);                                              \
        for (size_t j = 0; j != sizeof...(A); ++j) {                                                                   \
          src[j] += DYND_THREAD_COUNT(0) * src_stride[j];                                                              \
        }                                                                                                              \
      }                                                                                                                \
    }                                                                                                                  \
                                                                                                                       \
    static intptr_t instantiate(char *DYND_UNUSED(static_data), char *DYND_UNUSED(data), void *ckb,                    \
                                intptr_t ckb_offset, const ndt::type &DYND_UNUSED(dst_tp),                             \
                                const char *DYND_UNUSED(dst_arrmeta), intptr_t DYND_UNUSED(nsrc),                      \
                                const ndt::type *src_tp, const char *const *src_arrmeta, kernel_request_t kernreq,     \
                                const eval::eval_context *DYND_UNUSED(ectx), intptr_t nkwd, const nd::array *kwds,     \
                                const std::map<std::string, ndt::type> &DYND_UNUSED(tp_vars))                          \
    {                                                                                                                  \
      self_type::make(ckb, kernreq, ckb_offset, args_type(src_tp, src_arrmeta, kwds), kwds_type(nkwd, kwds));          \
      return ckb_offset;                                                                                               \
    }                                                                                                                  \
  }

    APPLY_FUNCTION_CK();

#undef APPLY_FUNCTION_CK

    template <typename func_type, func_type func, int N = arity_of<func_type>::value>
    using as_apply_function_ck =
        apply_function_ck<func_type, func, typename return_of<func_type>::type, as_apply_arg_sequence<func_type, N>,
                          make_index_sequence<N>, as_apply_kwd_sequence<func_type, N>,
                          make_index_sequence<arity_of<func_type>::value - N>>;

    template <typename T, typename mem_func_type, typename R, typename A, typename I, typename K, typename J>
    struct apply_member_function_ck;

#define APPLY_MEMBER_FUNCTION_CK(...)                                                                                  \
  template <typename T, typename mem_func_type, typename R, typename... A, size_t... I, typename... K, size_t... J>    \
  struct apply_member_function_ck<                                                                                     \
      T *, mem_func_type, R, type_sequence<A...>, index_sequence<I...>, type_sequence<K...>,                           \
      index_sequence<J...>> : base_kernel<apply_member_function_ck<T *, mem_func_type, R, type_sequence<A...>,         \
                                                                   index_sequence<I...>, type_sequence<K...>,          \
                                                                   index_sequence<J...>>,                              \
                                          sizeof...(A)>,                                                               \
                              apply_args<type_sequence<A...>, index_sequence<I...>>,                                   \
                              apply_kwds<type_sequence<K...>, index_sequence<J...>> {                                  \
    typedef apply_member_function_ck self_type;                                                                        \
    typedef apply_args<type_sequence<A...>, index_sequence<I...>> args_type;                                           \
    typedef apply_kwds<type_sequence<K...>, index_sequence<J...>> kwds_type;                                           \
    typedef std::pair<T *, mem_func_type> data_type;                                                                   \
                                                                                                                       \
    T *obj;                                                                                                            \
    mem_func_type mem_func;                                                                                            \
                                                                                                                       \
    __VA_ARGS__ apply_member_function_ck(T *obj, mem_func_type mem_func, args_type args, kwds_type kwds)               \
        : args_type(args), kwds_type(kwds), obj(obj), mem_func(mem_func)                                               \
    {                                                                                                                  \
    }                                                                                                                  \
                                                                                                                       \
    __VA_ARGS__ void single(char *dst, char *const *DYND_IGNORE_UNUSED(src))                                           \
    {                                                                                                                  \
      *reinterpret_cast<R *>(dst) = (obj->*mem_func)(apply_arg<A, I>::get(src[I])..., apply_kwd<K, J>::get()...);      \
    }                                                                                                                  \
                                                                                                                       \
    __VA_ARGS__ void strided(char *dst, intptr_t dst_stride, char *const *DYND_IGNORE_UNUSED(src_copy),                \
                             const intptr_t *DYND_IGNORE_UNUSED(src_stride), size_t count)                             \
    {                                                                                                                  \
      dynd::detail::array_wrapper<char *, sizeof...(A)> src;                                                           \
                                                                                                                       \
      dst += DYND_THREAD_ID(0) * dst_stride;                                                                           \
      for (size_t j = 0; j != sizeof...(A); ++j) {                                                                     \
        src[j] = src_copy[j] + DYND_THREAD_ID(0) * src_stride[j];                                                      \
      }                                                                                                                \
                                                                                                                       \
      for (std::ptrdiff_t i = DYND_THREAD_ID(0); i < (std::ptrdiff_t)count; i += DYND_THREAD_COUNT(0)) {               \
        *reinterpret_cast<R *>(dst) = (obj->*mem_func)(apply_arg<A, I>::get(src[I])..., apply_kwd<K, J>::get()...);    \
        dst += DYND_THREAD_COUNT(0) * dst_stride;                                                                      \
        for (size_t j = 0; j != sizeof...(A); ++j) {                                                                   \
          src[j] += DYND_THREAD_COUNT(0) * src_stride[j];                                                              \
        }                                                                                                              \
      }                                                                                                                \
    }                                                                                                                  \
                                                                                                                       \
    static intptr_t instantiate_without_cuda_launch(char *static_data, char *DYND_UNUSED(data), void *ckb,             \
                                                    intptr_t ckb_offset, const ndt::type &DYND_UNUSED(dst_tp),         \
                                                    const char *DYND_UNUSED(dst_arrmeta), intptr_t DYND_UNUSED(nsrc),  \
                                                    const ndt::type *src_tp, const char *const *src_arrmeta,           \
                                                    kernel_request_t kernreq,                                          \
                                                    const eval::eval_context *DYND_UNUSED(ectx), intptr_t nkwd,        \
                                                    const nd::array *kwds,                                             \
                                                    const std::map<std::string, ndt::type> &DYND_UNUSED(tp_vars))      \
    {                                                                                                                  \
      self_type::make(ckb, kernreq, ckb_offset, reinterpret_cast<data_type *>(static_data)->first,                     \
                      dynd::detail::make_value_wrapper(reinterpret_cast<data_type *>(static_data)->second),            \
                      args_type(src_tp, src_arrmeta, kwds), kwds_type(nkwd, kwds));                                    \
      return ckb_offset;                                                                                               \
    }                                                                                                                  \
                                                                                                                       \
    static intptr_t instantiate(char *DYND_UNUSED(static_data), char *DYND_UNUSED(data), void *ckb,                    \
                                intptr_t ckb_offset, const ndt::type &dst_tp, const char *dst_arrmeta, intptr_t nsrc,  \
                                const ndt::type *src_tp, const char *const *src_arrmeta, kernel_request_t kernreq,     \
                                const eval::eval_context *ectx, intptr_t nkwd, const nd::array *kwds,                  \
                                const std::map<std::string, ndt::type> &tp_vars);                                      \
  };                                                                                                                   \
                                                                                                                       \
  template <typename T, typename mem_func_type, typename... A, size_t... I, typename... K, size_t... J>                \
  struct apply_member_function_ck<                                                                                     \
      T *, mem_func_type, void, type_sequence<A...>, index_sequence<I...>, type_sequence<K...>,                        \
      index_sequence<J...>> : base_kernel<apply_member_function_ck<T *, mem_func_type, void, type_sequence<A...>,      \
                                                                   index_sequence<I...>, type_sequence<K...>,          \
                                                                   index_sequence<J...>>,                              \
                                          sizeof...(A)>,                                                               \
                              apply_args<type_sequence<A...>, index_sequence<I...>>,                                   \
                              apply_kwds<type_sequence<K...>, index_sequence<J...>> {                                  \
    typedef apply_member_function_ck self_type;                                                                        \
    typedef apply_args<type_sequence<A...>, index_sequence<I...>> args_type;                                           \
    typedef apply_kwds<type_sequence<K...>, index_sequence<J...>> kwds_type;                                           \
    typedef std::pair<T *, mem_func_type> data_type;                                                                   \
                                                                                                                       \
    T *obj;                                                                                                            \
    mem_func_type mem_func;                                                                                            \
                                                                                                                       \
    __VA_ARGS__ apply_member_function_ck(T *obj, mem_func_type mem_func, args_type args, kwds_type kwds)               \
        : args_type(args), kwds_type(kwds), obj(obj), mem_func(mem_func)                                               \
    {                                                                                                                  \
    }                                                                                                                  \
                                                                                                                       \
    __VA_ARGS__ void single(char *DYND_UNUSED(dst), char *const *DYND_IGNORE_UNUSED(src))                              \
    {                                                                                                                  \
      (obj->*mem_func)(apply_arg<A, I>::get(src[I])..., apply_kwd<K, J>::get()...);                                    \
    }                                                                                                                  \
                                                                                                                       \
    __VA_ARGS__ void strided(char *DYND_UNUSED(dst), intptr_t DYND_UNUSED(dst_stride),                                 \
                             char *const *DYND_IGNORE_UNUSED(src_copy),                                                \
                             const intptr_t *DYND_IGNORE_UNUSED(src_stride), size_t count)                             \
    {                                                                                                                  \
      dynd::detail::array_wrapper<char *, sizeof...(A)> src;                                                           \
                                                                                                                       \
      for (size_t j = 0; j != sizeof...(A); ++j) {                                                                     \
        src[j] = src_copy[j] + DYND_THREAD_ID(0) * src_stride[j];                                                      \
      }                                                                                                                \
                                                                                                                       \
      for (std::ptrdiff_t i = DYND_THREAD_ID(0); i < (std::ptrdiff_t)count; i += DYND_THREAD_COUNT(0)) {               \
        (obj->*mem_func)(apply_arg<A, I>::get(src[I])..., apply_kwd<K, J>::get()...);                                  \
        for (size_t j = 0; j != sizeof...(A); ++j) {                                                                   \
          src[j] += DYND_THREAD_COUNT(0) * src_stride[j];                                                              \
        }                                                                                                              \
      }                                                                                                                \
    }                                                                                                                  \
                                                                                                                       \
    static intptr_t instantiate_without_cuda_launch(char *static_data, char *DYND_UNUSED(data), void *ckb,             \
                                                    intptr_t ckb_offset, const ndt::type &DYND_UNUSED(dst_tp),         \
                                                    const char *DYND_UNUSED(dst_arrmeta), intptr_t DYND_UNUSED(nsrc),  \
                                                    const ndt::type *src_tp, const char *const *src_arrmeta,           \
                                                    kernel_request_t kernreq,                                          \
                                                    const eval::eval_context *DYND_UNUSED(ectx), intptr_t nkwd,        \
                                                    const nd::array *kwds,                                             \
                                                    const std::map<std::string, ndt::type> &DYND_UNUSED(tp_vars))      \
    {                                                                                                                  \
      self_type::make(ckb, kernreq, ckb_offset, reinterpret_cast<data_type *>(static_data)->first,                     \
                      dynd::detail::make_value_wrapper(reinterpret_cast<data_type *>(static_data)->second),            \
                      args_type(src_tp, src_arrmeta, kwds), kwds_type(nkwd, kwds));                                    \
      return ckb_offset;                                                                                               \
    }                                                                                                                  \
                                                                                                                       \
    static intptr_t instantiate(char *DYND_UNUSED(static_data), char *DYND_UNUSED(data), void *ckb,                    \
                                intptr_t ckb_offset, const ndt::type &dst_tp, const char *dst_arrmeta, intptr_t nsrc,  \
                                const ndt::type *src_tp, const char *const *src_arrmeta, kernel_request_t kernreq,     \
                                const eval::eval_context *ectx, intptr_t nkwd, const nd::array *kwds,                  \
                                const std::map<std::string, ndt::type> &tp_vars);                                      \
  }

    APPLY_MEMBER_FUNCTION_CK();

    template <typename T, typename mem_func_type, typename R, typename... A, size_t... I, typename... K, size_t... J>
    intptr_t
    apply_member_function_ck<T *, mem_func_type, R, type_sequence<A...>, index_sequence<I...>, type_sequence<K...>,
                             index_sequence<J...>>::instantiate(char *static_data, char *DYND_UNUSED(data), void *ckb,
                                                                intptr_t ckb_offset, const ndt::type &dst_tp,
                                                                const char *dst_arrmeta, intptr_t nsrc,
                                                                const ndt::type *src_tp, const char *const *src_arrmeta,
                                                                kernel_request_t kernreq,
                                                                const eval::eval_context *ectx, intptr_t nkwd,
                                                                const nd::array *kwds,
                                                                const std::map<std::string, ndt::type> &tp_vars)
    {
      return instantiate_without_cuda_launch(static_data, NULL, ckb, ckb_offset, dst_tp, dst_arrmeta, nsrc, src_tp,
                                             src_arrmeta, kernreq, ectx, nkwd, kwds, tp_vars);
    }

    template <typename T, typename mem_func_type, typename... A, size_t... I, typename... K, size_t... J>
    intptr_t
    apply_member_function_ck<T *, mem_func_type, void, type_sequence<A...>, index_sequence<I...>, type_sequence<K...>,
                             index_sequence<J...>>::instantiate(char *static_data, char *DYND_UNUSED(data), void *ckb,
                                                                intptr_t ckb_offset, const ndt::type &dst_tp,
                                                                const char *dst_arrmeta, intptr_t nsrc,
                                                                const ndt::type *src_tp, const char *const *src_arrmeta,
                                                                kernel_request_t kernreq,
                                                                const eval::eval_context *ectx, intptr_t nkwd,
                                                                const nd::array *kwds,
                                                                const std::map<std::string, ndt::type> &tp_vars)
    {
      return instantiate_without_cuda_launch(static_data, NULL, ckb, ckb_offset, dst_tp, dst_arrmeta, nsrc, src_tp,
                                             src_arrmeta, kernreq, ectx, nkwd, kwds, tp_vars);
    }

#undef APPLY_MEMBER_FUNCTION_CK

    template <typename T, typename mem_func_type, int N>
    using as_apply_member_function_ck =
        apply_member_function_ck<T, mem_func_type, typename return_of<mem_func_type>::type,
                                 as_apply_arg_sequence<mem_func_type, N>, make_index_sequence<N>,
                                 as_apply_kwd_sequence<mem_func_type, N>,
                                 make_index_sequence<arity_of<mem_func_type>::value - N>>;

    template <typename func_type, typename R, typename A, typename I, typename K, typename J>
    struct apply_callable_ck;

#define APPLY_CALLABLE_CK(...)                                                                                         \
  template <typename func_type, typename R, typename... A, size_t... I, typename... K, size_t... J>                    \
  struct apply_callable_ck<                                                                                            \
      func_type, R, type_sequence<A...>, index_sequence<I...>, type_sequence<K...>,                                    \
      index_sequence<J...>> : base_kernel<apply_callable_ck<func_type, R, type_sequence<A...>, index_sequence<I...>,   \
                                                            type_sequence<K...>, index_sequence<J...>>,                \
                                          sizeof...(A)>,                                                               \
                              apply_args<type_sequence<A...>, index_sequence<I...>>,                                   \
                              apply_kwds<type_sequence<K...>, index_sequence<J...>> {                                  \
    typedef apply_callable_ck self_type;                                                                               \
    typedef apply_args<type_sequence<A...>, index_sequence<I...>> args_type;                                           \
    typedef apply_kwds<type_sequence<K...>, index_sequence<J...>> kwds_type;                                           \
                                                                                                                       \
    func_type func;                                                                                                    \
                                                                                                                       \
    __VA_ARGS__ apply_callable_ck(func_type func, args_type args, kwds_type kwds)                                      \
        : args_type(args), kwds_type(kwds), func(func)                                                                 \
    {                                                                                                                  \
    }                                                                                                                  \
                                                                                                                       \
    __VA_ARGS__ void single(char *dst, char *const *DYND_IGNORE_UNUSED(src))                                           \
    {                                                                                                                  \
      *reinterpret_cast<R *>(dst) = func(apply_arg<A, I>::get(src[I])..., apply_kwd<K, J>::get()...);                  \
    }                                                                                                                  \
                                                                                                                       \
    __VA_ARGS__ void strided(char *dst, intptr_t dst_stride, char *const *DYND_IGNORE_UNUSED(src_copy),                \
                             const intptr_t *DYND_IGNORE_UNUSED(src_stride), size_t count)                             \
    {                                                                                                                  \
      dynd::detail::array_wrapper<char *, sizeof...(A)> src;                                                           \
                                                                                                                       \
      dst += DYND_THREAD_ID(0) * dst_stride;                                                                           \
      for (size_t j = 0; j != sizeof...(A); ++j) {                                                                     \
        src[j] = src_copy[j] + DYND_THREAD_ID(0) * src_stride[j];                                                      \
      }                                                                                                                \
                                                                                                                       \
      for (std::ptrdiff_t i = DYND_THREAD_ID(0); i < (std::ptrdiff_t)count; i += DYND_THREAD_COUNT(0)) {               \
        *reinterpret_cast<R *>(dst) = func(apply_arg<A, I>::get(src[I])..., apply_kwd<K, J>::get()...);                \
        dst += DYND_THREAD_COUNT(0) * dst_stride;                                                                      \
        for (size_t j = 0; j != sizeof...(A); ++j) {                                                                   \
          src[j] += DYND_THREAD_COUNT(0) * src_stride[j];                                                              \
        }                                                                                                              \
      }                                                                                                                \
    }                                                                                                                  \
                                                                                                                       \
    static intptr_t instantiate_without_cuda_launch(char *static_data, char *DYND_UNUSED(data), void *ckb,             \
                                                    intptr_t ckb_offset, const ndt::type &DYND_UNUSED(dst_tp),         \
                                                    const char *DYND_UNUSED(dst_arrmeta), intptr_t DYND_UNUSED(nsrc),  \
                                                    const ndt::type *src_tp, const char *const *src_arrmeta,           \
                                                    kernel_request_t kernreq,                                          \
                                                    const eval::eval_context *DYND_UNUSED(ectx), intptr_t nkwd,        \
                                                    const nd::array *kwds,                                             \
                                                    const std::map<std::string, ndt::type> &DYND_UNUSED(tp_vars))      \
    {                                                                                                                  \
      self_type::make(ckb, kernreq, ckb_offset,                                                                        \
                      dynd::detail::make_value_wrapper(*reinterpret_cast<func_type *>(static_data)),                   \
                      args_type(src_tp, src_arrmeta, kwds), kwds_type(nkwd, kwds));                                    \
      return ckb_offset;                                                                                               \
    }                                                                                                                  \
                                                                                                                       \
    static intptr_t instantiate(char *DYND_UNUSED(static_data), char *DYND_UNUSED(data), void *ckb,                    \
                                intptr_t ckb_offset, const ndt::type &dst_tp, const char *dst_arrmeta, intptr_t nsrc,  \
                                const ndt::type *src_tp, const char *const *src_arrmeta, kernel_request_t kernreq,     \
                                const eval::eval_context *ectx, intptr_t nkwd, const nd::array *kwds,                  \
                                const std::map<std::string, ndt::type> &tp_vars);                                      \
  };                                                                                                                   \
                                                                                                                       \
  template <typename func_type, typename... A, size_t... I, typename... K, size_t... J>                                \
  struct apply_callable_ck<                                                                                            \
      func_type, void, type_sequence<A...>, index_sequence<I...>, type_sequence<K...>,                                 \
      index_sequence<J...>> : base_kernel<apply_callable_ck<func_type, void, type_sequence<A...>,                      \
                                                            index_sequence<I...>, type_sequence<K...>,                 \
                                                            index_sequence<J...>>,                                     \
                                          sizeof...(A)>,                                                               \
                              apply_args<type_sequence<A...>, index_sequence<I...>>,                                   \
                              apply_kwds<type_sequence<K...>, index_sequence<J...>> {                                  \
    typedef apply_callable_ck self_type;                                                                               \
    typedef apply_args<type_sequence<A...>, index_sequence<I...>> args_type;                                           \
    typedef apply_kwds<type_sequence<K...>, index_sequence<J...>> kwds_type;                                           \
                                                                                                                       \
    func_type func;                                                                                                    \
                                                                                                                       \
    __VA_ARGS__ apply_callable_ck(func_type func, args_type args, kwds_type kwds)                                      \
        : args_type(args), kwds_type(kwds), func(func)                                                                 \
    {                                                                                                                  \
    }                                                                                                                  \
                                                                                                                       \
    __VA_ARGS__ void single(char *DYND_UNUSED(dst), char *const *DYND_IGNORE_UNUSED(src))                              \
    {                                                                                                                  \
      func(apply_arg<A, I>::get(src[I])..., apply_kwd<K, J>::get()...);                                                \
    }                                                                                                                  \
                                                                                                                       \
    __VA_ARGS__ void strided(char *DYND_UNUSED(dst), intptr_t DYND_UNUSED(dst_stride),                                 \
                             char *const *DYND_IGNORE_UNUSED(src_copy),                                                \
                             const intptr_t *DYND_IGNORE_UNUSED(src_stride), size_t count)                             \
    {                                                                                                                  \
      dynd::detail::array_wrapper<char *, sizeof...(A)> src;                                                           \
                                                                                                                       \
      for (size_t j = 0; j != sizeof...(A); ++j) {                                                                     \
        src[j] = src_copy[j] + DYND_THREAD_ID(0) * src_stride[j];                                                      \
      }                                                                                                                \
                                                                                                                       \
      for (std::ptrdiff_t i = DYND_THREAD_ID(0); i < (std::ptrdiff_t)count; i += DYND_THREAD_COUNT(0)) {               \
        func(apply_arg<A, I>::get(src[I])..., apply_kwd<K, J>::get()...);                                              \
        for (size_t j = 0; j != sizeof...(A); ++j) {                                                                   \
          src[j] += DYND_THREAD_COUNT(0) * src_stride[j];                                                              \
        }                                                                                                              \
      }                                                                                                                \
    }                                                                                                                  \
                                                                                                                       \
    static intptr_t instantiate_without_cuda_launch(char *static_data, char *DYND_UNUSED(data), void *ckb,             \
                                                    intptr_t ckb_offset, const ndt::type &DYND_UNUSED(dst_tp),         \
                                                    const char *DYND_UNUSED(dst_arrmeta), intptr_t DYND_UNUSED(nsrc),  \
                                                    const ndt::type *src_tp, const char *const *src_arrmeta,           \
                                                    kernel_request_t kernreq,                                          \
                                                    const eval::eval_context *DYND_UNUSED(ectx), intptr_t nkwd,        \
                                                    const nd::array *kwds,                                             \
                                                    const std::map<std::string, ndt::type> &DYND_UNUSED(tp_vars))      \
    {                                                                                                                  \
      self_type::make(ckb, kernreq, ckb_offset,                                                                        \
                      dynd::detail::make_value_wrapper(*reinterpret_cast<func_type *>(static_data)),                   \
                      args_type(src_tp, src_arrmeta, kwds), kwds_type(nkwd, kwds));                                    \
      return ckb_offset;                                                                                               \
    }                                                                                                                  \
                                                                                                                       \
    static intptr_t instantiate(char *static_data, char *DYND_UNUSED(data), void *ckb, intptr_t ckb_offset,            \
                                const ndt::type &dst_tp, const char *dst_arrmeta, intptr_t nsrc,                       \
                                const ndt::type *src_tp, const char *const *src_arrmeta, kernel_request_t kernreq,     \
                                const eval::eval_context *ectx, intptr_t nkwd, const nd::array *kwds,                  \
                                const std::map<std::string, ndt::type> &tp_vars);                                      \
  }

    APPLY_CALLABLE_CK();

    template <typename func_type, typename R, typename... A, size_t... I, typename... K, size_t... J>
    intptr_t
    apply_callable_ck<func_type, R, type_sequence<A...>, index_sequence<I...>, type_sequence<K...>,
                      index_sequence<J...>>::instantiate(char *static_data, char *DYND_UNUSED(data), void *ckb,
                                                         intptr_t ckb_offset, const ndt::type &dst_tp,
                                                         const char *dst_arrmeta, intptr_t nsrc,
                                                         const ndt::type *src_tp, const char *const *src_arrmeta,
                                                         kernel_request_t kernreq, const eval::eval_context *ectx,
                                                         intptr_t nkwd, const nd::array *kwds,
                                                         const std::map<std::string, ndt::type> &tp_vars)
    {
      return instantiate_without_cuda_launch(static_data, NULL, ckb, ckb_offset, dst_tp, dst_arrmeta, nsrc, src_tp,
                                             src_arrmeta, kernreq, ectx, nkwd, kwds, tp_vars);
    }

#undef APPLY_CALLABLE_CK

#define APPLY_CALLABLE_CK(...)                                                                                         \
  template <typename func_type, typename R, typename... A, size_t... I, typename... K, size_t... J>                    \
  struct apply_callable_ck<                                                                                            \
      func_type *, R, type_sequence<A...>, index_sequence<I...>, type_sequence<K...>,                                  \
      index_sequence<J...>> : base_kernel<apply_callable_ck<func_type *, R, type_sequence<A...>, index_sequence<I...>, \
                                                            type_sequence<K...>, index_sequence<J...>>,                \
                                          sizeof...(A)>,                                                               \
                              apply_args<type_sequence<A...>, index_sequence<I...>>,                                   \
                              apply_kwds<type_sequence<K...>, index_sequence<J...>> {                                  \
    typedef apply_callable_ck self_type;                                                                               \
    typedef apply_args<type_sequence<A...>, index_sequence<I...>> args_type;                                           \
    typedef apply_kwds<type_sequence<K...>, index_sequence<J...>> kwds_type;                                           \
                                                                                                                       \
    func_type *func;                                                                                                   \
                                                                                                                       \
    __VA_ARGS__ apply_callable_ck(func_type *func, args_type args, kwds_type kwds)                                     \
        : args_type(args), kwds_type(kwds), func(func)                                                                 \
    {                                                                                                                  \
    }                                                                                                                  \
                                                                                                                       \
    __VA_ARGS__ void single(char *dst, char *const *DYND_IGNORE_UNUSED(src))                                           \
    {                                                                                                                  \
      *reinterpret_cast<R *>(dst) = (*func)(apply_arg<A, I>::get(src[I])..., apply_kwd<K, J>::get()...);               \
    }                                                                                                                  \
                                                                                                                       \
    __VA_ARGS__ void strided(char *dst, intptr_t dst_stride, char *const *DYND_IGNORE_UNUSED(src_copy),                \
                             const intptr_t *DYND_IGNORE_UNUSED(src_stride), size_t count)                             \
    {                                                                                                                  \
      dynd::detail::array_wrapper<char *, sizeof...(A)> src;                                                           \
                                                                                                                       \
      dst += DYND_THREAD_ID(0) * dst_stride;                                                                           \
      for (size_t j = 0; j != sizeof...(A); ++j) {                                                                     \
        src[j] = src_copy[j] + DYND_THREAD_ID(0) * src_stride[j];                                                      \
      }                                                                                                                \
                                                                                                                       \
      for (std::ptrdiff_t i = DYND_THREAD_ID(0); i < (std::ptrdiff_t)count; i += DYND_THREAD_COUNT(0)) {               \
        *reinterpret_cast<R *>(dst) = (*func)(apply_arg<A, I>::get(src[I])..., apply_kwd<K, J>::get()...);             \
        dst += DYND_THREAD_COUNT(0) * dst_stride;                                                                      \
        for (size_t j = 0; j != sizeof...(A); ++j) {                                                                   \
          src[j] += DYND_THREAD_COUNT(0) * src_stride[j];                                                              \
        }                                                                                                              \
      }                                                                                                                \
    }                                                                                                                  \
                                                                                                                       \
    static intptr_t instantiate_without_cuda_launch(char *static_data, char *DYND_UNUSED(data), void *ckb,             \
                                                    intptr_t ckb_offset, const ndt::type &DYND_UNUSED(dst_tp),         \
                                                    const char *DYND_UNUSED(dst_arrmeta), intptr_t DYND_UNUSED(nsrc),  \
                                                    const ndt::type *src_tp, const char *const *src_arrmeta,           \
                                                    kernel_request_t kernreq,                                          \
                                                    const eval::eval_context *DYND_UNUSED(ectx), intptr_t nkwd,        \
                                                    const nd::array *kwds,                                             \
                                                    const std::map<std::string, ndt::type> &DYND_UNUSED(tp_vars))      \
    {                                                                                                                  \
                                                                                                                       \
      self_type::make(ckb, kernreq, ckb_offset, *reinterpret_cast<func_type **>(static_data),                          \
                      args_type(src_tp, src_arrmeta, kwds), kwds_type(nkwd, kwds));                                    \
      return ckb_offset;                                                                                               \
    }                                                                                                                  \
                                                                                                                       \
    static intptr_t instantiate(char *DYND_UNUSED(static_data), char *DYND_UNUSED(data), void *ckb,                    \
                                intptr_t ckb_offset, const ndt::type &dst_tp, const char *dst_arrmeta, intptr_t nsrc,  \
                                const ndt::type *src_tp, const char *const *src_arrmeta, kernel_request_t kernreq,     \
                                const eval::eval_context *ectx, intptr_t nkwd, const nd::array *kwds,                  \
                                const std::map<std::string, ndt::type> &tp_vars);                                      \
  };                                                                                                                   \
                                                                                                                       \
  template <typename func_type, typename... A, size_t... I, typename... K, size_t... J>                                \
  struct apply_callable_ck<                                                                                            \
      func_type *, void, type_sequence<A...>, index_sequence<I...>, type_sequence<K...>,                               \
      index_sequence<J...>> : base_kernel<apply_callable_ck<func_type *, void, type_sequence<A...>,                    \
                                                            index_sequence<I...>, type_sequence<K...>,                 \
                                                            index_sequence<J...>>,                                     \
                                          sizeof...(A)>,                                                               \
                              apply_args<type_sequence<A...>, index_sequence<I...>>,                                   \
                              apply_kwds<type_sequence<K...>, index_sequence<J...>> {                                  \
    typedef apply_callable_ck self_type;                                                                               \
    typedef apply_args<type_sequence<A...>, index_sequence<I...>> args_type;                                           \
    typedef apply_kwds<type_sequence<K...>, index_sequence<J...>> kwds_type;                                           \
                                                                                                                       \
    func_type *func;                                                                                                   \
                                                                                                                       \
    __VA_ARGS__ apply_callable_ck(func_type *func, args_type args, kwds_type kwds)                                     \
        : args_type(args), kwds_type(kwds), func(func)                                                                 \
    {                                                                                                                  \
    }                                                                                                                  \
                                                                                                                       \
    __VA_ARGS__ void single(char *DYND_UNUSED(dst), char *const *DYND_IGNORE_UNUSED(src))                              \
    {                                                                                                                  \
      (*func)(apply_arg<A, I>::get(src[I])..., apply_kwd<K, J>::get()...);                                             \
    }                                                                                                                  \
                                                                                                                       \
    __VA_ARGS__ void strided(char *DYND_UNUSED(dst), intptr_t DYND_UNUSED(dst_stride),                                 \
                             char *const *DYND_IGNORE_UNUSED(src_copy),                                                \
                             const intptr_t *DYND_IGNORE_UNUSED(src_stride), size_t count)                             \
    {                                                                                                                  \
      dynd::detail::array_wrapper<char *, sizeof...(A)> src;                                                           \
                                                                                                                       \
      for (size_t j = 0; j != sizeof...(A); ++j) {                                                                     \
        src[j] = src_copy[j] + DYND_THREAD_ID(0) * src_stride[j];                                                      \
      }                                                                                                                \
                                                                                                                       \
      for (std::ptrdiff_t i = DYND_THREAD_ID(0); i < (std::ptrdiff_t)count; i += DYND_THREAD_COUNT(0)) {               \
        (*func)(apply_arg<A, I>::get(src[I])..., apply_kwd<K, J>::get()...);                                           \
        for (size_t j = 0; j != sizeof...(A); ++j) {                                                                   \
          src[j] += DYND_THREAD_COUNT(0) * src_stride[j];                                                              \
        }                                                                                                              \
      }                                                                                                                \
    }                                                                                                                  \
                                                                                                                       \
    static intptr_t instantiate_without_cuda_launch(char *static_data, char *DYND_UNUSED(data), void *ckb,             \
                                                    intptr_t ckb_offset, const ndt::type &DYND_UNUSED(dst_tp),         \
                                                    const char *DYND_UNUSED(dst_arrmeta), intptr_t DYND_UNUSED(nsrc),  \
                                                    const ndt::type *src_tp, const char *const *src_arrmeta,           \
                                                    kernel_request_t kernreq,                                          \
                                                    const eval::eval_context *DYND_UNUSED(ectx), intptr_t nkwd,        \
                                                    const nd::array *kwds,                                             \
                                                    const std::map<std::string, ndt::type> &DYND_UNUSED(tp_vars))      \
    {                                                                                                                  \
                                                                                                                       \
      self_type::make(ckb, kernreq, ckb_offset, *reinterpret_cast<func_type **>(static_data),                          \
                      args_type(src_tp, src_arrmeta, kwds), kwds_type(nkwd, kwds));                                    \
      return ckb_offset;                                                                                               \
    }                                                                                                                  \
                                                                                                                       \
    static intptr_t instantiate(char *DYND_UNUSED(static_data), char *DYND_UNUSED(data), void *ckb,                    \
                                intptr_t ckb_offset, const ndt::type &dst_tp, const char *dst_arrmeta, intptr_t nsrc,  \
                                const ndt::type *src_tp, const char *const *src_arrmeta, kernel_request_t kernreq,     \
                                const eval::eval_context *ectx, intptr_t nkwd, const nd::array *kwds,                  \
                                const std::map<std::string, ndt::type> &tp_vars);                                      \
  }

    APPLY_CALLABLE_CK();

    template <typename func_type, typename R, typename... A, size_t... I, typename... K, size_t... J>
    intptr_t
    apply_callable_ck<func_type *, R, type_sequence<A...>, index_sequence<I...>, type_sequence<K...>,
                      index_sequence<J...>>::instantiate(char *static_data, char *DYND_UNUSED(data), void *ckb,
                                                         intptr_t ckb_offset, const ndt::type &dst_tp,
                                                         const char *dst_arrmeta, intptr_t nsrc,
                                                         const ndt::type *src_tp, const char *const *src_arrmeta,
                                                         kernel_request_t kernreq, const eval::eval_context *ectx,
                                                         intptr_t nkwd, const nd::array *kwds,
                                                         const std::map<std::string, ndt::type> &tp_vars)
    {
      return instantiate_without_cuda_launch(static_data, NULL, ckb, ckb_offset, dst_tp, dst_arrmeta, nsrc, src_tp,
                                             src_arrmeta, kernreq, ectx, nkwd, kwds, tp_vars);
    }

    template <typename func_type, typename... A, size_t... I, typename... K, size_t... J>
    intptr_t
    apply_callable_ck<func_type *, void, type_sequence<A...>, index_sequence<I...>, type_sequence<K...>,
                      index_sequence<J...>>::instantiate(char *static_data, char *DYND_UNUSED(data), void *ckb,
                                                         intptr_t ckb_offset, const ndt::type &dst_tp,
                                                         const char *dst_arrmeta, intptr_t nsrc,
                                                         const ndt::type *src_tp, const char *const *src_arrmeta,
                                                         kernel_request_t kernreq, const eval::eval_context *ectx,
                                                         intptr_t nkwd, const nd::array *kwds,
                                                         const std::map<std::string, ndt::type> &tp_vars)
    {
      return instantiate_without_cuda_launch(static_data, NULL, ckb, ckb_offset, dst_tp, dst_arrmeta, nsrc, src_tp,
                                             src_arrmeta, kernreq, ectx, nkwd, kwds, tp_vars);
    }

#undef APPLY_CALLABLE_CK

    template <typename func_type, int N>
    using as_apply_callable_ck =
        apply_callable_ck<func_type, typename return_of<func_type>::type, as_apply_arg_sequence<func_type, N>,
                          make_index_sequence<N>, as_apply_kwd_sequence<func_type, N>,
                          make_index_sequence<arity_of<func_type>::value - N>>;

    template <typename func_type, typename R, typename A, typename I, typename K, typename J>
    struct construct_then_apply_callable_ck;

#define CONSTRUCT_THEN_APPLY_CALLABLE_CK(...)                                                                          \
  template <typename func_type, typename R, typename... A, size_t... I, typename... K, size_t... J>                    \
  struct construct_then_apply_callable_ck<                                                                             \
      func_type, R, type_sequence<A...>, index_sequence<I...>, type_sequence<K...>,                                    \
      index_sequence<J...>> : base_kernel<construct_then_apply_callable_ck<func_type, R, type_sequence<A...>,          \
                                                                           index_sequence<I...>, type_sequence<K...>,  \
                                                                           index_sequence<J...>>,                      \
                                          sizeof...(A)>,                                                               \
                              apply_args<type_sequence<A...>, index_sequence<I...>> {                                  \
    typedef construct_then_apply_callable_ck self_type;                                                                \
    typedef apply_args<type_sequence<A...>, index_sequence<I...>> args_type;                                           \
    typedef apply_kwds<type_sequence<K...>, index_sequence<J...>> kwds_type;                                           \
                                                                                                                       \
    func_type func;                                                                                                    \
                                                                                                                       \
    __VA_ARGS__ construct_then_apply_callable_ck(args_type args, kwds_type DYND_IGNORE_UNUSED(kwds))                   \
        : args_type(args), func(kwds.apply_kwd<K, J>::get()...)                                                        \
    {                                                                                                                  \
    }                                                                                                                  \
                                                                                                                       \
    __VA_ARGS__ void single(char *dst, char *const *DYND_IGNORE_UNUSED(src))                                           \
    {                                                                                                                  \
      *reinterpret_cast<R *>(dst) = func(apply_arg<A, I>::get(src[I])...);                                             \
    }                                                                                                                  \
                                                                                                                       \
    __VA_ARGS__ void strided(char *dst, intptr_t dst_stride, char *const *DYND_IGNORE_UNUSED(src_copy),                \
                             const intptr_t *DYND_IGNORE_UNUSED(src_stride), size_t count)                             \
    {                                                                                                                  \
      dynd::detail::array_wrapper<char *, sizeof...(A)> src;                                                           \
                                                                                                                       \
      dst += DYND_THREAD_ID(0) * dst_stride;                                                                           \
      for (size_t j = 0; j != sizeof...(A); ++j) {                                                                     \
        src[j] = src_copy[j] + DYND_THREAD_ID(0) * src_stride[j];                                                      \
      }                                                                                                                \
                                                                                                                       \
      for (std::ptrdiff_t i = DYND_THREAD_ID(0); i < (std::ptrdiff_t)count; i += DYND_THREAD_COUNT(0)) {               \
        *reinterpret_cast<R *>(dst) = func(apply_arg<A, I>::get(src[I])...);                                           \
        dst += DYND_THREAD_COUNT(0) * dst_stride;                                                                      \
        for (size_t j = 0; j != sizeof...(A); ++j) {                                                                   \
          src[j] += DYND_THREAD_COUNT(0) * src_stride[j];                                                              \
        }                                                                                                              \
      }                                                                                                                \
    }                                                                                                                  \
                                                                                                                       \
    static intptr_t                                                                                                    \
    instantiate_without_cuda_launch(char *DYND_UNUSED(static_data), char *DYND_UNUSED(data), void *ckb,                \
                                    intptr_t ckb_offset, const ndt::type &DYND_UNUSED(dst_tp),                         \
                                    const char *DYND_UNUSED(dst_arrmeta), intptr_t DYND_UNUSED(nsrc),                  \
                                    const ndt::type *src_tp, const char *const *src_arrmeta, kernel_request_t kernreq, \
                                    const eval::eval_context *DYND_UNUSED(ectx), intptr_t nkwd, const nd::array *kwds, \
                                    const std::map<std::string, ndt::type> &DYND_UNUSED(tp_vars))                      \
    {                                                                                                                  \
      self_type::make(ckb, kernreq, ckb_offset, args_type(src_tp, src_arrmeta, kwds), kwds_type(nkwd, kwds));          \
      return ckb_offset;                                                                                               \
    }                                                                                                                  \
                                                                                                                       \
    static intptr_t instantiate(char *DYND_UNUSED(static_data), char *DYND_UNUSED(data), void *ckb,                    \
                                intptr_t ckb_offset, const ndt::type &dst_tp, const char *dst_arrmeta, intptr_t nsrc,  \
                                const ndt::type *src_tp, const char *const *src_arrmeta, kernel_request_t kernreq,     \
                                const eval::eval_context *ectx, intptr_t nkwd, const nd::array *kwds,                  \
                                const std::map<std::string, ndt::type> &tp_vars);                                      \
  };                                                                                                                   \
                                                                                                                       \
  template <typename func_type, typename... A, size_t... I, typename... K, size_t... J>                                \
  struct construct_then_apply_callable_ck<                                                                             \
      func_type, void, type_sequence<A...>, index_sequence<I...>, type_sequence<K...>,                                 \
      index_sequence<J...>> : base_kernel<construct_then_apply_callable_ck<func_type, void, type_sequence<A...>,       \
                                                                           index_sequence<I...>, type_sequence<K...>,  \
                                                                           index_sequence<J...>>,                      \
                                          sizeof...(A)>,                                                               \
                              apply_args<type_sequence<A...>, index_sequence<I...>> {                                  \
    typedef construct_then_apply_callable_ck self_type;                                                                \
    typedef apply_args<type_sequence<A...>, index_sequence<I...>> args_type;                                           \
    typedef apply_kwds<type_sequence<K...>, index_sequence<J...>> kwds_type;                                           \
                                                                                                                       \
    func_type func;                                                                                                    \
                                                                                                                       \
    __VA_ARGS__ construct_then_apply_callable_ck(args_type args, kwds_type DYND_IGNORE_UNUSED(kwds))                   \
        : args_type(args), func(kwds.apply_kwd<K, J>::get()...)                                                        \
    {                                                                                                                  \
    }                                                                                                                  \
                                                                                                                       \
    __VA_ARGS__ void single(char *DYND_UNUSED(dst), char *const *DYND_IGNORE_UNUSED(src))                              \
    {                                                                                                                  \
      func(apply_arg<A, I>::get(src[I])...);                                                                           \
    }                                                                                                                  \
                                                                                                                       \
    __VA_ARGS__ void strided(char *DYND_UNUSED(dst), intptr_t DYND_UNUSED(dst_stride),                                 \
                             char *const *DYND_IGNORE_UNUSED(src_copy),                                                \
                             const intptr_t *DYND_IGNORE_UNUSED(src_stride), size_t count)                             \
    {                                                                                                                  \
      dynd::detail::array_wrapper<char *, sizeof...(A)> src;                                                           \
                                                                                                                       \
      for (size_t j = 0; j != sizeof...(A); ++j) {                                                                     \
        src[j] = src_copy[j] + DYND_THREAD_ID(0) * src_stride[j];                                                      \
      }                                                                                                                \
                                                                                                                       \
      for (std::ptrdiff_t i = DYND_THREAD_ID(0); i < (std::ptrdiff_t)count; i += DYND_THREAD_COUNT(0)) {               \
        func(apply_arg<A, I>::get(src[I])...);                                                                         \
        for (size_t j = 0; j != sizeof...(A); ++j) {                                                                   \
          src[j] += DYND_THREAD_COUNT(0) * src_stride[j];                                                              \
        }                                                                                                              \
      }                                                                                                                \
    }                                                                                                                  \
                                                                                                                       \
    static intptr_t instantiate_without_cuda_launch(                                                                   \
        char *DYND_UNUSED(static_data), size_t DYND_UNUSED(data_size), char *DYND_UNUSED(data), void *ckb,             \
        intptr_t ckb_offset, const ndt::type &DYND_UNUSED(dst_tp), const char *DYND_UNUSED(dst_arrmeta),               \
        intptr_t DYND_UNUSED(nsrc), const ndt::type *src_tp, const char *const *src_arrmeta, kernel_request_t kernreq, \
        const eval::eval_context *DYND_UNUSED(ectx), intptr_t nkwd, const nd::array *kwds,                             \
        const std::map<std::string, ndt::type> &DYND_UNUSED(tp_vars))                                                  \
    {                                                                                                                  \
      self_type::make(ckb, kernreq, ckb_offset, args_type(src_tp, src_arrmeta, kwds), kwds_type(nkwd, kwds));          \
      return ckb_offset;                                                                                               \
    }                                                                                                                  \
                                                                                                                       \
    static intptr_t instantiate(char *DYND_UNUSED(static_data), char *DYND_UNUSED(data), void *ckb,                    \
                                intptr_t ckb_offset, const ndt::type &dst_tp, const char *dst_arrmeta, intptr_t nsrc,  \
                                const ndt::type *src_tp, const char *const *src_arrmeta, kernel_request_t kernreq,     \
                                const eval::eval_context *ectx, intptr_t nkwd, const nd::array *kwds,                  \
                                const std::map<std::string, ndt::type> &tp_vars);                                      \
  }

    CONSTRUCT_THEN_APPLY_CALLABLE_CK();

    template <typename func_type, typename R, typename... A, size_t... I, typename... K, size_t... J>
    intptr_t construct_then_apply_callable_ck<
        func_type, R, type_sequence<A...>, index_sequence<I...>, type_sequence<K...>,
        index_sequence<J...>>::instantiate(char *static_data, char *DYND_UNUSED(data), void *ckb, intptr_t ckb_offset,
                                           const ndt::type &dst_tp, const char *dst_arrmeta, intptr_t nsrc,
                                           const ndt::type *src_tp, const char *const *src_arrmeta,
                                           kernel_request_t kernreq, const eval::eval_context *ectx, intptr_t nkwd,
                                           const nd::array *kwds, const std::map<std::string, ndt::type> &tp_vars)
    {
      return instantiate_without_cuda_launch(static_data, NULL, ckb, ckb_offset, dst_tp, dst_arrmeta, nsrc, src_tp,
                                             src_arrmeta, kernreq, ectx, nkwd, kwds, tp_vars);
    }

#undef CONSTRUCT_THEN_APPLY_CALLABLE_CK

    template <typename func_type, typename... K>
    using as_construct_then_apply_callable_ck = construct_then_apply_callable_ck<
        func_type, typename return_of<func_type>::type, as_apply_arg_sequence<func_type, arity_of<func_type>::value>,
        make_index_sequence<arity_of<func_type>::value>, type_sequence<K...>, make_index_sequence<sizeof...(K)>>;

  } // namespace dynd::nd::functional
} // namespace dynd::nd
} // namespace dynd
