//
// Copyright (C) 2011-15 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#pragma once

#include <dynd/kernels/apply.hpp>
#include <dynd/kernels/base_strided_kernel.hpp>

namespace dynd {
namespace nd {
  namespace functional {
    namespace detail {

      template <typename func_type, func_type func, typename R, typename A, typename I, typename K, typename J>
      struct apply_function_kernel;

#define APPLY_FUNCTION_KERNEL(...)                                                                                     \
  template <typename func_type, func_type func, typename R, typename... A, size_t... I, typename... K, size_t... J>    \
  struct apply_function_kernel<func_type, func, R, type_sequence<A...>, index_sequence<I...>, type_sequence<K...>,     \
                               index_sequence<J...>>                                                                   \
      : base_strided_kernel<apply_function_kernel<func_type, func, R, type_sequence<A...>, index_sequence<I...>,       \
                                                  type_sequence<K...>, index_sequence<J...>>,                          \
                            sizeof...(A)>,                                                                             \
        apply_args<type_sequence<A...>, index_sequence<I...>>,                                                         \
        apply_kwds<type_sequence<K...>, index_sequence<J...>> {                                                        \
    typedef apply_args<type_sequence<A...>, index_sequence<I...>> args_type;                                           \
    typedef apply_kwds<type_sequence<K...>, index_sequence<J...>> kwds_type;                                           \
                                                                                                                       \
    __VA_ARGS__ apply_function_kernel(args_type args, kwds_type kwds) : args_type(args), kwds_type(kwds) {}            \
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
    static void instantiate(char *DYND_UNUSED(static_data), char *DYND_UNUSED(data), kernel_builder *ckb,              \
                            const ndt::type &DYND_UNUSED(dst_tp), const char *DYND_UNUSED(dst_arrmeta),                \
                            intptr_t DYND_UNUSED(nsrc), const ndt::type *src_tp, const char *const *src_arrmeta,       \
                            kernel_request_t kernreq, intptr_t nkwd, const nd::array *kwds,                            \
                            const std::map<std::string, ndt::type> &DYND_UNUSED(tp_vars))                              \
    {                                                                                                                  \
      ckb->emplace_back<apply_function_kernel>(kernreq, args_type(src_tp, src_arrmeta, kwds), kwds_type(nkwd, kwds));  \
    }                                                                                                                  \
  };                                                                                                                   \
                                                                                                                       \
  template <typename func_type, func_type func, typename... A, size_t... I, typename... K, size_t... J>                \
  struct apply_function_kernel<func_type, func, void, type_sequence<A...>, index_sequence<I...>, type_sequence<K...>,  \
                               index_sequence<J...>>                                                                   \
      : base_strided_kernel<apply_function_kernel<func_type, func, void, type_sequence<A...>, index_sequence<I...>,    \
                                                  type_sequence<K...>, index_sequence<J...>>,                          \
                            sizeof...(A)>,                                                                             \
        apply_args<type_sequence<A...>, index_sequence<I...>>,                                                         \
        apply_kwds<type_sequence<K...>, index_sequence<J...>> {                                                        \
    typedef apply_args<type_sequence<A...>, index_sequence<I...>> args_type;                                           \
    typedef apply_kwds<type_sequence<K...>, index_sequence<J...>> kwds_type;                                           \
                                                                                                                       \
    __VA_ARGS__ apply_function_kernel(args_type args, kwds_type kwds) : args_type(args), kwds_type(kwds) {}            \
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
    static void instantiate(char *DYND_UNUSED(static_data), char *DYND_UNUSED(data), kernel_builder *ckb,              \
                            const ndt::type &DYND_UNUSED(dst_tp), const char *DYND_UNUSED(dst_arrmeta),                \
                            intptr_t DYND_UNUSED(nsrc), const ndt::type *src_tp, const char *const *src_arrmeta,       \
                            kernel_request_t kernreq, intptr_t nkwd, const nd::array *kwds,                            \
                            const std::map<std::string, ndt::type> &DYND_UNUSED(tp_vars))                              \
    {                                                                                                                  \
      ckb->emplace_back<apply_function_kernel>(kernreq, args_type(src_tp, src_arrmeta, kwds), kwds_type(nkwd, kwds));  \
    }                                                                                                                  \
  }

      APPLY_FUNCTION_KERNEL();

#undef APPLY_FUNCTION_KERNEL

    } // namespace dynd::nd::functional::detail

    template <typename func_type, func_type func, int N = arity_of<func_type>::value>
    using apply_function_kernel =
        detail::apply_function_kernel<func_type, func, typename return_of<func_type>::type,
                                      as_apply_arg_sequence<func_type, N>, make_index_sequence<N>,
                                      as_apply_kwd_sequence<func_type, N>,
                                      make_index_sequence<arity_of<func_type>::value - N>>;

  } // namespace dynd::nd::functional
} // namespace dynd::nd
} // namespace dynd
