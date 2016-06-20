//
// Copyright (C) 2011-16 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#pragma once

#include <dynd/kernels/base_kernel.hpp>
#include <dynd/kernels/cuda_launch.hpp>
#include <dynd/types/state_type.hpp>

namespace dynd {

typedef ndt::type (*resolve_t)(size_t, const ndt::type *, size_t, const nd::array *);

template <typename ReturnType>
ndt::type default_resolve(size_t DYND_UNUSED(nsrc), const ndt::type *DYND_UNUSED(src_tp), size_t DYND_UNUSED(nkwd),
                          const nd::array *DYND_UNUSED(kwds)) {
  return ndt::make_type<ReturnType>();
}

template <typename ReturnType, resolve_t Resolve = default_resolve<ReturnType>>
struct return_wrapper {
  ReturnType *ptr;

  return_wrapper(ReturnType &ref) : ptr(&ref) {}

  return_wrapper(const return_wrapper &other) = default;

  ReturnType &get() const { return *ptr; }

  operator ReturnType &() const { return *ptr; }
};

namespace nd {
  namespace functional {

    template <typename ArgType, size_t I, typename Enable = void>
    struct apply_arg;

    template <typename A, size_t I>
    struct apply_arg<A, I, std::enable_if_t<ndt::traits<A>::is_same_layout>> {
      apply_arg(char *DYND_UNUSED(data), const char *DYND_UNUSED(arrmeta)) {}

      A &assign(char *data) { return *reinterpret_cast<A *>(data); }
    };

    template <typename ArgType, size_t I>
    struct apply_arg<ArgType, I, std::enable_if_t<!ndt::traits<ArgType>::is_same_layout>> {
      ArgType value;

      apply_arg(char *DYND_UNUSED(data), const char *arrmeta) : value(arrmeta, NULL) {}

      ArgType &assign(char *data) { return value.assign(data); }
    };

    template <typename ReturnType, resolve_t Resolve, size_t I>
    struct apply_arg<return_wrapper<ReturnType, Resolve>, I> : apply_arg<ReturnType, I> {
      using apply_arg<ReturnType, I>::apply_arg;

      return_wrapper<ReturnType, Resolve> assign(char *data) {
        return return_wrapper<ReturnType, Resolve>(apply_arg<ReturnType, I>::assign(data));
      }
    };

    template <size_t I>
    struct apply_arg<state, I> {
      size_t &it;

      apply_arg(char *data, const char *DYND_UNUSED(arrmeta)) : it(*reinterpret_cast<size_t *>(data)) {}

      state &assign(char *data) { return *reinterpret_cast<state *>(data); }

      size_t &begin() {
        it = 0;
        return it;
      }
    };

    template <typename T, size_t I>
    struct apply_arg<T &, I> : apply_arg<T, I> {
      using apply_arg<T, I>::apply_arg;
    };

    template <typename func_type, int N = args_of<typename funcproto_of<func_type>::type>::type::size>
    using args_for = typename to<typename args_of<typename funcproto_of<func_type>::type>::type, N>::type;

    template <typename A, typename I = std::make_index_sequence<A::size>>
    struct apply_args;

    template <typename... A, size_t... I>
    struct apply_args<type_sequence<A...>, std::index_sequence<I...>> : apply_arg<A, I>... {
      apply_args(char *DYND_IGNORE_UNUSED(data), const char *DYND_UNUSED(dst_arrmeta),
                 const char *const *DYND_IGNORE_UNUSED(src_arrmeta))
          : apply_arg<A, I>(data, src_arrmeta[I])... {}

      apply_args(const apply_args &) = default;
    };

    template <typename R, resolve_t Resolve, typename... A, size_t I0, size_t... I>
    struct apply_args<type_sequence<return_wrapper<R, Resolve>, A...>, std::index_sequence<I0, I...>>
        : apply_arg<return_wrapper<R, Resolve>, I0>, apply_arg<A, I>... {
      apply_args(char *DYND_IGNORE_UNUSED(data), const char *dst_arrmeta,
                 const char *const *DYND_IGNORE_UNUSED(src_arrmeta))
          : apply_arg<return_wrapper<R, Resolve>, I0>(data, dst_arrmeta), apply_arg<A, I>(data, src_arrmeta[I - 1])... {
      }

      apply_args(const apply_args &) = default;
    };

    template <typename T, size_t I>
    struct apply_kwd {
      T m_val;

      apply_kwd(nd::array val) : m_val(val.as<T>()) {}

      T get() { return m_val; }
    };

    template <typename K, typename J = std::make_index_sequence<K::size>>
    struct apply_kwds;

    template <>
    struct apply_kwds<type_sequence<>, std::index_sequence<>> {
      apply_kwds(intptr_t DYND_UNUSED(nkwd), const nd::array *DYND_UNUSED(kwds)) {}
    };

    template <typename... K, size_t... J>
    struct apply_kwds<type_sequence<K...>, std::index_sequence<J...>> : apply_kwd<K, J>... {
      apply_kwds(intptr_t DYND_UNUSED(nkwd), const nd::array *kwds) : apply_kwd<K, J>(kwds[J])... {}
    };

    template <typename func_type, int N>
    using as_apply_kwd_sequence =
        typename from<typename args_of<typename funcproto_of<func_type>::type>::type, N>::type;

  } // namespace dynd::nd::functional
} // namespace dynd::nd
} // namespace dynd

#include <dynd/kernels/apply_callable_kernel.hpp>
#include <dynd/kernels/apply_function_kernel.hpp>
#include <dynd/kernels/apply_member_function_kernel.hpp>
#include <dynd/kernels/construct_then_apply_callable_kernel.hpp>
