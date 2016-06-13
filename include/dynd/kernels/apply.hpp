//
// Copyright (C) 2011-16 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#pragma once

#include <dynd/kernels/base_kernel.hpp>
#include <dynd/kernels/cuda_launch.hpp>
#include <dynd/types/state_type.hpp>

namespace dynd {
namespace nd {
  namespace functional {

    template <typename A, size_t I>
    struct apply_arg {
      apply_arg(char *DYND_UNUSED(data), const char *DYND_UNUSED(arrmeta)) {}

      A &get(char *data) { return *reinterpret_cast<A *>(data); }
    };

    template <typename ElementType, size_t I>
    struct apply_arg<fixed_dim<ElementType>, I> {
      fixed_dim<ElementType> value;

      apply_arg(char *DYND_UNUSED(data), const char *arrmeta) : value(arrmeta, NULL) {}

      fixed_dim<ElementType> &get(char *data) {
        value.set_data(data);
        return value;
      }
    };

    template <size_t I>
    struct apply_arg<state, I> {
      size_t &it;

      apply_arg(char *data, const char *DYND_UNUSED(arrmeta)) : it(*reinterpret_cast<size_t *>(data)) {}

      state &get(char *data) { return *reinterpret_cast<state *>(data); }

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
    using as_apply_arg_sequence = typename to<typename args_of<typename funcproto_of<func_type>::type>::type, N>::type;

    template <typename A, typename I = std::make_index_sequence<A::size>>
    struct apply_args;

    template <typename... A, size_t... I>
    struct apply_args<type_sequence<A...>, std::index_sequence<I...>> : apply_arg<A, I>... {
      apply_args(char *DYND_IGNORE_UNUSED(data), const char *const *DYND_IGNORE_UNUSED(src_arrmeta))
          : apply_arg<A, I>(data, src_arrmeta[I])... {}

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

template <typename T>
struct return_wrapper : nd::functional::apply_arg<T, 0> {
  using nd::functional::apply_arg<T, 0>::apply_arg;
};

} // namespace dynd

#include <dynd/kernels/apply_callable_kernel.hpp>
#include <dynd/kernels/apply_function_kernel.hpp>
#include <dynd/kernels/apply_member_function_kernel.hpp>
#include <dynd/kernels/construct_then_apply_callable_kernel.hpp>
