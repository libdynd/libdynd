//
// Copyright (C) 2011-16 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#pragma once

#include <dynd/kernels/cuda_launch.hpp>
#include <dynd/kernels/base_kernel.hpp>
#include <dynd/types/iteration_type.hpp>

namespace dynd {
namespace nd {
  namespace functional {

    template <typename A, size_t I>
    struct apply_arg {
      typedef typename std::remove_cv<typename std::remove_reference<A>::type>::type D;

      apply_arg(char *DYND_UNUSED(data), const char *DYND_UNUSED(arrmeta)) {}

      D &get(char *data) { return *reinterpret_cast<D *>(data); }

      static char *at(char *const *args) { return args[I]; }
      static const char *at(const char *const *args) { return args[I]; }
    };

    template <typename ElementType, size_t I>
    struct apply_arg<fixed_dim<ElementType>, I> {
      fixed_dim<ElementType> value;

      apply_arg(char *DYND_UNUSED(data), const char *arrmeta) : value(arrmeta, NULL) {}

      fixed_dim<ElementType> &get(char *data) {
        value.set_data(data);
        return value;
      }

      static char *at(char *const *args) { return args[I]; }
      static const char *at(const char *const *args) { return args[I]; }
    };

    template <size_t I>
    struct apply_arg<state, I> {
      state st;
      size_t index[10];

      apply_arg(char *data, const char *DYND_UNUSED(arrmeta)) {
        st.ndim = reinterpret_cast<state *>(data)->ndim;
        st.index = index;
        for (size_t i = 0; i < st.ndim; ++i) {
          st.index[i] = 0;
        }
      }

      apply_arg(const apply_arg &other) {
        st.ndim = other.st.ndim;
        st.index = index;
        for (size_t i = 0; i < st.ndim; ++i) {
          st.index[i] = 0;
        }
      }

      state &get(char *DYND_UNUSED(data)) { return st; }

      size_t &begin() {
        st.index[st.ndim - 1] = 0;
        return st.index[st.ndim - 1];
      }

      static char *at(char *const *DYND_UNUSED(args)) { return nullptr; }
      static const char *at(const char *const *DYND_UNUSED(args)) { return nullptr; }
    };

    template <typename func_type, int N = args_of<typename funcproto_of<func_type>::type>::type::size>
    using as_apply_arg_sequence = typename to<typename args_of<typename funcproto_of<func_type>::type>::type, N>::type;

    template <typename A, typename I = make_index_sequence<A::size>>
    struct apply_args;

    template <typename... A, size_t... I>
    struct apply_args<type_sequence<A...>, index_sequence<I...>> : apply_arg<A, I>... {
      apply_args(char *DYND_IGNORE_UNUSED(data), const char *const *DYND_IGNORE_UNUSED(src_arrmeta))
          : apply_arg<A, I>(data, apply_arg<A, I>::at(src_arrmeta))... {}

      apply_args(const apply_args &) = default;
    };

    template <typename T, size_t I>
    struct apply_kwd {
      T m_val;

      apply_kwd(nd::array val) : m_val(val.as<T>()) {}

      T get() { return m_val; }
    };

    template <typename K, typename J = make_index_sequence<K::size>>
    struct apply_kwds;

    template <>
    struct apply_kwds<type_sequence<>, index_sequence<>> {
      apply_kwds(intptr_t DYND_UNUSED(nkwd), const nd::array *DYND_UNUSED(kwds)) {}
    };

    template <typename... K, size_t... J>
    struct apply_kwds<type_sequence<K...>, index_sequence<J...>> : apply_kwd<K, J>... {
      apply_kwds(intptr_t DYND_UNUSED(nkwd), const nd::array *kwds) : apply_kwd<K, J>(kwds[J])... {}
    };

    template <typename func_type, int N>
    using as_apply_kwd_sequence =
        typename from<typename args_of<typename funcproto_of<func_type>::type>::type, N>::type;

  } // namespace dynd::nd::functional
} // namespace dynd::nd
} // namespace dynd

#include <dynd/kernels/apply_function_kernel.hpp>
#include <dynd/kernels/apply_member_function_kernel.hpp>
#include <dynd/kernels/apply_callable_kernel.hpp>
#include <dynd/kernels/construct_then_apply_callable_kernel.hpp>
