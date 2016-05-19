//
// Copyright (C) 2011-16 DyND Developers
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

      template <typename func_type, func_type func, typename R, typename... A, size_t... I, typename... K, size_t... J>
      struct apply_function_kernel<func_type, func, R, type_sequence<A...>, std::index_sequence<I...>,
                                   type_sequence<K...>, std::index_sequence<J...>>
          : base_strided_kernel<
                apply_function_kernel<func_type, func, R, type_sequence<A...>, std::index_sequence<I...>,
                                      type_sequence<K...>, std::index_sequence<J...>>,
                sizeof...(A)>,
            apply_args<type_sequence<A...>, std::index_sequence<I...>>,
            apply_kwds<type_sequence<K...>, std::index_sequence<J...>> {
        typedef apply_args<type_sequence<A...>, std::index_sequence<I...>> args_type;
        typedef apply_kwds<type_sequence<K...>, std::index_sequence<J...>> kwds_type;

        apply_function_kernel(args_type args, kwds_type kwds) : args_type(args), kwds_type(kwds) {}

        void single(char *dst, char *const *DYND_IGNORE_UNUSED(src)) {
          *reinterpret_cast<R *>(dst) = func(apply_arg<A, I>::get(src[I])..., apply_kwd<K, J>::get()...);
        }
      };

      template <typename func_type, func_type func, typename... A, size_t... I, typename... K, size_t... J>
      struct apply_function_kernel<func_type, func, void, type_sequence<A...>, std::index_sequence<I...>,
                                   type_sequence<K...>, std::index_sequence<J...>>
          : base_strided_kernel<
                apply_function_kernel<func_type, func, void, type_sequence<A...>, std::index_sequence<I...>,
                                      type_sequence<K...>, std::index_sequence<J...>>,
                sizeof...(A)>,
            apply_args<type_sequence<A...>, std::index_sequence<I...>>,
            apply_kwds<type_sequence<K...>, std::index_sequence<J...>> {
        typedef apply_args<type_sequence<A...>, std::index_sequence<I...>> args_type;
        typedef apply_kwds<type_sequence<K...>, std::index_sequence<J...>> kwds_type;

        apply_function_kernel(args_type args, kwds_type kwds) : args_type(args), kwds_type(kwds) {}

        void single(char *DYND_UNUSED(dst), char *const *DYND_IGNORE_UNUSED(src)) {
          func(apply_arg<A, I>::get(src[I])..., apply_kwd<K, J>::get()...);
        }
      };

    } // namespace dynd::nd::functional::detail

    template <typename func_type, func_type func, int N = arity_of<func_type>::value>
    using apply_function_kernel =
        detail::apply_function_kernel<func_type, func, typename return_of<func_type>::type,
                                      as_apply_arg_sequence<func_type, N>, std::make_index_sequence<N>,
                                      as_apply_kwd_sequence<func_type, N>,
                                      std::make_index_sequence<arity_of<func_type>::value - N>>;

  } // namespace dynd::nd::functional
} // namespace dynd::nd
} // namespace dynd
