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

      template <typename ArgTypes>
      struct has_state {
        static const bool value = false;
      };

      template <typename... A, size_t I0, size_t... I>
      struct has_state<apply_args<type_sequence<state, A...>, std::index_sequence<I0, I...>>> {
        static const bool value = true;
      };

      template <typename A0, typename... A, size_t I0, size_t... I>
      struct has_state<apply_args<type_sequence<A0, A...>, std::index_sequence<I0, I...>>> {
        static const bool value = has_state<apply_args<type_sequence<A...>, std::index_sequence<I...>>>::value;
      };

      template <typename SelfType, typename ArgsType, size_t NArg>
      struct base_apply_kernel : base_strided_kernel<SelfType, NArg>, ArgsType {
        typedef
            typename std::conditional<has_state<ArgsType>::value, ArgsType, base_strided_kernel<SelfType, NArg>>::type
                T;
        using T::begin;

        base_apply_kernel(ArgsType args) : ArgsType(args) {}
      };

      template <typename func_type, typename R, typename A, typename I, typename K, typename J, typename Enable = void>
      struct apply_callable_kernel;

      template <typename func_type, typename R, typename... A, size_t... I, typename... K, size_t... J>
      struct apply_callable_kernel<func_type, R, type_sequence<A...>, std::index_sequence<I...>, type_sequence<K...>,
                                   std::index_sequence<J...>>
          : base_apply_kernel<apply_callable_kernel<func_type, R, type_sequence<A...>, std::index_sequence<I...>,
                                                    type_sequence<K...>, std::index_sequence<J...>>,
                              apply_args<type_sequence<A...>, std::index_sequence<I...>>, sizeof...(A)>,
            apply_kwds<type_sequence<K...>, std::index_sequence<J...>> {

        typedef base_apply_kernel<apply_callable_kernel<func_type, R, type_sequence<A...>, std::index_sequence<I...>,
                                                        type_sequence<K...>, std::index_sequence<J...>>,
                                  apply_args<type_sequence<A...>, std::index_sequence<I...>>, sizeof...(A)>
            base_type;

        typedef apply_args<type_sequence<A...>, std::index_sequence<I...>> args_type;
        typedef apply_kwds<type_sequence<K...>, std::index_sequence<J...>> kwds_type;

        func_type func;

        apply_callable_kernel(func_type func, args_type args, kwds_type kwds)
            : base_type(args), kwds_type(kwds), func(func) {}

        void single(char *dst, char *const *DYND_IGNORE_UNUSED(src)) {
          *reinterpret_cast<R *>(dst) = func(apply_arg<A, I>::assign(src[I])..., apply_kwd<K, J>::get()...);
        }
      };

      template <typename func_type, typename... A, size_t... I, typename... K, size_t... J>
      struct apply_callable_kernel<func_type, void, type_sequence<A...>, std::index_sequence<I...>, type_sequence<K...>,
                                   std::index_sequence<J...>>
          : base_strided_kernel<apply_callable_kernel<func_type, void, type_sequence<A...>, std::index_sequence<I...>,
                                                      type_sequence<K...>, std::index_sequence<J...>>,
                                sizeof...(A)>,
            apply_args<type_sequence<A...>, std::index_sequence<I...>>,
            apply_kwds<type_sequence<K...>, std::index_sequence<J...>> {
        typedef apply_args<type_sequence<A...>, std::index_sequence<I...>> args_type;
        typedef apply_kwds<type_sequence<K...>, std::index_sequence<J...>> kwds_type;

        func_type func;

        apply_callable_kernel(func_type func, args_type args, kwds_type kwds)
            : args_type(args), kwds_type(kwds), func(func) {}

        void single(char *DYND_UNUSED(dst), char *const *DYND_IGNORE_UNUSED(src)) {
          func(apply_arg<A, I>::assign(src[I])..., apply_kwd<K, J>::get()...);
        }
      };

      template <typename func_type, typename R, resolve_t Resolve, typename... A, size_t I0, size_t... I, typename... K,
                size_t... J>
      struct apply_callable_kernel<func_type, void, type_sequence<return_wrapper<R, Resolve>, A...>,
                                   std::index_sequence<I0, I...>, type_sequence<K...>, std::index_sequence<J...>>
          : base_strided_kernel<
                apply_callable_kernel<func_type, void, type_sequence<return_wrapper<R, Resolve>, A...>,
                                      std::index_sequence<I0, I...>, type_sequence<K...>, std::index_sequence<J...>>,
                sizeof...(A)>,
            apply_args<type_sequence<return_wrapper<R, Resolve>, A...>, std::index_sequence<I0, I...>>,
            apply_kwds<type_sequence<K...>, std::index_sequence<J...>> {
        typedef apply_args<type_sequence<return_wrapper<R, Resolve>, A...>, std::index_sequence<I0, I...>> args_type;
        typedef apply_kwds<type_sequence<K...>, std::index_sequence<J...>> kwds_type;

        func_type func;

        apply_callable_kernel(func_type func, args_type args, kwds_type kwds)
            : args_type(args), kwds_type(kwds), func(func) {}

        void single(char *dst, char *const *DYND_IGNORE_UNUSED(src)) {
          func(apply_arg<return_wrapper<R, Resolve>, I0>::assign(dst), apply_arg<A, I>::assign(src[I - 1])...,
               apply_kwd<K, J>::get()...);
        }
      };

    } // namespace dynd::nd::functional::detail

    template <typename func_type, int N>
    using apply_callable_kernel =
        detail::apply_callable_kernel<func_type, typename return_of<func_type>::type, args_for<func_type, N>,
                                      std::make_index_sequence<N>, as_apply_kwd_sequence<func_type, N>,
                                      std::make_index_sequence<arity_of<func_type>::value - N>>;

  } // namespace dynd::nd::functional
} // namespace dynd::nd
} // namespace dynd
