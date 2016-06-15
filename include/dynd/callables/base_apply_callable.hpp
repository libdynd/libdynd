//
// Copyright (C) 2011-16 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#pragma once

#include <dynd/callables/base_callable.hpp>
#include <dynd/kernels/apply.hpp>

namespace dynd {
namespace nd {
  namespace functional {

    template <typename ReturnType, typename ArgTypes, typename KwdTypes, typename Enable = void>
    class base_apply_callable;

    template <typename ReturnType, typename... ArgTypes, typename... KwdTypes>
    class base_apply_callable<ReturnType, type_sequence<ArgTypes...>, type_sequence<KwdTypes...>,
                              std::enable_if_t<!has_return_wrapper<ArgTypes...>::value>> : public base_callable {
    public:
      template <typename... S>
      base_apply_callable(S &&... names)
          : base_callable(
                ndt::make_type<ndt::callable_type>(ndt::make_type<ReturnType>(), {ndt::make_type<ArgTypes>()...},
                                                   {{ndt::make_type<KwdTypes>(), std::forward<S>(names)}...})) {}

      ndt::type resolve_return_type(const ndt::type &dst_tp, size_t DYND_UNUSED(nsrc),
                                    const ndt::type *DYND_UNUSED(src_tp)) {
        return dst_tp;
      }
    };

    template <typename Arg0Type, typename... ArgTypes, typename... KwdTypes>
    class base_apply_callable<
        void, type_sequence<Arg0Type, ArgTypes...>, type_sequence<KwdTypes...>,
        std::enable_if_t<has_return_wrapper<remove_reference_then_cv_t<Arg0Type>, ArgTypes...>::value>>
        : public base_callable {
    public:
      template <typename... S>
      base_apply_callable(S &&... names)
          : base_callable(ndt::make_type<ndt::callable_type>(
                ndt::make_type<typename remove_reference_then_cv_t<Arg0Type>::type>(), {ndt::make_type<ArgTypes>()...},
                {{ndt::make_type<KwdTypes>(), std::forward<S>(names)}...})) {}

      ndt::type resolve_return_type(const ndt::type &DYND_UNUSED(dst_tp), size_t nsrc, const ndt::type *src_tp) {
        return remove_reference_then_cv_t<Arg0Type>::resolve(nsrc, src_tp);
      }
    };

  } // namespace dynd::nd::functional
} // namespace dynd::nd
} // namespace dynd
