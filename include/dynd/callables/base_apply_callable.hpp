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
    namespace detail {

      template <typename Signature>
      class base_apply_callable;

      template <typename Signature>
      class base_apply_callable : public base_callable {
      public:
        template <typename... S>
        base_apply_callable(S &&... names) : base_callable(ndt::make_type<Signature>(std::forward<S>(names)...)) {}

        ndt::type resolve_return_type(const ndt::type &dst_tp, size_t DYND_UNUSED(nsrc),
                                      const ndt::type *DYND_UNUSED(src_tp)) {
          return dst_tp;
        }
      };

      template <typename T0, resolve_t Resolve, typename... T>
      class base_apply_callable<void(return_wrapper<T0, Resolve>, T...)> : public base_callable {
      public:
        template <typename... S>
        base_apply_callable(S &&... names) : base_callable(ndt::make_type<T0(T...)>(std::forward<S>(names)...)) {}

        ndt::type resolve_return_type(const ndt::type &DYND_UNUSED(dst_tp), size_t nsrc, const ndt::type *src_tp) {
          return Resolve(nsrc, src_tp);
        }
      };

    } // namespace dynd::nd::functional::detail

    template <typename CallableType>
    using base_apply_callable = detail::base_apply_callable<typename funcproto_of<CallableType>::type>;

  } // namespace dynd::nd::functional
} // namespace dynd::nd
} // namespace dynd
