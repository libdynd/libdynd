//
// Copyright (C) 2011-16 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#pragma once

#include <dynd/callables/base_callable.hpp>

namespace dynd {
namespace nd {
  namespace functional {

    template <typename SignatureType>
    class base_apply_callable : public base_callable {
    public:
      template <typename... T>
      base_apply_callable(T &&... names)
          : base_callable(ndt::make_type<typename funcproto_of<SignatureType>::type>(std::forward<T>(names)...)) {}

      ndt::type resolve_return_type(const ndt::type &dst_tp) { return dst_tp; }
    };

  } // namespace dynd::nd::functional
} // namespace dynd::nd
} // namespace dynd
