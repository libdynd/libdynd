//
// Copyright (C) 2011-15 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#pragma once

#include <numeric>

#include <dynd/callables/adapt_callable.hpp>
#include <dynd/callables/call_callable.hpp>
#include <dynd/callables/forward_na_callable.hpp>
#include <dynd/func/elwise.hpp>
#include <dynd/func/reduction.hpp>
#include <dynd/func/outer.hpp>
#include <dynd/iterator.hpp>
#include <dynd/callable.hpp>
#include <dynd/callables/base_dispatch_callable.hpp>
#include <dynd/callables/arithmetic_dispatch_callable.hpp>

namespace dynd {
namespace nd {
  namespace functional {

    template <typename FuncType>
    callable call(const ndt::type &tp)
    {
      return make_callable<call_callable<FuncType>>(tp);
    }

    inline callable adapt(const ndt::type &value_tp, const callable &forward)
    {
      return make_callable<adapt_callable>(value_tp, forward);
    }

    template <int... I>
    callable forward_na(const callable &child)
    {
      ndt::type tp = ndt::callable_type::make(ndt::make_type<ndt::option_type>(child.get_ret_type()),
                                              {ndt::type("Any"), ndt::type("Any")});
      return make_callable<forward_na_callable<I...>>(tp, child);
    }

  } // namespace dynd::nd::functional
} // namespace dynd::nd
} // namespace dynd
