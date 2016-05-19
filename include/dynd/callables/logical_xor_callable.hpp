//
// Copyright (C) 2011-16 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#pragma once

#include <dynd/callables/apply_function_callable.hpp>
#include <dynd/kernels/arithmetic.hpp>

namespace dynd {
namespace nd {

  template <typename Arg0Type, typename Arg1Type>
  using logical_xor_callable =
      functional::apply_function_callable<decltype(&dynd::detail::inline_logical_xor<Arg0Type, Arg1Type>::f),
                                          &dynd::detail::inline_logical_xor<Arg0Type, Arg1Type>::f>;

} // namespace dynd::nd
} // namespace dynd
