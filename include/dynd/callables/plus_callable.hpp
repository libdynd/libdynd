//
// Copyright (C) 2011-16 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#pragma once

#include <dynd/callables/apply_function_callable.hpp>
#include <dynd/kernels/arithmetic.hpp>

namespace dynd {
namespace nd {

  template <type_id_t Arg0ID>
  using plus_callable =
      functional::apply_function_callable<decltype(&detail::inline_plus<Arg0ID>::f), &detail::inline_plus<Arg0ID>::f>;

} // namespace dynd::nd
} // namespace dynd
