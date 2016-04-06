//
// Copyright (C) 2011-16 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#pragma once

#include <dynd/callables/apply_function_callable.hpp>
#include <dynd/kernels/arithmetic.hpp>

namespace dynd {
namespace nd {

  template <type_id_t Arg0ID, type_id_t Arg1ID>
  using divide_callable = functional::apply_function_callable<decltype(&detail::inline_divide<Arg0ID, Arg1ID>::f),
                                                              &detail::inline_divide<Arg0ID, Arg1ID>::f>;

} // namespace dynd::nd
} // namespace dynd
