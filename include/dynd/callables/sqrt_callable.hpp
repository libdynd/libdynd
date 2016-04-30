//
// Copyright (C) 2011-16 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#pragma once

#include <dynd/callables/apply_function_callable.hpp>
#include <dynd/kernels/arithmetic.hpp>

namespace dynd {
namespace nd {

  template <type_id_t Src0TypeID>
  using sqrt_callable = functional::apply_function_callable<decltype(&dynd::detail::inline_sqrt<Src0TypeID>::f),
                                                            &dynd::detail::inline_sqrt<Src0TypeID>::f>;

} // namespace dynd::nd
} // namespace dynd
