//
// Copyright (C) 2011-16 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <dynd/callables/plus_callable.hpp>
#include <dynd/unary_arithmetic.hpp>

DYND_API nd::callable nd::plus = make_unary_arithmetic<nd::plus_callable, dynd::detail::isdef_plus, arithmetic_types>();
