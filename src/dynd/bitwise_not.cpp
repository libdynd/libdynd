//
// Copyright (C) 2011-16 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <dynd/callables/bitwise_not_callable.hpp>
#include <dynd/unary_arithmetic.hpp>

DYND_API nd::callable nd::bitwise_not =
    make_unary_arithmetic<nd::bitwise_not_callable, dynd::detail::isdef_bitwise_not, integral_types>();
