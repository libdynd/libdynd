//
// Copyright (C) 2011-16 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <dynd/binary_arithmetic.hpp>
#include <dynd/callables/multiply_callable.hpp>

DYND_API nd::callable nd::multiply =
    make_binary_arithmetic<nd::multiply_callable, dynd::detail::isdef_multiply, arithmetic_types>();
