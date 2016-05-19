//
// Copyright (C) 2011-16 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <dynd/binary_arithmetic.hpp>
#include <dynd/callables/bitwise_and_callable.hpp>

DYND_API nd::callable nd::bitwise_and =
    make_binary_arithmetic<nd::bitwise_and_callable, dynd::detail::isdef_bitwise_and, integral_types>();
