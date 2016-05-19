//
// Copyright (C) 2011-16 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <dynd/binary_arithmetic.hpp>
#include <dynd/callables/logical_and_callable.hpp>

DYND_API nd::callable nd::logical_and =
    make_binary_arithmetic<nd::logical_and_callable, dynd::detail::isdef_logical_and, arithmetic_types>();
