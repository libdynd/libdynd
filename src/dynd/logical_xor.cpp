//
// Copyright (C) 2011-16 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <dynd/binary_arithmetic.hpp>
#include <dynd/callables/logical_xor_callable.hpp>

DYND_API nd::callable nd::logical_xor =
    make_binary_arithmetic<nd::logical_xor_callable, dynd::detail::isdef_logical_xor, arithmetic_types>();
