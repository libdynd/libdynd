//
// Copyright (C) 2011-16 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <dynd/binary_arithmetic.hpp>
#include <dynd/callables/subtract_callable.hpp>

DYND_API nd::callable nd::subtract =
    make_binary_arithmetic<nd::subtract_callable, dynd::detail::isdef_subtract, arithmetic_types>();
