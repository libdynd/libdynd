//
// Copyright (C) 2011-16 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <dynd/binary_arithmetic.hpp>
#include <dynd/callables/left_shift_callable.hpp>

DYND_API nd::callable nd::left_shift =
    make_binary_arithmetic<nd::left_shift_callable, dynd::detail::isdef_left_shift, integral_types>();
