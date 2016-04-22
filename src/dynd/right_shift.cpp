//
// Copyright (C) 2011-16 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <dynd/binary_arithmetic.hpp>
#include <dynd/callables/right_shift_callable.hpp>

DYND_API nd::callable nd::right_shift =
    make_binary_arithmetic<nd::right_shift_callable, dynd::detail::isdef_right_shift, integral_ids>();
