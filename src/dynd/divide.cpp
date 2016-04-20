//
// Copyright (C) 2011-16 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <dynd/binary_arithmetic.hpp>
#include <dynd/callables/divide_callable.hpp>

DYND_API nd::callable nd::divide =
    make_binary_arithmetic<nd::divide_callable, dynd::detail::isdef_divide, arithmetic_ids>();
