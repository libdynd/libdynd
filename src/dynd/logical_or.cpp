//
// Copyright (C) 2011-16 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <dynd/binary_arithmetic.hpp>
#include <dynd/callables/logical_or_callable.hpp>

DYND_API nd::callable nd::logical_or =
    make_binary_arithmetic<nd::logical_or_callable, dynd::detail::isdef_logical_or, arithmetic_ids>();
