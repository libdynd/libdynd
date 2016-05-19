//
// Copyright (C) 2011-16 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <dynd/binary_arithmetic.hpp>
#include <dynd/callables/add_callable.hpp>

DYND_API nd::callable nd::add = make_binary_arithmetic<nd::add_callable, dynd::detail::isdef_add, arithmetic_types>();
