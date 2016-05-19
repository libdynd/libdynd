//
// Copyright (C) 2011-16 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <dynd/binary_arithmetic.hpp>
#include <dynd/callables/mod_callable.hpp>

DYND_API nd::callable nd::mod = make_binary_arithmetic<nd::mod_callable, dynd::detail::isdef_mod, integral_types>();
