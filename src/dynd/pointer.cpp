//
// Copyright (C) 2011-15 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <dynd/pointer.hpp>
#include <dynd/callables/dereference_callable.hpp>

using namespace std;
using namespace dynd;

DYND_API nd::callable nd::dereference = nd::make_callable<nd::dereference_callable>();
