//
// Copyright (C) 2011-16 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <dynd/range.hpp>
#include <dynd/callables/range_callable.hpp>

using namespace std;
using namespace dynd;

DYND_API nd::callable nd::range = nd::make_callable<nd::range_dispatch_callable>();
