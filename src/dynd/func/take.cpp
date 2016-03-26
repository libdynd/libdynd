//
// Copyright (C) 2011-15 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <dynd/func/take.hpp>
#include <dynd/callables/take_dispatch_callable.hpp>

using namespace std;
using namespace dynd;

DYND_API nd::callable nd::take = nd::make_callable<nd::take_dispatch_callable>();
