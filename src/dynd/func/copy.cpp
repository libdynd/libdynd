//
// Copyright (C) 2011-15 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <dynd/func/copy.hpp>
#include <dynd/callables/copy_callable.hpp>

using namespace std;
using namespace dynd;

DYND_API nd::callable nd::copy::make() { return make_callable<copy_callable>(); }

DYND_DEFAULT_DECLFUNC_GET(nd::copy)

DYND_API struct nd::copy nd::copy;
