//
// Copyright (C) 2011-15 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <dynd/func/view.hpp>
#include <dynd/callables/view_callable.hpp>

using namespace std;
using namespace dynd;

DYND_API nd::callable nd::view = nd::make_callable<nd::view_callable>();
