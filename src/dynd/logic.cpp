//
// Copyright (C) 2011-16 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <dynd/callables/all_callable.hpp>
#include <dynd/functional.hpp>
#include <dynd/logic.hpp>

using namespace std;
using namespace dynd;

DYND_API nd::callable nd::all = nd::functional::reduction(true, nd::make_callable<nd::all_callable>());
