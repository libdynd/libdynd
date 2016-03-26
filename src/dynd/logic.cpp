//
// Copyright (C) 2011-15 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <dynd/functional.hpp>
#include <dynd/logic.hpp>
#include <dynd/callables/all_callable.hpp>

using namespace std;
using namespace dynd;

DYND_API nd::callable nd::all = functional::reduction(make_callable<all_callable>());
