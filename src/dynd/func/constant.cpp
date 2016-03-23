//
// Copyright (C) 2011-15 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <dynd/func/constant.hpp>
#include <dynd/callables/constant_callable.hpp>

using namespace std;
using namespace dynd;

nd::callable nd::functional::constant(const array &val) { return make_callable<constant_callable>(val); }
