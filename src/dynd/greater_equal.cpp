//
// Copyright (C) 2011-16 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <dynd/callables/greater_equal_callable.hpp>
#include <dynd/comparison_common.hpp>

using namespace std;
using namespace dynd;

namespace {
nd::callable make_greater_equal() { return make_comparison_callable<nd::greater_equal_callable>(); }
}

DYND_API nd::callable nd::greater_equal = make_greater_equal();
