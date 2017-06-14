//
// Copyright (C) 2011-16 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <dynd/callables/less_callable.hpp>
#include <dynd/comparison_common.hpp>

using namespace std;
using namespace dynd;

namespace {
nd::callable make_less() { return make_comparison_callable<nd::less_callable>(); }
}

DYND_API nd::callable nd::less = make_less();
