//
// Copyright (C) 2011-15 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <dynd/func/pointer.hpp>
#include <dynd/functional.hpp>
#include <dynd/callables/dereference_callable.hpp>

using namespace std;
using namespace dynd;

DYND_API nd::callable nd::dereference::make() { return make_callable<dereference_callable>(); }

DYND_DEFAULT_DECLFUNC_GET(nd::dereference)

DYND_API struct nd::dereference nd::dereference;
