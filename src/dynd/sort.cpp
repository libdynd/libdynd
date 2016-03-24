//
// Copyright (C) 2011-15 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <dynd/callables/sort_callable.hpp>
#include <dynd/callables/unique_callable.hpp>
#include <dynd/sort.hpp>

using namespace std;
using namespace dynd;

DYND_API nd::callable nd::sort::make() { return make_callable<sort_callable>(); }

DYND_DEFAULT_DECLFUNC_GET(nd::sort)

DYND_API struct nd::sort nd::sort;

DYND_API nd::callable nd::unique::make() { return make_callable<unique_callable>(); }

DYND_DEFAULT_DECLFUNC_GET(nd::unique)

DYND_API struct nd::unique nd::unique;
