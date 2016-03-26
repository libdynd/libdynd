//
// Copyright (C) 2011-15 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <dynd/callables/sort_callable.hpp>
#include <dynd/callables/unique_callable.hpp>
#include <dynd/sort.hpp>

using namespace std;
using namespace dynd;

DYND_API nd::callable nd::sort = nd::make_callable<nd::sort_callable>();

DYND_API nd::callable nd::unique = nd::make_callable<nd::unique_callable>();
