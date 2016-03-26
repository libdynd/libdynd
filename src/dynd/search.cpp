//
// Copyright (C) 2011-15 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <dynd/callables/binary_search_callable.hpp>
#include <dynd/search.hpp>

using namespace std;
using namespace dynd;

DYND_API nd::callable nd::binary_search = nd::make_callable<nd::binary_search_callable>();
