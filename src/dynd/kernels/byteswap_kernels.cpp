//
// Copyright (C) 2011-15 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <dynd/callables/byteswap_callable.hpp>

using namespace std;
using namespace dynd;

DYND_API nd::callable nd::byteswap = nd::make_callable<nd::byteswap_callable>();
DYND_API nd::callable nd::pairwise_byteswap = nd::make_callable<nd::pairwise_byteswap_callable>();
