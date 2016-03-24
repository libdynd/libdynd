//
// Copyright (C) 2011-15 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <dynd/callables/byteswap_callable.hpp>

using namespace std;
using namespace dynd;

nd::callable nd::byteswap::make() { return make_callable<byteswap_callable>(); }

DYND_DEFAULT_DECLFUNC_GET(nd::byteswap)

struct nd::byteswap nd::byteswap;

nd::callable nd::pairwise_byteswap::make() { return make_callable<pairwise_byteswap_callable>(); }

DYND_DEFAULT_DECLFUNC_GET(nd::pairwise_byteswap)

struct nd::pairwise_byteswap nd::pairwise_byteswap;
