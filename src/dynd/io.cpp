//
// Copyright (C) 2011-16 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <dynd/callables/serialize_callable.hpp>
#include <dynd/functional.hpp>
#include <dynd/io.hpp>

using namespace std;
using namespace dynd;

DYND_API nd::callable nd::serialize =
    nd::functional::reduction(nd::make_callable<nd::serialize_callable<ndt::scalar_kind_type>>());
