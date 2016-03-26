//
// Copyright (C) 2011-15 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <dynd/functional.hpp>
#include <dynd/io.hpp>
#include <dynd/callables/serialize_callable.hpp>

using namespace std;
using namespace dynd;

DYND_API nd::callable nd::serialize =
    nd::functional::reduction(nd::make_callable<nd::serialize_callable<scalar_kind_id>>());
