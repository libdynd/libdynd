//
// Copyright (C) 2011-16 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <dynd/array.hpp>
#include <dynd/callable.hpp>
#include <dynd/callables/field_access_callable.hpp>
#include <dynd/functional.hpp>
#include <dynd/kernels/field_access_kernel.hpp>
#include <dynd/access.hpp>
#include <dynd/types/callable_type.hpp>

using namespace std;
using namespace dynd;

DYND_API nd::callable nd::access = nd::make_callable<nd::get_array_field_callable>();

DYND_API nd::callable nd::field_access = nd::make_callable<nd::field_access_callable>();
