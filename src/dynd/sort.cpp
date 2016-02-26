//
// Copyright (C) 2011-15 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <dynd/kernels/sort_kernel.hpp>
#include <dynd/kernels/unique_kernel.hpp>
#include <dynd/sort.hpp>

using namespace std;
using namespace dynd;

DYND_API nd::callable nd::sort::make() { return callable::make<sort_kernel>(); }

DYND_DEFAULT_DECLFUNC_GET(nd::sort)

DYND_API struct nd::sort nd::sort;

DYND_API nd::callable nd::unique::make() { return callable::make<unique_kernel>(); }

DYND_DEFAULT_DECLFUNC_GET(nd::unique)

DYND_API struct nd::unique nd::unique;
