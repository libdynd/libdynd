//
// Copyright (C) 2011-15 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <dynd/func/reduction.hpp>
#include <dynd/kernels/all_kernel.hpp>
#include <dynd/logic.hpp>

using namespace std;
using namespace dynd;

DYND_API nd::callable nd::all::make() { return functional::reduction(callable::make<all_kernel>()); }

DYND_DEFAULT_DECLFUNC_GET(nd::all)

DYND_API struct nd::all nd::all;
