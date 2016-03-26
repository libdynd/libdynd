//
// Copyright (C) 2011-15 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <dynd/array.hpp>
#include <dynd/func/mean.hpp>
#include <dynd/func/reduction.hpp>
#include <dynd/kernels/base_kernel.hpp>
#include <dynd/kernels/mean_kernel.hpp>
#include <dynd/kernels/sum_kernel.hpp>
#include <dynd/callables/mean_callable.hpp>

using namespace std;
using namespace dynd;

DYND_API nd::callable nd::mean = nd::make_callable<nd::mean_callable>(ndt::type(int64_id));
