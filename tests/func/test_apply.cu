//
// Copyright (C) 2011-14 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <iostream>
#include <stdexcept>
#include <algorithm>
#include <cmath>

#include "inc_gtest.hpp"
#include "dynd_assertions.hpp"

#include <dynd/array.hpp>
#include <dynd/func/arrfunc.hpp>
#include <dynd/kernels/apply_kernels.hpp>

using namespace std;
using namespace dynd;

__device__ int func(int x, int y)
{
	return x + y;
}

__global__ void kern()
{
	typedef kernels::apply_function_ck<decltype(&func), &func, 2, int, int, int> ck_type;
	ck_type *ck;
}

TEST(Apply, CUDAFunction)
{
	kern<<<1, 1>>>();

//	std::exit(-1);
}