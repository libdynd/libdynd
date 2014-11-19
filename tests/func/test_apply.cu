//
// Copyright (C) 2011-14 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include "test_apply.cpp"

TEST(Apply, CUDAFunction)
{

}

/*
__global__ void kern()
{
	typedef kernels::apply_function_ck<decltype(&func), &func, 2, int, int, int> ck_type;
	ck_type *ck;
}

TEST(Apply, CUDAFunction)
{
	nd::arrfunc af;

	af = nd::make_apply_arrfunc<decltype(&func0), &func0>();
//	EXPECT_EQ(4, af(5, 3).as<int>());

//	std::exit(-1);
}
*/