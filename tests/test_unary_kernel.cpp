//
// Copyright (C) 2011-12, Dynamic NDArray Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <iostream>
#include <stdexcept>
#include <algorithm>
#include <cmath>
#include "inc_gtest.hpp"

#include "dnd/kernels/unary_kernel_instance.hpp"

using namespace std;
using namespace dynd;

TEST(UnaryKernel, Specialization) {
	
	//        result                                                              dst_stride dst_size src_stride src_size
	EXPECT_EQ(scalar_to_contiguous_unary_specialization, get_unary_specialization(2,         2,       0,         1        ));
	EXPECT_EQ(scalar_to_contiguous_unary_specialization, get_unary_specialization(2,         2,       0,         2        ));
	EXPECT_EQ(scalar_to_contiguous_unary_specialization, get_unary_specialization(2,         2,       0,         3        ));
	EXPECT_EQ(scalar_to_contiguous_unary_specialization, get_unary_specialization(2,         2,       0,         4        ));
	EXPECT_EQ(scalar_to_contiguous_unary_specialization, get_unary_specialization(4,         4,       0,         1        ));
	EXPECT_EQ(scalar_to_contiguous_unary_specialization, get_unary_specialization(4,         4,       0,         2        ));
	EXPECT_EQ(scalar_to_contiguous_unary_specialization, get_unary_specialization(4,         4,       0,         3        ));
	EXPECT_EQ(scalar_to_contiguous_unary_specialization, get_unary_specialization(4,         4,       0,         4        ));
	EXPECT_EQ(scalar_to_contiguous_unary_specialization, get_unary_specialization(8,         8,       0,         1        ));
	EXPECT_EQ(scalar_to_contiguous_unary_specialization, get_unary_specialization(8,         8,       0,         2        ));
	EXPECT_EQ(scalar_to_contiguous_unary_specialization, get_unary_specialization(8,         8,       0,         3        ));
	EXPECT_EQ(scalar_to_contiguous_unary_specialization, get_unary_specialization(8,         8,       0,         4        ));


	//        result                                                    dst_stride dst_size src_stride src_size
	EXPECT_EQ(contiguous_unary_specialization, get_unary_specialization(2,         2,       1,         1        ));
	EXPECT_EQ(contiguous_unary_specialization, get_unary_specialization(2,         2,       2,         2        ));
	EXPECT_EQ(contiguous_unary_specialization, get_unary_specialization(2,         2,       3,         3        ));
	EXPECT_EQ(contiguous_unary_specialization, get_unary_specialization(2,         2,       4,         4        ));
	EXPECT_EQ(contiguous_unary_specialization, get_unary_specialization(4,         4,       1,         1        ));
	EXPECT_EQ(contiguous_unary_specialization, get_unary_specialization(4,         4,       2,         2        ));
	EXPECT_EQ(contiguous_unary_specialization, get_unary_specialization(4,         4,       3,         3        ));
	EXPECT_EQ(contiguous_unary_specialization, get_unary_specialization(4,         4,       4,         4        ));
	EXPECT_EQ(contiguous_unary_specialization, get_unary_specialization(8,         8,       1,         1        ));
	EXPECT_EQ(contiguous_unary_specialization, get_unary_specialization(8,         8,       2,         2        ));
	EXPECT_EQ(contiguous_unary_specialization, get_unary_specialization(8,         8,       3,         3        ));
	EXPECT_EQ(contiguous_unary_specialization, get_unary_specialization(8,         8,       4,         4        ));

	//        result                                                dst_stride dst_size src_stride src_size
	EXPECT_EQ(scalar_unary_specialization, get_unary_specialization(0,         2,       0,         1        ));
	EXPECT_EQ(scalar_unary_specialization, get_unary_specialization(0,         2,       0,         2        ));
	EXPECT_EQ(scalar_unary_specialization, get_unary_specialization(0,         2,       0,         3        ));
	EXPECT_EQ(scalar_unary_specialization, get_unary_specialization(0,         2,       0,         4        ));
	EXPECT_EQ(scalar_unary_specialization, get_unary_specialization(0,         4,       0,         1        ));
	EXPECT_EQ(scalar_unary_specialization, get_unary_specialization(0,         4,       0,         2        ));
	EXPECT_EQ(scalar_unary_specialization, get_unary_specialization(0,         4,       0,         3        ));
	EXPECT_EQ(scalar_unary_specialization, get_unary_specialization(0,         4,       0,         4        ));
	EXPECT_EQ(scalar_unary_specialization, get_unary_specialization(0,         8,       0,         1        ));
	EXPECT_EQ(scalar_unary_specialization, get_unary_specialization(0,         8,       0,         2        ));
	EXPECT_EQ(scalar_unary_specialization, get_unary_specialization(0,         8,       0,         3        ));
	EXPECT_EQ(scalar_unary_specialization, get_unary_specialization(0,         8,       0,         4        ));

	//        result                                                 dst_stride dst_size src_stride src_size
	EXPECT_EQ(general_unary_specialization, get_unary_specialization(2,         1,       0,         1        ));
	EXPECT_EQ(general_unary_specialization, get_unary_specialization(2,         4,       0,         2        ));
	EXPECT_EQ(general_unary_specialization, get_unary_specialization(2,         8,       0,         3        ));
	EXPECT_EQ(general_unary_specialization, get_unary_specialization(4,         1,       0,         1        ));
	EXPECT_EQ(general_unary_specialization, get_unary_specialization(4,         2,       0,         2        ));
	EXPECT_EQ(general_unary_specialization, get_unary_specialization(4,         8,       0,         4        ));
	EXPECT_EQ(general_unary_specialization, get_unary_specialization(8,         1,       0,         1        ));
	EXPECT_EQ(general_unary_specialization, get_unary_specialization(8,         2,       0,         2        ));
	EXPECT_EQ(general_unary_specialization, get_unary_specialization(8,         4,       0,         3        ));


}

