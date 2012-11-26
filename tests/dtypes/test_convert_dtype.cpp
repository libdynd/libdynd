//
// Copyright (C) 2011-12, Dynamic NDArray Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <iostream>
#include <sstream>
#include <stdexcept>
#include "inc_gtest.hpp"

#include <dynd/dtype_assign.hpp>
#include <dynd/dtypes/byteswap_dtype.hpp>
#include <dynd/dtypes/convert_dtype.hpp>

using namespace std;
using namespace dynd;

TEST(ConvertDType, ExpressionInValue) {
    // When given an expression dtype as the destination, making a conversion dtype chains
    // the value dtype of the operand into the storage dtype of the desired result value
    dtype dt = make_convert_dtype(make_convert_dtype(make_dtype<float>(), make_dtype<int>()), make_dtype<float>());
    EXPECT_EQ(make_convert_dtype(make_dtype<float>(), make_convert_dtype<int, float>()), dt);
}

TEST(ConvertDType, CanonicalDType) {
    // The canonical dtype of a convert dtype is always the value
    EXPECT_EQ((make_dtype<float>()), (make_convert_dtype<float, int>().get_canonical_dtype()));
}