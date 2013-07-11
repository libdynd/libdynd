//
// Copyright (C) 2011-13 Mark Wiebe, DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <iostream>
#include <sstream>
#include <stdexcept>
#include "inc_gtest.hpp"

#include <dynd/dtype_assign.hpp>
#include <dynd/dtypes/byteswap_type.hpp>
#include <dynd/dtypes/convert_type.hpp>

using namespace std;
using namespace dynd;

TEST(ConvertDType, ExpressionInValue) {
    // When given an expression type as the destination, making a conversion dtype chains
    // the value dtype of the operand into the storage dtype of the desired result value
    ndt::type d = ndt::make_convert(ndt::make_convert(ndt::make_dtype<float>(), ndt::make_dtype<int>()), ndt::make_dtype<float>());
    EXPECT_EQ(ndt::make_convert(ndt::make_dtype<float>(), ndt::make_convert<int, float>()), d);
    EXPECT_TRUE(d.is_expression());
}

TEST(ConvertDType, CanonicalDType) {
    // The canonical type of a convert dtype is always the value
    EXPECT_EQ((ndt::make_dtype<float>()), (ndt::make_convert<float, int>().get_canonical_type()));
}

