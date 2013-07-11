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
    ndt::type d = make_convert_type(make_convert_type(ndt::make_dtype<float>(), ndt::make_dtype<int>()), ndt::make_dtype<float>());
    EXPECT_EQ(make_convert_type(ndt::make_dtype<float>(), make_convert_type<int, float>()), d);
    EXPECT_TRUE(d.is_expression());
}

TEST(ConvertDType, CanonicalDType) {
    // The canonical type of a convert dtype is always the value
    EXPECT_EQ((ndt::make_dtype<float>()), (make_convert_type<float, int>().get_canonical_type()));
}

