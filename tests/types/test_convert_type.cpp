//
// Copyright (C) 2011-13 Mark Wiebe, DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <iostream>
#include <sstream>
#include <stdexcept>
#include "inc_gtest.hpp"

#include <dynd/typed_data_assign.hpp>
#include <dynd/types/byteswap_type.hpp>
#include <dynd/types/convert_type.hpp>

using namespace std;
using namespace dynd;

TEST(ConvertDType, ExpressionInValue) {
    // When given an expression type as the destination, making a conversion type chains
    // the value type of the operand into the storage type of the desired result value
    ndt::type d = ndt::make_convert(ndt::make_convert(ndt::make_type<float>(), ndt::make_type<int>()), ndt::make_type<float>());
    EXPECT_EQ(ndt::make_convert(ndt::make_type<float>(), ndt::make_convert<int, float>()), d);
    EXPECT_TRUE(d.is_expression());
}

TEST(ConvertDType, CanonicalDType) {
    // The canonical type of a convert type is always the value
    EXPECT_EQ((ndt::make_type<float>()), (ndt::make_convert<float, int>().get_canonical_type()));
}

