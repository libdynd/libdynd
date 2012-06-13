#include <iostream>
#include <sstream>
#include <stdexcept>
#include <stdint.h>
#include "inc_gtest.hpp"

#include <dnd/dtype_assign.hpp>
#include <dnd/dtypes/byteswap_dtype.hpp>
#include <dnd/dtypes/conversion_dtype.hpp>

using namespace std;
using namespace dnd;

TEST(ConversionDType, ExpressionInValue) {
    // When given an expression dtype as the destination, making a conversion dtype chains
    // the value dtype of the operand into the storage dtype of the desired result value
    dtype dt = make_conversion_dtype(make_conversion_dtype(make_dtype<float>(), make_dtype<int>()), make_dtype<float>());
    EXPECT_EQ(make_conversion_dtype(make_dtype<float>(), make_conversion_dtype<int, float>()), dt);

    // Some expression dtypes, like byteswap, don't accept arbitrary changes to their operand, so they won't fit as the target value
    EXPECT_THROW(make_conversion_dtype(make_conversion_dtype(make_dtype<float>(), make_byteswap_dtype<int>()), make_dtype<float>()), runtime_error);
}
