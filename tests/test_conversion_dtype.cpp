#include <iostream>
#include <sstream>
#include <stdexcept>
#include <stdint.h>
#include "inc_gtest.hpp"

#include "dnd/dtype_assign.hpp"
#include "dnd/dtypes/conversion_dtype.hpp"

using namespace std;
using namespace dnd;

TEST(ConversionDType, LosslessCasting) {
}

TEST(ConversionDType, NoExpressionInValue) {
    // The value dtype cannot itself be an expression dtype
    ASSERT_THROW(make_conversion_dtype(make_conversion_dtype(make_dtype<float>(), make_dtype<int>()), make_dtype<float>()), runtime_error);
}
