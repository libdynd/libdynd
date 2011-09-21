#include <iostream>
#include <stdexcept>
#include <cmath>
#include <stdint.h>
#include <gtest/gtest.h>

#include "dnd/ndarray.hpp"

using namespace std;
using namespace dnd;

TEST(NDArray, AsScalar) {
    ndarray a;

    a = ndarray(make_dtype<float>());
    a.vassign(3.14f);
    EXPECT_EQ(3.14f, a.as_scalar<float>());
    EXPECT_EQ(3.14f, a.as_scalar<double>());
    EXPECT_THROW(a.as_scalar<int64_t>(), runtime_error);
    EXPECT_EQ(3, a.as_scalar<int64_t>(assign_error_overflow));
    EXPECT_THROW(a.as_scalar<dnd_bool>(), runtime_error);
    EXPECT_THROW(a.as_scalar<dnd_bool>(assign_error_overflow), runtime_error);
    EXPECT_EQ(true, a.as_scalar<dnd_bool>(assign_error_none));

    a = ndarray(make_dtype<double>());
    a.vassign(3.141592653589);
    EXPECT_EQ(3.141592653589, a.as_scalar<double>());
    EXPECT_THROW(a.as_scalar<float>(assign_error_inexact), runtime_error);
    EXPECT_EQ(3.141592653589f, a.as_scalar<float>());
}

