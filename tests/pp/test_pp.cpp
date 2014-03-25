//
// Copyright (C) 2011-14 Irwin Zaid, DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include "inc_gtest.hpp"

#include <dynd/pp/arithmetic.hpp>
#include <dynd/pp/comparision.hpp>
#include <dynd/pp/if.hpp>
#include <dynd/pp/list.hpp>
#include <dynd/pp/logical.hpp>

using namespace std;

TEST(PP, Cat) {
    EXPECT_EQ(DYND_PP_CAT(1, 2, 3), 123);
    EXPECT_EQ(DYND_PP_CAT(1, 2, 3, 4), 1234);
}


