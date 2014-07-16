//
// Copyright (C) 2011-14 Irwin Zaid, Mark Wiebe, DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <iostream>
#include <stdexcept>
#include <algorithm>
#include <cmath>

#include "inc_gtest.hpp"

#include <dynd/arrmeta_holder.hpp>
#include <dynd/types/fixed_dim_type.hpp>
#include <dynd/types/string_type.hpp>
#include <dynd/typed_data_assign.hpp>

using namespace std;
using namespace dynd;

TEST(ArrMetaHolder, Basic) {
    // The string type requires a memory block allocated in its
    // arrmeta, so this test checks that the arrmeta_holder can
    // allocate and manage that.
    string_type_data sarr[3];
    int iarr[3] = {-1234, 0, 999992};
    memset(sarr, 0, sizeof(sarr));
    intptr_t sarr_size = 3;

    arrmeta_holder smeta(ndt::type("strided * string"));
    EXPECT_EQ(smeta.get_type(), ndt::type("strided * string"));
    arrmeta_holder imeta(ndt::type("3 * int"));
    EXPECT_EQ(imeta.get_type(), ndt::type("fixed[3] * int32"));
    smeta.arrmeta_default_construct(1, &sarr_size);
    imeta.get_at<fixed_dim_type_arrmeta>(0)->dim_size = 3;
    imeta.get_at<fixed_dim_type_arrmeta>(0)->stride = sizeof(int);

    // Copy from iarr to sarr
    typed_data_assign(smeta.get_type(), smeta.get(),
                      reinterpret_cast<char *>(sarr), imeta.get_type(),
                      imeta.get(), reinterpret_cast<const char *>(iarr));
    EXPECT_EQ("-1234", string(sarr[0].begin, sarr[0].end));
    EXPECT_EQ("0", string(sarr[1].begin, sarr[1].end));
    EXPECT_EQ("999992", string(sarr[2].begin, sarr[2].end));
}
