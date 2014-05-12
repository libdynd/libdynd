//
// Copyright (C) 2011-14 Mark Wiebe, DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <iostream>
#include <sstream>
#include <stdexcept>
#include "inc_gtest.hpp"

#include <dynd/array.hpp>
#include <dynd/types/funcproto_type.hpp>

using namespace std;
using namespace dynd;

TEST(FuncProtoType, CreateSimple) {
    ndt::type tp;
    const funcproto_type *fpt;

    // Function prototype from C++ template parameter
    tp = ndt::make_funcproto<int64_t (float, int32_t, double)>();
    EXPECT_EQ(funcproto_type_id, tp.get_type_id());
    EXPECT_EQ(0u, tp.get_data_size());
    EXPECT_EQ(1u, tp.get_data_alignment());
    EXPECT_FALSE(tp.is_pod());
    fpt = tp.tcast<funcproto_type>();
    ASSERT_EQ(3u, fpt->get_param_count());
    EXPECT_EQ(ndt::make_type<float>(), fpt->get_param_types()[0]);
    EXPECT_EQ(ndt::make_type<int32_t>(), fpt->get_param_types()[1]);
    EXPECT_EQ(ndt::make_type<double>(), fpt->get_param_types()[2]);
    EXPECT_EQ(ndt::make_type<int64_t>(), fpt->get_return_type());
    // Roundtripping through a string
    EXPECT_EQ(tp, ndt::type(tp.str()));
    EXPECT_EQ("(float32, int32, float64) -> int64", tp.str());
}
