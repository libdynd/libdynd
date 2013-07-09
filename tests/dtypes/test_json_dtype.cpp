//
// Copyright (C) 2011-13 Mark Wiebe, DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <iostream>
#include <sstream>
#include <stdexcept>
#include "inc_gtest.hpp"

#include <dynd/array.hpp>
#include <dynd/dtypes/json_dtype.hpp>
#include <dynd/dtypes/string_dtype.hpp>

using namespace std;
using namespace dynd;

TEST(JSONDType, Create) {
    ndt::type d;

    // Strings with various encodings
    d = make_json_dtype();
    EXPECT_EQ(json_type_id, d.get_type_id());
    EXPECT_EQ(string_kind, d.get_kind());
    EXPECT_EQ(sizeof(void *), d.get_data_alignment());
    EXPECT_EQ(2*sizeof(void *), d.get_data_size());
    EXPECT_FALSE(d.is_expression());
}

TEST(JSONDType, Validation) {
    nd::array a;

    a = nd::array("[1,2,3]").ucast(make_json_dtype()).eval();
    EXPECT_EQ(make_json_dtype(), a.get_dtype());
    EXPECT_EQ("[1,2,3]", a.as<string>());

    EXPECT_THROW(nd::array("[1,2,3]#").ucast(make_json_dtype()).eval(), runtime_error);
}
