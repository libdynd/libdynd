//
// Copyright (C) 2011-13, DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <iostream>
#include <sstream>
#include <stdexcept>
#include "inc_gtest.hpp"

#include <dynd/ndobject.hpp>
#include <dynd/dtypes/json_dtype.hpp>
#include <dynd/dtypes/string_dtype.hpp>

using namespace std;
using namespace dynd;

TEST(JSONDType, Create) {
    dtype d;

    // Strings with various encodings
    d = make_json_dtype();
    EXPECT_EQ(json_type_id, d.get_type_id());
    EXPECT_EQ(string_kind, d.get_kind());
    EXPECT_EQ(sizeof(void *), d.get_alignment());
    EXPECT_EQ(2*sizeof(void *), d.get_data_size());
    EXPECT_FALSE(d.is_expression());
}

TEST(JSONDType, Validation) {
    ndobject a;

    a = ndobject("[1,2,3]").cast_scalars(make_json_dtype()).vals();
    EXPECT_EQ(make_json_dtype(), a.get_dtype());
    EXPECT_EQ("[1,2,3]", a.as<string>());

    EXPECT_THROW((a = ndobject("[1,2,3]#").cast_scalars(make_json_dtype()).vals()), runtime_error);
}
