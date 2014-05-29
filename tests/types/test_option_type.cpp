//
// Copyright (C) 2011-14 Mark Wiebe, DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <iostream>
#include <sstream>
#include <stdexcept>
#include "inc_gtest.hpp"

#include <dynd/array.hpp>
#include <dynd/types/option_type.hpp>
#include <dynd/types/string_type.hpp>

using namespace std;
using namespace dynd;

TEST(OptionType, Create) {
    ndt::type d;

    d = ndt::make_option<int16_t>();
    EXPECT_EQ(option_type_id, d.get_type_id());
    EXPECT_EQ(option_kind, d.get_kind());
    EXPECT_EQ(2u, d.get_data_alignment());
    EXPECT_EQ(2u, d.get_data_size());
    EXPECT_EQ(ndt::make_type<int16_t>(),
              d.tcast<option_type>()->get_value_type());
    EXPECT_FALSE(d.is_expression());
    // Roundtripping through a string
    EXPECT_EQ(d, ndt::type(d.str()));
    EXPECT_EQ("?int16", d.str());
    EXPECT_EQ(d, ndt::type("?int16"));
    EXPECT_EQ(d, ndt::type("option[int16]"));

    d = ndt::make_option(ndt::make_string());
    EXPECT_EQ(option_type_id, d.get_type_id());
    EXPECT_EQ(option_kind, d.get_kind());
    EXPECT_EQ(ndt::make_string().get_data_alignment(), d.get_data_alignment());
    EXPECT_EQ(ndt::make_string().get_data_size(), d.get_data_size());
    EXPECT_EQ(ndt::make_string(), d.tcast<option_type>()->get_value_type());
    EXPECT_FALSE(d.is_expression());
    // Roundtripping through a string
    EXPECT_EQ(d, ndt::type(d.str()));
    EXPECT_EQ("?string", d.str());
}
