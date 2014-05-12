//
// Copyright (C) 2011-14 Mark Wiebe, DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <iostream>
#include <sstream>
#include <stdexcept>
#include "inc_gtest.hpp"

#include <dynd/array.hpp>
#include <dynd/types/typevar_type.hpp>
#include <dynd/types/typevar_dim_type.hpp>

using namespace std;
using namespace dynd;

TEST(TypeVarType, CreateSimple) {
    ndt::type tp;
    const typevar_type *tvt;

    // Simple TypeVar
    tp = ndt::make_typevar("Blah");
    EXPECT_EQ(typevar_type_id, tp.get_type_id());
    EXPECT_EQ(0u, tp.get_data_size());
    EXPECT_EQ(1u, tp.get_data_alignment());
    EXPECT_FALSE(tp.is_pod());
    tvt = tp.tcast<typevar_type>();
    EXPECT_EQ("Blah", tvt->get_name_str());
    // Roundtripping through a string
    EXPECT_EQ(tp, ndt::type(tp.str()));
    EXPECT_EQ("Blah", tp.str());

    // The typevar name must start with a capital
    // and look like an identifier
    EXPECT_THROW(ndt::make_typevar("blah"), type_error);
    EXPECT_THROW(ndt::make_typevar("T "), type_error);
    EXPECT_THROW(ndt::make_typevar("123"), type_error);
    EXPECT_THROW(ndt::make_typevar("Two+"), type_error);
}
 
TEST(TypeVarDimType, CreateSimple) {
    ndt::type tp;
    const typevar_dim_type *tvt;

    // Simple Dimension TypeVar
    tp = ndt::make_typevar_dim("Blah", ndt::make_type<int>());
    EXPECT_EQ(typevar_dim_type_id, tp.get_type_id());
    EXPECT_EQ(0u, tp.get_data_size());
    EXPECT_EQ(1u, tp.get_data_alignment());
    EXPECT_FALSE(tp.is_pod());
    tvt = tp.tcast<typevar_dim_type>();
    EXPECT_EQ("Blah", tvt->get_name_str());
    EXPECT_EQ(ndt::make_type<int>(), tvt->get_element_type());
    // Roundtripping through a string
    EXPECT_EQ(tp, ndt::type(tp.str()));
    EXPECT_EQ("Blah * int32", tp.str());

    // The typevar name must start with a capital
    // and look like an identifier
    EXPECT_THROW(ndt::make_typevar_dim("blah", ndt::make_type<int>()),
                 type_error);
    EXPECT_THROW(ndt::make_typevar_dim("T ", ndt::make_type<int>()),
                 type_error);
    EXPECT_THROW(ndt::make_typevar_dim("123", ndt::make_type<int>()),
                 type_error);
    EXPECT_THROW(ndt::make_typevar_dim("Two+", ndt::make_type<int>()),
                 type_error);
}
 
