//
// Copyright (C) 2011-15 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <iostream>
#include <sstream>
#include <stdexcept>
#include "inc_gtest.hpp"

#include <dynd/array.hpp>
#include <dynd/types/tuple_type.hpp>
#include <dynd/types/ctuple_type.hpp>
#include <dynd/types/fixedstring_type.hpp>
#include <dynd/func/callable.hpp>

using namespace std;
using namespace dynd;

TEST(CTupleType, CreateSimple) {
    ndt::type tp;
    const ctuple_type *tt;

    // Tuple with one field
    tp = ndt::make_ctuple(ndt::make_type<int32_t>());
    EXPECT_EQ(ctuple_type_id, tp.get_type_id());
    EXPECT_EQ(4u, tp.get_data_size());
    EXPECT_EQ(4u, tp.get_data_alignment());
    EXPECT_TRUE(tp.is_pod());
    EXPECT_EQ(0u, (tp.get_flags()&(type_flag_blockref|type_flag_destructor)));
    tt = tp.extended<ctuple_type>();
    ASSERT_EQ(1, tt->get_field_count());
    EXPECT_EQ(ndt::make_type<int32_t>(), tt->get_field_type(0));
    EXPECT_EQ(0u, tt->get_data_offset(0));
    EXPECT_EQ(0u, tt->get_arrmeta_offset(0));
    // Roundtripping through a string
    EXPECT_EQ(tp, ndt::type(tp.str()));

    // Tuple with two fields
    tp = ndt::make_ctuple(ndt::make_type<int16_t>(), ndt::make_type<int32_t>());
    EXPECT_EQ(ctuple_type_id, tp.get_type_id());
    EXPECT_EQ(8u, tp.get_data_size());
    EXPECT_EQ(4u, tp.get_data_alignment());
    EXPECT_TRUE(tp.is_pod());
    EXPECT_EQ(0u, (tp.get_flags()&(type_flag_blockref|type_flag_destructor)));
    tt = tp.extended<ctuple_type>();
    ASSERT_EQ(2, tt->get_field_count());
    EXPECT_EQ(ndt::make_type<int16_t>(), tt->get_field_type(0));
    EXPECT_EQ(ndt::make_type<int32_t>(), tt->get_field_type(1));
    EXPECT_EQ(0u, tt->get_data_offset(0));
    EXPECT_EQ(4u, tt->get_data_offset(1));
    EXPECT_EQ(0u, tt->get_arrmeta_offset(0));
    EXPECT_EQ(0u, tt->get_arrmeta_offset(1));
    // Roundtripping through a string
    EXPECT_EQ(tp, ndt::type(tp.str()));
}

