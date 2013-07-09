//
// Copyright (C) 2011-13 Mark Wiebe, DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <iostream>
#include <sstream>
#include <stdexcept>
#include "inc_gtest.hpp"

#include <dynd/array.hpp>
#include <dynd/dtypes/bytes_dtype.hpp>
#include <dynd/dtypes/strided_dim_dtype.hpp>
#include <dynd/dtypes/fixedbytes_dtype.hpp>
#include <dynd/dtypes/convert_dtype.hpp>
#include <dynd/dtypes/string_dtype.hpp>

using namespace std;
using namespace dynd;

TEST(BytesDType, Create) {
    ndt::type d;

    // Strings with various alignments
    d = make_bytes_dtype(1);
    EXPECT_EQ(bytes_type_id, d.get_type_id());
    EXPECT_EQ(bytes_kind, d.get_kind());
    EXPECT_EQ(sizeof(void *), d.get_data_alignment());
    EXPECT_EQ(2*sizeof(void *), d.get_data_size());
    EXPECT_EQ(1u, static_cast<const bytes_dtype *>(d.extended())->get_target_alignment());
    EXPECT_EQ(1u, d.p("target_alignment").as<size_t>());
    EXPECT_FALSE(d.is_expression());

    d = make_bytes_dtype(2);
    EXPECT_EQ(bytes_type_id, d.get_type_id());
    EXPECT_EQ(bytes_kind, d.get_kind());
    EXPECT_EQ(sizeof(void *), d.get_data_alignment());
    EXPECT_EQ(2*sizeof(void *), d.get_data_size());
    EXPECT_EQ(2u, static_cast<const bytes_dtype *>(d.extended())->get_target_alignment());
    EXPECT_EQ(2u, d.p("target_alignment").as<size_t>());

    d = make_bytes_dtype(4);
    EXPECT_EQ(bytes_type_id, d.get_type_id());
    EXPECT_EQ(bytes_kind, d.get_kind());
    EXPECT_EQ(sizeof(void *), d.get_data_alignment());
    EXPECT_EQ(2*sizeof(void *), d.get_data_size());
    EXPECT_EQ(4u, static_cast<const bytes_dtype *>(d.extended())->get_target_alignment());
    EXPECT_EQ(4u, d.p("target_alignment").as<size_t>());

    d = make_bytes_dtype(8);
    EXPECT_EQ(bytes_type_id, d.get_type_id());
    EXPECT_EQ(bytes_kind, d.get_kind());
    EXPECT_EQ(sizeof(void *), d.get_data_alignment());
    EXPECT_EQ(2*sizeof(void *), d.get_data_size());
    EXPECT_EQ(8u, static_cast<const bytes_dtype *>(d.extended())->get_target_alignment());
    EXPECT_EQ(8u, d.p("target_alignment").as<size_t>());

    d = make_bytes_dtype(16);
    EXPECT_EQ(bytes_type_id, d.get_type_id());
    EXPECT_EQ(bytes_kind, d.get_kind());
    EXPECT_EQ(sizeof(void *), d.get_data_alignment());
    EXPECT_EQ(16u, static_cast<const bytes_dtype *>(d.extended())->get_target_alignment());
    EXPECT_EQ(16u, d.p("target_alignment").as<size_t>());
}

TEST(BytesDType, Assign) {
    nd::array a, b, c;

    // Round-trip a string through a bytes assignment
    a = nd::array("testing").view_scalars(make_bytes_dtype(1));
    EXPECT_EQ(a.get_dtype(), make_bytes_dtype(1));
    b = nd::empty(make_bytes_dtype(1));
    b.vals() = a;
    c = b.view_scalars(make_string_dtype());
    EXPECT_EQ(c.get_dtype(), make_string_dtype());
    EXPECT_EQ("testing", c.as<string>());
}

