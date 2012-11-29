//
// Copyright (C) 2011-12, Dynamic NDArray Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <iostream>
#include <sstream>
#include <stdexcept>
#include "inc_gtest.hpp"

#include <dynd/ndobject.hpp>
#include <dynd/dtypes/bytes_dtype.hpp>
#include <dynd/dtypes/strided_array_dtype.hpp>
#include <dynd/dtypes/fixedbytes_dtype.hpp>
#include <dynd/dtypes/convert_dtype.hpp>

using namespace std;
using namespace dynd;

TEST(BytesDType, Create) {
    dtype d;

    // Strings with various alignments
    d = make_bytes_dtype(1);
    EXPECT_EQ(bytes_type_id, d.get_type_id());
    EXPECT_EQ(bytes_kind, d.get_kind());
    EXPECT_EQ(sizeof(void *), d.get_alignment());
    EXPECT_EQ(2*sizeof(void *), d.get_element_size());
    EXPECT_EQ(1, static_cast<const bytes_dtype *>(d.extended())->get_data_alignment());
    EXPECT_FALSE(d.is_expression());

    d = make_bytes_dtype(2);
    EXPECT_EQ(bytes_type_id, d.get_type_id());
    EXPECT_EQ(bytes_kind, d.get_kind());
    EXPECT_EQ(sizeof(void *), d.get_alignment());
    EXPECT_EQ(2*sizeof(void *), d.get_element_size());
    EXPECT_EQ(2, static_cast<const bytes_dtype *>(d.extended())->get_data_alignment());

    d = make_bytes_dtype(4);
    EXPECT_EQ(bytes_type_id, d.get_type_id());
    EXPECT_EQ(bytes_kind, d.get_kind());
    EXPECT_EQ(sizeof(void *), d.get_alignment());
    EXPECT_EQ(2*sizeof(void *), d.get_element_size());
    EXPECT_EQ(4, static_cast<const bytes_dtype *>(d.extended())->get_data_alignment());

    d = make_bytes_dtype(8);
    EXPECT_EQ(bytes_type_id, d.get_type_id());
    EXPECT_EQ(bytes_kind, d.get_kind());
    EXPECT_EQ(sizeof(void *), d.get_alignment());
    EXPECT_EQ(2*sizeof(void *), d.get_element_size());
    EXPECT_EQ(8, static_cast<const bytes_dtype *>(d.extended())->get_data_alignment());

    d = make_bytes_dtype(16);
    EXPECT_EQ(bytes_type_id, d.get_type_id());
    EXPECT_EQ(bytes_kind, d.get_kind());
    EXPECT_EQ(sizeof(void *), d.get_alignment());
    EXPECT_EQ(16, static_cast<const bytes_dtype *>(d.extended())->get_data_alignment());
}
