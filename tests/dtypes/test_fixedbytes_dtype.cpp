//
// Copyright (C) 2011-13, DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <iostream>
#include <sstream>
#include <stdexcept>
#include "inc_gtest.hpp"

#include <dynd/ndobject.hpp>
#include <dynd/dtypes/fixedbytes_dtype.hpp>

using namespace std;
using namespace dynd;

TEST(FixedBytesDType, Create) {
    dtype d;

    // Strings with various encodings and sizes
    d = make_fixedbytes_dtype(7, 1);
    EXPECT_EQ(fixedbytes_type_id, d.get_type_id());
    EXPECT_EQ(bytes_kind, d.get_kind());
    EXPECT_EQ(1u, d.get_alignment());
    EXPECT_EQ(7u, d.get_data_size());
    EXPECT_FALSE(d.is_expression());

    // Strings with various encodings and sizes
    d = make_fixedbytes_dtype(12, 4);
    EXPECT_EQ(fixedbytes_type_id, d.get_type_id());
    EXPECT_EQ(bytes_kind, d.get_kind());
    EXPECT_EQ(4u, d.get_alignment());
    EXPECT_EQ(12u, d.get_data_size());
    EXPECT_FALSE(d.is_expression());

    // Larger element than data size
    EXPECT_THROW(make_fixedbytes_dtype(1, 2), runtime_error);
    // Invalid alignment
    EXPECT_THROW(make_fixedbytes_dtype(6, 3), runtime_error);
    EXPECT_THROW(make_fixedbytes_dtype(10, 5), runtime_error);
    // Alignment must divide size
    EXPECT_THROW(make_fixedbytes_dtype(9, 4), runtime_error);
}

TEST(FixedBytesDType, Assign) {
    char a[3] = {0, 0, 0};
    char b[3] = {1, 2, 3};

    // Assignment with fixedbytes
    dtype_assign(make_fixedbytes_dtype(3, 1), NULL, a,
                 make_fixedbytes_dtype(3, 1), NULL, b);
    EXPECT_EQ(1, a[0]);
    EXPECT_EQ(2, a[1]);
    EXPECT_EQ(3, a[2]);

    // Must be the same size
    EXPECT_THROW(dtype_assign(make_fixedbytes_dtype(2, 1), NULL, a,
                 make_fixedbytes_dtype(3, 1), NULL, b),
                    runtime_error);
}