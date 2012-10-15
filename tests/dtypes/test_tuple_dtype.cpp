//
// Copyright (C) 2011-12, Dynamic NDArray Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <iostream>
#include <sstream>
#include <stdexcept>
#include "inc_gtest.hpp"

#include <dnd/ndarray.hpp>
#include <dnd/dtypes/tuple_dtype.hpp>
#include <dnd/dtypes/fixedstring_dtype.hpp>

using namespace std;
using namespace dynd;

TEST(TupleDType, Create) {
    dtype dt;
    const tuple_dtype *tdt;

    // Tuple with one field
    dt = make_tuple_dtype(make_dtype<int32_t>());
    EXPECT_EQ(tuple_type_id, dt.type_id());
    EXPECT_EQ(4u, dt.element_size());
    EXPECT_EQ(4u, dt.alignment());
    EXPECT_EQ(pod_memory_management, dt.get_memory_management());
    tdt = static_cast<const tuple_dtype *>(dt.extended());
    EXPECT_TRUE(tdt->is_standard_layout());
    EXPECT_EQ(1u, tdt->get_fields().size());
    EXPECT_EQ(1u, tdt->get_offsets().size());
    EXPECT_EQ(make_dtype<int32_t>(), tdt->get_fields()[0]);
    EXPECT_EQ(0u, tdt->get_offsets()[0]);

    // Tuple with two fields
    dt = make_tuple_dtype(make_dtype<int64_t>(), make_dtype<int32_t>());
    EXPECT_EQ(tuple_type_id, dt.type_id());
    EXPECT_EQ(16u, dt.element_size());
    EXPECT_EQ(8u, dt.alignment());
    EXPECT_EQ(pod_memory_management, dt.get_memory_management());
    tdt = static_cast<const tuple_dtype *>(dt.extended());
    EXPECT_TRUE(tdt->is_standard_layout());
    EXPECT_EQ(2u, tdt->get_fields().size());
    EXPECT_EQ(2u, tdt->get_offsets().size());
    EXPECT_EQ(make_dtype<int64_t>(), tdt->get_fields()[0]);
    EXPECT_EQ(make_dtype<int32_t>(), tdt->get_fields()[1]);
    EXPECT_EQ(0u, tdt->get_offsets()[0]);
    EXPECT_EQ(8u, tdt->get_offsets()[1]);

    // Tuple with three fields
    dt = make_tuple_dtype(make_dtype<int64_t>(), make_dtype<int32_t>(), make_fixedstring_dtype(string_encoding_utf_8, 5));
    EXPECT_EQ(tuple_type_id, dt.type_id());
    EXPECT_EQ(24u, dt.element_size());
    EXPECT_EQ(8u, dt.alignment());
    EXPECT_EQ(pod_memory_management, dt.get_memory_management());
    tdt = static_cast<const tuple_dtype *>(dt.extended());
    EXPECT_TRUE(tdt->is_standard_layout());
    EXPECT_EQ(3u, tdt->get_fields().size());
    EXPECT_EQ(3u, tdt->get_offsets().size());
    EXPECT_EQ(make_dtype<int64_t>(), tdt->get_fields()[0]);
    EXPECT_EQ(make_dtype<int32_t>(), tdt->get_fields()[1]);
    EXPECT_EQ(make_fixedstring_dtype(string_encoding_utf_8, 5), tdt->get_fields()[2]);
    EXPECT_EQ(0u, tdt->get_offsets()[0]);
    EXPECT_EQ(8u, tdt->get_offsets()[1]);
    EXPECT_EQ(12u, tdt->get_offsets()[2]);

}
