//
// Copyright (C) 2011-12, Dynamic NDArray Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <iostream>
#include <sstream>
#include <stdexcept>
#include "inc_gtest.hpp"

#include <dynd/ndarray.hpp>
#include <dynd/dtypes/tuple_dtype.hpp>
#include <dynd/dtypes/fixedstring_dtype.hpp>

using namespace std;
using namespace dynd;

TEST(TupleDType, CreateOneField) {
cout << "line " << __LINE__ << endl;
    dtype dt;
cout << "line " << __LINE__ << endl;
    const tuple_dtype *tdt;
cout << "line " << __LINE__ << endl;

    // Tuple with one field
    dt = make_tuple_dtype(make_dtype<int32_t>());
cout << "line " << __LINE__ << endl;
    EXPECT_EQ(tuple_type_id, dt.type_id());
cout << "line " << __LINE__ << endl;
    EXPECT_EQ(4u, dt.element_size());
cout << "line " << __LINE__ << endl;
    EXPECT_EQ(4u, dt.alignment());
cout << "line " << __LINE__ << endl;
    EXPECT_EQ(pod_memory_management, dt.get_memory_management());
cout << "line " << __LINE__ << endl;
    tdt = static_cast<const tuple_dtype *>(dt.extended());
cout << "line " << __LINE__ << endl;
    EXPECT_TRUE(tdt->is_standard_layout());
cout << "line " << __LINE__ << endl;
    EXPECT_EQ(1u, tdt->get_fields().size());
cout << "line " << __LINE__ << endl;
    EXPECT_EQ(1u, tdt->get_offsets().size());
cout << "line " << __LINE__ << endl;
    EXPECT_EQ(make_dtype<int32_t>(), tdt->get_fields()[0]);
cout << "line " << __LINE__ << endl;
    EXPECT_EQ(0u, tdt->get_offsets()[0]);
cout << "line " << __LINE__ << endl;
}

TEST(TupleDType, CreateTwoField) {
cout << "line " << __LINE__ << endl;
    dtype dt;
cout << "line " << __LINE__ << endl;
    const tuple_dtype *tdt;
cout << "line " << __LINE__ << endl;

cout << "line " << __LINE__ << endl;

    // Tuple with two fields
    dt = make_tuple_dtype(make_dtype<int64_t>(), make_dtype<int32_t>());
cout << "line " << __LINE__ << endl;
    EXPECT_EQ(tuple_type_id, dt.type_id());
cout << "line " << __LINE__ << endl;
    EXPECT_EQ(16u, dt.element_size());
cout << "line " << __LINE__ << endl;
    EXPECT_EQ(8u, dt.alignment());
cout << "line " << __LINE__ << endl;
    EXPECT_EQ(pod_memory_management, dt.get_memory_management());
cout << "line " << __LINE__ << endl;
    tdt = static_cast<const tuple_dtype *>(dt.extended());
cout << "line " << __LINE__ << endl;
    EXPECT_TRUE(tdt->is_standard_layout());
cout << "line " << __LINE__ << endl;
    EXPECT_EQ(2u, tdt->get_fields().size());
cout << "line " << __LINE__ << endl;
    EXPECT_EQ(2u, tdt->get_offsets().size());
cout << "line " << __LINE__ << endl;
    EXPECT_EQ(make_dtype<int64_t>(), tdt->get_fields()[0]);
cout << "line " << __LINE__ << endl;
    EXPECT_EQ(make_dtype<int32_t>(), tdt->get_fields()[1]);
cout << "line " << __LINE__ << endl;
    EXPECT_EQ(0u, tdt->get_offsets()[0]);
cout << "line " << __LINE__ << endl;
    EXPECT_EQ(8u, tdt->get_offsets()[1]);
cout << "line " << __LINE__ << endl;
}

TEST(TupleDType, CreateThreeField) {
cout << "line " << __LINE__ << endl;
    dtype dt;
cout << "line " << __LINE__ << endl;
    const tuple_dtype *tdt;
cout << "line " << __LINE__ << endl;

    // Tuple with three fields
    dtype d1 = make_dtype<int64_t>();
    dtype d2 = make_dtype<int32_t>();
    dtype d3 = make_fixedstring_dtype(string_encoding_utf_8, 5);
    dt = make_tuple_dtype(d1, d2, d3);
cout << "line " << __LINE__ << endl;
    EXPECT_EQ(tuple_type_id, dt.type_id());
cout << "line " << __LINE__ << endl;
    EXPECT_EQ(24u, dt.element_size());
cout << "line " << __LINE__ << endl;
    EXPECT_EQ(8u, dt.alignment());
cout << "line " << __LINE__ << endl;
cout << "line " << __LINE__ << endl;
    EXPECT_EQ(pod_memory_management, dt.get_memory_management());
cout << "line " << __LINE__ << endl;
    tdt = static_cast<const tuple_dtype *>(dt.extended());
cout << "line " << __LINE__ << endl;
    EXPECT_TRUE(tdt->is_standard_layout());
cout << "line " << __LINE__ << endl;
    EXPECT_EQ(3u, tdt->get_fields().size());
cout << "line " << __LINE__ << endl;
    EXPECT_EQ(3u, tdt->get_offsets().size());
cout << "line " << __LINE__ << endl;
    EXPECT_EQ(make_dtype<int64_t>(), tdt->get_fields()[0]);
cout << "line " << __LINE__ << endl;
    EXPECT_EQ(make_dtype<int32_t>(), tdt->get_fields()[1]);
cout << "line " << __LINE__ << endl;
    EXPECT_EQ(make_fixedstring_dtype(string_encoding_utf_8, 5), tdt->get_fields()[2]);
cout << "line " << __LINE__ << endl;
    EXPECT_EQ(0u, tdt->get_offsets()[0]);
cout << "line " << __LINE__ << endl;
    EXPECT_EQ(8u, tdt->get_offsets()[1]);
cout << "line " << __LINE__ << endl;
    EXPECT_EQ(12u, tdt->get_offsets()[2]);
cout << "line " << __LINE__ << endl;

}
