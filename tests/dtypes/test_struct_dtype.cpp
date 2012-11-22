//
// Copyright (C) 2011-12, Dynamic NDArray Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <iostream>
#include <sstream>
#include <stdexcept>
#include "inc_gtest.hpp"

#include <dynd/ndarray.hpp>
#include <dynd/dtypes/struct_dtype.hpp>
#include <dynd/dtypes/fixedstring_dtype.hpp>

using namespace std;
using namespace dynd;

TEST(StructDType, CreateOneField) {
    dtype dt;
    const struct_dtype *tdt;

    // Struct with one field
    dt = make_struct_dtype(make_dtype<int32_t>(), "x");
    EXPECT_EQ(struct_type_id, dt.type_id());
    EXPECT_EQ(0u, dt.element_size()); // No size
    EXPECT_EQ(4u, dt.extended()->get_default_element_size(0, NULL));
    EXPECT_EQ(4u, dt.alignment());
    EXPECT_EQ(pod_memory_management, dt.get_memory_management());
    tdt = static_cast<const struct_dtype *>(dt.extended());
    EXPECT_EQ(1u, tdt->get_fields().size());
    EXPECT_EQ(make_dtype<int32_t>(), tdt->get_fields()[0]);
    EXPECT_EQ(1u, tdt->get_field_names().size());
    EXPECT_EQ("x", tdt->get_field_names()[0]);
}

TEST(StructDType, CreateTwoField) {
    dtype dt;
    const struct_dtype *tdt;

    // Struct with two fields
    dt = make_struct_dtype(make_dtype<int64_t>(), "a", make_dtype<int32_t>(), "b");
    EXPECT_EQ(struct_type_id, dt.type_id());
    EXPECT_EQ(0u, dt.element_size());
    EXPECT_EQ(16u, dt.extended()->get_default_element_size(0, NULL));
    EXPECT_EQ(8u, dt.alignment());
    EXPECT_EQ(pod_memory_management, dt.get_memory_management());
    tdt = static_cast<const struct_dtype *>(dt.extended());
    EXPECT_EQ(2u, tdt->get_fields().size());
    EXPECT_EQ(2u, tdt->get_field_names().size());
    EXPECT_EQ(make_dtype<int64_t>(), tdt->get_fields()[0]);
    EXPECT_EQ(make_dtype<int32_t>(), tdt->get_fields()[1]);
    EXPECT_EQ("a", tdt->get_field_names()[0]);
    EXPECT_EQ("b", tdt->get_field_names()[1]);
}

TEST(StructDType, CreateThreeField) {
    dtype dt;
    const struct_dtype *tdt;

    // Struct with three fields
    dtype d1 = make_dtype<int64_t>();
    dtype d2 = make_dtype<int32_t>();
    dtype d3 = make_fixedstring_dtype(string_encoding_utf_8, 5);
    dt = make_struct_dtype(d1, "x", d2, "y", d3, "z");
    EXPECT_EQ(struct_type_id, dt.type_id());
    EXPECT_EQ(0u, dt.element_size());
    EXPECT_EQ(24u, dt.extended()->get_default_element_size(0, NULL));
    EXPECT_EQ(8u, dt.alignment());
    EXPECT_EQ(pod_memory_management, dt.get_memory_management());
    tdt = static_cast<const struct_dtype *>(dt.extended());
    EXPECT_EQ(3u, tdt->get_fields().size());
    EXPECT_EQ(3u, tdt->get_field_names().size());
    EXPECT_EQ(make_dtype<int64_t>(), tdt->get_fields()[0]);
    EXPECT_EQ(make_dtype<int32_t>(), tdt->get_fields()[1]);
    EXPECT_EQ(make_fixedstring_dtype(string_encoding_utf_8, 5), tdt->get_fields()[2]);
    EXPECT_EQ("x", tdt->get_field_names()[0]);
    EXPECT_EQ("y", tdt->get_field_names()[1]);
    EXPECT_EQ("z", tdt->get_field_names()[2]);
}

TEST(StructDType, ReplaceScalarTypes) {
    dtype dt, dt2;
    const struct_dtype *tdt;

    // Struct with three fields
    dtype d1 = make_dtype<std::complex<double> >();
    dtype d2 = make_dtype<int32_t>();
    dtype d3 = make_fixedstring_dtype(string_encoding_utf_8, 5);
    dt = make_struct_dtype(d1, "x", d2, "y", d3, "z");
    dt2 = dt.extended()->with_replaced_scalar_types(make_dtype<int16_t>());
    EXPECT_EQ(make_struct_dtype(make_dtype<int16_t>(), "x", make_dtype<int16_t>(), "y", make_dtype<int16_t>(), "z"), dt2);
}