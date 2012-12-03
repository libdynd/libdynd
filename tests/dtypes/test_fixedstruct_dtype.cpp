//
// Copyright (C) 2011-12, Dynamic NDArray Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <iostream>
#include <sstream>
#include <stdexcept>
#include "inc_gtest.hpp"

#include <dynd/ndobject.hpp>
#include <dynd/dtypes/fixedstruct_dtype.hpp>
#include <dynd/dtypes/struct_dtype.hpp>
#include <dynd/dtypes/fixedstring_dtype.hpp>
#include <dynd/dtypes/string_dtype.hpp>
#include <dynd/dtypes/convert_dtype.hpp>
#include <dynd/dtypes/byteswap_dtype.hpp>

using namespace std;
using namespace dynd;

TEST(FixedStructDType, CreateOneField) {
    dtype dt;
    const fixedstruct_dtype *tdt;

    // Struct with one field
    dt = make_fixedstruct_dtype(make_dtype<int32_t>(), "x");
    EXPECT_EQ(fixedstruct_type_id, dt.get_type_id());
    EXPECT_EQ(4u, dt.get_element_size());
    EXPECT_EQ(4u, dt.extended()->get_default_element_size(0, NULL));
    EXPECT_EQ(4u, dt.get_alignment());
    EXPECT_EQ(pod_memory_management, dt.get_memory_management());
    tdt = static_cast<const fixedstruct_dtype *>(dt.extended());
    EXPECT_EQ(1u, tdt->get_field_types().size());
    EXPECT_EQ(make_dtype<int32_t>(), tdt->get_field_types()[0]);
    EXPECT_EQ(1u, tdt->get_data_offsets().size());
    EXPECT_EQ(0u, tdt->get_data_offsets()[0]);
    EXPECT_EQ(1u, tdt->get_field_names().size());
    EXPECT_EQ("x", tdt->get_field_names()[0]);
}

TEST(FixedStructDType, CreateTwoField) {
    dtype dt;
    const fixedstruct_dtype *tdt;

    // Struct with two fields
    dt = make_fixedstruct_dtype(make_dtype<int64_t>(), "a", make_dtype<int32_t>(), "b");
    EXPECT_EQ(fixedstruct_type_id, dt.get_type_id());
    EXPECT_EQ(16u, dt.get_element_size());
    EXPECT_EQ(16u, dt.extended()->get_default_element_size(0, NULL));
    EXPECT_EQ(8u, dt.get_alignment());
    EXPECT_EQ(pod_memory_management, dt.get_memory_management());
    tdt = static_cast<const fixedstruct_dtype *>(dt.extended());
    EXPECT_EQ(2u, tdt->get_field_types().size());
    EXPECT_EQ(make_dtype<int64_t>(), tdt->get_field_types()[0]);
    EXPECT_EQ(make_dtype<int32_t>(), tdt->get_field_types()[1]);
    EXPECT_EQ(2u, tdt->get_data_offsets().size());
    EXPECT_EQ(0u, tdt->get_data_offsets()[0]);
    EXPECT_EQ(8u, tdt->get_data_offsets()[1]);
    EXPECT_EQ(2u, tdt->get_field_names().size());
    EXPECT_EQ("a", tdt->get_field_names()[0]);
    EXPECT_EQ("b", tdt->get_field_names()[1]);
}

TEST(FixedStructDType, CreateThreeField) {
    dtype dt;
    const fixedstruct_dtype *tdt;

    // Struct with three fields
    dtype d1 = make_dtype<int64_t>();
    dtype d2 = make_dtype<int32_t>();
    dtype d3 = make_fixedstring_dtype(string_encoding_utf_8, 5);
    dt = make_fixedstruct_dtype(d1, "x", d2, "y", d3, "z");
    EXPECT_EQ(fixedstruct_type_id, dt.get_type_id());
    EXPECT_EQ(24u, dt.get_element_size());
    EXPECT_EQ(24u, dt.extended()->get_default_element_size(0, NULL));
    EXPECT_EQ(8u, dt.get_alignment());
    EXPECT_EQ(pod_memory_management, dt.get_memory_management());
    tdt = static_cast<const fixedstruct_dtype *>(dt.extended());
    EXPECT_EQ(3u, tdt->get_field_types().size());
    EXPECT_EQ(make_dtype<int64_t>(), tdt->get_field_types()[0]);
    EXPECT_EQ(make_dtype<int32_t>(), tdt->get_field_types()[1]);
    EXPECT_EQ(make_fixedstring_dtype(string_encoding_utf_8, 5), tdt->get_field_types()[2]);
    EXPECT_EQ(3u, tdt->get_data_offsets().size());
    EXPECT_EQ(0u, tdt->get_data_offsets()[0]);
    EXPECT_EQ(8u, tdt->get_data_offsets()[1]);
    EXPECT_EQ(12u, tdt->get_data_offsets()[2]);
    EXPECT_EQ(3u, tdt->get_field_names().size());
    EXPECT_EQ("x", tdt->get_field_names()[0]);
    EXPECT_EQ("y", tdt->get_field_names()[1]);
    EXPECT_EQ("z", tdt->get_field_names()[2]);
}

TEST(FixedStructDType, ReplaceScalarTypes) {
    dtype dt, dt2;

    // Struct with three fields
    dtype d1 = make_dtype<std::complex<double> >();
    dtype d2 = make_dtype<int32_t>();
    dtype d3 = make_fixedstring_dtype(string_encoding_utf_8, 5);
    dt = make_fixedstruct_dtype(d1, "x", d2, "y", d3, "z");
    dt2 = dt.with_replaced_scalar_types(make_dtype<int16_t>());
    EXPECT_EQ(make_fixedstruct_dtype(
                make_convert_dtype(make_dtype<int16_t>(), d1), "x",
                make_convert_dtype(make_dtype<int16_t>(), d2), "y",
                make_convert_dtype(make_dtype<int16_t>(), d3), "z"),
        dt2);
}

TEST(FixedStructDType, DTypeAt) {
    dtype dt, dt2;

    // Struct with three fields
    dtype d1 = make_dtype<std::complex<double> >();
    dtype d2 = make_dtype<int32_t>();
    dtype d3 = make_fixedstring_dtype(string_encoding_utf_8, 5);
    dt = make_fixedstruct_dtype(d1, "x", d2, "y", d3, "z");

    // indexing into a dtype with a slice produces a
    // struct dtype (not fixedstruct dtype) with the subset of fields.
    EXPECT_EQ(make_struct_dtype(d1, "x", d2, "y"), dt.at(irange() < 2));
    EXPECT_EQ(make_struct_dtype(d1, "x", d3, "z"), dt.at(irange(0, 3, 2)));
    EXPECT_EQ(make_struct_dtype(d3, "z", d2, "y"), dt.at(irange(2, 0, -1)));
}

TEST(FixedStructDType, CanonicalDType) {
    dtype dt, dt2;

    // Struct with three fields
    dtype d1 = make_convert_dtype<std::complex<double>, float>();
    dtype d2 = make_byteswap_dtype<int32_t>();
    dtype d3 = make_fixedstring_dtype(string_encoding_utf_32, 5);
    dt = make_fixedstruct_dtype(d1, "x", d2, "y", d3, "z");
    EXPECT_EQ(make_fixedstruct_dtype(make_dtype<std::complex<double> >(), "x",
                                make_dtype<int32_t>(), "y",
                                d3, "z"),
            dt.get_canonical_dtype());
}

TEST(FixedStructDType, IsExpression) {
    dtype d1 = make_dtype<float>();
    dtype d2 = make_byteswap_dtype<int32_t>();
    dtype d3 = make_fixedstring_dtype(string_encoding_utf_32, 5);
    dtype d = make_fixedstruct_dtype(d1, "x", d2, "y", d3, "z");

    EXPECT_TRUE(d.is_expression());
    EXPECT_FALSE(d.at(irange(0, 3, 2)).is_expression());
}
