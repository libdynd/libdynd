//
// Copyright (C) 2011-13, DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <iostream>
#include <sstream>
#include <stdexcept>
#include "inc_gtest.hpp"

#include <dynd/ndobject.hpp>
#include <dynd/dtypes/cstruct_dtype.hpp>
#include <dynd/dtypes/struct_dtype.hpp>
#include <dynd/dtypes/fixedstring_dtype.hpp>
#include <dynd/dtypes/string_dtype.hpp>
#include <dynd/dtypes/convert_dtype.hpp>
#include <dynd/dtypes/byteswap_dtype.hpp>

using namespace std;
using namespace dynd;

TEST(CStructDType, Basic) {
    EXPECT_NE(make_cstruct_dtype(make_dtype<int>(), "x"),
                    make_cstruct_dtype(make_dtype<int>(), "y"));
    EXPECT_NE(make_cstruct_dtype(make_dtype<float>(), "x"),
                    make_cstruct_dtype(make_dtype<int>(), "x"));
    EXPECT_NE(dtype("{x: int32}"), dtype("{y: int32}"));
    EXPECT_NE(dtype("{x: float32}"), dtype("{x: int32}"));
}

TEST(CStructDType, CreateOneField) {
    dtype dt;
    const cstruct_dtype *tdt;

    // Struct with one field
    dt = make_cstruct_dtype(make_dtype<int32_t>(), "x");
    EXPECT_EQ(cstruct_type_id, dt.get_type_id());
    EXPECT_EQ(4u, dt.get_data_size());
    EXPECT_EQ(4u, dt.extended()->get_default_data_size(0, NULL));
    EXPECT_EQ(4u, dt.get_alignment());
    EXPECT_TRUE(dt.is_pod());
    tdt = static_cast<const cstruct_dtype *>(dt.extended());
    EXPECT_EQ(1u, tdt->get_field_count());
    EXPECT_EQ(make_dtype<int32_t>(), tdt->get_field_types()[0]);
    EXPECT_EQ(0u, tdt->get_data_offsets_vector()[0]);
    EXPECT_EQ("x", tdt->get_field_names()[0]);
}

TEST(CStructDType, CreateTwoField) {
    dtype dt;
    const cstruct_dtype *tdt;

    // Struct with two fields
    dt = make_cstruct_dtype(make_dtype<int64_t>(), "a", make_dtype<int32_t>(), "b");
    EXPECT_EQ(cstruct_type_id, dt.get_type_id());
    EXPECT_EQ(16u, dt.get_data_size());
    EXPECT_EQ(16u, dt.extended()->get_default_data_size(0, NULL));
    EXPECT_EQ(8u, dt.get_alignment());
    EXPECT_TRUE(dt.is_pod());
    tdt = static_cast<const cstruct_dtype *>(dt.extended());
    EXPECT_EQ(2u, tdt->get_field_count());
    EXPECT_EQ(make_dtype<int64_t>(), tdt->get_field_types()[0]);
    EXPECT_EQ(make_dtype<int32_t>(), tdt->get_field_types()[1]);
    EXPECT_EQ(0u, tdt->get_data_offsets_vector()[0]);
    EXPECT_EQ(8u, tdt->get_data_offsets_vector()[1]);
    EXPECT_EQ("a", tdt->get_field_names()[0]);
    EXPECT_EQ("b", tdt->get_field_names()[1]);
}

TEST(CStructDType, CreateThreeField) {
    dtype dt;
    const cstruct_dtype *tdt;

    // Struct with three fields
    dtype d1 = make_dtype<int64_t>();
    dtype d2 = make_dtype<int32_t>();
    dtype d3 = make_fixedstring_dtype(5, string_encoding_utf_8);
    dt = make_cstruct_dtype(d1, "x", d2, "y", d3, "z");
    EXPECT_EQ(cstruct_type_id, dt.get_type_id());
    EXPECT_EQ(24u, dt.get_data_size());
    EXPECT_EQ(24u, dt.extended()->get_default_data_size(0, NULL));
    EXPECT_EQ(8u, dt.get_alignment());
    EXPECT_TRUE(dt.is_pod());
    tdt = static_cast<const cstruct_dtype *>(dt.extended());
    EXPECT_EQ(3u, tdt->get_field_count());
    EXPECT_EQ(make_dtype<int64_t>(), tdt->get_field_types()[0]);
    EXPECT_EQ(make_dtype<int32_t>(), tdt->get_field_types()[1]);
    EXPECT_EQ(make_fixedstring_dtype(5, string_encoding_utf_8),
                    tdt->get_field_types()[2]);
    EXPECT_EQ(0u, tdt->get_data_offsets_vector()[0]);
    EXPECT_EQ(8u, tdt->get_data_offsets_vector()[1]);
    EXPECT_EQ(12u, tdt->get_data_offsets_vector()[2]);
    EXPECT_EQ("x", tdt->get_field_names()[0]);
    EXPECT_EQ("y", tdt->get_field_names()[1]);
    EXPECT_EQ("z", tdt->get_field_names()[2]);
}

TEST(CStructDType, ReplaceScalarTypes) {
    dtype dt, dt2;

    // Struct with three fields
    dtype d1 = make_dtype<std::complex<double> >();
    dtype d2 = make_dtype<int32_t>();
    dtype d3 = make_fixedstring_dtype(5, string_encoding_utf_8);
    dt = make_cstruct_dtype(d1, "x", d2, "y", d3, "z");
    dt2 = dt.with_replaced_scalar_types(make_dtype<int16_t>());
    EXPECT_EQ(make_cstruct_dtype(
                make_convert_dtype(make_dtype<int16_t>(), d1), "x",
                make_convert_dtype(make_dtype<int16_t>(), d2), "y",
                make_convert_dtype(make_dtype<int16_t>(), d3), "z"),
        dt2);
}

TEST(CStructDType, DTypeAt) {
    dtype dt, dt2;

    // Struct with three fields
    dtype d1 = make_dtype<std::complex<double> >();
    dtype d2 = make_dtype<int32_t>();
    dtype d3 = make_fixedstring_dtype(5, string_encoding_utf_8);
    dt = make_cstruct_dtype(d1, "x", d2, "y", d3, "z");

    // indexing into a dtype with a slice produces a
    // struct dtype (not cstruct dtype) with the subset of fields.
    EXPECT_EQ(make_struct_dtype(d1, "x", d2, "y"), dt.at(irange() < 2));
    EXPECT_EQ(make_struct_dtype(d1, "x", d3, "z"), dt.at(irange(0, 3, 2)));
    EXPECT_EQ(make_struct_dtype(d3, "z", d2, "y"), dt.at(irange(2, 0, -1)));
}

TEST(CStructDType, CanonicalDType) {
    dtype dt, dt2;

    // Struct with three fields
    dtype d1 = make_convert_dtype<std::complex<double>, float>();
    dtype d2 = make_byteswap_dtype<int32_t>();
    dtype d3 = make_fixedstring_dtype(5, string_encoding_utf_32);
    dt = make_cstruct_dtype(d1, "x", d2, "y", d3, "z");
    EXPECT_EQ(make_cstruct_dtype(make_dtype<std::complex<double> >(), "x",
                                make_dtype<int32_t>(), "y",
                                d3, "z"),
            dt.get_canonical_dtype());
}

TEST(CStructDType, IsExpression) {
    dtype d1 = make_dtype<float>();
    dtype d2 = make_byteswap_dtype<int32_t>();
    dtype d3 = make_fixedstring_dtype(5, string_encoding_utf_32);
    dtype d = make_cstruct_dtype(d1, "x", d2, "y", d3, "z");

    EXPECT_TRUE(d.is_expression());
    EXPECT_FALSE(d.at(irange(0, 3, 2)).is_expression());
}

TEST(CStructDType, PropertyAccess) {
    dtype dt = make_cstruct_dtype(make_dtype<int>(), "x", make_dtype<double>(), "y", make_dtype<short>(), "z");
    ndobject a = empty(dt);
    a.at(0).vals() = 3;
    a.at(1).vals() = 4.25;
    a.at(2).vals() = 5;
    EXPECT_EQ(3, a.p("x").as<int>());
    EXPECT_EQ(4.25, a.p("y").as<double>());
    EXPECT_EQ(5, a.p("z").as<short>());
    EXPECT_THROW(a.p("w"), runtime_error);
}

TEST(CStructDType, EqualDTypeAssign) {
    dtype dt = make_cstruct_dtype(make_dtype<int>(), "x", make_dtype<double>(), "y", make_dtype<short>(), "z");
    ndobject a = make_strided_ndobject(2, dt);
    a.at(0,0).vals() = 3;
    a.at(0,1).vals() = 4.25;
    a.at(0,2).vals() = 5;
    a.at(1,0).vals() = 6;
    a.at(1,1).vals() = 7.25;
    a.at(1,2).vals() = 8;

    ndobject b = make_strided_ndobject(2, dt);
    b.val_assign(a);
    EXPECT_EQ(3,    a.at(0,0).as<int>());
    EXPECT_EQ(4.25, a.at(0,1).as<double>());
    EXPECT_EQ(5,    a.at(0,2).as<short>());
    EXPECT_EQ(6,    a.at(1,0).as<int>());
    EXPECT_EQ(7.25, a.at(1,1).as<double>());
    EXPECT_EQ(8,    a.at(1,2).as<short>());
}

TEST(CStructDType, DifferentDTypeAssign) {
    dtype dt = make_cstruct_dtype(make_dtype<int>(), "x", make_dtype<double>(), "y", make_dtype<short>(), "z");
    ndobject a = make_strided_ndobject(2, dt);
    a.at(0,0).vals() = 3;
    a.at(0,1).vals() = 4.25;
    a.at(0,2).vals() = 5;
    a.at(1,0).vals() = 6;
    a.at(1,1).vals() = 7.25;
    a.at(1,2).vals() = 8;

    dtype dt2 = make_cstruct_dtype(make_dtype<float>(), "y", make_dtype<int>(), "z", make_dtype<uint8_t>(), "x");
    ndobject b = make_strided_ndobject(2, dt2);
    b.val_assign(a);
    EXPECT_EQ(3,    b.at(0,2).as<int>());
    EXPECT_EQ(4.25, b.at(0,0).as<double>());
    EXPECT_EQ(5,    b.at(0,1).as<short>());
    EXPECT_EQ(6,    b.at(1,2).as<int>());
    EXPECT_EQ(7.25, b.at(1,0).as<double>());
    EXPECT_EQ(8,    b.at(1,1).as<short>());
}

TEST(CStructDType, FromStructAssign) {
    dtype dt = make_struct_dtype(make_dtype<int>(), "x", make_dtype<double>(), "y", make_dtype<short>(), "z");
    ndobject a = make_strided_ndobject(2, dt);
    a.at(0,0).vals() = 3;
    a.at(0,1).vals() = 4.25;
    a.at(0,2).vals() = 5;
    a.at(1,0).vals() = 6;
    a.at(1,1).vals() = 7.25;
    a.at(1,2).vals() = 8;

    dtype dt2 = make_cstruct_dtype(make_dtype<float>(), "y", make_dtype<int>(), "z", make_dtype<uint8_t>(), "x");
    ndobject b = make_strided_ndobject(2, dt2);
    b.val_assign(a);
    EXPECT_EQ(3,    b.at(0,2).as<int>());
    EXPECT_EQ(4.25, b.at(0,0).as<double>());
    EXPECT_EQ(5,    b.at(0,1).as<short>());
    EXPECT_EQ(6,    b.at(1,2).as<int>());
    EXPECT_EQ(7.25, b.at(1,0).as<double>());
    EXPECT_EQ(8,    b.at(1,1).as<short>());
}

TEST(CStructDType, SingleCompare) {
    ndobject a, b;
    dtype sdt = make_cstruct_dtype(make_dtype<int32_t>(), "a",
                    make_dtype<float>(), "b", make_dtype<int64_t>(), "c");
    a = empty(sdt);
    b = empty(sdt);

    // Test lexicographic sorting

    // a == b
    a.p("a").vals() = 3;
    a.p("b").vals() = -2.25;
    a.p("c").vals() = 66;
    b.p("a").vals() = 3;
    b.p("b").vals() = -2.25;
    b.p("c").vals() = 66;
    EXPECT_FALSE(a.op_sorting_less(b));
    EXPECT_THROW((a < b), not_comparable_error);
    EXPECT_THROW((a <= b), not_comparable_error);
    EXPECT_TRUE(a == b);
    EXPECT_FALSE(a != b);
    EXPECT_THROW((a >= b), not_comparable_error);
    EXPECT_THROW((a > b), not_comparable_error);
    EXPECT_FALSE(b.op_sorting_less(a));
    EXPECT_THROW((b < a), not_comparable_error);
    EXPECT_THROW((b <= a), not_comparable_error);
    EXPECT_TRUE(b == a);
    EXPECT_FALSE(b != a);
    EXPECT_THROW((b >= a), not_comparable_error);
    EXPECT_THROW((b > a), not_comparable_error);

    // Different in the first field
    a.p("a").vals() = 3;
    a.p("b").vals() = -2.25;
    a.p("c").vals() = 66;
    b.p("a").vals() = 4;
    b.p("b").vals() = -2.25;
    b.p("c").vals() = 66;
    EXPECT_TRUE(a.op_sorting_less(b));
    EXPECT_THROW((a < b), not_comparable_error);
    EXPECT_THROW((a <= b), not_comparable_error);
    EXPECT_FALSE(a == b);
    EXPECT_TRUE(a != b);
    EXPECT_THROW((a >= b), not_comparable_error);
    EXPECT_THROW((a > b), not_comparable_error);
    EXPECT_FALSE(b.op_sorting_less(a));
    EXPECT_THROW((b < a), not_comparable_error);
    EXPECT_THROW((b <= a), not_comparable_error);
    EXPECT_FALSE(b == a);
    EXPECT_TRUE(b != a);
    EXPECT_THROW((b >= a), not_comparable_error);
    EXPECT_THROW((b > a), not_comparable_error);

    // Different in the second field
    a.p("a").vals() = 3;
    a.p("b").vals() = -2.25;
    a.p("c").vals() = 66;
    b.p("a").vals() = 3;
    b.p("b").vals() = -2.23;
    b.p("c").vals() = 66;
    EXPECT_TRUE(a.op_sorting_less(b));
    EXPECT_THROW((a < b), not_comparable_error);
    EXPECT_THROW((a <= b), not_comparable_error);
    EXPECT_FALSE(a == b);
    EXPECT_TRUE(a != b);
    EXPECT_THROW((a >= b), not_comparable_error);
    EXPECT_THROW((a > b), not_comparable_error);
    EXPECT_FALSE(b.op_sorting_less(a));
    EXPECT_THROW((b < a), not_comparable_error);
    EXPECT_THROW((b <= a), not_comparable_error);
    EXPECT_FALSE(b == a);
    EXPECT_TRUE(b != a);
    EXPECT_THROW((b >= a), not_comparable_error);
    EXPECT_THROW((b > a), not_comparable_error);


    // Different in the third field
    a.p("a").vals() = 3;
    a.p("b").vals() = -2.25;
    a.p("c").vals() = 66;
    b.p("a").vals() = 3;
    b.p("b").vals() = -2.25;
    b.p("c").vals() = 1000;
    EXPECT_TRUE(a.op_sorting_less(b));
    EXPECT_THROW((a < b), not_comparable_error);
    EXPECT_THROW((a <= b), not_comparable_error);
    EXPECT_FALSE(a == b);
    EXPECT_TRUE(a != b);
    EXPECT_THROW((a >= b), not_comparable_error);
    EXPECT_THROW((a > b), not_comparable_error);
    EXPECT_FALSE(b.op_sorting_less(a));
    EXPECT_THROW((b < a), not_comparable_error);
    EXPECT_THROW((b <= a), not_comparable_error);
    EXPECT_FALSE(b == a);
    EXPECT_TRUE(b != a);
    EXPECT_THROW((b >= a), not_comparable_error);
    EXPECT_THROW((b > a), not_comparable_error);
}