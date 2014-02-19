//
// Copyright (C) 2011-14 Mark Wiebe, DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <iostream>
#include <sstream>
#include <stdexcept>
#include "inc_gtest.hpp"

#include <dynd/array.hpp>
#include <dynd/types/struct_type.hpp>
#include <dynd/types/fixedstring_type.hpp>
#include <dynd/types/string_type.hpp>
#include <dynd/types/convert_type.hpp>
#include <dynd/types/byteswap_type.hpp>
#include <dynd/types/cstruct_type.hpp>
#include <dynd/json_parser.hpp>

using namespace std;
using namespace dynd;

TEST(StructDType, Basic) {
    EXPECT_NE(ndt::make_struct(ndt::make_type<int>(), "x"),
                    ndt::make_struct(ndt::make_type<int>(), "y"));
    EXPECT_NE(ndt::make_struct(ndt::make_type<float>(), "x"),
                    ndt::make_struct(ndt::make_type<int>(), "x"));
}

TEST(StructDType, CreateOneField) {
    ndt::type dt;
    const struct_type *tdt;

    // Struct with one field
    dt = ndt::make_struct(ndt::make_type<int32_t>(), "x");
    EXPECT_EQ(struct_type_id, dt.get_type_id());
    EXPECT_EQ(0u, dt.get_data_size()); // No size
    EXPECT_EQ(4u, dt.extended()->get_default_data_size(0, NULL));
    EXPECT_EQ(4u, dt.get_data_alignment());
    EXPECT_FALSE(dt.is_pod());
    EXPECT_EQ(0u, (dt.get_flags()&(type_flag_blockref|type_flag_destructor)));
    tdt = static_cast<const struct_type *>(dt.extended());
    EXPECT_EQ(1u, tdt->get_field_count());
    EXPECT_EQ(ndt::make_type<int32_t>(), tdt->get_field_types()[0]);
    EXPECT_EQ("x", tdt->get_field_names()[0]);
}

struct two_field_struct {
    int64_t a;
    int32_t b;
};

TEST(StructDType, CreateTwoField) {
    ndt::type dt;
    const struct_type *tdt;

    // Struct with two fields
    dt = ndt::make_struct(ndt::make_type<int64_t>(), "a", ndt::make_type<int32_t>(), "b");
    EXPECT_EQ(struct_type_id, dt.get_type_id());
    EXPECT_EQ(0u, dt.get_data_size());
    EXPECT_EQ(sizeof(two_field_struct), dt.extended()->get_default_data_size(0, NULL));
    EXPECT_EQ((size_t)scalar_align_of<two_field_struct>::value, dt.get_data_alignment());
    EXPECT_FALSE(dt.is_pod());
    EXPECT_EQ(0u, (dt.get_flags()&(type_flag_blockref|type_flag_destructor)));
    tdt = static_cast<const struct_type *>(dt.extended());
    EXPECT_EQ(2u, tdt->get_field_count());
    EXPECT_EQ(ndt::make_type<int64_t>(), tdt->get_field_types()[0]);
    EXPECT_EQ(ndt::make_type<int32_t>(), tdt->get_field_types()[1]);
    EXPECT_EQ("a", tdt->get_field_names()[0]);
    EXPECT_EQ("b", tdt->get_field_names()[1]);
}

struct three_field_struct {
    int64_t x;
    int32_t y;
    char z[5];
};

TEST(StructDType, CreateThreeField) {
    ndt::type dt;
    const struct_type *tdt;

    // Struct with three fields
    ndt::type d1 = ndt::make_type<int64_t>();
    ndt::type d2 = ndt::make_type<int32_t>();
    ndt::type d3 = ndt::make_fixedstring(5, string_encoding_utf_8);
    dt = ndt::make_struct(d1, "x", d2, "y", d3, "z");
    EXPECT_EQ(struct_type_id, dt.get_type_id());
    EXPECT_EQ(0u, dt.get_data_size());
    EXPECT_EQ(sizeof(three_field_struct), dt.extended()->get_default_data_size(0, NULL));
    EXPECT_EQ((size_t)scalar_align_of<two_field_struct>::value, dt.get_data_alignment());
    EXPECT_FALSE(dt.is_pod());
    EXPECT_EQ(0u, (dt.get_flags()&(type_flag_blockref|type_flag_destructor)));
    tdt = static_cast<const struct_type *>(dt.extended());
    EXPECT_EQ(3u, tdt->get_field_count());
    EXPECT_EQ(ndt::make_type<int64_t>(), tdt->get_field_types()[0]);
    EXPECT_EQ(ndt::make_type<int32_t>(), tdt->get_field_types()[1]);
    EXPECT_EQ(ndt::make_fixedstring(5, string_encoding_utf_8), tdt->get_field_types()[2]);
    EXPECT_EQ("x", tdt->get_field_names()[0]);
    EXPECT_EQ("y", tdt->get_field_names()[1]);
    EXPECT_EQ("z", tdt->get_field_names()[2]);
}

TEST(StructDType, ReplaceScalarTypes) {
    ndt::type dt, dt2;

    // Struct with three fields
    ndt::type d1 = ndt::make_type<dynd_complex<double> >();
    ndt::type d2 = ndt::make_type<int32_t>();
    ndt::type d3 = ndt::make_fixedstring(5, string_encoding_utf_8);
    dt = ndt::make_struct(d1, "x", d2, "y", d3, "z");
    dt2 = dt.with_replaced_scalar_types(ndt::make_type<int16_t>());
    EXPECT_EQ(ndt::make_struct(
                ndt::make_convert(ndt::make_type<int16_t>(), d1), "x",
                ndt::make_convert(ndt::make_type<int16_t>(), d2), "y",
                ndt::make_convert(ndt::make_type<int16_t>(), d3), "z"),
        dt2);
}

TEST(StructDType, DTypeAt) {
    ndt::type dt, dt2;

    // Struct with three fields
    ndt::type d1 = ndt::make_type<dynd_complex<double> >();
    ndt::type d2 = ndt::make_type<int32_t>();
    ndt::type d3 = ndt::make_fixedstring(5, string_encoding_utf_8);
    dt = ndt::make_struct(d1, "x", d2, "y", d3, "z");

    // indexing into a type with a slice produces another
    // struct type with the subset of fields.
    EXPECT_EQ(ndt::make_struct(d1, "x", d2, "y"), dt.at(irange() < 2));
    EXPECT_EQ(ndt::make_struct(d1, "x", d3, "z"), dt.at(irange(0, 3, 2)));
    EXPECT_EQ(ndt::make_struct(d3, "z", d2, "y"), dt.at(irange(2, 0, -1)));
}

TEST(StructDType, CanonicalDType) {
    ndt::type dt, dt2;

    // Struct with three fields
    ndt::type d1 = ndt::make_convert<dynd_complex<double>, float>();
    ndt::type d2 = ndt::make_byteswap<int32_t>();
    ndt::type d3 = ndt::make_fixedstring(5, string_encoding_utf_32);
    dt = ndt::make_struct(d1, "x", d2, "y", d3, "z");
    EXPECT_EQ(ndt::make_struct(ndt::make_type<dynd_complex<double> >(), "x",
                                ndt::make_type<int32_t>(), "y",
                                d3, "z"),
            dt.get_canonical_type());
}

TEST(StructDType, IsExpression) {
    ndt::type d1 = ndt::make_type<float>();
    ndt::type d2 = ndt::make_byteswap<int32_t>();
    ndt::type d3 = ndt::make_fixedstring(5, string_encoding_utf_32);
    ndt::type d = ndt::make_struct(d1, "x", d2, "y", d3, "z");

    EXPECT_TRUE(d.is_expression());
    EXPECT_FALSE(d.at(irange(0, 3, 2)).is_expression());
}

TEST(StructDType, PropertyAccess) {
    ndt::type dt = ndt::make_struct(ndt::make_type<int>(), "x", ndt::make_type<double>(), "y", ndt::make_type<short>(), "z");
    nd::array a = nd::empty(dt);
    a(0).vals() = 3;
    a(1).vals() = 4.25;
    a(2).vals() = 5;
    EXPECT_EQ(3, a.p("x").as<int>());
    EXPECT_EQ(4.25, a.p("y").as<double>());
    EXPECT_EQ(5, a.p("z").as<short>());
    EXPECT_THROW(a.p("w"), runtime_error);
}

TEST(StructDType, EqualDTypeAssign) {
    ndt::type dt = ndt::make_struct(ndt::make_type<int>(), "x", ndt::make_type<double>(), "y", ndt::make_type<short>(), "z");
    nd::array a = nd::make_strided_array(2, dt);
    a(0,0).vals() = 3;
    a(0,1).vals() = 4.25;
    a(0,2).vals() = 5;
    a(1,0).vals() = 6;
    a(1,1).vals() = 7.25;
    a(1,2).vals() = 8;

    nd::array b = nd::make_strided_array(2, dt);
    b.val_assign(a);
    EXPECT_EQ(3,    a(0,0).as<int>());
    EXPECT_EQ(4.25, a(0,1).as<double>());
    EXPECT_EQ(5,    a(0,2).as<short>());
    EXPECT_EQ(6,    a(1,0).as<int>());
    EXPECT_EQ(7.25, a(1,1).as<double>());
    EXPECT_EQ(8,    a(1,2).as<short>());
}

TEST(StructDType, DifferentDTypeAssign) {
    ndt::type dt = ndt::make_struct(ndt::make_type<int>(), "x", ndt::make_type<double>(), "y", ndt::make_type<short>(), "z");
    nd::array a = nd::make_strided_array(2, dt);
    a(0,0).vals() = 3;
    a(0,1).vals() = 4.25;
    a(0,2).vals() = 5;
    a(1,0).vals() = 6;
    a(1,1).vals() = 7.25;
    a(1,2).vals() = 8;

    ndt::type dt2 = ndt::make_struct(ndt::make_type<float>(), "y", ndt::make_type<int>(), "z", ndt::make_type<uint8_t>(), "x");
    nd::array b = nd::make_strided_array(2, dt2);
    b.val_assign(a);
    EXPECT_EQ(3,    b(0,2).as<int>());
    EXPECT_EQ(4.25, b(0,0).as<double>());
    EXPECT_EQ(5,    b(0,1).as<short>());
    EXPECT_EQ(6,    b(1,2).as<int>());
    EXPECT_EQ(7.25, b(1,0).as<double>());
    EXPECT_EQ(8,    b(1,1).as<short>());
}

TEST(StructDType, FromCStructAssign) {
    ndt::type dt = ndt::make_cstruct(ndt::make_type<int>(), "x", ndt::make_type<double>(), "y", ndt::make_type<short>(), "z");
    nd::array a = nd::make_strided_array(2, dt);
    a(0,0).vals() = 3;
    a(0,1).vals() = 4.25;
    a(0,2).vals() = 5;
    a(1,0).vals() = 6;
    a(1,1).vals() = 7.25;
    a(1,2).vals() = 8;

    ndt::type dt2 = ndt::make_struct(ndt::make_type<float>(), "y", ndt::make_type<int>(), "z", ndt::make_type<uint8_t>(), "x");
    nd::array b = nd::make_strided_array(2, dt2);
    b.val_assign(a);
    EXPECT_EQ(3,    b(0,2).as<int>());
    EXPECT_EQ(4.25, b(0,0).as<double>());
    EXPECT_EQ(5,    b(0,1).as<short>());
    EXPECT_EQ(6,    b(1,2).as<int>());
    EXPECT_EQ(7.25, b(1,0).as<double>());
    EXPECT_EQ(8,    b(1,1).as<short>());
}

TEST(StructDType, SingleCompare) {
    nd::array a, b;
    ndt::type sdt = ndt::make_struct(ndt::make_type<int32_t>(), "a",
                    ndt::make_type<float>(), "b", ndt::make_type<int64_t>(), "c");
    a = nd::empty(sdt);
    b = nd::empty(sdt);

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


TEST(StructDType, SingleCompareDifferentMetadata) {
    nd::array a, b;
    ndt::type sdt = ndt::make_struct(ndt::make_type<int32_t>(), "a",
                    ndt::make_type<float>(), "b", ndt::make_type<int64_t>(), "c");
    ndt::type sdt_reverse = ndt::make_struct(ndt::make_type<int64_t>(), "c",
                    ndt::make_type<float>(), "b", ndt::make_type<int32_t>(), "a");
    a = nd::empty(sdt);
    b = nd::empty(sdt_reverse)(irange().by(-1));

    // Confirm that the metadata is different
    EXPECT_EQ(a.get_type(), b.get_type());
    const struct_type *a_sdt = static_cast<const struct_type *>(a.get_type().extended());
    const struct_type *b_sdt = static_cast<const struct_type *>(b.get_type().extended());
    EXPECT_NE(a_sdt->get_data_offsets(a.get_ndo_meta())[0],
                    b_sdt->get_data_offsets(b.get_ndo_meta())[0]);
    EXPECT_NE(a_sdt->get_data_offsets(a.get_ndo_meta())[1],
                    b_sdt->get_data_offsets(b.get_ndo_meta())[1]);
    EXPECT_NE(a_sdt->get_data_offsets(a.get_ndo_meta())[2],
                    b_sdt->get_data_offsets(b.get_ndo_meta())[2]);

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

