//
// Copyright (C) 2011-13, DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <iostream>
#include <sstream>
#include <stdexcept>
#include "inc_gtest.hpp"

#include <dynd/ndobject.hpp>
#include <dynd/dtypes/dtype_dtype.hpp>
#include <dynd/dtypes/strided_dim_dtype.hpp>
#include <dynd/dtypes/fixed_dim_dtype.hpp>
#include <dynd/dtypes/var_dim_dtype.hpp>

using namespace std;
using namespace dynd;

TEST(DTypeDType, Create) {
    dtype d;

    // Strings with various encodings and sizes
    d = make_dtype_dtype();
    EXPECT_EQ(dtype_type_id, d.get_type_id());
    EXPECT_EQ(custom_kind, d.get_kind());
    EXPECT_EQ(dtype("dtype"), d);
    EXPECT_EQ(sizeof(const base_dtype *), d.get_alignment());
    EXPECT_EQ(sizeof(const base_dtype *), d.get_data_size());
    EXPECT_FALSE(d.is_expression());
}

TEST(DTypeDType, BasicNDobject) {
    ndobject a;

    a = dtype("int32");
    EXPECT_EQ(dtype_type_id, a.get_dtype().get_type_id());
    EXPECT_EQ(make_dtype<int32_t>(), a.as<dtype>());
}

TEST(DTypeDType, StringCasting) {
    ndobject a;

    a = ndobject("int32").ucast(make_dtype_dtype());
    a = a.eval();
    EXPECT_EQ(dtype_type_id, a.get_dtype().get_type_id());
    EXPECT_EQ(make_dtype<int32_t>(), a.as<dtype>());
    EXPECT_EQ("int32", a.as<string>());
}

TEST(DTypeDType, ScalarRefCount) {
    ndobject a;
    dtype d, d2;
    d = make_strided_dim_dtype(make_dtype<int>());

    a = empty(make_dtype_dtype());
    EXPECT_EQ(1, d.extended()->get_use_count());
    a.vals() = d;
    EXPECT_EQ(2, d.extended()->get_use_count());
    d2 = a.as<dtype>();
    EXPECT_EQ(3, d.extended()->get_use_count());
    d2 = dtype();
    EXPECT_EQ(2, d.extended()->get_use_count());
    // Assigning a new value in the ndobject should free the reference in 'a'
    a.vals() = dtype();
    EXPECT_EQ(1, d.extended()->get_use_count());
    a.vals() = d;
    EXPECT_EQ(2, d.extended()->get_use_count());
    // Assigning a new reference to 'a' should free the reference when
    // destructing the existing 'a'
    a = 1.0;
    EXPECT_EQ(1, d.extended()->get_use_count());
}

TEST(DTypeDType, StridedArrayRefCount) {
    ndobject a;
    dtype d;
    d = make_strided_dim_dtype(make_dtype<int>());

    // 1D Strided Array
    a = empty(10, make_strided_dim_dtype(make_dtype_dtype()));
    EXPECT_EQ(1, d.extended()->get_use_count());
    a.vals() = d;
    EXPECT_EQ(11, d.extended()->get_use_count());
    // Assigning one value should free one reference count
    a.vals_at(0) = make_dtype<float>();
    EXPECT_EQ(10, d.extended()->get_use_count());
    // Assigning all values should free all the reference counts
    a.vals() = make_dtype<int>();
    EXPECT_EQ(1, d.extended()->get_use_count());
    a.vals() = d;
    EXPECT_EQ(11, d.extended()->get_use_count());
    // Assigning a new reference to 'a' should free the references when
    // destructing the existing 'a'
    a = 1.0;
    EXPECT_EQ(1, d.extended()->get_use_count());

    // 2D Strided Array
    a = empty(3, 3, dtype("M, N, dtype"));
    EXPECT_EQ(strided_dim_type_id, a.get_dtype().get_type_id());
    EXPECT_EQ(1, d.extended()->get_use_count());
    a.vals() = d;
    EXPECT_EQ(10, d.extended()->get_use_count());
    // Assigning one value should free one reference count
    a.vals_at(0,1) = make_dtype<float>();
    EXPECT_EQ(9, d.extended()->get_use_count());
    // Assigning all values should free all the reference counts
    a.vals() = make_dtype<int>();
    EXPECT_EQ(1, d.extended()->get_use_count());
    a.vals() = d;
    EXPECT_EQ(10, d.extended()->get_use_count());
    // Assigning one slice should free several reference counts
    a.vals_at(1) = make_dtype<double>();
    EXPECT_EQ(7, d.extended()->get_use_count());
    // Assigning a new reference to 'a' should free the references when
    // destructing the existing 'a'
    a = 1.0;
    EXPECT_EQ(1, d.extended()->get_use_count());
}


TEST(DTypeDType, FixedArrayRefCount) {
    ndobject a;
    dtype d;
    d = make_strided_dim_dtype(make_dtype<int>());

    // 1D Fixed Array
    a = empty(make_fixed_dim_dtype(10, make_dtype_dtype()));
    EXPECT_EQ(1, d.extended()->get_use_count());
    a.vals() = d;
    EXPECT_EQ(11, d.extended()->get_use_count());
    // Assigning one value should free one reference count
    a.vals_at(0) = make_dtype<float>();
    EXPECT_EQ(10, d.extended()->get_use_count());
    // Assigning all values should free all the reference counts
    a.vals() = make_dtype<int>();
    EXPECT_EQ(1, d.extended()->get_use_count());
    a.vals() = d;
    EXPECT_EQ(11, d.extended()->get_use_count());
    // Assigning a new reference to 'a' should free the references when
    // destructing the existing 'a'
    a = 1.0;
    EXPECT_EQ(1, d.extended()->get_use_count());

    // 2D Fixed Array
    a = empty(dtype("3, 3, dtype"));
    EXPECT_EQ(fixed_dim_type_id, a.get_dtype().get_type_id());
    EXPECT_EQ(1, d.extended()->get_use_count());
    a.vals() = d;
    EXPECT_EQ(10, d.extended()->get_use_count());
    // Assigning one value should free one reference count
    a.vals_at(0,1) = make_dtype<float>();
    EXPECT_EQ(9, d.extended()->get_use_count());
    // Assigning all values should free all the reference counts
    a.vals() = make_dtype<int>();
    EXPECT_EQ(1, d.extended()->get_use_count());
    a.vals() = d;
    EXPECT_EQ(10, d.extended()->get_use_count());
    // Assigning one slice should free several reference counts
    a.vals_at(1) = make_dtype<double>();
    EXPECT_EQ(7, d.extended()->get_use_count());
    // Assigning a new reference to 'a' should free the references when
    // destructing the existing 'a'
    a = 1.0;
    EXPECT_EQ(1, d.extended()->get_use_count());
}

TEST(DTypeDType, VarArrayRefCount) {
    ndobject a;
    dtype d;
    d = make_strided_dim_dtype(make_dtype<int>());

    // 1D Var Array
    a = empty(make_var_dim_dtype(make_dtype_dtype()));
    // It should have an objectarray memory block type
    EXPECT_EQ((uint32_t)objectarray_memory_block_type,
                    reinterpret_cast<const var_dim_dtype_metadata *>(
                        a.get_ndo_meta())->blockref->m_type);
    a.vals() = empty("10, dtype");
    EXPECT_EQ(1, d.extended()->get_use_count());
    a.vals() = d;
    EXPECT_EQ(11, d.extended()->get_use_count());
    // Assigning one value should free one reference count
    a.at(0).vals() = make_dtype<float>();
    EXPECT_EQ(10, d.extended()->get_use_count());
    // Assigning all values should free all the reference counts
    a.vals() = make_dtype<int>();
    EXPECT_EQ(1, d.extended()->get_use_count());
    a.vals() = d;
    EXPECT_EQ(11, d.extended()->get_use_count());
    // Assigning a new reference to 'a' should free the references when
    // destructing the existing 'a'
    a = 1.0;
    EXPECT_EQ(1, d.extended()->get_use_count());

    // 2D Strided + Var Array
    a = empty(3, make_strided_dim_dtype(make_var_dim_dtype(make_dtype_dtype())));
    a.vals_at(0) = empty("2, dtype");
    a.vals_at(1) = empty("3, dtype");
    a.vals_at(2) = empty("4, dtype");
    EXPECT_EQ(1, d.extended()->get_use_count());
    a.vals() = d;
    EXPECT_EQ(10, d.extended()->get_use_count());
    // Assigning one value should free one reference count
    a.at(0,1).vals() = make_dtype<float>();
    EXPECT_EQ(9, d.extended()->get_use_count());
    // Assigning all values should free all the reference counts
    a.vals() = make_dtype<int>();
    EXPECT_EQ(1, d.extended()->get_use_count());
    a.vals() = d;
    EXPECT_EQ(10, d.extended()->get_use_count());
    // Assigning one slice should free several reference counts
    a.vals_at(2) = make_dtype<double>();
    EXPECT_EQ(6, d.extended()->get_use_count());
    // Assigning a new reference to 'a' should free the references when
    // destructing the existing 'a'
    a = 1.0;
    EXPECT_EQ(1, d.extended()->get_use_count());
}

TEST(DTypeDType, CStructRefCount) {
    ndobject a;
    dtype d;
    d = make_strided_dim_dtype(make_dtype<int>());

    // Single CStruct Instance
    a = empty("{dt: dtype; more: {a: int32; b: dtype}; other: string}");
    EXPECT_EQ(cstruct_type_id, a.get_dtype().get_type_id());
    EXPECT_EQ(1, d.extended()->get_use_count());
    a.p("dt").vals() = d;
    EXPECT_EQ(2, d.extended()->get_use_count());
    a.p("more").p("b").vals() = d;
    EXPECT_EQ(3, d.extended()->get_use_count());
    // Assigning one value should free one reference count
    a.vals_at(0) = dtype();
    EXPECT_EQ(2, d.extended()->get_use_count());
    a.vals_at(1,1) = make_dtype<int>();
    EXPECT_EQ(1, d.extended()->get_use_count());
    a.vals_at(0) = d;
    EXPECT_EQ(2, d.extended()->get_use_count());
    a.vals_at(1,1) = d;
    EXPECT_EQ(3, d.extended()->get_use_count());
    // Assigning a new reference to 'a' should free the references when
    // destructing the existing 'a'
    a = 1.0;
    EXPECT_EQ(1, d.extended()->get_use_count());

    // Array of CStruct Instance
    a = empty(10, "M, {dt: dtype; more: {a: int32; b: dtype}; other: string}");
    EXPECT_EQ(cstruct_type_id, a.at(0).get_dtype().get_type_id());
    EXPECT_EQ(1, d.extended()->get_use_count());
    a.p("dt").vals() = d;
    EXPECT_EQ(11, d.extended()->get_use_count());
    a.p("more").p("b").vals() = d;
    EXPECT_EQ(21, d.extended()->get_use_count());
    // Assigning one value should free one reference count
    a.vals_at(0,0) = dtype();
    EXPECT_EQ(20, d.extended()->get_use_count());
    a.vals_at(-1,1,1) = make_dtype<int>();
    EXPECT_EQ(19, d.extended()->get_use_count());
    // Assigning one slice should free several reference counts
    a.at(3 <= irange() < 6).p("dt").vals() = dtype();
    EXPECT_EQ(16, d.extended()->get_use_count());
    // Assigning a new reference to 'a' should free the references when
    // destructing the existing 'a'
    a = 1.0;
    EXPECT_EQ(1, d.extended()->get_use_count());
}


TEST(DTypeDType, StructRefCount) {
    ndobject a;
    dtype d;
    d = make_strided_dim_dtype(make_dtype<int>());

    // Single CStruct Instance
    a = empty("{dt: dtype; more: {a: int32; b: dtype}; other: string}").at(0 <= irange() < 2);
    EXPECT_EQ(struct_type_id, a.get_dtype().get_type_id());
    EXPECT_EQ(1, d.extended()->get_use_count());
    a.p("dt").vals() = d;
    EXPECT_EQ(2, d.extended()->get_use_count());
    a.p("more").p("b").vals() = d;
    EXPECT_EQ(3, d.extended()->get_use_count());
    // Assigning one value should free one reference count
    a.vals_at(0) = dtype();
    EXPECT_EQ(2, d.extended()->get_use_count());
    a.vals_at(1,1) = make_dtype<int>();
    EXPECT_EQ(1, d.extended()->get_use_count());
    a.vals_at(0) = d;
    EXPECT_EQ(2, d.extended()->get_use_count());
    a.vals_at(1,1) = d;
    EXPECT_EQ(3, d.extended()->get_use_count());
    // Assigning a new reference to 'a' should free the references when
    // destructing the existing 'a'
    a = 1.0;
    EXPECT_EQ(1, d.extended()->get_use_count());

    // Array of Struct Instance
    a = empty(10, "M, {dt: dtype; more: {a: int32; b: dtype}; other: string}").at(irange(), 0 <= irange() < 2);
    EXPECT_EQ(struct_type_id, a.at(0).get_dtype().get_type_id());
    EXPECT_EQ(1, d.extended()->get_use_count());
    a.p("dt").vals() = d;
    EXPECT_EQ(11, d.extended()->get_use_count());
    a.p("more").p("b").vals() = d;
    EXPECT_EQ(21, d.extended()->get_use_count());
    // Assigning one value should free one reference count
    a.vals_at(0,0) = dtype();
    EXPECT_EQ(20, d.extended()->get_use_count());
    a.vals_at(-1,1,1) = make_dtype<int>();
    EXPECT_EQ(19, d.extended()->get_use_count());
    // Assigning one slice should free several reference counts
    a.at(3 <= irange() < 6).p("dt").vals() = dtype();
    EXPECT_EQ(16, d.extended()->get_use_count());
    // Assigning a new reference to 'a' should free the references when
    // destructing the existing 'a'
    a = 1.0;
    EXPECT_EQ(1, d.extended()->get_use_count());
}
