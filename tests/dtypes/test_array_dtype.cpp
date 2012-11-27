//
// Copyright (C) 2011-12, Dynamic NDArray Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <iostream>
#include <sstream>
#include <stdexcept>
#include "inc_gtest.hpp"

#include "dynd/dtype_assign.hpp"
#include "dynd/dtypes/tuple_dtype.hpp"
#include "dynd/dtypes/array_dtype.hpp"

using namespace std;
using namespace dynd;

TEST(ArrayDType, DTypeSubscript) {
    dtype dfloat = make_dtype<float>();
    dtype darr1 = make_array_dtype(dfloat);
    dtype darr2 = make_array_dtype(darr1);
    dtype dtest;

    // Indexing an array like this creates a result with a known array size
    dtest = darr1.at(1 <= irange() < 3);
    // TODO!
    //EXPECT_EQ(make_strided_array_dtype(), dtest);
    //dtest = darr2.index(1, ss);
    //EXPECT_EQ(darr1, dtest);

}

TEST(ArrayDType, LosslessCasting) {
/*
    intptr_t shape_235[] = {2,3,5}, shape_215[] = {2,1,5}, shape_35[] = {3,5};
    dtype adt_int_235 = make_ndobject_dtype<int>(3, shape_235);
    dtype adt_int_215 = make_ndobject_dtype<int>(3, shape_215);
    dtype adt_int_35 = make_ndobject_dtype<int>(2, shape_35);

    dtype adt_int16_215 = make_ndobject_dtype<int16_t>(3, shape_215);
    dtype adt_int64_215 = make_ndobject_dtype<int64_t>(3, shape_215);

    // Broadcasting equal types treated as lossless
    EXPECT_TRUE(is_lossless_assignment(adt_int_235, adt_int_215));
    EXPECT_TRUE(is_lossless_assignment(adt_int_235, adt_int_35));
    EXPECT_FALSE(is_lossless_assignment(adt_int_215, adt_int_235));
    EXPECT_TRUE(is_lossless_assignment(adt_int_235, make_dtype<int>()));

    // Broadcasting unequal type follows the element type's rules
    EXPECT_TRUE(is_lossless_assignment(adt_int_235, adt_int16_215));
    EXPECT_FALSE(is_lossless_assignment(adt_int_235, adt_int64_215));

    // Scalars broadcast to arrays
    EXPECT_TRUE(is_lossless_assignment(adt_int_235, make_dtype<int16_t>()));
    EXPECT_FALSE(is_lossless_assignment(adt_int_235, make_dtype<int64_t>()));
    EXPECT_FALSE(is_lossless_assignment(make_dtype<int64_t>(), adt_int_235));
*/
}

TEST(ArrayDType, StringOutput) {
/*
    intptr_t shape_235[] = {2,3,5};
    dtype adt_int_235 = make_ndobject_dtype<int>(3, shape_235);

    // Verify the current string representation [note it does not include strides at the moment]
    stringstream ss;
    ss << adt_int_235;
    EXPECT_EQ("array<int32, (2,3,5)>", ss.str());
*/
}
