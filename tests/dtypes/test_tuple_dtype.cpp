//
// Copyright (C) 2011-13 Mark Wiebe, DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <iostream>
#include <sstream>
#include <stdexcept>
#include "inc_gtest.hpp"

#include <dynd/ndobject.hpp>
#include <dynd/dtypes/tuple_dtype.hpp>
#include <dynd/dtypes/fixedstring_dtype.hpp>

using namespace std;
using namespace dynd;

TEST(TupleDType, CreateOneField) {
    dtype dt;
    const tuple_dtype *tdt;

    // Tuple with one field
    dt = make_tuple_dtype(make_dtype<int32_t>());
    EXPECT_EQ(tuple_type_id, dt.get_type_id());
    EXPECT_EQ(4u, dt.get_data_size());
    EXPECT_EQ(4u, dt.get_alignment());
    EXPECT_TRUE(dt.is_pod());
    EXPECT_EQ(0u, (dt.get_flags()&(dtype_flag_blockref|dtype_flag_destructor)));
    tdt = static_cast<const tuple_dtype *>(dt.extended());
    EXPECT_TRUE(tdt->is_standard_layout());
    EXPECT_EQ(1u, tdt->get_fields().size());
    EXPECT_EQ(1u, tdt->get_offsets().size());
    EXPECT_EQ(make_dtype<int32_t>(), tdt->get_fields()[0]);
    EXPECT_EQ(0u, tdt->get_offsets()[0]);
}

