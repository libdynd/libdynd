//
// Copyright (C) 2011-13 Mark Wiebe, DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <iostream>
#include <stdexcept>
#include <algorithm>
#include <cmath>

#include "inc_gtest.hpp"

#include <dynd/types/fixedstring_type.hpp>
#include <dynd/kernels/ckernel_deferred.hpp>
#include <dynd/kernels/assignment_kernels.hpp>

using namespace std;
using namespace dynd;

TEST(CKernelDeferred, Assignment) {
    ckernel_deferred ckd;
    // Create a deferred ckernel for converting string to int
    make_ckernel_deferred_from_assignment(ndt::make_type<int>(), ndt::make_fixedstring(16),
                    unary_operation_funcproto, assign_error_default, ckd);
    // Validate that its types, etc are set right
    ASSERT_EQ(unary_operation_funcproto, ckd.ckernel_funcproto);
    ASSERT_EQ(2, ckd.data_types_size);
    ASSERT_EQ(ndt::make_type<int>(), ndt::type(ckd.data_dynd_types[0], true));
    ASSERT_EQ(ndt::make_fixedstring(16), ndt::type(ckd.data_dynd_types[1], true));

    const char *dynd_metadata[2] = {NULL, NULL};

    // Instantiate a single ckernel
    ckernel_builder ckb;
    ckd.instantiate_func(ckd.data_ptr, &ckb, 0, dynd_metadata, kernel_request_single);
    int int_out = 0;
    char str_in[16] = "3251";
    unary_single_operation_t usngo = ckb.get()->get_function<unary_single_operation_t>();
    usngo(reinterpret_cast<char *>(&int_out), str_in, ckb.get());
    EXPECT_EQ(3251, int_out);

    // Instantiate a strided ckernel
    ckb.reset();
    ckd.instantiate_func(ckd.data_ptr, &ckb, 0, dynd_metadata, kernel_request_strided);
    int ints_out[3] = {0, 0, 0};
    char strs_in[3][16] = {"123", "4567", "891029"};
    unary_strided_operation_t ustro = ckb.get()->get_function<unary_strided_operation_t>();
    ustro(reinterpret_cast<char *>(&ints_out), sizeof(int), strs_in[0], 16, 3, ckb.get());
    EXPECT_EQ(123, ints_out[0]);
    EXPECT_EQ(4567, ints_out[1]);
    EXPECT_EQ(891029, ints_out[2]);
}
