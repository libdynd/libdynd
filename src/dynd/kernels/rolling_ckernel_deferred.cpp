//
// Copyright (C) 2011-14 Mark Wiebe, DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <dynd/kernels/rolling_ckernel_deferred.hpp>

using namespace std;
using namespace dynd;

void make_rolling_ckernel_deferred(ckernel_deferred *out_ckd,
                                   const ndt::type &dst_tp,
                                   const ndt::type &src_tp,
                                   const nd::array &window_op, int window_size)
{
}