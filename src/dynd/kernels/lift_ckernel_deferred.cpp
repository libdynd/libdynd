//
// Copyright (C) 2011-13 Mark Wiebe, DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <dynd/kernels/lift_ckernel_deferred.hpp>

using namespace std;
using namespace dynd;

void dynd::lift_ckernel_deferred(ckernel_deferred *out_ckd,
                const nd::array& ckd,
                const std::vector<ndt::type>& lifted_types)
{
    throw runtime_error("lift_ckernel_deferred unimplemented");
}
