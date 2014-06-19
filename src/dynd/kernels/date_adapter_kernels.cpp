//
// Copyright (C) 2011-14 Mark Wiebe, DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <dynd/kernels/date_adapter_kernels.hpp>

using namespace std;
using namespace dynd;

bool dynd::make_date_adapter_arrfunc(const ndt::type &operand_tp,
                                     const nd::string &op,
                                     nd::arrfunc &out_forward,
                                     nd::arrfunc &out_reverse)
{
    return false;
}
