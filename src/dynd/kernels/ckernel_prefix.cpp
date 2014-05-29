//
// Copyright (C) 2011-14 Mark Wiebe, DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <dynd/kernels/ckernel_prefix.hpp>

using namespace std;
using namespace dynd;

std::ostream& dynd::operator<<(std::ostream& o, kernel_request_t kernreq)
{
    switch (kernreq) {
        case kernel_request_single:
            return (o << "kernel_request_single");
        case kernel_request_strided:
            return (o << "kernel_request_strided");
        default:
            return (o << "(unknown kernrel request " << (int)kernreq << ")");
    }
}
