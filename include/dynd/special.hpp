//
// Copyright (C) 2011-14 Irwin Zaid, DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#ifndef _DYND__SPECIAL_HPP_
#define _DYND__SPECIAL_HPP_

#include <dynd/array.hpp>
#include <dynd/cephes.hpp>

namespace dynd { 

double bessel_j0(double x) {
    return cephes::j0(x);
}

double bessel_j1(double x) {
    return cephes::j1(x);
}

double bessel_y0(double x) {
    return cephes::y0(x);
}

double bessel_y1(double x) {
    return cephes::y1(x);
}

} // namespace dynd

#endif // _DYND__SPECIAL_HPP_
