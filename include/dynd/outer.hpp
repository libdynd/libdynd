//
// Copyright (C) 2011-14 Mark Wiebe, Irwin Zaid, DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#ifndef _DYND__OUTER_HPP_
#define _DYND__OUTER_HPP_

#include <dynd/elwise.hpp>

namespace dynd { namespace nd {

template<typename A0, typename A1>
void outer_broadcast(const nd::array& a0, const nd::array& a1, intptr_t& res_ndim, dynd::dimvector& res_shape) {
    res_ndim = a0.get_ndim() + a1.get_ndim();
    res_shape.init(res_ndim);
}

template<typename R, typename A0, typename A1>
nd::array outer(void (*)(R&, A0, A1), const nd::array &a0, const nd::array& a1) {
    intptr_t res_ndim;
    dimvector res_shape;

    outer_broadcast<A0, A1>(a0, a1, res_ndim, res_shape);

//    nd::array b0 = a0;
//    nd::array b1 = a1;
    // view?


    return a0;
}

}} // namespace dynd::nd

#endif // _DYND__OUTER_HPP_
