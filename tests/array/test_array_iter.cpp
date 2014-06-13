//
// Copyright (C) 2011-14 Irwin Zaid, Mark Wiebe, DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <iostream>
#include <stdexcept>
#include <algorithm>
#include <cmath>

#include "inc_gtest.hpp"
#include "dynd_assertions.hpp"

#include <dynd/array_iter.hpp>

using namespace std;
using namespace dynd;

TEST(ArrayIter, Unknown) {
    int vals[2][3] = {{45, 1, 2}, {3, 4, 5}};

    nd::array a = nd::empty(2, 3, ndt::make_strided_dim(ndt::make_strided_dim(ndt::make_type<int>())));
    a.vals() = vals;

//    const intptr_t offset[2] = {0, 0};
    const intptr_t shape[2] = {2, 2};

    array_neighborhood_iter<0, 1> iter(a, shape);
    do {
        const char *data = iter.data();
        const char *neighbor_data = iter.neighbor_data();

        cout << "DEBUG: " << *reinterpret_cast<const int *>(data) << endl;
        cout << "DEBUG: " << *reinterpret_cast<const int *>(neighbor_data) << endl;
    } while(iter.next());
}
