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

    const intptr_t offset[2] = {-1, -1};
    const intptr_t shape[2] = {2, 2};

    array_neighborhood_iter<0, 1> iter(a, shape, offset);
  //  const intptr_t *index = iter.index();
//    const intptr_t *neighbor_index = iter.neighbor_index();
    do {
        const char *data = iter.data();
      //  cout << "(DEBUG) index is " << index[0] << ", " << index[1] << endl;
        cout << "(DEBUG) data is " << *reinterpret_cast<const int *>(data) << endl;

        do {
            const char *neighbor_data = iter.neighbor_data();
//            cout << "(DEBUG) relative_neighbor_index is " << neighbor_index[0] - 1 << ", " << neighbor_index[1] - 1 << endl;
  //          cout << "(DEBUG) neighbor_index is " << index[0] - 1 + neighbor_index[0] << ", " << index[1] - 1 + neighbor_index[1] << endl;
    //        cout << "(DEBUG) within_bounds is " << iter.within_bounds() << endl;
            if (neighbor_data != NULL) {
                cout << "(DEBUG) neighbor_data is " << *reinterpret_cast<const int *>(neighbor_data) << endl;
            }
        } while (iter.next_neighbor());
    } while(iter.next());
}

/*
TEST(ArrayIter, Mean) {
    int vals[2][3] = {{0, 1, 2}, {3, 4, 5}};

    nd::array a = nd::empty(2, 3, ndt::make_strided_dim(ndt::make_strided_dim(ndt::make_type<int>())));
    a.vals() = vals;

    const intptr_t neighborhood_shape[2] = {2, 2};
    const intptr_t neighborhood_offset[2] = {-1, -1};
}
*/
