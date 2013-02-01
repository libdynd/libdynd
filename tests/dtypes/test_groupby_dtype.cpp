//
// Copyright (C) 2011-13, DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <iostream>
#include <sstream>
#include <stdexcept>
#include "inc_gtest.hpp"

#include <dynd/ndobject.hpp>
#include <dynd/dtypes/categorical_dtype.hpp>

using namespace std;
using namespace dynd;

TEST(GroupByDType, Create) {
    int data[] = {1,2,3};
    int by[] = {1,1,2};
    int groups[] = {1,2};
    ndobject g = groupby(data, by, make_categorical_dtype(groups));
    //g.debug_print(cout);
    //g = g.vals();
    //g.debug_print(cout);
    //cout << g << endl;
}
