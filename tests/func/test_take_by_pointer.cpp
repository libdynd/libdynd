//
// Copyright (C) 2011-14 Irwin Zaid, DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <iostream>
#include <stdexcept>
#include <algorithm>
#include <cmath>

#include "inc_gtest.hpp"

#include <dynd/array.hpp>
#include <dynd/json_parser.hpp>
#include <dynd/func/lift_arrfunc.hpp>
#include <dynd/func/take_by_pointer_arrfunc.hpp>
#include <dynd/types/pointer_type.hpp>

using namespace std;
using namespace dynd;

TEST(TakeByPointer, Simple) {
    nd::array a = parse_json("4 * int",
        "[0, 1, 2, 3]");
    nd::array idx = parse_json("4 * int64",
        "[2, 1, 0, 3]");

    nd::arrfunc af = make_take_by_pointer_arrfunc();

    nd::array res = nd::empty(4, ndt::make_pointer(ndt::make_type<int>()));
    std::cout << af.get()->func_proto << std::endl;

    af.call_out(a, idx, res);
    std::cout << res << std::endl;

    a = parse_json("2 * 4 * int",
        "[[0, 1, 2, 3], [4, 5, 6, 7]]");
    idx = parse_json("2 * 2 * int64",
        "[[1, 0], [1, 1]]");
    res = nd::empty(2, 4, ndt::make_pointer(ndt::make_type<int>()));
    af.call_out(a, idx, res);

/*
    std::cout << res << std::endl;
*/


//    std::cout << func_proto << std::endl;

//    nd::array a = parse_json("3 * 2 * int",
  //      "[[0, 1], [2, 3], [4, 5]]");
    //nd::array idx = parse_

//    std::exit(-1);

    std::exit(-1);
}
