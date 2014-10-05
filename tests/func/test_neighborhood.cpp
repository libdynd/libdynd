//
// Copyright (C) 2011-14 Irwin Zaid, Mark Wiebe, DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <iostream>
#include <stdexcept>
#include <algorithm>
#include <cmath>

#include "inc_gtest.hpp"
#include "../dynd_assertions.hpp"

#include <dynd/json_parser.hpp>
#include <dynd/func/functor_arrfunc.hpp>
#include <dynd/func/neighborhood_arrfunc.hpp>
#include <dynd/types/struct_type.hpp>

using namespace std;
using namespace dynd;

template <int N>
int sum(const nd::strided_vals<int, N> &nh) {
    typedef nd::strided_vals<int, N> nh_type;

    int res = 0;
    for (typename nh_type::iterator it = nh.begin(); it != nh.end(); ++it) {
        res += *it;
    }

    return res;
}

TEST(Neighborhood, Sum1D) {
    nd::arrfunc af = make_neighborhood_arrfunc(nd::make_functor_arrfunc(sum<1>), 1);
    nd::array a;

    a = parse_json("4 * int",
        "[0, 1, 2, 3]");
    EXPECT_JSON_EQ_ARR("[3, 6, 5, 3]",
        af(a, kwds("shape", parse_json("1 * int", "[3]"))));

    EXPECT_JSON_EQ_ARR("[1, 3, 6, 5]",
        af(a, kwds("shape", parse_json("1 * int", "[3]"), "offset", parse_json("1 * int", "[-1]"))));

    a = parse_json("10 * int",
        "[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]");
    EXPECT_JSON_EQ_ARR("[15, 21, 27, 33, 39, 35, 30, 24, 17, 9]",
        af(a, kwds("shape", parse_json("1 * int", "[6]"))));

    EXPECT_JSON_EQ_ARR("[10, 15, 21, 27, 33, 39, 35, 30, 24, 17]",
        af(a, kwds("shape", parse_json("1 * int", "[6]"), "offset", parse_json("1 * int", "[-1]"))));
}

TEST(Neighborhood, Sum2D) {
    nd::arrfunc af = make_neighborhood_arrfunc(nd::make_functor_arrfunc(sum<2>), 2);
    nd::array a;

    a = parse_json("4 * 4 * int",
        "[[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11], [12, 13, 14, 15]]");
    EXPECT_JSON_EQ_ARR("[[45, 54, 39, 21], [81, 90, 63, 33], [66, 72, 50, 26], [39, 42, 29, 15]]",
        af(a, kwds("shape", parse_json("2 * int", "[3, 3]"))));

    EXPECT_JSON_EQ_ARR("[[10, 18, 24, 18], [27, 45, 54, 39], [51, 81, 90, 63], [42, 66, 72, 50]]",
        af(a, kwds("shape", parse_json("2 * int", "[3, 3]"), "offset", parse_json("2 * int", "[-1, -1]"))));

    a = parse_json("6 * 5 * int",
        "[[0, 1, 2, 3, 4], [5, 6, 7, 8, 9], [10, 11, 12, 13, 14],"
        "[15, 16, 17, 18, 19], [20, 21, 22, 23, 24], [25, 26, 27, 28, 29]]");
    EXPECT_JSON_EQ_ARR("[[300, 250, 195, 135, 70], [425, 350, 270, 185, 95], [390, 320, 246, 168, 86],"
        "[330, 270, 207, 141, 72], [245, 200, 153, 104, 53], [135, 110, 84, 57, 29]]",
        af(a, kwds("shape", parse_json("2 * int", "[5, 5]"))));

    EXPECT_JSON_EQ_ARR("[[54, 78, 105, 90, 72], [102, 144, 190, 160, 126], [165, 230, 300, 250, 195],"
        "[240, 330, 425, 350, 270], [222, 304, 390, 320, 246], [189, 258, 330, 270, 207]]",
        af(a, kwds("shape", parse_json("2 * int", "[5, 5]"), "offset", parse_json("2 * int", "[-2, -2]"))));

    EXPECT_JSON_EQ_ARR("[[32, 40, 33, 24, 13], [72, 80, 63, 44, 23], [112, 120, 93, 64, 33],"
        "[152, 160, 123, 84, 43], [192, 200, 153, 104, 53], [106, 110, 84, 57, 29]]",
        af(a, kwds("shape", parse_json("2 * int", "[2, 4]"))));

    EXPECT_JSON_EQ_ARR("[[0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [9, 7, 4, 0, 0],"
        "[33, 24, 13, 0, 0], [63, 44, 23, 0, 0], [93, 64, 33, 0, 0]]",
        af(a, kwds("shape", parse_json("2 * int", "[2, 4]"), "offset", parse_json("2 * int", "[-3, 2]"))));
}

TEST(Neighborhood, Sum3D) {
    nd::arrfunc af = make_neighborhood_arrfunc(nd::make_functor_arrfunc(sum<3>), 3);
    nd::array a;

    a = parse_json("4 * 4 * 4 * int",
        "[[[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11], [12, 13, 14, 15]],"
        "[[16, 17, 18, 19], [20, 21, 22, 23], [24, 25, 26, 27], [28, 29, 30, 31]],"
        "[[32, 33, 34, 35], [36, 37, 38, 39], [40, 41, 42, 43], [44, 45, 46, 47]],"
        "[[48, 49, 50, 51], [52, 53, 54, 55], [56, 57, 58, 59], [60, 61, 62, 63]]]");
    EXPECT_JSON_EQ_ARR("[[[567, 594, 405, 207], [675, 702, 477, 243], [486, 504, 342, 174], [261, 270, 183, 93]],"
        "[[999, 1026, 693, 351], [1107, 1134, 765, 387], [774, 792, 534, 270], [405, 414, 279, 141]],"
        "[[810, 828, 558, 282], [882, 900, 606, 306], [612, 624, 420, 212], [318, 324, 218, 110]],"
        "[[477, 486, 327, 165], [513, 522, 351, 177], [354, 360, 242, 122], [183, 186, 125, 63]]]",
        af(a, kwds("shape", parse_json("3 * int", "[3, 3, 3]"))));

    EXPECT_JSON_EQ_ARR("[[[84, 132, 144, 100], [150, 234, 252, 174], [198, 306, 324, 222], [148, 228, 240, 164]],"
        "[[222, 342, 360, 246], [369, 567, 594, 405], [441, 675, 702, 477], [318, 486, 504, 342]],"
        "[[414, 630, 648, 438], [657, 999, 1026, 693], [729, 1107, 1134, 765], [510, 774, 792, 534]],"
        "[[340, 516, 528, 356], [534, 810, 828, 558], [582, 882, 900, 606], [404, 612, 624, 420]]]",
        af(a, kwds("shape", parse_json("3 * int", "[3, 3, 3]"), "offset", parse_json("3 * int", "[-1, -1, -1]"))));

/*
    EXPECT_JSON_EQ_ARR("[[[1128, 864, 588, 300], [918, 702, 477, 243], [660, 504, 342, 174], [354, 270, 183, 93]],"
        "[[1896, 1440, 972, 492], [1494, 1134, 765, 387], [1044, 792, 534, 270], [546, 414, 279, 141]],"
        "[[1520, 1152, 776, 392], [1188, 900, 606, 306], [824, 624, 420, 212], [428, 324, 218, 110]],"
        "[[888, 672, 452, 228], [690, 522, 351, 177], [476, 360, 242, 122], [246, 186, 125, 63]]]",
        af(a, kwds("shape", parse_json("3 * int", "[3, 5, 7]"))));
*/
}
