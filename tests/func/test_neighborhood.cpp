//
// Copyright (C) 2011-14 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <iostream>
#include <stdexcept>
#include <algorithm>
#include <cmath>

#include "inc_gtest.hpp"
#include "../dynd_assertions.hpp"

#include <dynd/json_parser.hpp>
#include <dynd/func/apply_arrfunc.hpp>
#include <dynd/func/neighborhood_arrfunc.hpp>
#include <dynd/types/struct_type.hpp>
#include <dynd/func/call_callable.hpp>

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
    nd::arrfunc af = make_neighborhood_arrfunc(nd::make_apply_arrfunc(sum<1>), 1);
    nd::array a;

    a = parse_json("4 * int",
        "[0, 1, 2, 3]");

    EXPECT_JSON_EQ_ARR("[3, 6, 5, 3]",
        af(a, kwds("shape", parse_json("1 * int", "[3]"))));

    EXPECT_JSON_EQ_ARR("[1, 3, 6, 5]",
        af(a, kwds("shape", parse_json("1 * int", "[3]"), "offset", parse_json("1 * int", "[-1]"))));

    EXPECT_JSON_EQ_ARR("[3, 1, 2, 3]",
        af(a, kwds("mask", parse_json("4 * bool", "[true, false, false, true]"))));

    EXPECT_JSON_EQ_ARR("[3, 5, 3, 0]",
        af(a, kwds("mask", parse_json("4 * bool", "[false, true, true, false]"))));

    EXPECT_JSON_EQ_ARR("[2, 3, 1, 2]",
        af(a, kwds("mask", parse_json("4 * bool", "[true, false, false, true]"), "offset", parse_json("1 * int", "[-1]"))));

    EXPECT_JSON_EQ_ARR("[1, 3, 5, 3]",
        af(a, kwds("mask", parse_json("4 * bool", "[false, true, true, false]"), "offset", parse_json("1 * int", "[-1]"))));

    a = parse_json("10 * int",
        "[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]");

    EXPECT_JSON_EQ_ARR("[15, 21, 27, 33, 39, 35, 30, 24, 17, 9]",
        af(a, kwds("shape", parse_json("1 * int", "[6]"))));

    EXPECT_JSON_EQ_ARR("[6, 9, 12, 15, 18, 21, 14, 16, 8, 9]",
        af(a, kwds("mask", parse_json("6 * bool", "[true, false, true, false, true, false]"))));

    EXPECT_JSON_EQ_ARR("[10, 15, 21, 27, 33, 39, 35, 30, 24, 17]",
        af(a, kwds("shape", parse_json("1 * int", "[6]"), "offset", parse_json("1 * int", "[-1]"))));

    EXPECT_JSON_EQ_ARR("[3, 5, 7, 9, 11, 13, 15, 17, 9, 0]",
        af(a, kwds("mask", parse_json("6 * bool", "[false, false, true, true, false, false]"), "offset", parse_json("1 * int", "[-1]"))));
}

TEST(Neighborhood, Sum2D) {
    nd::arrfunc af = make_neighborhood_arrfunc(nd::make_apply_arrfunc(sum<2>), 2);
    nd::array a;

    a = parse_json("4 * 4 * int",
        "[[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11], [12, 13, 14, 15]]");

    EXPECT_JSON_EQ_ARR("[[45, 54, 39, 21], [81, 90, 63, 33], [66, 72, 50, 26], [39, 42, 29, 15]]",
        af(a, kwds("shape", parse_json("2 * int", "[3, 3]"))));

    EXPECT_JSON_EQ_ARR("[[25, 30, 27, 7], [45, 50, 43, 11], [48, 52, 40, 15], [13, 14, 15, 0]]",
        af(a, kwds("mask", parse_json("3 * 3 * bool", "[[false, true, false], [true, true, true], [false, true, false]]"))));

    EXPECT_JSON_EQ_ARR("[[10, 18, 24, 18], [27, 45, 54, 39], [51, 81, 90, 63], [42, 66, 72, 50]]",
        af(a, kwds("shape", parse_json("2 * int", "[3, 3]"), "offset", parse_json("2 * int", "[-1, -1]"))));

    EXPECT_JSON_EQ_ARR("[[5, 8, 12, 12], [17, 25, 30, 27], [33, 45, 50, 43], [33, 48, 52, 40]]",
        af(a, kwds("mask", parse_json("3 * 3 * bool", "[[false, true, false], [true, true, true], [false, true, false]]"),
        "offset", parse_json("2 * int", "[-1, -1]"))));

    a = parse_json("6 * 5 * int",
        "[[0, 1, 2, 3, 4], [5, 6, 7, 8, 9], [10, 11, 12, 13, 14],"
        "[15, 16, 17, 18, 19], [20, 21, 22, 23, 24], [25, 26, 27, 28, 29]]");

    EXPECT_JSON_EQ_ARR("[[300, 250, 195, 135, 70], [425, 350, 270, 185, 95], [390, 320, 246, 168, 86],"
        "[330, 270, 207, 141, 72], [245, 200, 153, 104, 53], [135, 110, 84, 57, 29]]",
        af(a, kwds("shape", parse_json("2 * int", "[5, 5]"))));

    EXPECT_JSON_EQ_ARR("[[30, 35, 40, 25, 18], [55, 60, 65, 40, 28], [80, 85, 90, 55, 38],"
        "[105, 110, 115, 70, 48], [68, 71, 74, 52, 24], [52, 54, 56, 28, 29]]",
        af(a, kwds("mask", parse_json("3 * 3 * bool", "[[true, false, true], [false, true, false], [true, false, true]]"))));

    EXPECT_JSON_EQ_ARR("[[69, 75, 81, 37, 19], [99, 105, 111, 47, 24], [129, 135, 141, 57, 29],"
        "[66, 69, 72, 0, 0], [49, 51, 53, 0, 0], [27, 28, 29, 0, 0]]",
        af(a, kwds("mask", parse_json("4 * 3 * bool", "[[false, false, true], [false, false, true], [false, false, true], [true, true, true]]"))));

    EXPECT_JSON_EQ_ARR("[[54, 78, 105, 90, 72], [102, 144, 190, 160, 126], [165, 230, 300, 250, 195],"
        "[240, 330, 425, 350, 270], [222, 304, 390, 320, 246], [189, 258, 330, 270, 207]]",
        af(a, kwds("shape", parse_json("2 * int", "[5, 5]"), "offset", parse_json("2 * int", "[-2, -2]"))));

    EXPECT_JSON_EQ_ARR("[[0, 1, 3, 6, 9], [5, 12, 20, 25, 30], [15, 28, 47, 54, 61],"
        "[30, 48, 82, 89, 96], [45, 68, 117, 124, 131], [60, 88, 152, 159, 166]]",
        af(a, kwds("mask", parse_json("3 * 3 * bool", "[[true, false, true], [true, false, true], [true, true, true]]"),
        "offset", parse_json("2 * int", "[-2, -2]"))));

    EXPECT_JSON_EQ_ARR("[[32, 40, 33, 24, 13], [72, 80, 63, 44, 23], [112, 120, 93, 64, 33],"
        "[152, 160, 123, 84, 43], [192, 200, 153, 104, 53], [106, 110, 84, 57, 29]]",
        af(a, kwds("shape", parse_json("2 * int", "[2, 4]"))));

    EXPECT_JSON_EQ_ARR("[[0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [9, 7, 4, 0, 0],"
        "[33, 24, 13, 0, 0], [63, 44, 23, 0, 0], [93, 64, 33, 0, 0]]",
        af(a, kwds("shape", parse_json("2 * int", "[2, 4]"), "offset", parse_json("2 * int", "[-3, 2]"))));

    EXPECT_JSON_EQ_ARR("[[0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [3, 4, 0, 0, 0],"
        "[14, 12, 4, 0, 0], [29, 22, 9, 0, 0], [44, 32, 14, 0, 0]]",
        af(a, kwds("mask", parse_json("2 * 3 * bool", "[[true, false, true], [false, true, false]]"),
        "offset", parse_json("2 * int", "[-3, 2]"))));   
}

TEST(Neighborhood, Sum3D) {
    nd::arrfunc af = make_neighborhood_arrfunc(nd::make_apply_arrfunc(sum<3>), 3);
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

    EXPECT_JSON_EQ_ARR("[[[294, 308, 202, 115], [350, 364, 238, 135], [241, 250, 171, 85], [145, 150, 91, 62]],"
        "[[518, 532, 346, 195], [574, 588, 382, 215], [385, 394, 267, 133], [225, 230, 139, 94]],"
        "[[397, 406, 279, 133], [433, 442, 303, 145], [306, 312, 210, 106], [151, 154, 109, 47]],"
        "[[265, 270, 163, 110], [285, 290, 175, 118], [175, 178, 121, 59], [122, 124, 62, 63]]]",
        af(a, kwds("mask", parse_json("3 * 3 * 3 * bool", "[[[true, false, true], [false, true, false], [true, false, true]],"
        "[[false, true, false], [true, false, true], [false, true, false]],"
        "[[true, false, true], [false, true, false], [true, false, true]]]"))));

    EXPECT_JSON_EQ_ARR("[[[84, 132, 144, 100], [150, 234, 252, 174], [198, 306, 324, 222], [148, 228, 240, 164]],"
        "[[222, 342, 360, 246], [369, 567, 594, 405], [441, 675, 702, 477], [318, 486, 504, 342]],"
        "[[414, 630, 648, 438], [657, 999, 1026, 693], [729, 1107, 1134, 765], [510, 774, 792, 534]],"
        "[[340, 516, 528, 356], [534, 810, 828, 558], [582, 882, 900, 606], [404, 612, 624, 420]]]",
        af(a, kwds("shape", parse_json("3 * int", "[3, 3, 3]"), "offset", parse_json("3 * int", "[-1, -1, -1]"))));

    EXPECT_JSON_EQ_ARR("[[[42, 66, 72, 50], [75, 125, 134, 87], [99, 161, 170, 111], [74, 114, 120, 82]],"
        "[[111, 173, 182, 123], [185, 294, 308, 202], [221, 350, 364, 238], [159, 241, 250, 171]],"
        "[[207, 317, 326, 219], [329, 518, 532, 346], [365, 574, 588, 382], [255, 385, 394, 267]],"
        "[[170, 258, 264, 178], [267, 397, 406, 279], [291, 433, 442, 303], [202, 306, 312, 210]]]",
        af(a, kwds("mask", parse_json("3 * 3 * 3 * bool", "[[[true, false, true], [false, true, false], [true, false, true]],"
        "[[false, true, false], [true, false, true], [false, true, false]],"
        "[[true, false, true], [false, true, false], [true, false, true]]]"), "offset", parse_json("3 * int", "[-1, -1, -1]"))));
}

/*
    Todo: Make this 3D test pass.

    EXPECT_JSON_EQ_ARR("[[[1128, 864, 588, 300], [918, 702, 477, 243], [660, 504, 342, 174], [354, 270, 183, 93]],"
        "[[1896, 1440, 972, 492], [1494, 1134, 765, 387], [1044, 792, 534, 270], [546, 414, 279, 141]],"
        "[[1520, 1152, 776, 392], [1188, 900, 606, 306], [824, 624, 420, 212], [428, 324, 218, 110]],"
        "[[888, 672, 452, 228], [690, 522, 351, 177], [476, 360, 242, 122], [246, 186, 125, 63]]]",
        af(a, kwds("shape", parse_json("3 * int", "[3, 5, 7]"))));
*/
