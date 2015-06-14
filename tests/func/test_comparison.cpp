//
// Copyright (C) 2011-15 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <iostream>
#include <stdexcept>
#include <cmath>
#include <inc_gtest.hpp>

#include "../test_memory_new.hpp"
#include "../dynd_assertions.hpp"

#include <dynd/func/arithmetic.hpp>
#include <dynd/func/comparison.hpp>
#include <dynd/func/elwise.hpp>
#include <dynd/array.hpp>
#include <dynd/json_parser.hpp>

using namespace dynd;

TEST(Comparison, Simple)
{
    nd::array a = {1, -1, 3};
    nd::array b = {0, 1, 2};

    std::cout << a << std::endl;
    std::cout << (a > b) << std::endl;
}