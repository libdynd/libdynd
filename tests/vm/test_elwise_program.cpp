//
// Copyright (C) 2011-12, Dynamic NDArray Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <iostream>
#include <stdexcept>
#include <algorithm>
#include <cmath>

#include "inc_gtest.hpp"

#include "dynd/vm/elwise_program.hpp"

using namespace std;
using namespace dynd;

TEST(VMElwiseProgram, Validation) {
    // Test that the validation of elementwise VM programs works
    int program1[] = {vm::opcode_add, 2, 3, 4};
    // Should fail if there are not enough arguments
    EXPECT_THROW(vm::validate_elwise_program(10, 1, program1), runtime_error);
    EXPECT_THROW(vm::validate_elwise_program(10, 2, program1), runtime_error);
    EXPECT_THROW(vm::validate_elwise_program(10, 3, program1), runtime_error);
    vm::validate_elwise_program(10, 4, program1);
    // Should fail if there are not enough registers
    EXPECT_THROW(vm::validate_elwise_program(4, 4, program1), runtime_error);
    vm::validate_elwise_program(5, 4, program1);

    // Should fail with out of bounds opcode
    int program2[] = {-1, 0, 0, 0};
    EXPECT_THROW(vm::validate_elwise_program(4, 4, program2), runtime_error);
    program2[0] = vm::opcode_count;
    EXPECT_THROW(vm::validate_elwise_program(4, 4, program2), runtime_error);

    // Should fail with out of bounds register
    int program3[] = {vm::opcode_add, 0, 0, -1};
    EXPECT_THROW(vm::validate_elwise_program(1, 4, program3), runtime_error);
    program3[3] = 1;
    EXPECT_THROW(vm::validate_elwise_program(1, 4, program3), runtime_error);

    // Two instructions
    int program4[] = {vm::opcode_add, 0, 0, 1, vm::opcode_divide, 0, 1, 2};
    vm::validate_elwise_program(3, 8, program4);
    EXPECT_THROW(vm::validate_elwise_program(3, 8, program3), runtime_error);

}
