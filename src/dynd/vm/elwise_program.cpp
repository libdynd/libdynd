//
// Copyright (C) 2012, Dynamic NDArray Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <stdexcept>
#include <sstream>

#include <dynd/vm/elwise_program.hpp>

using namespace std;
using namespace dynd;

const dynd::vm::opcode_info_t dynd::vm::opcode_info[opcode_count] = {
    {"add", 2},
    {"subtract", 2},
    {"multiply", 2},
    {"divide", 2}
};

void dynd::vm::validate_elwise_program(int reg_count, size_t program_size, const int *program)
{
    size_t i = 0;
    while (i < program_size) {
        int opcode = program[i];
        if (opcode < 0 || opcode >= opcode_count) {
            stringstream ss;
            ss << "DyND VM program contains invalid opcode " << opcode << " at position " << i;
            throw runtime_error(ss.str());
        }

        int arity = vm::opcode_info[opcode].arity;
        if (i + arity + 1 >= program_size) {
            stringstream ss;
            ss << "DyND VM program opcode " << vm::opcode_info[opcode].name << " at position " << i;
            ss << " does not have enough arguments";
            ss << " (require " << arity << ", but only provided " << program_size - i - 1 << ")";
            throw runtime_error(ss.str());
        }

        for (int j = 0; j < arity + 1; ++j) {
            int reg = program[i + j + 1];
            if (reg < 0 || reg >= reg_count) {
                stringstream ss;
                ss << "DyND VM program opcode " << vm::opcode_info[opcode].name << " at position " << i;
                ss << ", has argument register " << (j + 1) << " of " << arity << " out of bounds";
                ss << " (register number " << reg << ", number of registers " << reg_count << ")";
                throw runtime_error(ss.str());
            }
        }

        i += 2 + arity;
    }
}
