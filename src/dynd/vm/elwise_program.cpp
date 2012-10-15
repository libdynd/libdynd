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

void dynd::vm::validate_elwise_program(int input_count, int reg_count, size_t program_size, const int *program)
{
    size_t i = 0;
    bool wrote_to_output = false;
    while (i < program_size) {
        // Validate the opcode
        int opcode = program[i];
        if (opcode < 0 || opcode >= opcode_count) {
            stringstream ss;
            ss << "DyND VM program contains invalid opcode " << opcode << " at position " << i;
            throw runtime_error(ss.str());
        }

        // Validate that there are enough arguments
        int arity = vm::opcode_info[opcode].arity;
        if (i + arity + 1 >= program_size) {
            stringstream ss;
            ss << "DyND VM program opcode " << vm::opcode_info[opcode].name << " at position " << i;
            ss << " does not have enough arguments";
            ss << " (require " << arity << ", but only provided " << program_size - i - 1 << ")";
            throw runtime_error(ss.str());
        }

        // Validate that all the registers are in range and aren't writing to an input argument
        int reg;
        for (int j = 0; j < arity + 1; ++j) {
            reg = program[i + j + 1];
            if (reg < 0 || reg >= reg_count) {
                stringstream ss;
                ss << "DyND VM program opcode " << vm::opcode_info[opcode].name << " at position " << i;
                ss << ", has argument register " << (j + 1) << " of " << (arity + 1) << " out of bounds";
                ss << " (register number " << reg << ", number of registers " << reg_count << ")";
                throw runtime_error(ss.str());
            }
            if (j == 0) {
                // Output argument
                if (reg == 0) {
                    wrote_to_output = true;
                } else if (reg <= input_count) {
                    stringstream ss;
                    ss << "DyND VM program opcode " << vm::opcode_info[opcode].name << " at position " << i;
                    ss << ", has its output set to one of the input registers, which are read-only";
                    throw runtime_error(ss.str());
                }
            }
        }

        i += 2 + arity;
    }

    if (!wrote_to_output) {
        throw runtime_error("DyND VM program did not write to the output register (register 0)");
    }
}
