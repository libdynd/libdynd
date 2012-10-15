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

void dynd::vm::elwise_program::validate_program() const
{
    int program_size = m_program.size();
    int num_registers = m_regtypes.size();

    int i = 0;
    while (i < program_size) {
        int opcode = m_program[i];
        if (opcode < 0 || opcode > opcode_count) {
            stringstream ss;
            ss << "DyND VM program contains invalid opcode " << m_program[i] << " at position " << i;
            throw runtime_error(ss.str());
        }

        int arity = vm::opcode_info[opcode].arity;
        if (i + arity >= program_size) {
            stringstream ss;
            ss << "DyND VM program opcode " << vm::opcode_info[opcode].name << " at position " << i;
            ss << " does not have enough arguments";
            ss << " (require " << arity << ", but only provided " << program_size - i - 1 << ")";
            throw runtime_error(ss.str());
        }

        for (int j = 0; j < arity; ++j) {
            int reg = m_program[i + j + 1];
            if (reg < 0 || reg >= num_registers) {
                stringstream ss;
                ss << "DyND VM program opcode " << vm::opcode_info[opcode].name << " at position " << i;
                ss << ", has argument register " << (j + 1) << " of " << arity << " out of bounds";
                ss << " (register number " << reg << ", number of registers " << num_registers << ")";
                throw runtime_error(ss.str());
            }
        }

        i += 1 + arity;
    }
}
