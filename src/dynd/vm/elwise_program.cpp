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
    {"copy", 1},
    {"add", 2},
    {"subtract", 2},
    {"multiply", 2},
    {"divide", 2}
};

int dynd::vm::validate_elwise_program(int input_count, int reg_count, size_t program_size, const int *program)
{
    size_t i = 0;
    bool wrote_to_output = false;
    int instruction_count = 0;
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
                    ss << ", has its output set to register " << reg << ", which is a read-only input register";
                    throw runtime_error(ss.str());
                }
            }
        }

        ++instruction_count;
        i += 2 + arity;
    }

    if (!wrote_to_output) {
        throw runtime_error("DyND VM program did not write to the output register (register 0)");
    }

    return instruction_count;
}

static void print_register(std::ostream& o, int reg)
{
    o << "r";
    if (reg < 10) {
        o << "0";
    }
    o << reg;
}

void dynd::vm::elwise_program::debug_print(std::ostream& o, const std::string& indent) const
{
    // Print all the register types
    o << indent << "output register (0):\n";
    o << indent << "  " << m_regtypes[0] << "\n";
    if (m_input_count == 0) {
        o << indent << "no input registers\n";
    } else {
        o << indent << "input registers (1 to " << m_input_count << "):\n";
        for (int i = 1; i <= m_input_count; ++i) {
            o << indent << "  " << m_regtypes[i] << "\n";
        }
    }
    if (m_input_count + 1 == (int)m_regtypes.size()) {
        o << indent << "no temporary registers\n";
    } else {
        o << indent << "temporary registers (" << (m_input_count + 1) << " to " << (m_regtypes.size() - 1) << "):\n";
        for (int i = m_input_count + 1; i < (int)m_regtypes.size(); ++i) {
            o << indent << "  " << m_regtypes[i] << "\n";
        }
    }

    // Print all the instructions
    // NOTE: This assumes the program has previously been validated and not modified since
    o << indent << "program:\n";
    for (size_t ip = 0; ip < m_program.size();) {
        int opcode = m_program[ip];
        int arity = vm::opcode_info[opcode].arity;
        // operation
        o << indent << "  " << vm::opcode_info[opcode].name << " ";
        for (size_t i = 0, i_end = 12 - strlen(vm::opcode_info[opcode].name); i != i_end; ++i) {
            o << " ";
        }
        // output
        print_register(o, m_program[ip + 1]);
        if (arity > 0) {
            o << ",  ";
            for (int i = 1; i <= arity; ++i) {
                print_register(o, m_program[ip + 1 + i]);
                if (i != arity) {
                    o << ", ";
                }
            }
        }
        o << "\n";
        ip += 2 + arity;
    }
    o.flush();
}
