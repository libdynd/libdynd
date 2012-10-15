//
// Copyright (C) 2012, Dynamic NDArray Developers
// BSD 2-Clause License, see LICENSE.txt
//

#ifndef _DND__ELWISE_PROGRAM_HPP_
#define _DND__ELWISE_PROGRAM_HPP_

#include <vector>

#include <dynd/dtype.hpp>

namespace dynd { namespace vm {

enum opcode_t {
    opcode_add,
    opcode_subtract,
    opcode_multiply,
    opcode_divide
};
const int opcode_count = 4;

struct opcode_info_t {
    const char *name;
    int arity;
};

extern const opcode_info_t opcode_info[opcode_count];

/**
    * Validates that the program follows the VM structural rules.
    * Registers are [<output>, <input1>, ..., <inputN>, <temp1>, ..., <tempM>].
    * The output register is write-only, the input registers are read-only.
    */
void validate_elwise_program(int input_count, int reg_count, size_t program_size, const int *program);

class elwise_program {
    std::vector<dtype> m_regtypes;
    std::vector<int> m_program;
    int m_input_count;

public:
    /** Constructs an elementwise VM program, stealing the internal values of regtypes and program */
    elwise_program(int input_count, std::vector<dtype>& regtypes, std::vector<int>& program)
        : m_input_count(input_count)
    {
        validate_elwise_program(input_count, regtypes.size(), program.size(), &program[0]);
        m_regtypes.swap(regtypes);
        m_program.swap(program);
    }
};

}} // namespace dynd::vm

#endif // _DND__ELWISE_PROGRAM_HPP_