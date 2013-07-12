//
// Copyright (C) 2012, Dynamic NDArray Developers
// BSD 2-Clause License, see LICENSE.txt
//

#ifndef _DYND__ELWISE_PROGRAM_HPP_
#define _DYND__ELWISE_PROGRAM_HPP_

#include <vector>

#include <dynd/type.hpp>

namespace dynd { namespace vm {

enum opcode_t {
    opcode_copy,
    opcode_add,
    opcode_subtract,
    opcode_multiply,
    opcode_divide
};
const int opcode_count = opcode_divide + 1;

struct opcode_info_t {
    const char *name;
    int arity;
};

extern const opcode_info_t opcode_info[opcode_count];

/**
* Validates that the program follows the VM structural rules.
* Registers are [<output>, <input1>, ..., <inputN>, <temp1>, ..., <tempM>].
* The output register is write-only, the input registers are read-only.
*
* Returns the number of instructions in the program.
*/
int validate_elwise_program(int input_count, int reg_count, size_t program_size, const int *program);

class elwise_program {
    std::vector<ndt::type> m_regtypes;
    std::vector<int> m_program;
    int m_input_count, m_instruction_count;

public:
    elwise_program()
        : m_regtypes(), m_program(), m_input_count(0), m_instruction_count(0)
    {
    }

    /** Constructs an elementwise VM program, stealing the internal values of regtypes and program */
    elwise_program(int input_count, std::vector<ndt::type>& regtypes, std::vector<int>& program)
        : m_input_count(input_count)
    {
        m_instruction_count = validate_elwise_program(input_count, (int)regtypes.size(), (int)program.size(), &program[0]);
        m_regtypes.swap(regtypes);
        m_program.swap(program);
    }

    /** Sets the values of the elementwise VM program, stealing the internal values of regtypes and program */
    void set(int input_count, std::vector<ndt::type>& regtypes, std::vector<int>& program)
    {
        m_instruction_count = validate_elwise_program(input_count, (int)regtypes.size(), (int)program.size(), &program[0]);
        m_regtypes.swap(regtypes);
        m_program.swap(program);
        m_input_count = input_count;
    }

    /** Debug printing of the elwise program */
    void debug_print(std::ostream& o, const std::string& indent = "") const;

    /** Returns a const reference to the vector of register types */
    const std::vector<ndt::type>& get_register_types() const {
        return m_regtypes;
    }

    /** Returns a const reference to the program vector */
    const std::vector<int>& get_program() const {
        return m_program;
    }

    /** The number of inputs to the VM program */
    int get_input_count() const {
        return m_input_count;
    }

    /**
     * The number of instructions in the program
     * (one instruction = op code + registers/other data)
     */
    int get_instruction_count() const {
        return m_instruction_count;
    }
};

}} // namespace dynd::vm

#endif // _DYND__ELWISE_PROGRAM_HPP_

