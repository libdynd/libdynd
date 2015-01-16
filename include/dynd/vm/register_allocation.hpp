//
// Copyright (C) 2011-15 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#pragma once

#include <vector>

#include <dynd/vm/elwise_program.hpp>

namespace dynd { namespace vm {

class register_allocation {
    const std::vector<ndt::type>& m_regtypes;
    std::vector<char *> m_registers;
    std::vector<memory_block_ptr> m_blockrefs;
    char *m_allocated_memory;
public:
    register_allocation(const std::vector<ndt::type>& regtypes, intptr_t max_element_count, intptr_t max_byte_count);
    ~register_allocation();

    const std::vector<ndt::type>& get_regtypes() const {
        return m_regtypes;
    }

    const std::vector<char *>& get_registers() const {
        return m_registers;
    }
};

}} // namespace dynd::vm
