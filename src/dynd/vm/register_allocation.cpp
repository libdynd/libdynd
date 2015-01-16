//
// Copyright (C) 2011-15 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <dynd/vm/register_allocation.hpp>

using namespace std;
using namespace dynd;

dynd::vm::register_allocation::register_allocation(const std::vector<ndt::type>& regtypes,
                        intptr_t max_element_count, intptr_t max_byte_count)
    : m_regtypes(regtypes), m_registers(m_regtypes.size()), m_blockrefs(m_regtypes.size()), m_allocated_memory(NULL)
{
    if (regtypes.empty()) {
        throw runtime_error("Cannot do a register allocation with no registers");
    }

    // Get the number of bytes per element across all the registers
    intptr_t bytes_per_element = regtypes[0].get_data_size();
    for (size_t i = 1; i < regtypes.size(); ++i) {
        bytes_per_element += regtypes[i].get_data_size();
    }
    // Turn it into an element count, clamped to [1, max_element_count
    intptr_t element_count = max_byte_count / bytes_per_element;
    if (element_count == 0) {
        element_count = 1;
    }
    else if (element_count > max_element_count) {
        element_count = max_element_count;
    }
    // Allocate memory for the registers and padding bytes (maybe use more padding, to
    // preclude false cache sharing between CPUs when multithreading?)
    size_t memsize = bytes_per_element * element_count + 16 * regtypes.size();
    m_allocated_memory = (char *)malloc(memsize);
    if (m_allocated_memory == NULL) {
        throw bad_alloc();
    }
    // Create pointers to the individual register data
    size_t offset = 0;
    for (size_t i = 0; i < regtypes.size(); ++i) {
        const ndt::type& d = regtypes[i];
        // Align the pointer
        offset = inc_to_alignment(offset, d.get_data_alignment());
        m_registers[i] = m_allocated_memory + offset;
    }
}

dynd::vm::register_allocation::~register_allocation()
{
    if (m_allocated_memory != NULL) {
        free(m_allocated_memory);
    }
}
