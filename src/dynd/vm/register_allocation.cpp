//
// Copyright (C) 2012, Dynamic NDArray Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <dynd/vm/register_allocation.hpp>

using namespace std;
using namespace dynd;

dynd::vm::register_allocation::register_allocation(const std::vector<dtype>& regtypes,
                        intptr_t max_element_count, intptr_t max_byte_count)
    : m_regtypes(regtypes), m_registers(m_regtypes.size()), m_blockrefs(m_regtypes.size()), m_allocated_memory(NULL)
{
    // Get the number of bytes per element across all the registers
    intptr_t bytes_per_element = 0;
    for (size_t i = 0; i < regtypes.size(); ++i) {
        bytes_per_element += regtypes[i].element_size();
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
        throw bad_alloc("out of memory allocating registers for the dynd VM");
    }
    // Create pointers to the individual register data
    size_t offset = 0;
    for (size_t i = 0; i < regtypes.size(); ++i) {
        const dtype& d = regtypes[i];
        // Align the pointer
        offset = d.apply_alignment(offset);
        m_registers[i] = m_allocated_memory + offset;
    }
}

dynd::vm::register_allocation::~register_allocation()
{
    if (m_allocated_memory != NULL) {
        free(m_allocated_memory);
    }
}
