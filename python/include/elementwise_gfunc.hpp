//
// Copyright (C) 2011-12, Dynamic NDArray Developers
// BSD 2-Clause License, see LICENSE.txt
//

#ifndef _DND__ELEMENTWISE_GFUNC_HPP_
#define _DND__ELEMENTWISE_GFUNC_HPP_

#include <stdint.h>
#include <sstream>
#include <deque>
#include <vector>

#include <dnd/dtype.hpp>
#include <dnd/kernels/kernel_instance.hpp>
#include <dnd/codegen/codegen_cache.hpp>

#include <Python.h>

namespace pydnd {

class elementwise_gfunc_kernel {
public:
    dnd::dtype m_out, m_params[1];
    dnd::unary_specialization_kernel_instance m_kernel;
    PyObject *m_pyobj;

    elementwise_gfunc_kernel()
        : m_pyobj(0)
    {
    }

    ~elementwise_gfunc_kernel();

    void swap(elementwise_gfunc_kernel& rhs) {
        m_out.swap(rhs.m_out);
        m_params[0].swap(rhs.m_params[0]);
        m_kernel.swap(rhs.m_kernel);
    }
};

class elementwise_gfunc {
    std::string m_name;
    /**
     * This is a deque instead of a vector, because we are targetting C++98
     * and so cannot rely on C++11 move semantics.
     */
    std::deque<elementwise_gfunc_kernel> m_kernels;
    std::vector<dnd::memory_block_data *> m_blockrefs;
public:
    elementwise_gfunc(const char *name)
        : m_name(name)
    {
    }

    ~elementwise_gfunc();

    const std::string& get_name() const {
        return m_name;
    }

    /**
     * Adds a new memory_block for the elementwise_gfunc to
     * hold a reference to. For example, the executable
     * memory block for generated code should get added.
     */
    void add_blockref(dnd::memory_block_data *blockref);

    void add_kernel(dnd::codegen_cache& cgcache, PyObject *kernel);

    PyObject *call(PyObject *args, PyObject *kwargs);

    std::string debug_dump() const;
};

struct elementwise_gfunc_placement_wrapper {
    intptr_t dummy[(sizeof(elementwise_gfunc) + sizeof(intptr_t) - 1)/sizeof(intptr_t)];
};

inline void elementwise_gfunc_placement_new(elementwise_gfunc_placement_wrapper& v, const char *name)
{
    // Call placement new
    new (&v) elementwise_gfunc(name);
}

inline void elementwise_gfunc_placement_delete(elementwise_gfunc_placement_wrapper& v)
{
    // Call the destructor
    ((elementwise_gfunc *)(&v))->~elementwise_gfunc();
}

// dtype placement cast
inline elementwise_gfunc& GET(elementwise_gfunc_placement_wrapper& v)
{
    return *(elementwise_gfunc *)&v;
}

} // namespace pydnd

#endif // _DND__ELEMENTWISE_GFUNC_HPP_
