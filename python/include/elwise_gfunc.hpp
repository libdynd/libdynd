//
// Copyright (C) 2011-12, Dynamic NDArray Developers
// BSD 2-Clause License, see LICENSE.txt
//

#ifndef _DND__ELWISE_GFUNC_HPP_
#define _DND__ELWISE_GFUNC_HPP_

#include <stdint.h>
#include <sstream>
#include <deque>
#include <vector>

#include <dnd/dtype.hpp>
#include <dnd/kernels/kernel_instance.hpp>
#include <dnd/codegen/codegen_cache.hpp>

#include <Python.h>

namespace pydnd {

class elwise_gfunc_kernel {
public:
    std::vector<dnd::dtype> m_sig;
    dnd::unary_specialization_kernel_instance m_unary_kernel;
    dnd::kernel_instance<dnd::binary_operation_t> m_binary_kernel;
    PyObject *m_pyobj;

    elwise_gfunc_kernel()
        : m_pyobj(0)
    {
    }

    ~elwise_gfunc_kernel();

    void swap(elwise_gfunc_kernel& rhs) {
        m_sig.swap(rhs.m_sig);
        m_unary_kernel.swap(rhs.m_unary_kernel);
        m_binary_kernel.swap(rhs.m_binary_kernel);
        std::swap(m_pyobj, rhs.m_pyobj);
    }
};

class elwise_gfunc {
    std::string m_name;
    /**
     * This is a deque instead of a vector, because we are targetting C++98
     * and so cannot rely on C++11 move semantics.
     */
    std::deque<elwise_gfunc_kernel> m_kernels;
    std::vector<dnd::memory_block_data *> m_blockrefs;
public:
    elwise_gfunc(const char *name)
        : m_name(name)
    {
    }

    ~elwise_gfunc();

    const std::string& get_name() const {
        return m_name;
    }

    /**
     * Adds a new memory_block for the elwise_gfunc to
     * hold a reference to. For example, the executable
     * memory block for generated code should get added.
     */
    void add_blockref(dnd::memory_block_data *blockref);

    void add_kernel(dnd::codegen_cache& cgcache, PyObject *kernel);

    PyObject *call(PyObject *args, PyObject *kwargs);

    std::string debug_dump() const;
};

struct elwise_gfunc_placement_wrapper {
    intptr_t dummy[(sizeof(elwise_gfunc) + sizeof(intptr_t) - 1)/sizeof(intptr_t)];
};

inline void elwise_gfunc_placement_new(elwise_gfunc_placement_wrapper& v, const char *name)
{
    // Call placement new
    new (&v) elwise_gfunc(name);
}

inline void elwise_gfunc_placement_delete(elwise_gfunc_placement_wrapper& v)
{
    // Call the destructor
    ((elwise_gfunc *)(&v))->~elwise_gfunc();
}

// placement cast
inline elwise_gfunc& GET(elwise_gfunc_placement_wrapper& v)
{
    return *(elwise_gfunc *)&v;
}

} // namespace pydnd

#endif // _DND__ELWISE_GFUNC_HPP_
