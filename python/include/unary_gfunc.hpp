//
// Copyright (C) 2011-12, Dynamic NDArray Developers
// BSD 2-Clause License, see LICENSE.txt
//

#ifndef _DND__UNARY_GFUNC_HPP_
#define _DND__UNARY_GFUNC_HPP_

#include <stdint.h>
#include <sstream>
#include <deque>
#include <vector>

#include <dnd/dtype.hpp>
#include <dnd/kernels/kernel_instance.hpp>
#include <dnd/codegen/codegen_cache.hpp>

#include <Python.h>

namespace pydnd {

class unary_gfunc_kernel {
public:
    dnd::dtype m_out, m_params[1];
    dnd::unary_specialization_kernel_instance m_kernel;
    PyObject *m_pyobj;

    unary_gfunc_kernel()
        : m_pyobj(0)
    {
    }

    ~unary_gfunc_kernel();

    void swap(unary_gfunc_kernel& rhs) {
        m_out.swap(rhs.m_out);
        m_params[0].swap(rhs.m_params[0]);
        m_kernel.swap(rhs.m_kernel);
    }
};

class unary_gfunc {
    std::string m_name;
    /**
     * This is a deque instead of a vector, because we are targetting C++98
     * and so cannot rely on C++11 move semantics.
     */
    std::deque<unary_gfunc_kernel> m_kernels;
    std::vector<dnd::memory_block_data *> m_blockrefs;
public:
    unary_gfunc(const char *name)
        : m_name(name)
    {
    }

    ~unary_gfunc();

    const std::string& get_name() const {
        return m_name;
    }

    /**
     * Adds a new memory_block for the unary_gfunc to
     * hold a reference to. For example, the executable
     * memory block for generated code should get added.
     */
    void add_blockref(dnd::memory_block_data *blockref);

    void add_kernel(dnd::codegen_cache& cgcache, PyObject *kernel);

    PyObject *call(PyObject *args, PyObject *kwargs);

    std::string debug_dump() const;
};

struct unary_gfunc_placement_wrapper {
    intptr_t dummy[(sizeof(unary_gfunc) + sizeof(intptr_t) - 1)/sizeof(intptr_t)];
};

inline void unary_gfunc_placement_new(unary_gfunc_placement_wrapper& v, const char *name)
{
    // Call placement new
    new (&v) unary_gfunc(name);
}

inline void unary_gfunc_placement_delete(unary_gfunc_placement_wrapper& v)
{
    // Call the destructor
    ((unary_gfunc *)(&v))->~unary_gfunc();
}

// dtype placement cast
inline unary_gfunc& GET(unary_gfunc_placement_wrapper& v)
{
    return *(unary_gfunc *)&v;
}

} // namespace pydnd

#endif // _DND__UNARY_GFUNC_HPP_
