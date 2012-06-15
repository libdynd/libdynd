//
// Copyright (C) 2011-12, Dynamic NDArray Developers
// BSD 2-Clause License, see LICENSE.txt
//

#ifndef _DND__UNARY_GFUNC_HPP_
#define _DND__UNARY_GFUNC_HPP_

#include <stdint.h>
#include <sstream>
#include <deque>

#include <dnd/dtype.hpp>
#include <dnd/kernels/kernel_instance.hpp>

#include <Python.h>

namespace pydnd {

class unary_gfunc_kernel {
public:
    dnd::dtype m_out, m_params[1];
    dnd::kernel_instance<dnd::unary_operation_t> m_kernel;

    void swap(unary_gfunc_kernel& rhs) {
        m_out.swap(rhs.m_out);
        m_params[0].swap(rhs.m_params[0]);
        m_kernel.swap(rhs.m_kernel);
    }
};

class unary_gfunc {
    std::string m_name;
    std::deque<unary_gfunc_kernel> m_kernels;
public:
    unary_gfunc(const char *name)
        : m_name(name)
    {
    }

    const std::string& get_name() const {
        return m_name;
    }

    void add_kernel(PyObject *kernel);

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