//
// Copyright (C) 2011-12, Dynamic NDArray Developers
// BSD 2-Clause License, see LICENSE.txt
//

#ifndef _DND__ELWISE_REDUCE_GFUNC_HPP_
#define _DND__ELWISE_REDUCE_GFUNC_HPP_

#include <Python.h>

#include <stdint.h>
#include <sstream>
#include <deque>
#include <vector>

#include <dnd/dtype.hpp>
#include <dnd/ndarray.hpp>
#include <dnd/kernels/kernel_instance.hpp>
#include <dnd/codegen/codegen_cache.hpp>

namespace pydnd {

class elwise_reduce_gfunc_kernel {
public:
    /**
     * If the kernel is associative, evaluating right-to-left
     * and left-to-right are equivalent.
     */
    bool m_associative;
    /**
     * If the kernel is commutative, multidimensional reduction is ok,
     * and the left/right kernels are equivalent, so just a left associating
     * kernel is provided.
     */
    bool m_commutative;
    std::vector<dnd::dtype> m_sig;
    dnd::ndarray m_identity;
    /**
     * Does dst <- operation(dst, src), use when iterating from index 0 to N-1.
     */
    dnd::kernel_instance<dnd::unary_operation_t> m_left_associative_reduction_kernel;
    /**
     * Does dst <- operation(src, dst), use when iterating from index N-1 to 0.
     * If the kernel is flagged commutative, this kernel is never used so may be left empty.
     */
    dnd::kernel_instance<dnd::unary_operation_t> m_right_associative_reduction_kernel;
    PyObject *m_pyobj;

    elwise_reduce_gfunc_kernel()
        : m_pyobj(0)
    {
    }

    ~elwise_reduce_gfunc_kernel();

    void swap(elwise_reduce_gfunc_kernel& rhs) {
        std::swap(m_associative, rhs.m_associative);
        std::swap(m_commutative, rhs.m_commutative);
        m_sig.swap(rhs.m_sig);
        m_left_associative_reduction_kernel.swap(rhs.m_left_associative_reduction_kernel);
        m_right_associative_reduction_kernel.swap(rhs.m_right_associative_reduction_kernel);
        std::swap(m_pyobj, rhs.m_pyobj);
        m_identity.swap(rhs.m_identity);
    }
};

class elwise_reduce_gfunc {
    std::string m_name;
    /**
     * This is a deque instead of a vector, because we are targetting C++98
     * and so cannot rely on C++11 move semantics.
     */
    std::deque<elwise_reduce_gfunc_kernel> m_kernels;
    std::vector<dnd::memory_block_data *> m_blockrefs;
public:
    elwise_reduce_gfunc(const char *name)
        : m_name(name)
    {
    }

    ~elwise_reduce_gfunc();

    const std::string& get_name() const {
        return m_name;
    }

    /**
     * Adds a new memory_block for the elwise_gfunc to
     * hold a reference to. For example, the executable
     * memory block for generated code should get added.
     */
    void add_blockref(dnd::memory_block_data *blockref);

    void add_kernel(dnd::codegen_cache& cgcache, PyObject *kernel, bool associative, bool commutative, const dnd::ndarray& identity);

    PyObject *call(PyObject *args, PyObject *kwargs);

    std::string debug_dump() const;
};

struct elwise_reduce_gfunc_placement_wrapper {
    intptr_t dummy[(sizeof(elwise_reduce_gfunc) + sizeof(intptr_t) - 1)/sizeof(intptr_t)];
};

inline void elwise_reduce_gfunc_placement_new(elwise_reduce_gfunc_placement_wrapper& v, const char *name)
{
    // Call placement new
    new (&v) elwise_reduce_gfunc(name);
}

inline void elwise_reduce_gfunc_placement_delete(elwise_reduce_gfunc_placement_wrapper& v)
{
    // Call the destructor
    ((elwise_reduce_gfunc *)(&v))->~elwise_reduce_gfunc();
}

// placement cast
inline elwise_reduce_gfunc& GET(elwise_reduce_gfunc_placement_wrapper& v)
{
    return *(elwise_reduce_gfunc *)&v;
}

} // namespace pydnd

#endif // _DND__ELWISE_REDUCE_GFUNC_HPP_
