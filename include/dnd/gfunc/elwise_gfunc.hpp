//
// Copyright (C) 2011-12, Dynamic NDArray Developers
// BSD 2-Clause License, see LICENSE.txt
//

#ifndef _DND__ELWISE_GFUNC_HPP_
#define _DND__ELWISE_GFUNC_HPP_

#include <sstream>
#include <deque>
#include <vector>

#include <dnd/dtype.hpp>
#include <dnd/kernels/kernel_instance.hpp>
#include <dnd/codegen/codegen_cache.hpp>

namespace dynd { namespace gfunc {

class elwise_kernel {
public:
    dtype m_returntype;
    std::vector<dynd::dtype> m_paramtypes;
    dynd::unary_specialization_kernel_instance m_unary_kernel;
    dynd::kernel_instance<dynd::binary_operation_t> m_binary_kernel;

    void swap(elwise_kernel& rhs) {
        m_returntype.swap(rhs.m_returntype);
        m_paramtypes.swap(rhs.m_paramtypes);
        m_unary_kernel.swap(rhs.m_unary_kernel);
        m_binary_kernel.swap(rhs.m_binary_kernel);
    }
};

class elwise {
    std::string m_name;
    /**
     * This is a deque instead of a vector, because we are targetting C++98
     * and so cannot rely on C++11 move semantics.
     */
    std::deque<elwise_kernel> m_kernels;
    std::vector<dynd::memory_block_data *> m_blockrefs;
public:
    elwise(const char *name)
        : m_name(name)
    {
    }

    const std::string& get_name() const {
        return m_name;
    }

    /**
     * Searches for a kernel which matches all the parameter types.
     */
    const elwise_kernel *find_matching_kernel(const std::vector<dtype>& paramtypes) const;

    /**
     * Adds the provided kernel to the gfunc. This swaps it out of the provided
     * variable to avoid extra copies.
     */
    void add_kernel(elwise_kernel& egk);

    void debug_dump(std::ostream& o, const std::string& indent = "") const;
};

}} // namespace dynd::gfunc

#endif // _DND__ELWISE_GFUNC_HPP_
