//
// Copyright (C) 2011-14 Mark Wiebe, DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#ifndef _DYND__ELWISE_GFUNC_HPP_
#define _DYND__ELWISE_GFUNC_HPP_

#include <sstream>
#include <deque>
#include <vector>

#include <dynd/type.hpp>
#include <dynd/codegen/codegen_cache.hpp>

namespace dynd { namespace gfunc {

class elwise_kernel {
public:
    ndt::type m_returntype;
    std::vector<dynd::ndt::type> m_paramtypes;
    //dynd::kernel_instance<unary_operation_pair_t> m_unary_kernel;
    //dynd::kernel_instance<dynd::binary_operation_pair_t> m_binary_kernel;

    void swap(elwise_kernel& rhs) {
        m_returntype.swap(rhs.m_returntype);
        m_paramtypes.swap(rhs.m_paramtypes);
//        m_unary_kernel.swap(rhs.m_unary_kernel);
//        m_binary_kernel.swap(rhs.m_binary_kernel);
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
    const elwise_kernel *find_matching_kernel(const std::vector<ndt::type>& paramtypes) const;

    /**
     * Adds the provided kernel to the gfunc. This swaps it out of the provided
     * variable to avoid extra copies.
     */
    void add_kernel(elwise_kernel& egk);

    void debug_print(std::ostream& o, const std::string& indent = "") const;
};

}} // namespace dynd::gfunc

#endif // _DYND__ELWISE_GFUNC_HPP_
