//
// Copyright (C) 2011-14 Mark Wiebe, DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#ifndef _DYND__ELWISE_REDUCE_GFUNC_HPP_
#define _DYND__ELWISE_REDUCE_GFUNC_HPP_

#include <sstream>
#include <deque>
#include <vector>

#include <dynd/type.hpp>
#include <dynd/array.hpp>
#include <dynd/codegen/codegen_cache.hpp>

namespace dynd { namespace gfunc {

class elwise_reduce_kernel {
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
    ndt::type m_returntype;
    std::vector<dynd::ndt::type> m_paramtypes;
    dynd::nd::array m_identity;
    /**
     * Does dst <- operation(dst, src), use when iterating from index 0 to N-1.
     */
//    dynd::kernel_instance<dynd::unary_operation_pair_t> m_left_associative_reduction_kernel;
    /**
     * Does dst <- operation(src, dst), use when iterating from index N-1 to 0.
     * If the kernel is flagged commutative, this kernel is never used so may be left empty.
     */
//    dynd::kernel_instance<dynd::unary_operation_pair_t> m_right_associative_reduction_kernel;

    void swap(elwise_reduce_kernel& rhs) {
        std::swap(m_associative, rhs.m_associative);
        std::swap(m_commutative, rhs.m_commutative);
        m_returntype.swap(rhs.m_returntype);
        m_paramtypes.swap(rhs.m_paramtypes);
//        m_left_associative_reduction_kernel.swap(rhs.m_left_associative_reduction_kernel);
//        m_right_associative_reduction_kernel.swap(rhs.m_right_associative_reduction_kernel);
        m_identity.swap(rhs.m_identity);
    }
};

class elwise_reduce {
    std::string m_name;
    /**
     * This is a deque instead of a vector, because we are targeting C++98
     * and so cannot rely on C++11 move semantics.
     */
    std::deque<elwise_reduce_kernel> m_kernels;
public:
    elwise_reduce(const char *name)
        : m_name(name)
    {
    }

    const std::string& get_name() const {
        return m_name;
    }

    /**
     * Searches for a kernel which matches all the parameter types.
     */
    const elwise_reduce_kernel *find_matching_kernel(const std::vector<ndt::type>& paramtypes) const;

    /**
     * Adds the provided kernel to the gfunc. This swaps it out of the provided
     * variable to avoid extra copies.
     */
    void add_kernel(elwise_reduce_kernel& ergk);

    void debug_print(std::ostream& o, const std::string& indent = "") const;
};

}} // namespace dynd::gfunc

#endif // _DYND__ELWISE_REDUCE_GFUNC_HPP_
