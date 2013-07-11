//
// Copyright (C) 2011-13 Mark Wiebe, DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

// Implement a number of string algorithms. Note that the initial 

#ifndef _DYND__STRING_ALGORITHM_KERNELS_HPP_
#define _DYND__STRING_ALGORITHM_KERNELS_HPP_

#include <dynd/kernels/kernel_instance.hpp>
#include <dynd/kernels/hierarchical_kernels.hpp>
#include <dynd/typed_data_assign.hpp>
#include <dynd/string_encodings.hpp>

namespace dynd { namespace kernels {

/**
 * String concatenation kernel.
 *
 * (string, string, ...) -> string
 */
struct string_concatenation_kernel {
    typedef string_concatenation_kernel extra_type;

    kernel_data_prefix m_base;
    // The number of input operands
    size_t m_nop;
    // The kernel borrows this reference from the dst metadata
    memory_block_data *m_dst_blockref;

    kernel_data_prefix& base() {
        return m_base;
    }

    /**
     * Initializes the kernel data.
     *
     * \param nop  This must be >= 2.
     * \param dst_metadata  Must be the destination for a "string" type (utf-8 string type).
     * \param src_metadata  Must be the two sources for "string" types.
     */
    void init(size_t nop, const char *dst_metadata, const char **src_metadata);

    static void single(char *dst, const char * const *src,
                kernel_data_prefix *extra);
    static void strided(char *dst, intptr_t dst_stride,
                const char * const *src, const intptr_t *src_stride,
                size_t count, kernel_data_prefix *extra);
};

}} // namespace dynd::kernels

#endif // _DYND__STRING_ALGORITHM_KERNELS_HPP_
