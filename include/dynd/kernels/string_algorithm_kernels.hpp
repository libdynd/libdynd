//
// Copyright (C) 2011-14 Mark Wiebe, DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

// Implement a number of string algorithms. Note that the initial 

#ifndef _DYND__STRING_ALGORITHM_KERNELS_HPP_
#define _DYND__STRING_ALGORITHM_KERNELS_HPP_

#include <dynd/kernels/ckernel_builder.hpp>
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

    ckernel_prefix m_base;
    // The number of input operands
    size_t m_nop;
    // The kernel borrows this reference from the dst arrmeta
    memory_block_data *m_dst_blockref;

    ckernel_prefix& base() {
        return m_base;
    }

    /**
     * Initializes the kernel data.
     *
     * \param nop  This must be >= 2.
     * \param dst_arrmeta  Must be the destination for a "string" type (utf-8 string type).
     * \param src_arrmeta  Must be the two sources for "string" types.
     */
    void init(size_t nop, const char *dst_arrmeta, const char **src_arrmeta);

    static void single(char *dst, const char *const *src,
                       ckernel_prefix *extra);
    static void strided(char *dst, intptr_t dst_stride, const char *const *src,
                        const intptr_t *src_stride, size_t count,
                        ckernel_prefix *extra);
};

/**
 * String find kernel, which searches the whole string.
 *
 * (string, string) -> intp
 */
struct string_find_kernel {
    typedef string_find_kernel extra_type;

    ckernel_prefix m_base;
    // The string type being searched through
    const base_string_type *m_str_type;
    const char *m_str_arrmeta;
    // The substring type being searched for
    const base_string_type *m_sub_type;
    const char *m_sub_arrmeta;

    ckernel_prefix& base() {
        return m_base;
    }

    /**
     * Initializes the kernel data.
     *
     * \param src_tp        The array of two src types.
     * \param src_arrmeta  The array of two src arrmeta.
     */
    void init(const ndt::type *src_tp, const char *const *src_arrmeta);

    static void destruct(ckernel_prefix *extra);

    static void single(char *dst, const char *const *src,
                       ckernel_prefix *extra);
    static void strided(char *dst, intptr_t dst_stride, const char *const *src,
                        const intptr_t *src_stride, size_t count,
                        ckernel_prefix *extra);
};


}} // namespace dynd::kernels

#endif // _DYND__STRING_ALGORITHM_KERNELS_HPP_
