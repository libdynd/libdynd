//
// Copyright (C) 2011-14 Mark Wiebe, DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#ifndef _DYND__CKERNEL_PREFIX_HPP_
#define _DYND__CKERNEL_PREFIX_HPP_

#include <dynd/config.hpp>

namespace dynd {

/**
 * This is the struct which begins the memory layout
 * of all ckernels. First comes the function pointer,
 * which has a context-specific prototype, such as
 * `unary_single_operation_t`, and then comes the
 * destructor.
 *
 * The ckernel is defined in terms of a C ABI definition,
 * and must satisfy alignments, movability, and
 * concurrency requirements to be valid. See the
 * document 
 */
struct ckernel_prefix {
    typedef void (*destructor_fn_t)(ckernel_prefix *);

    void *function;
    destructor_fn_t destructor;

    /**
     * To help with generic code a bit, structs which
     * begin with a ckernel_prefix can define this
     * base() function which returns that ckernel_prefix.
     */
    inline ckernel_prefix& base() {
        return *this;
    }

    /**
     * Call to get the kernel function pointer, whose type
     * must be known by the context.
     *
     *      kdp->get_function<unary_single_operation_t>()
     */
    template<typename T>
    inline T get_function() const {
        return reinterpret_cast<T>(function);
    }

    template<typename T>
    inline void set_function(T fnptr) {
        function = reinterpret_cast<void *>(fnptr);
    }

    /**
     * Calls the destructor of the ckernel if it is
     * non-NULL.
     */
    inline void destroy() {
        if (destructor != NULL) {
            destructor(this);
        }
    }

    /**
     * Returns the pointer to a child ckernel at the provided
     * offset.
     */
    inline ckernel_prefix *get_child_ckernel(size_t offset) {
        return reinterpret_cast<ckernel_prefix *>(
            reinterpret_cast<char *>(this) + offset);
    }

    /**
     * If the provided offset is non-zero, destroys
     * a ckernel at the given offset from `this`.
     */
    inline void destroy_child_ckernel(size_t offset) {
        if (offset != 0) {
            ckernel_prefix *child = reinterpret_cast<ckernel_prefix *>(
                reinterpret_cast<char *>(this) + offset);
            child->destroy();
        }
    }
};

} // namespace dynd

#endif // _DYND__CKERNEL_PREFIX_HPP_
