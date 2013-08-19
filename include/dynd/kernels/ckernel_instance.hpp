//
// Copyright (C) 2011-13 Mark Wiebe, DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#ifndef _DYND__CKERNEL_INSTANCE_HPP_
#define _DYND__CKERNEL_INSTANCE_HPP_

#include <dynd/config.hpp>
#include <dynd/kernels/ckernel_prefix.hpp>

namespace dynd {

/**
 * This is a struct designed for interoperability at
 * the C ABI level. It contains enough information
 * to pass kernels from one library to another with
 * no dependencies between them.
 *
 * To free a ckernel instance, one must
 * first call the destructor in the ckernel_prefix,
 * then call the free_func to deallocate the memory.
 * The function free_ckernel_instance is provided
 * here for this purpose.
 */
struct ckernel_instance {
    /** Pointer to dynamically allocated kernel data */
    ckernel_prefix *kernel;
    /**
     * How many bytes in the kernel data. Because the
     * kernel data must be movable, one may move the
     * data somewhere else, and then free the kernel memory
     * without calling the kernel destructor.
     *
     * This allows for ckernels which are not complete,
     * e.g. a kernel which handles some leading dimensions,
     * and is expecting a child kernel immediately after
     * it to handle the elements.
     */
    size_t kernel_size;
    /** Pointer to a function for freeing 'kernel'. */
    void (*free_func)(void *);
};

inline void free_ckernel_instance(ckernel_instance& cki)
{
    if (cki.kernel != NULL) {
        if (cki.kernel->destructor != NULL) {
            cki.kernel->destructor(cki.kernel);
        }
        cki.free_func(cki.kernel);
        cki.kernel = NULL;
    }
}

} // namespace dynd

#endif // _DYND__CKERNEL_INSTANCE_HPP_
