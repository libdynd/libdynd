//
// Copyright (C) 2011-14 Mark Wiebe, DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#ifndef _DYND__CKERNEL_PREFIX_HPP_
#define _DYND__CKERNEL_PREFIX_HPP_

#include <iostream>
#include <sstream>

#include <dynd/config.hpp>

namespace dynd {

struct ckernel_prefix;

/** Typedef for a unary operation on a single element */
typedef void (*unary_single_operation_t)(char *dst, const char *src,
                                         ckernel_prefix *self);
/** Typedef for a unary operation on a strided segment of elements */
typedef void (*unary_strided_operation_t)(char *dst, intptr_t dst_stride,
                                          const char *src, intptr_t src_stride,
                                          size_t count, ckernel_prefix *self);

typedef void (*expr_single_operation_t)(char *dst, const char *const *src,
                                        ckernel_prefix *extra);
typedef void (*expr_strided_operation_t)(char *dst, intptr_t dst_stride,
                                         const char *const *src,
                                         const intptr_t *src_stride,
                                         size_t count, ckernel_prefix *extra);

enum kernel_request_t {
    /** Kernel function unary_single_operation_t or expr_single_operation_t */
    kernel_request_single,
    /** Kernel function unary_strided_operation_t or expr_strided_operation_t*/
    kernel_request_strided,
    /**
     * Kernel function unary_single_operation_t,
     * but the data in the kernel at position 'offset_out'
     * is for data that describes the accumulation
     * of multiple strided dimensions that work
     * in a simple NumPy fashion.
     */
//    kernel_request_single_multistride,
    /**
     * Kernel function unary_strided_operation_t,
     * but the data in the kernel at position 'offset_out'
     * is for data that describes the accumulation
     * of multiple strided dimensions that work
     * in a simple NumPy fashion.
     */
//    kernel_request_strided_multistride
};

std::ostream& operator<<(std::ostream& o, kernel_request_t kernreq);

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

    inline void set_unary_function(kernel_request_t kernreq,
                                   unary_single_operation_t single,
                                   unary_strided_operation_t strided)
    {
        if (kernreq == kernel_request_single) {
            function = reinterpret_cast<void *>(single);
        } else if (kernreq == kernel_request_strided) {
            function = reinterpret_cast<void *>(strided);
        } else {
            std::stringstream ss;
            ss << "unrecognized unary kernel request type " << kernreq;
            throw std::runtime_error(ss.str());
        }
    }

    template<class T>
    inline void set_unary_function(kernel_request_t kernreq)
    {
        set_unary_function(kernreq, &T::single, &T::strided);
    }

    inline void set_expr_function(kernel_request_t kernreq,
                                  expr_single_operation_t single,
                                  expr_strided_operation_t strided)
    {
        if (kernreq == kernel_request_single) {
            function = reinterpret_cast<void *>(single);
        } else if (kernreq == kernel_request_strided) {
            function = reinterpret_cast<void *>(strided);
        } else {
            std::stringstream ss;
            ss << "unrecognized expr kernel request type " << kernreq;
            throw std::runtime_error(ss.str());
        }
    }

    template<class T>
    inline void set_expr_function(kernel_request_t kernreq)
    {
        set_expr_function(kernreq, &T::single, &T::strided);
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
