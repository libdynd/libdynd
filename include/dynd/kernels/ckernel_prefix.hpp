//
// Copyright (C) 2011-13 Mark Wiebe, DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#ifndef _DYND__CKERNEL_PREFIX_HPP_
#define _DYND__CKERNEL_PREFIX_HPP_

// For this header, we define both a C and a C++
// interface
#if !defined(__cplusplus)

struct dynd_ckernel_prefix_ {
    void *function;
    void (*destructor)(struct dynd_ckernel_prefix_ *);
};

typedef void (*dynd_ckernel_destructor_fn_t)(struct dynd_ckernel_prefix_ *);
typedef struct dynd_ckernel_prefix_ dynd_ckernel_prefix;

#else // if defined(__cplusplus)

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
    ckernel_prefix& base() {
        return *this;
    }

    /**
     * Call to get the kernel function pointer, whose type
     * must be known by the context.
     *
     *      kdp->get_function<unary_single_operation_t>()
     */
    template<typename T>
    T get_function() const {
        return reinterpret_cast<T>(function);
    }

    template<typename T>
    void set_function(T fnptr) {
        function = reinterpret_cast<void *>(fnptr);
    }

    template<typename T, typename U, typename V>
    void init(const T&, const U&, const V&) {
    }
};

} // namespace dynd

#endif // defined(__cplusplus)

#endif // _DYND__CKERNEL_PREFIX_HPP_
