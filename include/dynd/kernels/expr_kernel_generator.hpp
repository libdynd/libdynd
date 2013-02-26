//
// Copyright (C) 2011-13, DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#ifndef _DYND__EXPR_KERNEL_GENERATOR_HPP_
#define _DYND__EXPR_KERNEL_GENERATOR_HPP_

#include <dynd/config.hpp>
#include <dynd/atomic_refcount.hpp>
#include <dynd/dtypes/type_id.hpp>
#include <dynd/kernels/assignment_kernels.hpp>
#include <dynd/eval/eval_context.hpp>

namespace dynd {

class dtype;
class expr_kernel_generator;

typedef void (*expr_single_operation_t)(
                char *dst, const char * const *src,
                kernel_data_prefix *extra);
typedef void (*expr_strided_operation_t)(
                char *dst, intptr_t dst_stride,
                const char * const *src, const intptr_t *src_stride,
                size_t count, kernel_data_prefix *extra);


/**
 * This is the memory structure for an object which
 * can generate an expression kernel.
 */
class expr_kernel_generator {
    /** Embedded reference counting */
    mutable atomic_refcount m_use_count;
public:
    expr_kernel_generator()
        : m_use_count(1)
    {
    }

    virtual ~expr_kernel_generator();

    virtual size_t make_expr_kernel(
                assignment_kernel *out, size_t offset_out,
                const dtype& dst_dt, const char *dst_metadata,
                size_t src_count, const dtype *src_dt, const char **src_metadata,
                kernel_request_t kernreq, const eval::eval_context *ectx) const = 0;

    friend void expr_kernel_generator_incref(const expr_kernel_generator *ed);
    friend void expr_kernel_generator_decref(const expr_kernel_generator *ed);
};

/**
 * Increments the reference count of a memory block object.
 */
inline void expr_kernel_generator_incref(const expr_kernel_generator *ed)
{
    ++ed->m_use_count;
}

/**
 * Decrements the reference count of a memory block object,
 * freeing it if the count reaches zero.
 */
inline void expr_kernel_generator_decref(const expr_kernel_generator *ed)
{
    if (--ed->m_use_count == 0) {
        delete ed;
    }
}

} // namespace dynd

#endif // _DYND__EXPR_KERNEL_GENERATOR_HPP_
