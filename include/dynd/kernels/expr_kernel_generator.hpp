//
// Copyright (C) 2011-15 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#pragma once

#include <dynd/config.hpp>
#include <dynd/types/type_id.hpp>
#include <dynd/eval/eval_context.hpp>
#include <dynd/kernels/base_kernel.hpp>

namespace dynd {

namespace ndt {
    class type;
} // namespace ndt
class DYND_API expr_kernel_generator;

struct DYND_API expr_operation_pair {
    expr_single_t single;
    expr_strided_t strided;
};

/**
 * This is the memory structure for an object which
 * can generate an expression kernel.
 */
class DYND_API expr_kernel_generator {
    /** Embedded reference counting */
    mutable std::atomic_long m_use_count;
    bool m_elwise;
public:
    expr_kernel_generator(bool elwise)
        : m_use_count(1), m_elwise(elwise)
    {
    }

    virtual ~expr_kernel_generator();

    inline bool is_elwise() const {
        return m_elwise;
    }

    virtual size_t make_expr_kernel(
                void *ckb, intptr_t ckb_offset,
                const ndt::type& dst_tp, const char *dst_arrmeta,
                size_t src_count, const ndt::type *src_dt, const char *const*src_arrmeta,
                kernel_request_t kernreq, const eval::eval_context *ectx) const = 0;

    /** Used to print information about the kernel in the type */
    virtual void print_type(std::ostream& o) const = 0;

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
