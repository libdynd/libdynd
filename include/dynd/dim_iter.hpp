//
// Copyright (C) 2011-14 Mark Wiebe, DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#ifndef _DYND__DIM_ITER_HPP_
#define _DYND__DIM_ITER_HPP_

#include <dynd/config.hpp>
#include <dynd/types/base_type.hpp>

namespace dynd {

struct dim_iter;

enum dim_iter_flags {
    dim_iter_restartable = 0x001,
    dim_iter_seekable = 0x002,
    dim_iter_contiguous = 0x004
};

/**
 * Table of functions for a `dim_iter` instance.
 */
struct dim_iter_vtable {
    /** Destructor */
    void (*destructor)(dim_iter *self);
    /** Function to advance the iterator. Return 1 if elements are availabe, 0 otherwise */
    int (*next)(dim_iter *self);
    /** Function to seek the iterator to an index */
    void (*seek)(dim_iter *self, intptr_t i);
};

/**
 * The `dim_iter` object provides an iterator abstraction defined
 * in terms of the C ABI of the system. It's intended to be produced
 * and consumed both within DyND and by external C and dynamically JIT
 * compiled code using DyND. One way to think of it is analogous to
 * NumPy's NpyIter, restricted to one operand and one dimension.
 *
 * This iterator iterates in one dimension, with some capabilities flags.
 */
struct dim_iter {
    /** The table of functions*/
    const dim_iter_vtable *vtable;
    /** Pointer to the data. May be inside the array, or a temporary buffer */
    const char *data_ptr;
    /** The number of elements available in the buffer */
    intptr_t data_elcount;
    /** The stride, measured in bytes, for advancing the data pointer */
    intptr_t data_stride;
    /**
     * Flags about the iterator.
     *   0x01 : dim_iter_restartable : can call seek(0) on the iterator
     *   0x02 : dim_iter_seekable    : can call seek(N) on the iterator
     *   0x04 : dim_iter_contiguous  : the stride of the iterator will always be contiguous
     */
    uint64_t flags;
    /** The type of one element */
    const base_type *eltype;
    /** Array arrmeta that each element conforms to. */
    const char *el_arrmeta;
    /**
     * Some space the creator of the iterator can use. If all additional data
     * fits here, a memory allocation can be avoided, otherwise a dynamically
     * allocated buffer can be referenced from here.
     */
    uintptr_t custom[8];

    dim_iter()
        : vtable(NULL)
    {
    }

    inline void destroy() {
        if (vtable) {
            vtable->destructor(this);
            vtable = NULL;
        }
    }

    ~dim_iter() {
        destroy();
    }
};

/**
 * Creates a dim_iter for a strided dimension, which iterates directly
 * on the strided element memory.
 *
 * If the type is an expression type,
 * and you want to iterate over value elements, use
 * make_buffered_strided_dim_iter.
 *
 * \param out_di  An uninitialized dim_iter object. The function
 *                populates it assuming it is filled with garbage.
 * \param tp  The type of the elements being iterated
 * \param arrmeta  The arrmeta corresponding to `tp`.
 * \param data_ptr  The data pointer of element 0.
 * \param size  The dimension size.
 * \param stride  The stride between elements.
 * \param ref  A reference which holds the memory.
 */
void make_strided_dim_iter(
    dim_iter *out_di,
    const ndt::type& tp, const char *arrmeta,
    const char *data_ptr, intptr_t size, intptr_t stride,
    const memory_block_ptr& ref);

/**
 * Creates a dim_iter for a strided dimension, using buffering to
 * provide elements of the requested value type. Typically, val_tp
 * will be the value type of mem_tp, but this is not required.
 *
 * \param out_di  An uninitialized dim_iter object. The function
 *                populates it assuming it is filled with garbage.
 * \param val_tp  The type of the elements the iterator should produce.
 * \param mem_tp  The type of elements in memory.
 * \param mem_arrmeta  The arrmeta for mem_tp.
 * \param data_ptr  The data pointer of element 0.
 * \param size  The dimension size.
 * \param stride  The stride between elements.
 * \param ref  A reference which holds the memory.
 * \param buffer_max_mem  The maximum amount of memory to use for the temporary buffer.
 * \param ectx  The evaluation context.
 */
void make_buffered_strided_dim_iter(
    dim_iter *out_di,
    const ndt::type& val_tp,
    const ndt::type& mem_tp, const char *mem_arrmeta,
    const char *data_ptr, intptr_t size, intptr_t stride,
    const memory_block_ptr& ref, intptr_t buffer_max_mem = 65536,
    const eval::eval_context *ectx = &eval::default_eval_context);

/**
 * Makes an iterator which is empty.
 *
 * \param out_di  An uninitialized dim_iter object. The function
 *                populates it assuming it is filled with garbage.
 * \param tp  The type of the elements being iterated.
 * \param arrmeta  The arrmeta corresponding to `tp`.
 */
inline void make_empty_dim_iter(
    dim_iter *out_di,
    const ndt::type& tp,
    const char *arrmeta)
{
    make_strided_dim_iter(out_di, tp, arrmeta,
        NULL, 0, 0, memory_block_ptr());
}

} // namespace dynd

#endif // _DYND__DIM_ITER_HPP_
