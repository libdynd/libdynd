//
// Copyright (C) 2011-13 Mark Wiebe, DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#ifndef _DYND__CKERNEL_DEFERRED_HPP_
#define _DYND__CKERNEL_DEFERRED_HPP_

#include <dynd/config.hpp>
#include <dynd/types/base_type.hpp>
#include <dynd/kernels/ckernel_prefix.hpp>

namespace dynd {

enum deferred_ckernel_funcproto_t {
    unary_operation_funcproto,
    expr_operation_funcproto,
    binary_predicate_funcproto
};

/**
 * Function prototype for instantiating a ckernel from a
 * ckernel_deferred (ckd). To use this function, the
 * caller should first allocate the appropriate
 * amount of memory (ckd->ckernel_size) with the alignment
 * required (sizeof(void *)). When the data types of the kernel
 * require metadata, such as for 'strided' or 'var' dimension types,
 * the metadata must be provided as well.
 *
 * \param self_data_ptr  This is ckd->data_ptr.
 * \param out_ckernel  This is where the ckernel is placed.
 * \param dynd_metadata  An array of dynd metadata pointers,
 *                       corresponding to ckd->data_dynd_types.
 * \param kerntype  Either dynd::kernel_request_single or dynd::kernel_request_strided,
 *                  as required by the caller.
 */
typedef void (*instantiate_deferred_ckernel_fn_t)(void *self_data_ptr,
                dynd::ckernel_prefix *out_ckernel,
                const char *const* dynd_metadata, uint32_t kerntype);



/**
 * This is a struct designed for interoperability at
 * the C ABI level. It contains enough information
 * to pass deferred kernels from one library to another
 * with no dependencies between them.
 *
 * The deferred kernel can produce a ckernel with with a few
 * variations, like choosing between a single
 * operation and a strided operation, or constructing
 * with different array metadata.
 */
struct ckernel_deferred {
    /** A value from the enumeration `deferred_ckernel_funcproto_t`. */
    size_t ckernel_funcproto;
    /** The size of the ckernel which this object instantiates. */
    size_t ckernel_size;
    /**
     * The number of types in the data_types array. This is used to
     * determine how many operands there are for the `expr_operation_funcproto`,
     * for example.
     */
    size_t data_types_size;
    /**
     * An array of dynd types for the kernel's data pointers.
     * Note that the builtin dynd types are stored as
     * just the type ID, so cases like bool, int float
     * can be done very simply.
     *
     * This data for this array should be either be static,
     * or contained within the memory of data_ptr.
     */
    const dynd::base_type * const* data_dynd_types;
    /**
     * A pointer to typically heap-allocated memory for
     * the deferred ckernel. This is the value to be passed
     * in when calling instantiate_func and free_func.
     */
    void *data_ptr;
    /**
     * The function which instantiates a ckernel. See the documentation
     * for the function typedef for more details.
     */
    instantiate_deferred_ckernel_fn_t instantiate_func;
    /**
     * A function which deallocates the memory behind data_ptr after
     * freeing any additional resources it might contain.
     */
    void (*free_func)(void *self_data_ptr);
};


} // namespace dynd

#endif // _DYND__CKERNEL_DEFERRED_HPP_
