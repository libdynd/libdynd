//
// Copyright (C) 2011-13 Mark Wiebe, DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#ifndef _DYND__CKERNEL_DEFERRED_HPP_
#define _DYND__CKERNEL_DEFERRED_HPP_

#include <dynd/config.hpp>
#include <dynd/types/base_type.hpp>
#include <dynd/kernels/ckernel_builder.hpp>

namespace dynd {

enum deferred_ckernel_funcproto_t {
    unary_operation_funcproto,
    expr_operation_funcproto,
    binary_predicate_funcproto
};

/**
 * Function prototype for instantiating a ckernel from a
 * ckernel_deferred (ckd). To use this function, the
 * caller should first allocate a `ckernel_builder` instance,
 * either from C++ normally or by reserving appropriately aligned/sized
 * data and calling the C function constructor dynd provides. When the
 * data types of the kernel require metadata, such as for 'strided'
 * or 'var' dimension types, the metadata must be provided as well.
 *
 * \param self_data_ptr  This is ckd->data_ptr.
 * \param out_ckb  A ckernel_builder instance where the kernel is placed.
 * \param ckb_offset  The offset into the output ckernel_builder `out_ckb`
 *                    where the kernel should be placed.
 * \param dynd_metadata  An array of dynd metadata pointers,
 *                       corresponding to ckd->data_dynd_types.
 * \param kerntype  Either dynd::kernel_request_single or dynd::kernel_request_strided,
 *                  as required by the caller.
 */
typedef intptr_t (*instantiate_deferred_ckernel_fn_t)(void *self_data_ptr,
                dynd::ckernel_builder *out_ckb, intptr_t ckb_offset,
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
    /**
     * The number of types in the data_types array. This is used to
     * determine how many operands there are for the `expr_operation_funcproto`,
     * for example.
     */
    intptr_t data_types_size;
    /**
     * An array of dynd types for the kernel's data pointers.
     * Note that the builtin dynd types are stored as
     * just the type ID, so cases like bool, int float
     * can be done very simply.
     *
     * This data for this array should be either be static,
     * or contained within the memory of data_ptr.
     */
    const ndt::type *data_dynd_types;
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

    // Default to all NULL, so the destructor works correctly
    inline ckernel_deferred()
        : ckernel_funcproto(0), data_types_size(0), data_dynd_types(0),
            data_ptr(0), instantiate_func(0), free_func(0)
    {
    }

    // If it contains a deferred ckernel, free it
    inline ~ckernel_deferred()
    {
        if (free_func && data_ptr) {
            free_func(data_ptr);
        }
    }
};

/**
 * Creates a deferred ckernel which does the assignment from
 * data of src_tp to dst_tp.
 *
 * \param dst_tp  The type of the destination.
 * \param src_tp  The type of the source.
 * \param src_prop_tp  If different from src_tp, this is a type whose
 * \param funcproto  The function prototype to generate (must be
 *                   unary_operation_funcproto or expr_operation_funcproto).
 * \param errmode  The error mode to use for the assignment.
 * \param out_ckd  The output `ckernel_deferred` struct to be populated.
 * \param ectx  The evaluation context.
 */
void make_ckernel_deferred_from_assignment(
                const ndt::type& dst_tp, const ndt::type& src_tp, const ndt::type& src_prop_tp,
                deferred_ckernel_funcproto_t funcproto,
                assign_error_mode errmode, ckernel_deferred& out_ckd,
                const dynd::eval::eval_context *ectx = &dynd::eval::default_eval_context);

/**
 * Creates a deferred ckernel which does the assignment from
 * data of `tp` to its property `propname`
 *
 * \param tp  The type of the source.
 * \param propname  The name of the property.
 * \param funcproto  The function prototype to generate (must be
 *                   unary_operation_funcproto or expr_operation_funcproto).
 * \param errmode  The error mode to use for the assignment.
 * \param out_ckd  The output `ckernel_deferred` struct to be populated.
 * \param ectx  The evaluation context.
 */
void make_ckernel_deferred_from_property(const ndt::type& tp, const std::string& propname,
                deferred_ckernel_funcproto_t funcproto,
                assign_error_mode errmode, ckernel_deferred& out_ckd,
                const dynd::eval::eval_context *ectx = &dynd::eval::default_eval_context);

} // namespace dynd

#endif // _DYND__CKERNEL_DEFERRED_HPP_
