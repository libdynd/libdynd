//
// Copyright (C) 2011-14 Mark Wiebe, DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#ifndef _DYND__ARRFUNC_HPP_
#define _DYND__ARRFUNC_HPP_

#include <dynd/config.hpp>
#include <dynd/eval/eval_context.hpp>
#include <dynd/types/base_type.hpp>
#include <dynd/types/funcproto_type.hpp>
#include <dynd/types/arrfunc_type.hpp>
#include <dynd/kernels/ckernel_builder.hpp>

namespace dynd {

enum arrfunc_proto_t {
    unary_operation_funcproto,
    expr_operation_funcproto,
    binary_predicate_funcproto
};

struct arrfunc_type_data;

/**
 * Function prototype for instantiating a ckernel from an
 * arrfunc. To use this function, the
 * caller should first allocate a `ckernel_builder` instance,
 * either from C++ normally or by reserving appropriately aligned/sized
 * data and calling the C function constructor dynd provides. When the
 * data types of the kernel require metadata, such as for 'strided'
 * or 'var' dimension types, the metadata must be provided as well.
 *
 * \param self  The arrfunc.
 * \param ckb  A ckernel_builder instance where the kernel is placed.
 * \param ckb_offset  The offset into the output ckernel_builder `out_ckb`
 *                    where the kernel should be placed.
 * \param dst_tp  The destination type of the ckernel to generate. This may be
 *                different from the one in the function prototype, but must
 *                match its pattern.
 * \param dst_arrmeta  The destination arrmeta.
 * \param src_tp  An array of the source types of the ckernel to generate. These may be
 *                different from the ones in the function prototype, but must
 *                match the patterns.
 * \param src_arrmeta  An array of dynd arrmeta pointers,
 *                     corresponding to the source types.
 * \param kernreq  Either dynd::kernel_request_single or dynd::kernel_request_strided,
 *                  as required by the caller.
 * \param ectx  The evaluation context.
 *
 * \returns  The offset into ``ckb`` immediately after the instantiated ckernel.
 */
typedef intptr_t (*arrfunc_instantiate_t)(
    const arrfunc_type_data *self, dynd::ckernel_builder *ckb,
    intptr_t ckb_offset, const ndt::type &dst_tp, const char *dst_arrmeta,
    const ndt::type *src_tp, const char *const *src_arrmeta, uint32_t kernreq,
    const eval::eval_context *ectx);

/**
 * Resolves the destination type for this arrfunc based on the types
 * of the source parameters.
 *
 * \param self  The arrfunc.
 * \param out_dst_tp  To be filled with the destination type.
 * \param src_tp  An array of the source types.
 * \param throw_on_error  If true, should throw when there's an error, if
 *                        false, should return 0 when there's an error.
 *
 * \returns  True on success, false on error (if throw_on_error was false).
 */
typedef int (*arrfunc_resolve_dst_type_t)(const arrfunc_type_data *self,
                                          ndt::type &out_dst_tp,
                                          const ndt::type *src_tp,
                                          int throw_on_error);

/**
 * This is a struct designed for interoperability at
 * the C ABI level. It contains enough information
 * to pass arrfuncs from one library to another
 * with no dependencies between them.
 *
 * The arrfunc can produce a ckernel with with a few
 * variations, like choosing between a single
 * operation and a strided operation, or constructing
 * with different array metadata.
 */
struct arrfunc_type_data {
    /** The function prototype of the arrfunc */
    ndt::type func_proto;
    /** A value from the enumeration `arrfunc_proto_t`. */
    size_t ckernel_funcproto;
    /**
     * A pointer to typically heap-allocated memory for
     * the arrfunc. This is the value to be passed
     * in when calling instantiate_func and free_func.
     */
    void *data_ptr;
    /**
     * The function which instantiates a ckernel. See the documentation
     * for the function typedef for more details.
     */
    arrfunc_instantiate_t instantiate;
    arrfunc_resolve_dst_type_t resolve_dst_type;
    /**
     * A function which deallocates the memory behind data_ptr after
     * freeing any additional resources it might contain.
     */
    void (*free_func)(void *self_data_ptr);

    // Default to all NULL, so the destructor works correctly
    inline arrfunc_type_data()
        : func_proto(), ckernel_funcproto(0),
            data_ptr(0), instantiate(0), free_func(0)
    {
    }

    // If it contains an arrfunc, free it
    inline ~arrfunc_type_data()
    {
        if (free_func && data_ptr) {
            free_func(data_ptr);
        }
    }

    inline size_t get_param_count() const {
        return func_proto.tcast<funcproto_type>()->get_param_count();
    }

    inline const ndt::type *get_param_types() const {
        return func_proto.tcast<funcproto_type>()->get_param_types_raw();
    }

    inline const ndt::type &get_param_type(intptr_t i) const {
        return get_param_types()[i];
    }

    inline const ndt::type &get_return_type() const {
        return func_proto.tcast<funcproto_type>()->get_return_type();
    }
};

namespace nd {
/**
    * Holds a single instance of an arrfunc in an immutable nd::array,
    * providing some more direct convenient interface.
    */
class arrfunc {
    nd::array m_value;
public:
    inline arrfunc() : m_value() {}
    inline arrfunc(const arrfunc &rhs) : m_value(rhs.m_value) {}
    /**
     * Constructor from an nd::array. Validates that the input
     * has "arrfunc" type and is immutable.
     */
    arrfunc(const nd::array& rhs);

    inline arrfunc& operator=(const arrfunc& rhs) {
        m_value = rhs.m_value;
        return *this;
    }

    inline bool is_null() const {
        return m_value.is_null();
    }

    inline const arrfunc_type_data *get() const {
        return !m_value.is_null() ? reinterpret_cast<const arrfunc_type_data *>(
                                        m_value.get_readonly_originptr())
                                  : NULL;
    }

    inline operator nd::array() const {
        return m_value;
    }
};
} // namespace nd

/**
 * Creates an arrfunc which does the assignment from
 * data of src_tp to dst_tp.
 *
 * \param dst_tp  The type of the destination.
 * \param src_tp  The type of the source.
 * \param funcproto  The function prototype to generate (must be
 *                   unary_operation_funcproto or expr_operation_funcproto).
 * \param errmode  The error mode to use for the assignment.
 * \param out_af  The output `arrfunc` struct to be populated.
 */
void make_arrfunc_from_assignment(const ndt::type &dst_tp,
                                  const ndt::type &src_tp,
                                  arrfunc_proto_t funcproto,
                                  assign_error_mode errmode,
                                  arrfunc_type_data &out_af);

inline nd::arrfunc make_arrfunc_from_assignment(const ndt::type &dst_tp,
                                                const ndt::type &src_tp,
                                                arrfunc_proto_t funcproto,
                                                assign_error_mode errmode)
{
    nd::array af = nd::empty(ndt::make_arrfunc());
    make_arrfunc_from_assignment(
        dst_tp, src_tp, funcproto, errmode,
        *reinterpret_cast<arrfunc_type_data *>(af.get_readwrite_originptr()));
    af.flag_as_immutable();
    return af;
}

/**
 * Creates an arrfunc which does the assignment from
 * data of `tp` to its property `propname`
 *
 * \param tp  The type of the source.
 * \param propname  The name of the property.
 * \param funcproto  The function prototype to generate (must be
 *                   unary_operation_funcproto or expr_operation_funcproto).
 * \param out_af  The output `arrfunc` struct to be populated.
 */
void make_arrfunc_from_property(const ndt::type &tp,
                                const std::string &propname,
                                arrfunc_proto_t funcproto,
                                arrfunc_type_data &out_af);

inline nd::arrfunc make_arrfunc_from_property(const ndt::type &tp,
                                              const std::string &propname,
                                              arrfunc_proto_t funcproto)
{
    nd::array af = nd::empty(ndt::make_arrfunc());
    make_arrfunc_from_property(tp, propname, funcproto,
        *reinterpret_cast<arrfunc_type_data *>(af.get_readwrite_originptr()));
    af.flag_as_immutable();
    return af;
}

} // namespace dynd

#endif // _DYND__ARRFUNC_HPP_
