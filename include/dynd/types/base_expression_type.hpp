//
// Copyright (C) 2011-13 Mark Wiebe, DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#ifndef _DYND__BASE_EXPRESSION_TYPE_HPP_
#define _DYND__BASE_EXPRESSION_TYPE_HPP_

#include <dynd/types/base_type.hpp>

namespace dynd {

/**
 * Base class for all types of expression_kind, and for the pointer_type.
 */
class base_expression_type : public base_type {
public:
    inline base_expression_type(type_id_t type_id, type_kind_t kind,
                    size_t data_size, size_t alignment, flags_type flags, size_t metadata_size, size_t undim=0)
        : base_type(type_id, kind, data_size, alignment, flags, metadata_size, undim)
    {}

    virtual ~base_expression_type();

    /**
     * Should return a reference to the type representing the value which
     * is for calculation. This should never be an expression type.
     */
    virtual const ndt::type& get_value_type() const = 0;
    /**
     * Should return a reference to a type representing the data this type
     * uses to produce the value.
     */
    virtual const ndt::type& get_operand_type() const = 0;

    /**
     * Returns a flags value which inherits the appropriate flags from
     * the value and operand types.
     */
    static inline flags_type inherited_flags(flags_type value_flags,
                    flags_type operand_flags)
    {
        return (value_flags&type_flags_value_inherited)|
                        (operand_flags&type_flags_operand_inherited);
    }

    /**
     * This method is for expression types, and is a way to substitute
     * the storage type (deepest operand type) of an existing type.
     *
     * The value_type of the replacement should match the storage type
     * of this instance. Implementations of this should raise an exception
     * when this is not true.
     */
    virtual ndt::type with_replaced_storage_type(const ndt::type& replacement_type) const = 0;

    // Always return true for expression types
    bool is_expression() const;

    // The canonical type for expression types is always the value type
    ndt::type get_canonical_type() const;

    // Expression types use the values from their operand type.
    void metadata_default_construct(char *metadata, intptr_t ndim, const intptr_t* shape) const;
    void metadata_copy_construct(char *dst_metadata, const char *src_metadata, memory_block_data *embedded_reference) const;
    void metadata_destruct(char *metadata) const;
    void metadata_debug_print(const char *metadata, std::ostream& o, const std::string& indent) const;

    // Expression types stop the iterdata chain
    // TODO: Maybe it should be more flexible?
    size_t get_iterdata_size(intptr_t ndim) const;

    /** Makes a kernel which converts from (operand_type().value_type()) to (value_type()) */
    virtual size_t make_operand_to_value_assignment_kernel(
                    ckernel_builder *out, size_t offset_out,
                    const char *dst_metadata, const char *src_metadata,
                    kernel_request_t kernreq, const eval::eval_context *ectx) const;

    /** Makes a kernel which converts from (value_type()) to (operand_type().value_type()) */
    virtual size_t make_value_to_operand_assignment_kernel(
                    ckernel_builder *out, size_t offset_out,
                    const char *dst_metadata, const char *src_metadata,
                    kernel_request_t kernreq, const eval::eval_context *ectx) const;

    size_t make_assignment_kernel(
                    ckernel_builder *out, size_t offset_out,
                    const ndt::type& dst_tp, const char *dst_metadata,
                    const ndt::type& src_tp, const char *src_metadata,
                    kernel_request_t kernreq, assign_error_mode errmode,
                    const eval::eval_context *ectx) const;

    size_t make_comparison_kernel(
                    ckernel_builder *out, size_t offset_out,
                    const ndt::type& src0_dt, const char *src0_metadata,
                    const ndt::type& src1_dt, const char *src1_metadata,
                    comparison_type_t comptype,
                    const eval::eval_context *ectx) const;
};

} // namespace dynd

#endif // _DYND__BASE_EXPRESSION_TYPE_HPP_
