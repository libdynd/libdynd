//
// Copyright (C) 2011-15 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#pragma once

#include <dynd/types/base_type.hpp>

namespace dynd {
namespace ndt {

  /**
   * Base class for all types of expr_kind, and for the pointer_type.
   */
  class DYND_API base_expr_type : public base_type {
  public:
    base_expr_type(type_id_t type_id, type_kind_t kind, size_t data_size,
                   size_t alignment, flags_type flags, size_t arrmeta_size,
                   size_t ndim = 0)
        : base_type(type_id, kind, data_size, alignment, flags, arrmeta_size,
                    ndim, 0)
    {
    }

    virtual ~base_expr_type();

    /**
     * Should return a reference to the type representing the value which
     * is for calculation. This should never be an expression type.
     */
    virtual const type &get_value_type() const = 0;
    /**
     * Should return a reference to a type representing the data this type
     * uses to produce the value.
     */
    virtual const type &get_operand_type() const = 0;

    /**
     * Returns a flags value which inherits the appropriate flags from
     * the value and operand types.
     */
    static inline flags_type inherited_flags(flags_type value_flags,
                                             flags_type operand_flags)
    {
      return (value_flags & type_flags_value_inherited) |
             (operand_flags & type_flags_operand_inherited);
    }

    /**
     * This method is for expression types, and is a way to substitute
     * the storage type (deepest operand type) of an existing type.
     *
     * The value_type of the replacement should match the storage type
     * of this instance. Implementations of this should raise an exception
     * when this is not true.
     */
    virtual type
    with_replaced_storage_type(const type &replacement_type) const = 0;

    // Always return true for expression types
    bool is_expression() const;

    // The canonical type for expression types is always the value type
    type get_canonical_type() const;

    // Expression types use the values from their operand type.
    void arrmeta_default_construct(char *arrmeta, bool blockref_alloc) const;
    void arrmeta_copy_construct(char *dst_arrmeta, const char *src_arrmeta,
                                const intrusive_ptr<memory_block_data> &embedded_reference) const;
    void arrmeta_destruct(char *arrmeta) const;
    void arrmeta_debug_print(const char *arrmeta, std::ostream &o,
                             const std::string &indent) const;

    // Expression types stop the iterdata chain
    // TODO: Maybe it should be more flexible?
    size_t get_iterdata_size(intptr_t ndim) const;

    /** Makes a kernel which converts from (operand_type().value_type()) to
     * (value_type()) */
    virtual size_t make_operand_to_value_assignment_kernel(
        void *ckb, intptr_t ckb_offset, const char *dst_arrmeta,
        const char *src_arrmeta, kernel_request_t kernreq,
        const eval::eval_context *ectx) const;

    /** Makes a kernel which converts from (value_type()) to
     * (operand_type().value_type()) */
    virtual size_t make_value_to_operand_assignment_kernel(
        void *ckb, intptr_t ckb_offset, const char *dst_arrmeta,
        const char *src_arrmeta, kernel_request_t kernreq,
        const eval::eval_context *ectx) const;

    virtual intptr_t
    make_assignment_kernel(void *ckb, intptr_t ckb_offset, const type &dst_tp,
                           const char *dst_arrmeta, const type &src_tp,
                           const char *src_arrmeta, kernel_request_t kernreq,
                           const eval::eval_context *ectx) const;

    size_t make_comparison_kernel(void *ckb, intptr_t ckb_offset,
                                  const type &src0_dt, const char *src0_arrmeta,
                                  const type &src1_dt, const char *src1_arrmeta,
                                  comparison_type_t comptype,
                                  const eval::eval_context *ectx) const;
  };

} // namespace dynd::ndt
} // namespace dynd
