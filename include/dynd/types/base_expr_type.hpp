//
// Copyright (C) 2011-15 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#pragma once

#include <dynd/types/base_type.hpp>

namespace dynd {
namespace ndt {

  /**
   * Base class for all types of expr_kind.
   */
  class DYNDT_API base_expr_type : public base_type {
  public:
    base_expr_type(type_id_t type_id, size_t data_size, size_t alignment, uint32_t flags, size_t arrmeta_size,
                   size_t ndim = 0);

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
     * Should return a reference to a type representing the data this type
     * uses to produce the value.
     */
    virtual const type &get_storage_type() const
    {
      throw std::runtime_error("get_storage_type is not implemented for this type");
    }

    /**
     * Returns a flags value which inherits the appropriate flags from
     * the value and operand types.
     */
    static inline uint32_t inherited_flags(uint32_t value_flags, uint32_t operand_flags)
    {
      return (value_flags & type_flags_value_inherited) | (operand_flags & type_flags_operand_inherited);
    }

    /**
     * This method is for expression types, and is a way to substitute
     * the storage type (deepest operand type) of an existing type.
     *
     * The value_type of the replacement should match the storage type
     * of this instance. Implementations of this should raise an exception
     * when this is not true.
     */
    virtual type with_replaced_storage_type(const type &replacement_type) const = 0;

    // Always return true for expression types
    bool is_expression() const;

    // The canonical type for expression types is always the value type
    type get_canonical_type() const;

    // Expression types use the values from their operand type.
    void arrmeta_default_construct(char *arrmeta, bool blockref_alloc) const;
    void arrmeta_copy_construct(char *dst_arrmeta, const char *src_arrmeta,
                                const intrusive_ptr<memory_block_data> &embedded_reference) const;
    void arrmeta_destruct(char *arrmeta) const;
    void arrmeta_debug_print(const char *arrmeta, std::ostream &o, const std::string &indent) const;

    // Expression types stop the iterdata chain
    // TODO: Maybe it should be more flexible?
    size_t get_iterdata_size(intptr_t ndim) const;
  };

} // namespace dynd::ndt
} // namespace dynd
