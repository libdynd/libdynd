//
// Copyright (C) 2011-14 Mark Wiebe, DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

/**
 * The pointer type contains C/C++ raw pointers
 * pointing at data in other memory_blocks, using
 * blockrefs to manage the memory.
 *
 * This type operates in a "gather/scatter" fashion,
 * exposing itself as an expression type whose expression
 * copies the data to/from the pointer targets.
 */

#ifndef _DYND__POINTER_TYPE_HPP_
#define _DYND__POINTER_TYPE_HPP_

#include <dynd/type.hpp>
#include <dynd/types/void_pointer_type.hpp>

namespace dynd {

struct pointer_type_arrmeta {
    /**
     * A reference to the memory block which contains the data.
     */
    memory_block_data *blockref;
    /* Each pointed-to destination is offset by this amount */
    intptr_t offset;
};

class pointer_type : public base_expr_type {
    ndt::type m_target_tp;

public:
    pointer_type(const ndt::type& target_tp);

    virtual ~pointer_type();

    const ndt::type& get_value_type() const {
        return m_target_tp.value_type();
    }
    const ndt::type& get_operand_type() const;

    const ndt::type& get_target_type() const {
        return m_target_tp;
    }

    void print_data(std::ostream& o, const char *arrmeta, const char *data) const;

    void print_type(std::ostream& o) const;

    inline bool is_type_subarray(const ndt::type& subarray_tp) const {
        // Uniform dimensions can share one implementation
        return (!subarray_tp.is_builtin() && (*this) == (*subarray_tp.extended())) ||
                        m_target_tp.is_type_subarray(subarray_tp);
    }

    bool is_expression() const;
    bool is_unique_data_owner(const char *arrmeta) const;
    void transform_child_types(type_transform_fn_t transform_fn, void *extra,
                    ndt::type& out_transformed_tp, bool& out_was_transformed) const;
    ndt::type get_canonical_type() const;

    ndt::type apply_linear_index(intptr_t nindices, const irange *indices,
                size_t current_i, const ndt::type& root_tp, bool leading_dimension) const;
    intptr_t apply_linear_index(intptr_t nindices, const irange *indices, const char *arrmeta,
                    const ndt::type& result_tp, char *out_arrmeta,
                    memory_block_data *embedded_reference,
                    size_t current_i, const ndt::type& root_tp,
                    bool leading_dimension, char **inout_data,
                    memory_block_data **inout_dataref) const;
    ndt::type at_single(intptr_t i0, const char **inout_arrmeta, const char **inout_data) const;

    ndt::type get_type_at_dimension(char **inout_arrmeta, intptr_t i, intptr_t total_ndim = 0) const;

        void get_shape(intptr_t ndim, intptr_t i, intptr_t *out_shape, const char *arrmeta, const char *data) const;

    axis_order_classification_t classify_axis_order(const char *arrmeta) const;

    bool is_lossless_assignment(const ndt::type& dst_tp, const ndt::type& src_tp) const;

    bool operator==(const base_type& rhs) const;

    ndt::type with_replaced_storage_type(const ndt::type& replacement_type) const;

    void arrmeta_default_construct(char *arrmeta, intptr_t ndim,
                                   const intptr_t *shape,
                                   bool blockref_alloc) const;
    void arrmeta_copy_construct(char *dst_arrmeta, const char *src_arrmeta, memory_block_data *embedded_reference) const;
    void arrmeta_reset_buffers(char *arrmeta) const;
    void arrmeta_finalize_buffers(char *arrmeta) const;
    void arrmeta_destruct(char *arrmeta) const;
    void arrmeta_debug_print(const char *arrmeta, std::ostream& o, const std::string& indent) const;

    size_t make_assignment_kernel(ckernel_builder *ckb, intptr_t ckb_offset,
                                  const ndt::type &dst_tp,
                                  const char *dst_arrmeta,
                                  const ndt::type &src_tp,
                                  const char *src_arrmeta,
                                  kernel_request_t kernreq,
                                  const eval::eval_context *ectx) const;

    void get_dynamic_type_properties(
                    const std::pair<std::string, gfunc::callable> **out_properties,
                    size_t *out_count) const;
};

namespace ndt {
    ndt::type make_pointer(const ndt::type& target_tp);

    template<typename Tnative>
    inline ndt::type make_pointer() {
        return make_pointer(ndt::make_type<Tnative>());
    }
} // namespace ndt

} // namespace dynd

#endif // _DYND__POINTER_TYPE_HPP_
