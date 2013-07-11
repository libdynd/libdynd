//
// Copyright (C) 2011-13 Mark Wiebe, DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

/**
 * The pointer dtype contains C/C++ raw pointers
 * pointing at data in other memory_blocks, using
 * blockrefs to manage the memory.
 *
 * This dtype operates in a "gather/scatter" fashion,
 * exposing itself as an expression type whose expression
 * copies the data to/from the pointer targets.
 */

#ifndef _DYND__POINTER_TYPE_HPP_
#define _DYND__POINTER_TYPE_HPP_

#include <dynd/type.hpp>
#include <dynd/dtypes/void_pointer_type.hpp>

namespace dynd {

struct pointer_type_metadata {
    /**
     * A reference to the memory block which contains the data.
     */
    memory_block_data *blockref;
    /* Each pointed-to destination is offset by this amount */
    intptr_t offset;
};

class pointer_type : public base_expression_type {
    ndt::type m_target_dtype;
    static ndt::type m_void_pointer_type;

public:
    pointer_type(const ndt::type& target_dtype);

    virtual ~pointer_type();

    const ndt::type& get_value_type() const {
        return m_target_dtype.value_type();
    }
    const ndt::type& get_operand_type() const {
        if (m_target_dtype.get_type_id() == pointer_type_id) {
            return m_target_dtype;
        } else {
            return m_void_pointer_type;
        }
    }

    const ndt::type& get_target_dtype() const {
        return m_target_dtype;
    }

    void print_data(std::ostream& o, const char *metadata, const char *data) const;

    void print_dtype(std::ostream& o) const;

    bool is_expression() const;
    bool is_unique_data_owner(const char *metadata) const;
    void transform_child_types(type_transform_fn_t transform_fn, void *extra,
                    ndt::type& out_transformed_dtype, bool& out_was_transformed) const;
    ndt::type get_canonical_type() const;

    ndt::type apply_linear_index(size_t nindices, const irange *indices,
                size_t current_i, const ndt::type& root_dt, bool leading_dimension) const;
    intptr_t apply_linear_index(size_t nindices, const irange *indices, const char *metadata,
                    const ndt::type& result_dtype, char *out_metadata,
                    memory_block_data *embedded_reference,
                    size_t current_i, const ndt::type& root_dt,
                    bool leading_dimension, char **inout_data,
                    memory_block_data **inout_dataref) const;
    ndt::type at_single(intptr_t i0, const char **inout_metadata, const char **inout_data) const;

    ndt::type get_type_at_dimension(char **inout_metadata, size_t i, size_t total_ndim = 0) const;

    void get_shape(size_t ndim, size_t i, intptr_t *out_shape, const char *metadata) const;

    axis_order_classification_t classify_axis_order(const char *metadata) const;

    bool is_lossless_assignment(const ndt::type& dst_dt, const ndt::type& src_dt) const;

    bool operator==(const base_type& rhs) const;

    ndt::type with_replaced_storage_type(const ndt::type& replacement_type) const;

    void metadata_default_construct(char *metadata, size_t ndim, const intptr_t* shape) const;
    void metadata_copy_construct(char *dst_metadata, const char *src_metadata, memory_block_data *embedded_reference) const;
    void metadata_reset_buffers(char *metadata) const;
    void metadata_finalize_buffers(char *metadata) const;
    void metadata_destruct(char *metadata) const;
    void metadata_debug_print(const char *metadata, std::ostream& o, const std::string& indent) const;

    void get_dynamic_dtype_properties(
                    const std::pair<std::string, gfunc::callable> **out_properties,
                    size_t *out_count) const;
};

namespace ndt {
    inline ndt::type make_pointer(const ndt::type& target_dtype) {
        if (target_dtype.get_type_id() != void_type_id) {
            return ndt::type(new pointer_type(target_dtype), false);
        } else {
            return ndt::type(new void_pointer_type(), false);
        }
    }

    template<typename Tnative>
    ndt::type make_pointer() {
        return make_pointer(ndt::make_type<Tnative>());
    }
} // namespace ndt

} // namespace dynd

#endif // _DYND__POINTER_TYPE_HPP_
