//
// Copyright (C) 2011-14 Mark Wiebe, DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#ifndef _DYND__TUPLE_TYPE_HPP_
#define _DYND__TUPLE_TYPE_HPP_

#include <vector>
#include <string>

#include <dynd/type.hpp>
#include <dynd/types/base_tuple_type.hpp>
#include <dynd/memblock/memory_block.hpp>

namespace dynd {

class tuple_type : public base_tuple_type {
    std::vector<ndt::type> m_field_types;
    std::vector<size_t> m_metadata_offsets;
    std::vector<std::pair<std::string, gfunc::callable> > m_array_properties;

public:
    tuple_type(size_t field_count, const ndt::type *field_types);

    virtual ~tuple_type();

    const ndt::type *get_field_types() const {
        return &m_field_types[0];
    }

    const std::vector<ndt::type> get_field_types_vector() const {
        return m_field_types;
    }

    const size_t *get_metadata_offsets() const {
        return &m_metadata_offsets[0];
    }

    const std::vector<size_t>& get_metadata_offsets_vector() const {
        return m_metadata_offsets;
    }

    const size_t *get_data_offsets(const char *metadata) const {
        return reinterpret_cast<const size_t *>(metadata);
    }

    void print_data(std::ostream& o, const char *metadata, const char *data) const;

    void print_type(std::ostream& o) const;

    bool is_expression() const;
    bool is_unique_data_owner(const char *metadata) const;
    void transform_child_types(type_transform_fn_t transform_fn, void *extra,
                    ndt::type& out_transformed_tp, bool& out_was_transformed) const;
    ndt::type get_canonical_type() const;

    ndt::type apply_linear_index(intptr_t nindices, const irange *indices,
                size_t current_i, const ndt::type& root_tp, bool leading_dimension) const;
    intptr_t apply_linear_index(intptr_t nindices, const irange *indices, const char *metadata,
                    const ndt::type& result_tp, char *out_metadata,
                    memory_block_data *embedded_reference,
                    size_t current_i, const ndt::type& root_tp,
                    bool leading_dimension, char **inout_data,
                    memory_block_data **inout_dataref) const;

    bool is_lossless_assignment(const ndt::type& dst_tp, const ndt::type& src_tp) const;

    bool operator==(const base_type& rhs) const;

    void metadata_default_construct(char *metadata, intptr_t ndim, const intptr_t* shape) const;
    void metadata_copy_construct(char *dst_metadata, const char *src_metadata, memory_block_data *embedded_reference) const;
    void metadata_reset_buffers(char *metadata) const;
    void metadata_finalize_buffers(char *metadata) const;
    void metadata_destruct(char *metadata) const;
    void metadata_debug_print(const char *metadata, std::ostream& o, const std::string& indent) const;


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

    void foreach_leading(const char *metadata, char *data,
                         foreach_fn_t callback, void *callback_data) const;

    void get_dynamic_type_properties(const std::pair<std::string, gfunc::callable> **out_properties, size_t *out_count) const;
}; // class tuple_type

namespace ndt {
    /** Makes a ctuple type with the specified types */
    inline ndt::type make_tuple(size_t field_count, const ndt::type *field_types) {
        return ndt::type(new tuple_type(field_count, field_types), false);
    }

    /** Makes a ctuple type with the specified types */
    inline ndt::type make_tuple(const ndt::type& tp0)
    {
        return ndt::make_tuple(1, &tp0);
    }

    /** Makes a ctuple type with the specified types */
    inline ndt::type make_tuple(const ndt::type& tp0, const ndt::type& tp1)
    {
        ndt::type field_types[2];
        field_types[0] = tp0;
        field_types[1] = tp1;
        return ndt::make_tuple(2, field_types);
    }

    /** Makes a ctuple type with the specified types */
    inline ndt::type make_tuple(const ndt::type& tp0, const ndt::type& tp1, const ndt::type& tp2)
    {
        ndt::type field_types[3];
        field_types[0] = tp0;
        field_types[1] = tp1;
        field_types[2] = tp2;
        return ndt::make_tuple(3, field_types);
    }

    /** Makes a ctuple type with the specified types */
    inline ndt::type make_tuple(const ndt::type& tp0,
                    const ndt::type& tp1, const ndt::type& tp2,
                    const ndt::type& tp3)
    {
        ndt::type field_types[4];
        field_types[0] = tp0;
        field_types[1] = tp1;
        field_types[2] = tp2;
        field_types[3] = tp3;
        return ndt::make_tuple(4, field_types);
    }

    /** Makes a ctuple type with the specified types */
    inline ndt::type make_tuple(const ndt::type& tp0,
                    const ndt::type& tp1, const ndt::type& tp2,
                    const ndt::type& tp3, const ndt::type& tp4)
    {
        ndt::type field_types[5];
        field_types[0] = tp0;
        field_types[1] = tp1;
        field_types[2] = tp2;
        field_types[3] = tp3;
        field_types[4] = tp4;
        return ndt::make_tuple(5, field_types);
    }

    /** Makes a ctuple type with the specified types */
    inline ndt::type make_tuple(const ndt::type& tp0,
                    const ndt::type& tp1, const ndt::type& tp2,
                    const ndt::type& tp3, const ndt::type& tp4,
                    const ndt::type& tp5)
    {
        ndt::type field_types[6];
        field_types[0] = tp0;
        field_types[1] = tp1;
        field_types[2] = tp2;
        field_types[3] = tp3;
        field_types[4] = tp4;
        field_types[5] = tp5;
        return ndt::make_tuple(6, field_types);
    }

    /** Makes a ctuple type with the specified types */
    inline ndt::type make_tuple(const ndt::type& tp0,
                    const ndt::type& tp1, const ndt::type& tp2,
                    const ndt::type& tp3, const ndt::type& tp4,
                    const ndt::type& tp5, const ndt::type& tp6)
    {
        ndt::type field_types[7];
        field_types[0] = tp0;
        field_types[1] = tp1;
        field_types[2] = tp2;
        field_types[3] = tp3;
        field_types[4] = tp4;
        field_types[5] = tp5;
        field_types[6] = tp6;
        return ndt::make_tuple(7, field_types);
    }
} // namespace ndt

} // namespace dynd

#endif // _DYND__TUPLE_TYPE_HPP_
