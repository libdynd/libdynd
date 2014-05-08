//
// Copyright (C) 2011-14 Mark Wiebe, DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#ifndef _DYND__CSTRUCT_TYPE_HPP_
#define _DYND__CSTRUCT_TYPE_HPP_

#include <vector>
#include <string>

#include <dynd/type.hpp>
#include <dynd/types/base_struct_type.hpp>
#include <dynd/memblock/memory_block.hpp>

namespace dynd {

/**
 * This defines a C-style structure type, which follows
 * C rules for the field layout. For the various builtin
 * types, a cstruct type should have a layout which matches
 * the equivalent in C/C++. This type works together with
 * the struct type, whose field layout is defined in the
 * metadata and hence supports viewing existing structs
 * with reordered and missing fields.
 *
 * Note that DyND doesn't support bitfields,
 * for example, so there isn't a way to match 100% of all
 * C structs.
 */
class cstruct_type : public base_struct_type {
    std::vector<ndt::type> m_field_types;
    std::vector<std::string> m_field_names;
    std::vector<size_t> m_data_offsets;
    std::vector<size_t> m_metadata_offsets;
    std::vector<std::pair<std::string, gfunc::callable> > m_array_properties;

    void create_array_properties();

    // Special constructor to break the property parameter cycle in
    // create_array_properties
    cstruct_type(int, int);
public:
    cstruct_type(size_t field_count, const ndt::type *field_types,
                    const std::string *field_names);

    virtual ~cstruct_type();

    size_t get_default_data_size(intptr_t DYND_UNUSED(ndim), const intptr_t *DYND_UNUSED(shape)) const {
        return get_data_size();
    }

    const ndt::type *get_field_types() const {
        return &m_field_types[0];
    }

    const std::vector<ndt::type> get_field_types_vector() const {
        return m_field_types;
    }

    const std::string *get_field_names() const {
        return &m_field_names[0];
    }

    const std::vector<std::string>& get_field_names_vector() const {
        return m_field_names;
    }

    intptr_t get_field_index(const std::string& field_name) const;

    const size_t *get_data_offsets(const char *DYND_UNUSED(metadata)) const {
        return &m_data_offsets[0];
    }

    inline const size_t *get_data_offsets() const {
        return &m_data_offsets[0];
    }

    const std::vector<size_t>& get_data_offsets_vector() const {
        return m_data_offsets;
    }

    const size_t *get_metadata_offsets() const {
        return &m_metadata_offsets[0];
    }

    const std::vector<size_t>& get_metadata_offsets_vector() const {
        return m_metadata_offsets;
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
    ndt::type at_single(intptr_t i0, const char **inout_metadata, const char **inout_data) const;

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

    void get_dynamic_type_properties(
                    const std::pair<std::string, gfunc::callable> **out_properties,
                    size_t *out_count) const;
    void get_dynamic_array_properties(
                    const std::pair<std::string, gfunc::callable> **out_properties,
                    size_t *out_count) const;
}; // class cstruct_type

namespace ndt {
    /** Makes a struct type with the specified fields */
    inline ndt::type make_cstruct(size_t field_count, const ndt::type *field_types,
                    const std::string *field_names) {
        return ndt::type(new cstruct_type(field_count, field_types, field_names), false);
    }

    /** Makes a struct type with the specified fields */
    inline ndt::type make_cstruct(const ndt::type& tp0, const std::string& name0)
    {
        return ndt::make_cstruct(1, &tp0, &name0);
    }

    /** Makes a struct type with the specified fields */
    inline ndt::type make_cstruct(const ndt::type& tp0, const std::string& name0, const ndt::type& tp1, const std::string& name1)
    {
        ndt::type field_types[2];
        std::string field_names[2];
        field_types[0] = tp0;
        field_types[1] = tp1;
        field_names[0] = name0;
        field_names[1] = name1;
        return ndt::make_cstruct(2, field_types, field_names);
    }

    /** Makes a struct type with the specified fields */
    inline ndt::type make_cstruct(const ndt::type& tp0, const std::string& name0, const ndt::type& tp1, const std::string& name1, const ndt::type& tp2, const std::string& name2)
    {
        ndt::type field_types[3];
        std::string field_names[3];
        field_types[0] = tp0;
        field_types[1] = tp1;
        field_types[2] = tp2;
        field_names[0] = name0;
        field_names[1] = name1;
        field_names[2] = name2;
        return ndt::make_cstruct(3, field_types, field_names);
    }

    /** Makes a struct type with the specified fields */
    inline ndt::type make_cstruct(const ndt::type& tp0, const std::string& name0,
                    const ndt::type& tp1, const std::string& name1, const ndt::type& tp2, const std::string& name2,
                    const ndt::type& tp3, const std::string& name3)
    {
        ndt::type field_types[4];
        std::string field_names[4];
        field_types[0] = tp0;
        field_types[1] = tp1;
        field_types[2] = tp2;
        field_types[3] = tp3;
        field_names[0] = name0;
        field_names[1] = name1;
        field_names[2] = name2;
        field_names[3] = name3;
        return ndt::make_cstruct(4, field_types, field_names);
    }

    /** Makes a struct type with the specified fields */
    inline ndt::type make_cstruct(const ndt::type& tp0, const std::string& name0,
                    const ndt::type& tp1, const std::string& name1, const ndt::type& tp2, const std::string& name2,
                    const ndt::type& tp3, const std::string& name3, const ndt::type& tp4, const std::string& name4)
    {
        ndt::type field_types[5];
        std::string field_names[5];
        field_types[0] = tp0;
        field_types[1] = tp1;
        field_types[2] = tp2;
        field_types[3] = tp3;
        field_types[4] = tp4;
        field_names[0] = name0;
        field_names[1] = name1;
        field_names[2] = name2;
        field_names[3] = name3;
        field_names[4] = name4;
        return ndt::make_cstruct(5, field_types, field_names);
    }

    /** Makes a struct type with the specified fields */
    inline ndt::type make_cstruct(const ndt::type& tp0, const std::string& name0,
                    const ndt::type& tp1, const std::string& name1, const ndt::type& tp2, const std::string& name2,
                    const ndt::type& tp3, const std::string& name3, const ndt::type& tp4, const std::string& name4,
                    const ndt::type& tp5, const std::string& name5)
    {
        ndt::type field_types[6];
        std::string field_names[6];
        field_types[0] = tp0;
        field_types[1] = tp1;
        field_types[2] = tp2;
        field_types[3] = tp3;
        field_types[4] = tp4;
        field_types[5] = tp5;
        field_names[0] = name0;
        field_names[1] = name1;
        field_names[2] = name2;
        field_names[3] = name3;
        field_names[4] = name4;
        field_names[5] = name5;
        return ndt::make_cstruct(6, field_types, field_names);
    }

    /** Makes a struct type with the specified fields */
    inline ndt::type make_cstruct(const ndt::type& tp0, const std::string& name0,
                    const ndt::type& tp1, const std::string& name1, const ndt::type& tp2, const std::string& name2,
                    const ndt::type& tp3, const std::string& name3, const ndt::type& tp4, const std::string& name4,
                    const ndt::type& tp5, const std::string& name5, const ndt::type& tp6, const std::string& name6)
    {
        ndt::type field_types[7];
        std::string field_names[7];
        field_types[0] = tp0;
        field_types[1] = tp1;
        field_types[2] = tp2;
        field_types[3] = tp3;
        field_types[4] = tp4;
        field_types[5] = tp5;
        field_types[6] = tp6;
        field_names[0] = name0;
        field_names[1] = name1;
        field_names[2] = name2;
        field_names[3] = name3;
        field_names[4] = name4;
        field_names[5] = name5;
        field_names[6] = name6;
        return ndt::make_cstruct(7, field_types, field_names);
    }

    /**
     * \brief Checks whether a set of offsets can be used for cstruct.
     *
     * Because cstruct does not support customizable offset (use struct for
     * that), this function can be used to check that offsets are compatible with
     * cstruct.
     *
     * \param field_count  The number of array entries in `field_types` and `field_offsets`
     * \param field_types  An array of the field types.
     * \param field_offsets  The offsets corresponding to the types.
     * \param total_size  The total size of the struct in bytes.
     *
     * \returns  True if constructing a cstruct with the same types and field offsets will
     *           produce the provided offsets.
     */
    inline bool is_cstruct_compatible_offsets(size_t field_count,
                    const ndt::type *field_types, const size_t *field_offsets, size_t total_size)
    {
        size_t offset = 0, max_alignment = 1;
        for (size_t i = 0; i != field_count; ++i) {
            size_t field_data_alignment = field_types[i].get_data_alignment();
            size_t field_data_size = field_types[i].get_data_size();
            offset = inc_to_alignment(offset, field_data_alignment);
            if (field_offsets[i] != offset || field_data_size == 0) {
                return false;
            }
            max_alignment = (field_data_alignment > max_alignment) ? field_data_alignment : max_alignment;
            offset += field_data_size;
        }
        offset = inc_to_alignment(offset, max_alignment);
        return total_size == offset;
    }

} // namespace ndt

} // namespace dynd

#endif // _DYND__CSTRUCT_TYPE_HPP_
