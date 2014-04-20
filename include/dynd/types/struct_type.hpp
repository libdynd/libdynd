//
// Copyright (C) 2011-14 Mark Wiebe, DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#ifndef _DYND__STRUCT_TYPE_HPP_
#define _DYND__STRUCT_TYPE_HPP_

#include <vector>
#include <string>

#include <dynd/type.hpp>
#include <dynd/types/base_struct_type.hpp>
#include <dynd/memblock/memory_block.hpp>

namespace dynd {

class struct_type : public base_struct_type {
    std::vector<ndt::type> m_field_types;
    std::vector<std::string> m_field_names;
    std::vector<size_t> m_metadata_offsets;
    std::vector<std::pair<std::string, gfunc::callable> > m_array_properties;

    void create_array_properties();

    // Used as the parameters type for the nd::array properties callables
    static ndt::type array_parameters_type;
public:
    struct_type(size_t field_count, const ndt::type *field_types,
                const std::string *field_names);

    virtual ~struct_type();

    const ndt::type *get_field_types() const {
        return &m_field_types[0];
    }

    const std::string *get_field_names() const {
        return &m_field_names[0];
    }

    const std::vector<std::string>& get_field_names_vector() const {
        return m_field_names;
    }

    intptr_t get_field_index(const std::string& field_name) const;

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

    void foreach_leading(char *data, const char *metadata, foreach_fn_t callback, void *callback_data) const;

    void get_dynamic_type_properties(const std::pair<std::string, gfunc::callable> **out_properties, size_t *out_count) const;
    void get_dynamic_array_properties(const std::pair<std::string, gfunc::callable> **out_properties, size_t *out_count) const;
}; // class struct_type

namespace ndt {
    /** Makes a struct type with the specified fields */
    inline ndt::type make_struct(size_t field_count, const ndt::type *field_types,
                    const std::string *field_names) {
        return ndt::type(new struct_type(field_count, field_types, field_names), false);
    }


    /** Makes a struct type with the specified fields */
    inline ndt::type make_struct(const ndt::type& tp0, const std::string& name0)
    {
        return ndt::make_struct(1, &tp0, &name0);
    }

    /** Makes a struct type with the specified fields */
    inline ndt::type make_struct(const ndt::type& tp0, const std::string& name0, const ndt::type& tp1, const std::string& name1)
    {
        ndt::type field_types[2];
        std::string field_names[2];
        field_types[0] = tp0;
        field_types[1] = tp1;
        field_names[0] = name0;
        field_names[1] = name1;
        return ndt::make_struct(2, field_types, field_names);
    }

    /** Makes a struct type with the specified fields */
    inline ndt::type make_struct(const ndt::type& tp0, const std::string& name0, const ndt::type& tp1, const std::string& name1, const ndt::type& tp2, const std::string& name2)
    {
        ndt::type field_types[3];
        std::string field_names[3];
        field_types[0] = tp0;
        field_types[1] = tp1;
        field_types[2] = tp2;
        field_names[0] = name0;
        field_names[1] = name1;
        field_names[2] = name2;
        return ndt::make_struct(3, field_types, field_names);
    }

    /** Makes a struct type with the specified fields */
    inline ndt::type make_struct(const ndt::type& tp0, const std::string& name0,
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
        return ndt::make_struct(4, field_types, field_names);
    }

    /** Makes a struct type with the specified fields */
    inline ndt::type make_struct(const ndt::type& tp0, const std::string& name0,
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
        return ndt::make_struct(5, field_types, field_names);
    }

    /** Makes a struct type with the specified fields */
    inline ndt::type make_struct(const ndt::type& tp0, const std::string& name0,
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
        return ndt::make_struct(6, field_types, field_names);
    }

    /** Makes a struct type with the specified fields */
    inline ndt::type make_struct(const ndt::type& tp0, const std::string& name0,
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
        return ndt::make_struct(7, field_types, field_names);
    }
} // namespace ndt

} // namespace dynd

#endif // _DYND__STRUCT_TYPE_HPP_
