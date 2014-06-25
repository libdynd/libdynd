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
#include <dynd/types/type_type.hpp>
#include <dynd/memblock/memory_block.hpp>

namespace dynd {

/**
 * This defines a C-style structure type, which follows
 * C rules for the field layout. For the various builtin
 * types, a cstruct type should have a layout which matches
 * the equivalent in C/C++. This type works together with
 * the struct type, whose field layout is defined in the
 * arrmeta and hence supports viewing existing structs
 * with reordered and missing fields.
 *
 * Note that DyND doesn't support bitfields,
 * for example, so there isn't a way to match 100% of all
 * C structs.
 */
class cstruct_type : public base_struct_type {
    nd::array m_data_offsets;
    std::vector<std::pair<std::string, gfunc::callable> > m_array_properties;

    void create_array_properties();

    // Special constructor to break the property parameter cycle in
    // create_array_properties
    cstruct_type(int, int);
public:
    cstruct_type(const nd::array &field_names, const nd::array &field_types);
    virtual ~cstruct_type();

    size_t get_default_data_size(intptr_t DYND_UNUSED(ndim), const intptr_t *DYND_UNUSED(shape)) const {
        return get_data_size();
    }

    inline const nd::array &get_data_offsets() const {
        return m_data_offsets;
    }

    const uintptr_t *get_data_offsets(const char *DYND_UNUSED(arrmeta)) const {
        return reinterpret_cast<const uintptr_t *>(
            m_data_offsets.get_readonly_originptr());
    }

    inline const uintptr_t *get_data_offsets_raw() const {
        return reinterpret_cast<const uintptr_t *>(
            m_data_offsets.get_readonly_originptr());
    }
    inline const uintptr_t& get_data_offset(intptr_t i) const {
        return get_data_offsets_raw()[i];
    }

    void print_type(std::ostream& o) const;

    void transform_child_types(type_transform_fn_t transform_fn, void *extra,
                    ndt::type& out_transformed_tp, bool& out_was_transformed) const;
    ndt::type get_canonical_type() const;

    ndt::type at_single(intptr_t i0, const char **inout_arrmeta, const char **inout_data) const;

    bool is_lossless_assignment(const ndt::type& dst_tp, const ndt::type& src_tp) const;

    bool operator==(const base_type& rhs) const;

    void arrmeta_debug_print(const char *arrmeta, std::ostream& o, const std::string& indent) const;

    size_t make_assignment_kernel(ckernel_builder *ckb, intptr_t ckb_offset,
                                  const ndt::type &dst_tp,
                                  const char *dst_arrmeta,
                                  const ndt::type &src_tp,
                                  const char *src_arrmeta,
                                  kernel_request_t kernreq,
                                  const eval::eval_context *ectx) const;

    size_t make_comparison_kernel(ckernel_builder *ckb, intptr_t ckb_offset,
                                  const ndt::type &src0_dt,
                                  const char *src0_arrmeta,
                                  const ndt::type &src1_dt,
                                  const char *src1_arrmeta,
                                  comparison_type_t comptype,
                                  const eval::eval_context *ectx) const;

    void get_dynamic_type_properties(
                    const std::pair<std::string, gfunc::callable> **out_properties,
                    size_t *out_count) const;
    void get_dynamic_array_properties(
                    const std::pair<std::string, gfunc::callable> **out_properties,
                    size_t *out_count) const;
}; // class cstruct_type

namespace ndt {
    /** Makes a cstruct type with the specified fields */
    inline ndt::type make_cstruct(const nd::array &field_names,
                                 const nd::array &field_types)
    {
        return ndt::type(new cstruct_type(field_names, field_types), false);
    }


    /** Makes a cstruct type with the specified fields */
    inline ndt::type make_cstruct(const ndt::type &tp0, const std::string &name0)
    {
        const std::string *names[1] = {&name0};
        nd::array field_names = nd::make_strided_string_array(names, 1);
        intptr_t one = 1;
        nd::array field_types = nd::typed_empty(1, &one, ndt::make_strided_of_type());
        unchecked_strided_dim_get_rw<ndt::type>(field_types, 0) = tp0;
        field_types.flag_as_immutable();
        return ndt::make_cstruct(field_names, field_types);
    }

    /** Makes a cstruct type with the specified fields */
    inline ndt::type make_cstruct(const ndt::type &tp0, const std::string &name0,
                                 const ndt::type &tp1, const std::string &name1)
    {
        const std::string *names[2] = {&name0, &name1};
        nd::array field_names = nd::make_strided_string_array(names, 2);
        intptr_t two = 2;
        nd::array field_types = nd::typed_empty(1, &two, ndt::make_strided_of_type());
        unchecked_strided_dim_get_rw<ndt::type>(field_types, 0) = tp0;
        unchecked_strided_dim_get_rw<ndt::type>(field_types, 1) = tp1;
        field_types.flag_as_immutable();
        return ndt::make_cstruct(field_names, field_types);
    }

    /** Makes a cstruct type with the specified fields */
    inline ndt::type make_cstruct(const ndt::type &tp0, const std::string &name0,
                                 const ndt::type &tp1, const std::string &name1,
                                 const ndt::type &tp2, const std::string &name2)
    {
        const std::string *names[3] = {&name0, &name1, &name2};
        nd::array field_names = nd::make_strided_string_array(names, 3);
        intptr_t three = 3;
        nd::array field_types = nd::typed_empty(1, &three, ndt::make_strided_of_type());
        unchecked_strided_dim_get_rw<ndt::type>(field_types, 0) = tp0;
        unchecked_strided_dim_get_rw<ndt::type>(field_types, 1) = tp1;
        unchecked_strided_dim_get_rw<ndt::type>(field_types, 2) = tp2;
        field_types.flag_as_immutable();
        return ndt::make_cstruct(field_names, field_types);
    }

    /** Makes a cstruct type with the specified fields */
    inline ndt::type make_cstruct(const ndt::type &tp0, const std::string &name0,
                                 const ndt::type &tp1, const std::string &name1,
                                 const ndt::type &tp2, const std::string &name2,
                                 const ndt::type &tp3, const std::string &name3)
    {
        const std::string *names[4] = {&name0, &name1, &name2, &name3};
        nd::array field_names = nd::make_strided_string_array(names, 4);
        intptr_t four = 4;
        nd::array field_types = nd::typed_empty(1, &four, ndt::make_strided_of_type());
        unchecked_strided_dim_get_rw<ndt::type>(field_types, 0) = tp0;
        unchecked_strided_dim_get_rw<ndt::type>(field_types, 1) = tp1;
        unchecked_strided_dim_get_rw<ndt::type>(field_types, 2) = tp2;
        unchecked_strided_dim_get_rw<ndt::type>(field_types, 3) = tp3;
        field_types.flag_as_immutable();
        return ndt::make_cstruct(field_names, field_types);
    }

    /** Makes a cstruct type with the specified fields */
    inline ndt::type make_cstruct(const ndt::type &tp0, const std::string &name0,
                                 const ndt::type &tp1, const std::string &name1,
                                 const ndt::type &tp2, const std::string &name2,
                                 const ndt::type &tp3, const std::string &name3,
                                 const ndt::type &tp4, const std::string &name4)
    {
        const std::string *names[5] = {&name0, &name1, &name2, &name3, &name4};
        nd::array field_names = nd::make_strided_string_array(names, 5);
        intptr_t five = 5;
        nd::array field_types = nd::typed_empty(1, &five, ndt::make_strided_of_type());
        unchecked_strided_dim_get_rw<ndt::type>(field_types, 0) = tp0;
        unchecked_strided_dim_get_rw<ndt::type>(field_types, 1) = tp1;
        unchecked_strided_dim_get_rw<ndt::type>(field_types, 2) = tp2;
        unchecked_strided_dim_get_rw<ndt::type>(field_types, 3) = tp3;
        unchecked_strided_dim_get_rw<ndt::type>(field_types, 4) = tp4;
        field_types.flag_as_immutable();
        return ndt::make_cstruct(field_names, field_types);
    }

    /** Makes a cstruct type with the specified fields */
    inline ndt::type make_cstruct(const ndt::type &tp0, const std::string &name0,
                                 const ndt::type &tp1, const std::string &name1,
                                 const ndt::type &tp2, const std::string &name2,
                                 const ndt::type &tp3, const std::string &name3,
                                 const ndt::type &tp4, const std::string &name4,
                                 const ndt::type &tp5, const std::string &name5)
    {
        const std::string *names[6] = {&name0, &name1, &name2,
                                       &name3, &name4, &name5};
        nd::array field_names = nd::make_strided_string_array(names, 6);
        intptr_t six = 6;
        nd::array field_types = nd::typed_empty(1, &six, ndt::make_strided_of_type());
        unchecked_strided_dim_get_rw<ndt::type>(field_types, 0) = tp0;
        unchecked_strided_dim_get_rw<ndt::type>(field_types, 1) = tp1;
        unchecked_strided_dim_get_rw<ndt::type>(field_types, 2) = tp2;
        unchecked_strided_dim_get_rw<ndt::type>(field_types, 3) = tp3;
        unchecked_strided_dim_get_rw<ndt::type>(field_types, 4) = tp4;
        unchecked_strided_dim_get_rw<ndt::type>(field_types, 5) = tp5;
        field_types.flag_as_immutable();
        return ndt::make_cstruct(field_names, field_types);
    }

    /** Makes a cstruct type with the specified fields */
    inline ndt::type make_cstruct(const ndt::type &tp0, const std::string &name0,
                                 const ndt::type &tp1, const std::string &name1,
                                 const ndt::type &tp2, const std::string &name2,
                                 const ndt::type &tp3, const std::string &name3,
                                 const ndt::type &tp4, const std::string &name4,
                                 const ndt::type &tp5, const std::string &name5,
                                 const ndt::type &tp6, const std::string &name6)
    {
        const std::string *names[7] = {&name0, &name1, &name2,
                                       &name3, &name4, &name5, &name6};
        nd::array field_names = nd::make_strided_string_array(names, 7);
        intptr_t seven = 7;
        nd::array field_types = nd::typed_empty(1, &seven, ndt::make_strided_of_type());
        unchecked_strided_dim_get_rw<ndt::type>(field_types, 0) = tp0;
        unchecked_strided_dim_get_rw<ndt::type>(field_types, 1) = tp1;
        unchecked_strided_dim_get_rw<ndt::type>(field_types, 2) = tp2;
        unchecked_strided_dim_get_rw<ndt::type>(field_types, 3) = tp3;
        unchecked_strided_dim_get_rw<ndt::type>(field_types, 4) = tp4;
        unchecked_strided_dim_get_rw<ndt::type>(field_types, 5) = tp5;
        unchecked_strided_dim_get_rw<ndt::type>(field_types, 6) = tp6;
        field_types.flag_as_immutable();
        return ndt::make_cstruct(field_names, field_types);
    }


    /**
     * Checks whether a set of offsets can be used for cstruct.
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
                    const ndt::type *field_types, const uintptr_t *field_offsets, size_t total_size)
    {
        size_t offset = 0, max_alignment = 1;
        for (size_t i = 0; i != field_count; ++i) {
            uintptr_t field_data_alignment = field_types[i].get_data_alignment();
            uintptr_t field_data_size = field_types[i].get_data_size();
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
