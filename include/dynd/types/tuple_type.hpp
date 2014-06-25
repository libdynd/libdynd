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
#include <dynd/types/type_type.hpp>
#include <dynd/types/strided_dim_type.hpp>
#include <dynd/memblock/memory_block.hpp>

namespace dynd {

class tuple_type : public base_tuple_type {
    std::vector<std::pair<std::string, gfunc::callable> > m_array_properties;
protected:

    uintptr_t *get_arrmeta_data_offsets(char *arrmeta) const {
        return reinterpret_cast<uintptr_t *>(arrmeta);
    }
public:
    inline tuple_type(const nd::array &field_types)
        : base_tuple_type(tuple_type_id, field_types, type_flag_none, true)
    {
    }

    virtual ~tuple_type();

    inline const uintptr_t *get_data_offsets(const char *arrmeta) const {
        return reinterpret_cast<const uintptr_t *>(arrmeta);
    }

    void print_type(std::ostream& o) const;

    void transform_child_types(type_transform_fn_t transform_fn, void *extra,
                    ndt::type& out_transformed_tp, bool& out_was_transformed) const;
    ndt::type get_canonical_type() const;

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

    void get_dynamic_type_properties(const std::pair<std::string, gfunc::callable> **out_properties, size_t *out_count) const;
}; // class tuple_type

namespace ndt {
    /** Makes a tuple type with the specified types */
    inline ndt::type make_tuple(const nd::array& field_types) {
        return ndt::type(new tuple_type(field_types), false);
    }

    /** Makes a tuple type with the specified types */
    inline ndt::type make_tuple(const ndt::type& tp0)
    {
        intptr_t one = 1;
        nd::array field_types =
            nd::typed_empty(1, &one, ndt::make_strided_of_type());
        unchecked_strided_dim_get_rw<ndt::type>(field_types, 0) = tp0;
        field_types.flag_as_immutable();
        return ndt::make_tuple(field_types);
    }

    /** Makes a ctuple type with the specified types */
    inline ndt::type make_tuple(const ndt::type& tp0, const ndt::type& tp1)
    {
        intptr_t two = 2;
        nd::array field_types = nd::typed_empty(1, &two, ndt::make_strided_of_type());
        unchecked_strided_dim_get_rw<ndt::type>(field_types, 0) = tp0;
        unchecked_strided_dim_get_rw<ndt::type>(field_types, 1) = tp1;
        field_types.flag_as_immutable();
        return ndt::make_tuple(field_types);
    }

    /** Makes a ctuple type with the specified types */
    inline ndt::type make_tuple(const ndt::type& tp0, const ndt::type& tp1, const ndt::type& tp2)
    {
        intptr_t three = 3;
        nd::array field_types = nd::typed_empty(1, &three, ndt::make_strided_of_type());
        unchecked_strided_dim_get_rw<ndt::type>(field_types, 0) = tp0;
        unchecked_strided_dim_get_rw<ndt::type>(field_types, 1) = tp1;
        unchecked_strided_dim_get_rw<ndt::type>(field_types, 2) = tp2;
        field_types.flag_as_immutable();
        return ndt::make_tuple(field_types);
    }

    /** Makes a tuple type with the specified types */
    inline ndt::type make_tuple(const ndt::type& tp0,
                    const ndt::type& tp1, const ndt::type& tp2,
                    const ndt::type& tp3)
    {
        intptr_t four = 4;
        nd::array field_types = nd::typed_empty(1, &four, ndt::make_strided_of_type());
        unchecked_strided_dim_get_rw<ndt::type>(field_types, 0) = tp0;
        unchecked_strided_dim_get_rw<ndt::type>(field_types, 1) = tp1;
        unchecked_strided_dim_get_rw<ndt::type>(field_types, 2) = tp2;
        unchecked_strided_dim_get_rw<ndt::type>(field_types, 3) = tp3;
        field_types.flag_as_immutable();
        return ndt::make_tuple(field_types);
    }

    /** Makes a tuple type with the specified types */
    inline ndt::type make_tuple(const ndt::type& tp0,
                    const ndt::type& tp1, const ndt::type& tp2,
                    const ndt::type& tp3, const ndt::type& tp4)
    {
        intptr_t five = 5;
        nd::array field_types = nd::typed_empty(1, &five, ndt::make_strided_of_type());
        unchecked_strided_dim_get_rw<ndt::type>(field_types, 0) = tp0;
        unchecked_strided_dim_get_rw<ndt::type>(field_types, 1) = tp1;
        unchecked_strided_dim_get_rw<ndt::type>(field_types, 2) = tp2;
        unchecked_strided_dim_get_rw<ndt::type>(field_types, 3) = tp3;
        unchecked_strided_dim_get_rw<ndt::type>(field_types, 4) = tp4;
        field_types.flag_as_immutable();
        return ndt::make_tuple(field_types);
    }

    /** Makes a tuple type with the specified types */
    inline ndt::type make_tuple(const ndt::type& tp0,
                    const ndt::type& tp1, const ndt::type& tp2,
                    const ndt::type& tp3, const ndt::type& tp4,
                    const ndt::type& tp5)
    {
        intptr_t six = 6;
        nd::array field_types = nd::typed_empty(1, &six, ndt::make_strided_of_type());
        unchecked_strided_dim_get_rw<ndt::type>(field_types, 0) = tp0;
        unchecked_strided_dim_get_rw<ndt::type>(field_types, 1) = tp1;
        unchecked_strided_dim_get_rw<ndt::type>(field_types, 2) = tp2;
        unchecked_strided_dim_get_rw<ndt::type>(field_types, 3) = tp3;
        unchecked_strided_dim_get_rw<ndt::type>(field_types, 4) = tp4;
        unchecked_strided_dim_get_rw<ndt::type>(field_types, 5) = tp5;
        field_types.flag_as_immutable();
        return ndt::make_tuple(field_types);
    }

    /** Makes a tuple type with the specified types */
    inline ndt::type make_tuple(const ndt::type& tp0,
                    const ndt::type& tp1, const ndt::type& tp2,
                    const ndt::type& tp3, const ndt::type& tp4,
                    const ndt::type& tp5, const ndt::type& tp6)
    {
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
        return ndt::make_tuple(field_types);
    }
} // namespace ndt

} // namespace dynd

#endif // _DYND__TUPLE_TYPE_HPP_
