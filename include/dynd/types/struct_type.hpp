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
#include <dynd/types/pointer_type.hpp>
#include <dynd/types/type_type.hpp>
#include <dynd/memblock/memory_block.hpp>
#include <dynd/pp/list.hpp>
#include <dynd/pp/meta.hpp>

namespace dynd {

class struct_type : public base_struct_type {
    std::vector<std::pair<std::string, gfunc::callable> > m_array_properties;

    void create_array_properties();

protected:

    uintptr_t *get_arrmeta_data_offsets(char *arrmeta) const {
        return reinterpret_cast<uintptr_t *>(arrmeta);
    }
public:
    inline struct_type(const nd::array &field_names, const nd::array &field_types)
      : base_struct_type(struct_type_id, field_names, field_types,
                         type_flag_none, true)
    {
        create_array_properties();
    }

    virtual ~struct_type();

    intptr_t get_field_index(const std::string& field_name) const;

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

    void get_dynamic_type_properties(
        const std::pair<std::string, gfunc::callable> **out_properties,
        size_t *out_count) const;
    void get_dynamic_array_properties(
        const std::pair<std::string, gfunc::callable> **out_properties,
        size_t *out_count) const;
}; // class struct_type

namespace ndt {
    /** Makes a struct type with the specified fields */
    inline ndt::type make_struct(const nd::array &field_names,
                                 const nd::array &field_types)
    {
        return ndt::type(new struct_type(field_names, field_types), false);
    }


    /** Makes a struct type with the specified fields */
    inline ndt::type make_struct(const ndt::type &tp0, const std::string &name0)
    {
        const std::string *names[1] = {&name0};
        nd::array field_names = nd::make_strided_string_array(names, 1);
        intptr_t one = 1;
        nd::array field_types = nd::typed_empty(1, &one, ndt::make_strided_of_type());
        unchecked_strided_dim_get_rw<ndt::type>(field_types, 0) = tp0;
        field_types.flag_as_immutable();
        return ndt::make_struct(field_names, field_types);
    }

    /** Makes a struct type with the specified fields */
    inline ndt::type make_struct(const ndt::type &tp0, const std::string &name0,
                                 const ndt::type &tp1, const std::string &name1)
    {
        const std::string *names[2] = {&name0, &name1};
        nd::array field_names = nd::make_strided_string_array(names, 2);
        intptr_t two = 2;
        nd::array field_types = nd::typed_empty(1, &two, ndt::make_strided_of_type());
        unchecked_strided_dim_get_rw<ndt::type>(field_types, 0) = tp0;
        unchecked_strided_dim_get_rw<ndt::type>(field_types, 1) = tp1;
        field_types.flag_as_immutable();
        return ndt::make_struct(field_names, field_types);
    }

    /** Makes a struct type with the specified fields */
    inline ndt::type make_struct(const ndt::type &tp0, const std::string &name0,
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
        return ndt::make_struct(field_names, field_types);
    }

    /** Makes a struct type with the specified fields */
    inline ndt::type make_struct(const ndt::type &tp0, const std::string &name0,
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
        return ndt::make_struct(field_names, field_types);
    }

    /** Makes a struct type with the specified fields */
    inline ndt::type make_struct(const ndt::type &tp0, const std::string &name0,
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
        return ndt::make_struct(field_names, field_types);
    }

    /** Makes a struct type with the specified fields */
    inline ndt::type make_struct(const ndt::type &tp0, const std::string &name0,
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
        return ndt::make_struct(field_names, field_types);
    }

    /** Makes a struct type with the specified fields */
    inline ndt::type make_struct(const ndt::type &tp0, const std::string &name0,
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
        return ndt::make_struct(field_names, field_types);
    }
} // namespace ndt

namespace detail {

template <typename T>
struct pack {
    static ndt::type make_type(const T &DYND_UNUSED(val)) {
        return ndt::make_type<T>();
    }

    static void insert(nd::array &a, const std::string &name, const T &val) {
        a.p(name).vals() = val;
    }
};

template <>
struct pack<nd::array> {
    static ndt::type make_type(const nd::array &val) {
        return ndt::make_pointer(val.get_type());
    }

    static void insert(nd::array &a, const std::string &name, const nd::array &val) {
        a.p(name).val_assign(combine_into_tuple(1, &val)(0));
    }
};

template <typename T>
void pack_insert(nd::array &a, const std::string &name, const T &val) {
    pack<T>::insert(a, name, val);
}

} // namespace detail

template <typename T>
ndt::type make_pack_type(const T &val) {
    return detail::pack<T>::make_type(val);
}

#define PACK_ARG(N) DYND_PP_ELWISE_1(DYND_PP_ID, \
    DYND_PP_MAP_1(DYND_PP_META_DECL_CONST_STR_REF, DYND_PP_META_NAME_RANGE(name, N)), \
    DYND_PP_ELWISE_1(DYND_PP_META_DECL_CONST_REF, DYND_PP_META_NAME_RANGE(A, N), DYND_PP_META_NAME_RANGE(a, N)))

#define PACK(N) \
    template <DYND_PP_JOIN_MAP_1(DYND_PP_META_TYPENAME, (,), DYND_PP_META_NAME_RANGE(A, N))> \
    nd::array pack PACK_ARG(N) { \
        const std::string *field_names[N] = {DYND_PP_JOIN_1((,), DYND_PP_META_NAME_RANGE(&name, N))}; \
        const ndt::type field_types[N] = {DYND_PP_JOIN_MAP_1(make_pack_type, (,), DYND_PP_META_NAME_RANGE(a, N))}; \
\
        nd::array res = nd::empty(ndt::make_struct(field_names, field_types)); \
        DYND_PP_JOIN_ELWISE_1(detail::pack_insert, (;), \
            DYND_PP_REPEAT_1(res, N), DYND_PP_META_NAME_RANGE(name, N), DYND_PP_META_NAME_RANGE(a, N)); \
\
        return res; \
    }

DYND_PP_JOIN_MAP(PACK, (), DYND_PP_RANGE(1, DYND_PP_INC(DYND_ARG_MAX)))

#undef PACK

#define PACK(N) \
    template <DYND_PP_JOIN_MAP_1(DYND_PP_META_TYPENAME, (,), DYND_PP_META_NAME_RANGE(A, N))> \
    nd::array pack DYND_PP_PREPEND(const nd::array &aux, PACK_ARG(N)) { \
        if (aux.is_null()) { \
            return pack DYND_PP_ELWISE_1(DYND_PP_ID, DYND_PP_META_NAME_RANGE(name, N), DYND_PP_META_NAME_RANGE(a, N)); \
        } \
\
        const nd::array &old_field_names = aux.get_dtype().tcast<base_struct_type>()->get_field_names(); \
        const std::string *new_field_names[N] = {DYND_PP_JOIN_1((,), DYND_PP_META_NAME_RANGE(&name, N))}; \
        nd::array field_names = nd::empty(old_field_names.get_dim_size() + N, ndt::make_string()); \
        field_names(irange() < old_field_names.get_dim_size()).val_assign(old_field_names); \
        field_names(old_field_names.get_dim_size() <= irange()).val_assign(new_field_names); \
\
        const nd::array &old_field_types = aux.get_dtype().tcast<base_struct_type>()->get_field_types(); \
        const ndt::type new_field_types[N] = {DYND_PP_JOIN_MAP_1(make_pack_type, (,), DYND_PP_META_NAME_RANGE(a, N))}; \
        nd::array field_types = nd::empty(old_field_types.get_dim_size() + N, ndt::make_type()); \
        field_types(irange() < old_field_types.get_dim_size()).val_assign(old_field_types); \
        field_types(old_field_types.get_dim_size() <= irange()).val_assign(new_field_types); \
\
        nd::array res = nd::empty(ndt::make_struct(field_names, field_types)); \
        res(irange() < aux.get_dim_size()).val_assign(aux); \
        DYND_PP_JOIN_ELWISE_1(detail::pack_insert, (;), \
            DYND_PP_REPEAT_1(res, N), DYND_PP_META_NAME_RANGE(name, N), DYND_PP_META_NAME_RANGE(a, N)); \
\
        return res; \
    }

DYND_PP_JOIN_MAP(PACK, (), DYND_PP_RANGE(1, DYND_PP_INC(DYND_ARG_MAX)))

#undef PACK

#undef PACK_ARG

} // namespace dynd

#endif // _DYND__STRUCT_TYPE_HPP_
