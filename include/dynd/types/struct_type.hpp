//
// Copyright (C) 2011-14 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#pragma once

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

    void transform_child_types(type_transform_fn_t transform_fn,
                               intptr_t arrmeta_offset, void *extra,
                               ndt::type &out_transformed_tp,
                               bool &out_was_transformed) const;
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
        nd::array field_types = nd::empty(1, ndt::make_type());
        unchecked_fixed_dim_get_rw<ndt::type>(field_types, 0) = tp0;
        field_types.flag_as_immutable();
        return ndt::make_struct(field_names, field_types);
    }

    /** Makes a struct type with the specified fields */
    inline ndt::type make_struct(const ndt::type &tp0, const std::string &name0,
                                 const ndt::type &tp1, const std::string &name1)
    {
        const std::string *names[2] = {&name0, &name1};
        nd::array field_names = nd::make_strided_string_array(names, 2);
        nd::array field_types = nd::empty(2, ndt::make_type());
        unchecked_fixed_dim_get_rw<ndt::type>(field_types, 0) = tp0;
        unchecked_fixed_dim_get_rw<ndt::type>(field_types, 1) = tp1;
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
        nd::array field_types = nd::empty(3, ndt::make_type());
        unchecked_fixed_dim_get_rw<ndt::type>(field_types, 0) = tp0;
        unchecked_fixed_dim_get_rw<ndt::type>(field_types, 1) = tp1;
        unchecked_fixed_dim_get_rw<ndt::type>(field_types, 2) = tp2;
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
        nd::array field_types = nd::empty(4, ndt::make_type());
        unchecked_fixed_dim_get_rw<ndt::type>(field_types, 0) = tp0;
        unchecked_fixed_dim_get_rw<ndt::type>(field_types, 1) = tp1;
        unchecked_fixed_dim_get_rw<ndt::type>(field_types, 2) = tp2;
        unchecked_fixed_dim_get_rw<ndt::type>(field_types, 3) = tp3;
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
        nd::array field_types = nd::empty(5, ndt::make_type());
        unchecked_fixed_dim_get_rw<ndt::type>(field_types, 0) = tp0;
        unchecked_fixed_dim_get_rw<ndt::type>(field_types, 1) = tp1;
        unchecked_fixed_dim_get_rw<ndt::type>(field_types, 2) = tp2;
        unchecked_fixed_dim_get_rw<ndt::type>(field_types, 3) = tp3;
        unchecked_fixed_dim_get_rw<ndt::type>(field_types, 4) = tp4;
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
        nd::array field_types = nd::empty(6, ndt::make_type());
        unchecked_fixed_dim_get_rw<ndt::type>(field_types, 0) = tp0;
        unchecked_fixed_dim_get_rw<ndt::type>(field_types, 1) = tp1;
        unchecked_fixed_dim_get_rw<ndt::type>(field_types, 2) = tp2;
        unchecked_fixed_dim_get_rw<ndt::type>(field_types, 3) = tp3;
        unchecked_fixed_dim_get_rw<ndt::type>(field_types, 4) = tp4;
        unchecked_fixed_dim_get_rw<ndt::type>(field_types, 5) = tp5;
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
        nd::array field_types = nd::empty(7, ndt::make_type());
        unchecked_fixed_dim_get_rw<ndt::type>(field_types, 0) = tp0;
        unchecked_fixed_dim_get_rw<ndt::type>(field_types, 1) = tp1;
        unchecked_fixed_dim_get_rw<ndt::type>(field_types, 2) = tp2;
        unchecked_fixed_dim_get_rw<ndt::type>(field_types, 3) = tp3;
        unchecked_fixed_dim_get_rw<ndt::type>(field_types, 4) = tp4;
        unchecked_fixed_dim_get_rw<ndt::type>(field_types, 5) = tp5;
        unchecked_fixed_dim_get_rw<ndt::type>(field_types, 6) = tp6;
        field_types.flag_as_immutable();
        return ndt::make_struct(field_names, field_types);
    }
} // namespace ndt

/**
 * Concatenates the fields of two structs together into one.
 */
nd::array struct_concat(nd::array lhs, nd::array rhs);

namespace detail {

/**
  * A struct with some metafunctions to help packing values into structs and
  * tuples. For each value to pack into the struct/tuple, one first calls
  *     tp = pack<T>::make_type(val)
  * to get the ndt::type packing will use, then
  *     pack<T>::insert(val, tp, arrmeta, data);
  * which must initialize the output arrmeta if it is non-empty,
  * and copy the value into the target of the data pointer.
  */
template <typename T>
struct pack {
  /**
   * Returns the type to use for packing this specific value. The value
   * is allowed to affect the type, e.g. for packing a std::vector
   */
  static ndt::type make_type(const T &DYND_UNUSED(val))
  {
    // Default case is for when T and the ndt::type have identical
    // memory layout, which is guaranteed by make_exact_type<T>().
    return ndt::make_exact_type<T>();
  }

  /**
   * Packs a value into memory allocated to store it via the ``make_type(val)``
   * call. Because the destination arrmeta is guaranteed to be for only one
   * data element
   */
  static void insert(const T &val, const ndt::type &DYND_UNUSED(tp),
                     char *DYND_UNUSED(out_arrmeta), char *out_data)
  {
    *reinterpret_cast<T *>(out_data) = val;
  }
};

template <>
struct pack<nd::array> {
  static ndt::type make_type(const nd::array &val)
  {
    if ((val.get_access_flags()&nd::write_access_flag) == 0) {
      throw std::runtime_error("TODO: how to handle readonly/immutable arrays "
                               "in struct/tuple packing");
    }
    return ndt::make_pointer(val.get_type());
  }

  static void insert(const nd::array &val, const ndt::type &DYND_UNUSED(tp),
                     char *out_arrmeta, char *out_data)
  {
    pointer_type_arrmeta *am =
        reinterpret_cast<pointer_type_arrmeta *>(out_arrmeta);
    // Insert the reference in the destination pointer's arrmeta
    am->blockref = val.get_data_memblock().get();
    memory_block_incref(am->blockref);
    // Copy the rest of the arrmeta after the pointer's arrmeta
    const ndt::type &val_tp = val.get_type();
    if (val_tp.get_arrmeta_size() > 0) {
      val_tp.extended()->arrmeta_copy_construct(
          out_arrmeta + sizeof(pointer_type_arrmeta), val.get_arrmeta(),
          val.get_memblock().get());
    }
    // Copy the pointer
    *reinterpret_cast<char **>(out_data) = val.get_readwrite_originptr();
  }
};

template <typename T>
void pack_insert(const T &val, const ndt::type &tp, char *out_arrmeta,
                 char *out_data)
{
  pack<T>::insert(val, tp, out_arrmeta, out_data);
}

} // namespace detail

template <typename T>
ndt::type make_pack_type(const T &val) {
    return detail::pack<T>::make_type(val);
}

#define PACK_ARG(N)                                                            \
  DYND_PP_ELWISE_1(DYND_PP_ID,                                                 \
                   DYND_PP_MAP_1(DYND_PP_META_DECL_CONST_STR_REF,              \
                                 DYND_PP_META_NAME_RANGE(name, N)),            \
                   DYND_PP_ELWISE_1(DYND_PP_META_DECL_CONST_REF,               \
                                    DYND_PP_META_NAME_RANGE(A, N),             \
                                    DYND_PP_META_NAME_RANGE(a, N)))

#define PACK(N)                                                                \
  template <DYND_PP_JOIN_MAP_1(DYND_PP_META_TYPENAME, (, ),                    \
                               DYND_PP_META_NAME_RANGE(A, N))>                 \
  nd::array pack PACK_ARG(N)                                                   \
  {                                                                            \
    const std::string *field_names[N] = {                                      \
        DYND_PP_JOIN_1((, ), DYND_PP_META_NAME_RANGE(&name, N))};              \
    const ndt::type field_types[N] = {DYND_PP_JOIN_MAP_1(                      \
        make_pack_type, (, ), DYND_PP_META_NAME_RANGE(a, N))};                 \
                                                                               \
    /* Allocate res with empty_shell, leaves unconstructed arrmeta */          \
    nd::array res =                                                            \
        nd::empty_shell(ndt::make_struct(field_names, field_types));           \
    /* The struct's arrmeta includes data offsets, init them to default */     \
    struct_type::fill_default_data_offsets(                                    \
        N, field_types, reinterpret_cast<uintptr_t *>(res.get_arrmeta()));     \
    const uintptr_t *field_arrmeta_offsets =                                   \
        res.get_type().tcast<base_struct_type>()->get_arrmeta_offsets_raw();   \
    const uintptr_t *field_data_offsets =                                      \
        res.get_type().tcast<base_struct_type>()->get_data_offsets(            \
            res.get_arrmeta());                                                \
    char *res_arrmeta = res.get_arrmeta();                                     \
    char *res_data = res.get_readwrite_originptr();                            \
    /* Each pack_insert initializes the arrmeta and the data */                \
    DYND_PP_JOIN_ELWISE_1(                                                     \
        detail::pack_insert, (;), DYND_PP_META_NAME_RANGE(a, N),               \
        DYND_PP_META_AT_RANGE(field_types, N),                                 \
        DYND_PP_ELWISE_1(DYND_PP_META_ADD, DYND_PP_REPEAT_1(res_arrmeta, N),   \
                         DYND_PP_META_AT_RANGE(field_arrmeta_offsets, N)),     \
        DYND_PP_ELWISE_1(DYND_PP_META_ADD, DYND_PP_REPEAT_1(res_data, N),      \
                         DYND_PP_META_AT_RANGE(field_data_offsets, N)));       \
                                                                               \
    return res;                                                                \
  }

DYND_PP_JOIN_MAP(PACK, (), DYND_PP_RANGE(1, DYND_PP_INC(DYND_ARG_MAX)))

#undef PACK

#define PACK(N)                                                                \
  template <DYND_PP_JOIN_MAP_1(DYND_PP_META_TYPENAME, (, ),                    \
                               DYND_PP_META_NAME_RANGE(A, N))>                 \
  nd::array pack DYND_PP_PREPEND(const nd::array &aux, PACK_ARG(N))            \
  {                                                                            \
    return struct_concat(                                                      \
        aux,                                                                   \
        pack DYND_PP_ELWISE_1(DYND_PP_ID, DYND_PP_META_NAME_RANGE(name, N),    \
                              DYND_PP_META_NAME_RANGE(a, N)));                 \
  }

DYND_PP_JOIN_MAP(PACK, (), DYND_PP_RANGE(1, DYND_PP_INC(DYND_ARG_MAX)))

#undef PACK

#undef PACK_ARG

} // namespace dynd
