//
// Copyright (C) 2011-15 DyND Developers
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
namespace ndt {

  class struct_type : public base_struct_type {
    std::vector<std::pair<std::string, gfunc::callable>> m_array_properties;

    void create_array_properties();

    // Special constructor to break the property parameter cycle in
    // create_array_properties
    struct_type(int, int);

  protected:
    uintptr_t *get_arrmeta_data_offsets(char *arrmeta) const
    {
      return reinterpret_cast<uintptr_t *>(arrmeta);
    }

  public:
    struct_type(const nd::array &field_names, const nd::array &field_types,
                bool variadic);

    virtual ~struct_type();

    inline const uintptr_t *get_data_offsets(const char *arrmeta) const
    {
      return reinterpret_cast<const uintptr_t *>(arrmeta);
    }

    void print_type(std::ostream &o) const;

    void transform_child_types(type_transform_fn_t transform_fn,
                               intptr_t arrmeta_offset, void *extra,
                               type &out_transformed_tp,
                               bool &out_was_transformed) const;
    type get_canonical_type() const;

    type at_single(intptr_t i0, const char **inout_arrmeta,
                   const char **inout_data) const;

    bool is_lossless_assignment(const type &dst_tp, const type &src_tp) const;

    bool operator==(const base_type &rhs) const;

    void arrmeta_debug_print(const char *arrmeta, std::ostream &o,
                             const std::string &indent) const;

    virtual intptr_t make_assignment_kernel(
        const arrfunc_type_data *self, const arrfunc_type *af_tp, void *ckb,
        intptr_t ckb_offset, const type &dst_tp, const char *dst_arrmeta,
        const type &src_tp, const char *src_arrmeta, kernel_request_t kernreq,
        const eval::eval_context *ectx, const nd::array &kwds) const;

    size_t make_comparison_kernel(void *ckb, intptr_t ckb_offset,
                                  const type &src0_dt, const char *src0_arrmeta,
                                  const type &src1_dt, const char *src1_arrmeta,
                                  comparison_type_t comptype,
                                  const eval::eval_context *ectx) const;

    void get_dynamic_type_properties(
        const std::pair<std::string, gfunc::callable> **out_properties,
        size_t *out_count) const;
    void get_dynamic_array_properties(
        const std::pair<std::string, gfunc::callable> **out_properties,
        size_t *out_count) const;
  };

  inline type make_empty_struct(bool variadic = false)
  {
    // TODO: return a static instance
    nd::array field_names = nd::empty(0, make_string());
    nd::array field_types = nd::empty(0, make_type());
    return type(new struct_type(field_names, field_types, variadic), false);
  }

  /** Makes a struct type with the specified fields */
  inline type make_struct(const nd::array &field_names,
                          const nd::array &field_types, bool variadic = false)
  {
    return type(new struct_type(field_names, field_types, variadic), false);
  }

  /** Makes a struct type with the specified fields */
  inline type make_struct(const type &tp0, const std::string &name0)
  {
    const std::string *names[1] = {&name0};
    nd::array field_names = nd::make_strided_string_array(names, 1);
    nd::array field_types = nd::empty(1, make_type());
    unchecked_fixed_dim_get_rw<type>(field_types, 0) = tp0;
    field_types.flag_as_immutable();
    return make_struct(field_names, field_types);
  }

  /** Makes a struct type with the specified fields */
  inline type make_struct(const type &tp0, const std::string &name0,
                          const type &tp1, const std::string &name1)
  {
    const std::string *names[2] = {&name0, &name1};
    nd::array field_names = nd::make_strided_string_array(names, 2);
    nd::array field_types = nd::empty(2, make_type());
    unchecked_fixed_dim_get_rw<type>(field_types, 0) = tp0;
    unchecked_fixed_dim_get_rw<type>(field_types, 1) = tp1;
    field_types.flag_as_immutable();
    return make_struct(field_names, field_types);
  }

  /** Makes a struct type with the specified fields */
  inline type make_struct(const type &tp0, const std::string &name0,
                          const type &tp1, const std::string &name1,
                          const type &tp2, const std::string &name2)
  {
    const std::string *names[3] = {&name0, &name1, &name2};
    nd::array field_names = nd::make_strided_string_array(names, 3);
    nd::array field_types = nd::empty(3, make_type());
    unchecked_fixed_dim_get_rw<type>(field_types, 0) = tp0;
    unchecked_fixed_dim_get_rw<type>(field_types, 1) = tp1;
    unchecked_fixed_dim_get_rw<type>(field_types, 2) = tp2;
    field_types.flag_as_immutable();
    return make_struct(field_names, field_types);
  }

  /** Makes a struct type with the specified fields */
  inline type make_struct(const type &tp0, const std::string &name0,
                          const type &tp1, const std::string &name1,
                          const type &tp2, const std::string &name2,
                          const type &tp3, const std::string &name3)
  {
    const std::string *names[4] = {&name0, &name1, &name2, &name3};
    nd::array field_names = nd::make_strided_string_array(names, 4);
    nd::array field_types = nd::empty(4, make_type());
    unchecked_fixed_dim_get_rw<type>(field_types, 0) = tp0;
    unchecked_fixed_dim_get_rw<type>(field_types, 1) = tp1;
    unchecked_fixed_dim_get_rw<type>(field_types, 2) = tp2;
    unchecked_fixed_dim_get_rw<type>(field_types, 3) = tp3;
    field_types.flag_as_immutable();
    return make_struct(field_names, field_types);
  }

  /** Makes a struct type with the specified fields */
  inline type make_struct(const type &tp0, const std::string &name0,
                          const type &tp1, const std::string &name1,
                          const type &tp2, const std::string &name2,
                          const type &tp3, const std::string &name3,
                          const type &tp4, const std::string &name4)
  {
    const std::string *names[5] = {&name0, &name1, &name2, &name3, &name4};
    nd::array field_names = nd::make_strided_string_array(names, 5);
    nd::array field_types = nd::empty(5, make_type());
    unchecked_fixed_dim_get_rw<type>(field_types, 0) = tp0;
    unchecked_fixed_dim_get_rw<type>(field_types, 1) = tp1;
    unchecked_fixed_dim_get_rw<type>(field_types, 2) = tp2;
    unchecked_fixed_dim_get_rw<type>(field_types, 3) = tp3;
    unchecked_fixed_dim_get_rw<type>(field_types, 4) = tp4;
    field_types.flag_as_immutable();
    return make_struct(field_names, field_types);
  }

  /** Makes a struct type with the specified fields */
  inline type make_struct(const type &tp0, const std::string &name0,
                          const type &tp1, const std::string &name1,
                          const type &tp2, const std::string &name2,
                          const type &tp3, const std::string &name3,
                          const type &tp4, const std::string &name4,
                          const type &tp5, const std::string &name5)
  {
    const std::string *names[6] = {&name0, &name1, &name2,
                                   &name3, &name4, &name5};
    nd::array field_names = nd::make_strided_string_array(names, 6);
    nd::array field_types = nd::empty(6, make_type());
    unchecked_fixed_dim_get_rw<type>(field_types, 0) = tp0;
    unchecked_fixed_dim_get_rw<type>(field_types, 1) = tp1;
    unchecked_fixed_dim_get_rw<type>(field_types, 2) = tp2;
    unchecked_fixed_dim_get_rw<type>(field_types, 3) = tp3;
    unchecked_fixed_dim_get_rw<type>(field_types, 4) = tp4;
    unchecked_fixed_dim_get_rw<type>(field_types, 5) = tp5;
    field_types.flag_as_immutable();
    return make_struct(field_names, field_types);
  }

  /** Makes a struct type with the specified fields */
  inline type make_struct(const type &tp0, const std::string &name0,
                          const type &tp1, const std::string &name1,
                          const type &tp2, const std::string &name2,
                          const type &tp3, const std::string &name3,
                          const type &tp4, const std::string &name4,
                          const type &tp5, const std::string &name5,
                          const type &tp6, const std::string &name6)
  {
    const std::string *names[7] = {&name0, &name1, &name2, &name3,
                                   &name4, &name5, &name6};
    nd::array field_names = nd::make_strided_string_array(names, 7);
    nd::array field_types = nd::empty(7, make_type());
    unchecked_fixed_dim_get_rw<type>(field_types, 0) = tp0;
    unchecked_fixed_dim_get_rw<type>(field_types, 1) = tp1;
    unchecked_fixed_dim_get_rw<type>(field_types, 2) = tp2;
    unchecked_fixed_dim_get_rw<type>(field_types, 3) = tp3;
    unchecked_fixed_dim_get_rw<type>(field_types, 4) = tp4;
    unchecked_fixed_dim_get_rw<type>(field_types, 5) = tp5;
    unchecked_fixed_dim_get_rw<type>(field_types, 6) = tp6;
    field_types.flag_as_immutable();
    return make_struct(field_names, field_types);
  }

} // namespace dynd::ndt

/**
 * Concatenates the fields of two structs together into one.
 */
nd::array struct_concat(nd::array lhs, nd::array rhs);

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
        ndt::get_forward_type, (, ), DYND_PP_META_NAME_RANGE(a, N))};          \
                                                                               \
    /* Allocate res with empty_shell, leaves unconstructed arrmeta */          \
    nd::array res =                                                            \
        nd::empty_shell(ndt::make_struct(field_names, field_types));           \
    /* The struct's arrmeta includes data offsets, init them to default */     \
    ndt::struct_type::fill_default_data_offsets(                               \
        N, field_types, reinterpret_cast<uintptr_t *>(res.get_arrmeta()));     \
    const uintptr_t *field_arrmeta_offsets =                                   \
        res.get_type()                                                         \
            .extended<ndt::base_struct_type>()                                 \
            ->get_arrmeta_offsets_raw();                                       \
    const uintptr_t *field_data_offsets =                                      \
        res.get_type().extended<ndt::base_struct_type>()->get_data_offsets(    \
            res.get_arrmeta());                                                \
    char *res_arrmeta = res.get_arrmeta();                                     \
    char *res_data = res.get_readwrite_originptr();                            \
    /* Each pack_insert initializes the arrmeta and the data */                \
    DYND_PP_JOIN_ELWISE_1(                                                     \
        nd::forward_as_array, (;), DYND_PP_META_AT_RANGE(field_types, N),      \
        DYND_PP_ELWISE_1(DYND_PP_META_ADD, DYND_PP_REPEAT_1(res_arrmeta, N),   \
                         DYND_PP_META_AT_RANGE(field_arrmeta_offsets, N)),     \
        DYND_PP_ELWISE_1(DYND_PP_META_ADD, DYND_PP_REPEAT_1(res_data, N),      \
                         DYND_PP_META_AT_RANGE(field_data_offsets, N)),        \
        DYND_PP_META_NAME_RANGE(a, N));                                        \
                                                                               \
    return res;                                                                \
  }

DYND_PP_JOIN_MAP(PACK, (), DYND_PP_RANGE(1, DYND_PP_INC(DYND_ARG_MAX)))

#undef PACK

#undef PACK_ARG

} // namespace dynd
