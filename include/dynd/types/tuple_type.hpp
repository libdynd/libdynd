//
// Copyright (C) 2011-15 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#pragma once

#include <vector>
#include <string>

#include <dynd/type.hpp>
#include <dynd/types/base_tuple_type.hpp>
#include <dynd/types/type_type.hpp>
#include <dynd/types/fixed_dim_type.hpp>
#include <dynd/memblock/memory_block.hpp>

namespace dynd {
namespace ndt {

  class tuple_type : public base_tuple_type {
    std::vector<std::pair<std::string, gfunc::callable>> m_array_properties;

  protected:
    uintptr_t *get_arrmeta_data_offsets(char *arrmeta) const
    {
      return reinterpret_cast<uintptr_t *>(arrmeta);
    }

  public:
    inline tuple_type(const nd::array &field_types, bool variadic)
        : base_tuple_type(tuple_type_id, field_types, type_flag_none, true,
                          variadic)
    {
    }

    virtual ~tuple_type();

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

    bool is_lossless_assignment(const type &dst_tp, const type &src_tp) const;

    bool operator==(const base_type &rhs) const;

    void arrmeta_debug_print(const char *arrmeta, std::ostream &o,
                             const std::string &indent) const;

    intptr_t make_assignment_kernel(void *ckb, intptr_t ckb_offset,
                                    const type &dst_tp, const char *dst_arrmeta,
                                    const type &src_tp, const char *src_arrmeta,
                                    kernel_request_t kernreq,
                                    const eval::eval_context *ectx) const;

    size_t make_comparison_kernel(void *ckb, intptr_t ckb_offset,
                                  const type &src0_dt, const char *src0_arrmeta,
                                  const type &src1_dt, const char *src1_arrmeta,
                                  comparison_type_t comptype,
                                  const eval::eval_context *ectx) const;

    void get_dynamic_type_properties(
        const std::pair<std::string, gfunc::callable> **out_properties,
        size_t *out_count) const;
  }; // class tuple_type

  /** Makes a tuple type with the specified types */
  inline type make_tuple(const nd::array &field_types, bool variadic = false)
  {
    // field_types.flag_as_immutable();
    return type(new tuple_type(field_types, variadic), false);
  }

  inline type make_empty_tuple()
  {
    // TODO: return a static instance
    nd::array field_types = nd::empty(0, make_type());
    return make_tuple(field_types);
  }

  inline type make_tuple() { return make_empty_tuple(); }

  /** Makes a tuple type with the specified types */
  inline type make_tuple(const type &tp0)
  {
    nd::array field_types = nd::empty(1, make_type());
    unchecked_fixed_dim_get_rw<type>(field_types, 0) = tp0;
    field_types.flag_as_immutable();
    return make_tuple(field_types);
  }

  /** Makes a tuple type with the specified types */
  inline type make_tuple(const type &tp0, const type &tp1)
  {
    nd::array field_types = nd::empty(2, make_type());
    unchecked_fixed_dim_get_rw<type>(field_types, 0) = tp0;
    unchecked_fixed_dim_get_rw<type>(field_types, 1) = tp1;
    field_types.flag_as_immutable();
    return make_tuple(field_types);
  }

  /** Makes a tuple type with the specified types */
  inline type make_tuple(const type &tp0, const type &tp1, const type &tp2)
  {
    nd::array field_types = nd::empty(3, make_type());
    unchecked_fixed_dim_get_rw<type>(field_types, 0) = tp0;
    unchecked_fixed_dim_get_rw<type>(field_types, 1) = tp1;
    unchecked_fixed_dim_get_rw<type>(field_types, 2) = tp2;
    field_types.flag_as_immutable();
    return make_tuple(field_types);
  }

  /** Makes a tuple type with the specified types */
  inline type make_tuple(const type &tp0, const type &tp1, const type &tp2,
                         const type &tp3)
  {
    nd::array field_types = nd::empty(4, make_type());
    unchecked_fixed_dim_get_rw<type>(field_types, 0) = tp0;
    unchecked_fixed_dim_get_rw<type>(field_types, 1) = tp1;
    unchecked_fixed_dim_get_rw<type>(field_types, 2) = tp2;
    unchecked_fixed_dim_get_rw<type>(field_types, 3) = tp3;
    field_types.flag_as_immutable();
    return make_tuple(field_types);
  }

  /** Makes a tuple type with the specified types */
  inline type make_tuple(const type &tp0, const type &tp1, const type &tp2,
                         const type &tp3, const type &tp4)
  {
    nd::array field_types = nd::empty(5, ndt::make_type());
    unchecked_fixed_dim_get_rw<type>(field_types, 0) = tp0;
    unchecked_fixed_dim_get_rw<type>(field_types, 1) = tp1;
    unchecked_fixed_dim_get_rw<type>(field_types, 2) = tp2;
    unchecked_fixed_dim_get_rw<type>(field_types, 3) = tp3;
    unchecked_fixed_dim_get_rw<type>(field_types, 4) = tp4;
    field_types.flag_as_immutable();
    return ndt::make_tuple(field_types);
  }

  /** Makes a tuple type with the specified types */
  inline type make_tuple(const type &tp0, const type &tp1, const type &tp2,
                         const type &tp3, const type &tp4, const type &tp5)
  {
    nd::array field_types = nd::empty(6, make_type());
    unchecked_fixed_dim_get_rw<type>(field_types, 0) = tp0;
    unchecked_fixed_dim_get_rw<type>(field_types, 1) = tp1;
    unchecked_fixed_dim_get_rw<type>(field_types, 2) = tp2;
    unchecked_fixed_dim_get_rw<type>(field_types, 3) = tp3;
    unchecked_fixed_dim_get_rw<type>(field_types, 4) = tp4;
    unchecked_fixed_dim_get_rw<type>(field_types, 5) = tp5;
    field_types.flag_as_immutable();
    return make_tuple(field_types);
  }

  /** Makes a tuple type with the specified types */
  inline type make_tuple(const type &tp0, const type &tp1, const type &tp2,
                         const type &tp3, const type &tp4, const type &tp5,
                         const type &tp6)
  {
    nd::array field_types = nd::empty(7, ndt::make_type());
    unchecked_fixed_dim_get_rw<type>(field_types, 0) = tp0;
    unchecked_fixed_dim_get_rw<type>(field_types, 1) = tp1;
    unchecked_fixed_dim_get_rw<type>(field_types, 2) = tp2;
    unchecked_fixed_dim_get_rw<type>(field_types, 3) = tp3;
    unchecked_fixed_dim_get_rw<type>(field_types, 4) = tp4;
    unchecked_fixed_dim_get_rw<type>(field_types, 5) = tp5;
    unchecked_fixed_dim_get_rw<type>(field_types, 6) = tp6;
    field_types.flag_as_immutable();
    return make_tuple(field_types);
  }

  nd::array pack(intptr_t field_count, const nd::array *field_vals);

} // namespace dynd::ndt
} // namespace dynd