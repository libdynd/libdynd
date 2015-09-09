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

  class DYND_API tuple_type : public base_tuple_type {
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
        const std::pair<std::string, nd::callable> **out_properties,
        size_t *out_count) const;

    /** Makes a tuple type with the specified types */
    static type make(const nd::array &field_types, bool variadic = false)
    {
      return type(new tuple_type(field_types, variadic), false);
    }

    /** Makes an empty tuple */
    static type make(bool variadic = false)
    {
      return make(nd::empty(0, make_type()), variadic);
    }
  };

  DYND_API nd::array pack(intptr_t field_count, const nd::array *field_vals);

} // namespace dynd::ndt
} // namespace dynd
