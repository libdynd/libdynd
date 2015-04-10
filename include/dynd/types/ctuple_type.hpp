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
#include <dynd/memblock/memory_block.hpp>

namespace dynd {

class ctuple_type : public base_tuple_type {
  nd::array m_data_offsets;
  std::vector<std::pair<std::string, gfunc::callable>> m_array_properties;

public:
  ctuple_type(const nd::array &field_types);

  virtual ~ctuple_type();

  inline const nd::array &get_data_offsets() const { return m_data_offsets; }

  const uintptr_t *get_data_offsets(const char *DYND_UNUSED(arrmeta)) const
  {
    return reinterpret_cast<const uintptr_t *>(
        m_data_offsets.get_readonly_originptr());
  }

  inline const uintptr_t *get_data_offsets_raw() const
  {
    return reinterpret_cast<const uintptr_t *>(
        m_data_offsets.get_readonly_originptr());
  }
  inline const uintptr_t &get_data_offset(intptr_t i) const
  {
    return get_data_offsets_raw()[i];
  }

  void print_type(std::ostream &o) const;

  void transform_child_types(type_transform_fn_t transform_fn,
                             intptr_t arrmeta_offset, void *extra,
                             ndt::type &out_transformed_tp,
                             bool &out_was_transformed) const;
  ndt::type get_canonical_type() const;

  ndt::type at_single(intptr_t i0, const char **inout_arrmeta,
                      const char **inout_data) const;

  bool is_lossless_assignment(const ndt::type &dst_tp,
                              const ndt::type &src_tp) const;

  bool operator==(const base_type &rhs) const;

  void arrmeta_debug_print(const char *arrmeta, std::ostream &o,
                           const std::string &indent) const;

  intptr_t make_assignment_kernel(
      const arrfunc_type_data *self, const arrfunc_type *af_tp, void *ckb,
      intptr_t ckb_offset, const ndt::type &dst_tp, const char *dst_arrmeta,
      const ndt::type &src_tp, const char *src_arrmeta,
      kernel_request_t kernreq, const eval::eval_context *ectx,
      const nd::array &kwds) const;

  size_t make_comparison_kernel(void *ckb, intptr_t ckb_offset,
                                const ndt::type &src0_dt,
                                const char *src0_arrmeta,
                                const ndt::type &src1_dt,
                                const char *src1_arrmeta,
                                comparison_type_t comptype,
                                const eval::eval_context *ectx) const;

  void get_dynamic_type_properties(
      const std::pair<std::string, gfunc::callable> **out_properties,
      size_t *out_count) const;
}; // class ctuple_type

} // namespace dynd
