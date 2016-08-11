//
// Copyright (C) 2011-16 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#pragma once

#include <dynd/types/base_abstract_type.hpp>
#include <dynd/types/base_type.hpp>

namespace dynd {
namespace ndt {

  class DYNDT_API any_kind_type : public base_abstract_type {
  public:
    any_kind_type(type_id_t id) : base_abstract_type(id, type_flag_variadic, 0, 0) {}

    void print_type(std::ostream &o) const { o << "Any"; }

    bool is_expression() const;
    bool is_unique_data_owner(const char *arrmeta) const;
    void transform_child_types(type_transform_fn_t transform_fn, intptr_t arrmeta_offset, void *extra,
                               type &out_transformed_tp, bool &out_was_transformed) const;
    type get_canonical_type() const;

    type at_single(intptr_t i0, const char **inout_arrmeta, const char **inout_data) const;

    type get_type_at_dimension(char **inout_arrmeta, intptr_t i, intptr_t total_ndim = 0) const;

    intptr_t get_dim_size(const char *arrmeta, const char *data) const;
    void get_shape(intptr_t ndim, intptr_t i, intptr_t *out_shape, const char *arrmeta, const char *data) const;

    bool is_lossless_assignment(const type &dst_tp, const type &src_tp) const;

    bool operator==(const base_type &rhs) const { return this == &rhs || rhs.get_id() == any_kind_id; }

    bool match(const type &DYND_UNUSED(candidate_tp), std::map<std::string, type> &DYND_UNUSED(tp_vars)) const {
      return true;
    }
  };

  template <>
  struct id_of<any_kind_type> : std::integral_constant<type_id_t, any_kind_id> {};

} // namespace dynd::ndt
} // namespace dynd
