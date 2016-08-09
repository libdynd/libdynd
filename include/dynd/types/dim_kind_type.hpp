//
// Copyright (C) 2011-16 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#pragma once

#include <dynd/types/scalar_kind_type.hpp>

namespace dynd {
namespace ndt {

  class DYNDT_API dim_kind_type : public base_dim_type {
  public:
    using base_dim_type::base_dim_type;

    dim_kind_type(type_id_t id, const type &element_tp = make_type<any_kind_type>());

    bool match(const type &candidate_tp, std::map<std::string, type> &tp_vars) const;

    void print_data(std::ostream &o, const char *arrmeta, const char *data) const;

    size_t arrmeta_copy_construct_onedim(char *dst_arrmeta, const char *src_arrmeta,
                                         const nd::memory_block &embedded_reference) const;

    void print_type(std::ostream &o) const;

    type get_type_at_dimension(char **inout_arrmeta, intptr_t i, intptr_t total_ndim = 0) const;

    intptr_t get_dim_size(const char *arrmeta, const char *data) const;

    type with_element_type(const type &element_tp) const;

    bool operator==(const base_type &rhs) const;

    static ndt::type construct_type(type_id_t id, const nd::buffer &args, const ndt::type &element_type);
  };

  template <>
  struct id_of<dim_kind_type> : std::integral_constant<type_id_t, dim_kind_id> {};

} // namespace dynd::ndt
} // namespace dynd
