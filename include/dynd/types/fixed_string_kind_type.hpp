//
// Copyright (C) 2011-16 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#pragma once

#include <dynd/type.hpp>
#include <dynd/types/base_string_type.hpp>

namespace dynd {
namespace ndt {

  class DYNDT_API fixed_string_kind_type : public base_string_type {
  public:
    fixed_string_kind_type(type_id_t id) : base_string_type(id, 0, 0, type_flag_symbolic, 0) {}

    size_t get_default_data_size() const;

    void print_data(std::ostream &o, const char *arrmeta, const char *data) const;

    void print_type(std::ostream &o) const;

    bool is_expression() const;
    bool is_unique_data_owner(const char *arrmeta) const;
    type get_canonical_type() const;

    bool is_lossless_assignment(const type &dst_tp, const type &src_tp) const;

    string_encoding_t get_encoding() const;

    void get_string_range(const char **out_begin, const char **out_end, const char *arrmeta, const char *data) const;

    bool operator==(const base_type &rhs) const;

    void arrmeta_default_construct(char *arrmeta, bool blockref_alloc) const;
    void arrmeta_copy_construct(char *dst_arrmeta, const char *src_arrmeta,
                                const nd::memory_block &embedded_reference) const;
    void arrmeta_reset_buffers(char *arrmeta) const;
    void arrmeta_finalize_buffers(char *arrmeta) const;
    void arrmeta_destruct(char *arrmeta) const;
    void arrmeta_debug_print(const char *arrmeta, std::ostream &o, const std::string &indent) const;

    void data_destruct(const char *arrmeta, char *data) const;
    void data_destruct_strided(const char *arrmeta, char *data, intptr_t stride, size_t count) const;

    bool match(const type &candidate_tp, std::map<std::string, type> &tp_vars) const;
  };

  template <>
  struct id_of<fixed_string_kind_type> : std::integral_constant<type_id_t, fixed_string_kind_id> {};

} // namespace dynd::ndt
} // namespace dynd
