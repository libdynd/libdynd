//
// Copyright (C) 2011-16 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#pragma once

#include <dynd/type.hpp>
#include <dynd/types/scalar_kind_type.hpp>

namespace dynd {
namespace ndt {

  /**
   * A type whose instance represents a type itself.
   */
  class DYNDT_API type_type : public base_type {
  public:
    typedef type data_type;

    type_type(type_id_t id)
        : base_type(id, sizeof(ndt::type), sizeof(ndt::type), type_flag_zeroinit | type_flag_destructor, 0, 0, 0) {}

    void print_data(std::ostream &o, const char *arrmeta, const char *data) const;

    void print_type(std::ostream &o) const;

    bool operator==(const base_type &rhs) const;

    void arrmeta_default_construct(char *arrmeta, bool blockref_alloc) const;
    void arrmeta_copy_construct(char *dst_arrmeta, const char *src_arrmeta,
                                const nd::memory_block &embedded_reference) const;
    void arrmeta_reset_buffers(char *arrmeta) const;
    void arrmeta_finalize_buffers(char *arrmeta) const;
    void arrmeta_destruct(char *arrmeta) const;
    void arrmeta_debug_print(const char *DYND_UNUSED(arrmeta), std::ostream &DYND_UNUSED(o),
                             const std::string &DYND_UNUSED(indent)) const {}

    void data_destruct(const char *arrmeta, char *data) const;
    void data_destruct_strided(const char *arrmeta, char *data, intptr_t stride, size_t count) const;
  };

  template <>
  struct id_of<type_type> : std::integral_constant<type_id_t, type_id> {};

  template <>
  struct id_of<type> : std::integral_constant<type_id_t, type_id> {};

} // namespace dynd::ndt
} // namespace dynd
