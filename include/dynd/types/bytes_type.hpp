//
// Copyright (C) 2011-16 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#pragma once

#include <dynd/bytes.hpp>
#include <dynd/type.hpp>
#include <dynd/types/base_bytes_type.hpp>
#include <dynd/types/bytes_kind_type.hpp>

namespace dynd {
namespace ndt {

  /**
   * The bytes type uses memory_block references to store
   * arbitrarily sized runs of bytes.
   */
  class DYNDT_API bytes_type : public base_bytes_type {
    size_t m_alignment;

  public:
    typedef bytes data_type;

    bytes_type(type_id_t id, size_t alignment = 1)
        : base_bytes_type(id, sizeof(bytes), alignof(bytes), type_flag_zeroinit | type_flag_destructor, 0),
          m_alignment(alignment) {
      if (alignment != 1 && alignment != 2 && alignment != 4 && alignment != 8 && alignment != 16) {
        std::stringstream ss;
        ss << "Cannot make a dynd bytes type with alignment " << alignment << ", it must be a small power of two";
        throw std::runtime_error(ss.str());
      }
    }

    /** Alignment of the bytes data being pointed to. */
    size_t get_target_alignment() const { return m_alignment; }

    void print_data(std::ostream &o, const char *arrmeta, const char *data) const;

    void print_type(std::ostream &o) const;

    void get_bytes_range(const char **out_begin, const char **out_end, const char *arrmeta, const char *data) const;
    void set_bytes_data(const char *arrmeta, char *data, const char *bytes_begin, const char *bytes_end) const;

    bool is_unique_data_owner(const char *arrmeta) const;
    type get_canonical_type() const;

    void get_shape(intptr_t ndim, intptr_t i, intptr_t *out_shape, const char *arrmeta, const char *data) const;

    bool is_lossless_assignment(const type &dst_tp, const type &src_tp) const;

    bool operator==(const base_type &rhs) const;

    void data_destruct(const char *arrmeta, char *data) const;
    void data_destruct_strided(const char *arrmeta, char *data, intptr_t stride, size_t count) const;

    std::map<std::string, std::pair<ndt::type, const char *>> get_dynamic_type_properties() const;
  };

  template <>
  struct id_of<bytes> : std::integral_constant<type_id_t, bytes_id> {};

  template <>
  struct id_of<bytes_type> : std::integral_constant<type_id_t, bytes_id> {};

  template <>
  struct traits<bytes> {
    static const size_t ndim = 0;

    static const bool is_same_layout = true;

    static type equivalent() { return make_type<bytes_type>(); }

    static bytes na() { return bytes(); }
  };

} // namespace dynd::ndt
} // namespace dynd
