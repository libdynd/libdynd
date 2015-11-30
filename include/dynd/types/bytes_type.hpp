//
// Copyright (C) 2011-15 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#pragma once

#include <dynd/bytes.hpp>
#include <dynd/type.hpp>
#include <dynd/types/base_bytes_type.hpp>
#include <dynd/typed_data_assign.hpp>

namespace dynd {

typedef bytes bytes_type_data;

namespace ndt {

  /**
   * The bytes type uses memory_block references to store
   * arbitrarily sized runs of bytes.
   */
  class DYND_API bytes_type : public base_bytes_type {
    size_t m_alignment;

  public:
    bytes_type(size_t alignment);

    virtual ~bytes_type();

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

    void get_dynamic_type_properties(const std::pair<std::string, nd::callable> **out_properties,
                                     size_t *out_count) const;

    static const type &make()
    {
      static const type bytes_tp(new bytes_type(1), false);
      return *reinterpret_cast<const type *>(&bytes_tp);
    }

    static type make(size_t alignment) { return type(new bytes_type(alignment), false); }
  };

} // namespace dynd::ndt
} // namespace dynd
