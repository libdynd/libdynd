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

struct DYND_API bytes_type_arrmeta {
  /**
   * A reference to the memory block which contains the byte's data.
   * NOTE: This is identical to string_type_arrmeta, by design. Maybe
   *       both should become a typedef to a common class?
   */
  memory_block_data *blockref;
};

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
    size_t get_target_alignment() const
    {
      return m_alignment;
    }

    void print_data(std::ostream &o, const char *arrmeta, const char *data) const;

    void print_type(std::ostream &o) const;

    void get_bytes_range(const char **out_begin, const char **out_end, const char *arrmeta, const char *data) const;
    void set_bytes_data(const char *arrmeta, char *data, const char *bytes_begin, const char *bytes_end) const;

    bool is_unique_data_owner(const char *arrmeta) const;
    type get_canonical_type() const;

    void get_shape(intptr_t ndim, intptr_t i, intptr_t *out_shape, const char *arrmeta, const char *data) const;

    bool is_lossless_assignment(const type &dst_tp, const type &src_tp) const;

    bool operator==(const base_type &rhs) const;

    void arrmeta_default_construct(char *arrmeta, bool blockref_alloc) const;
    void arrmeta_copy_construct(char *dst_arrmeta, const char *src_arrmeta,
                                memory_block_data *embedded_reference) const;
    void arrmeta_reset_buffers(char *arrmeta) const;
    void arrmeta_finalize_buffers(char *arrmeta) const;
    void arrmeta_destruct(char *arrmeta) const;
    void arrmeta_debug_print(const char *arrmeta, std::ostream &o, const std::string &indent) const;

    intptr_t make_assignment_kernel(void *ckb, intptr_t ckb_offset, const type &dst_tp, const char *dst_arrmeta,
                                    const type &src_tp, const char *src_arrmeta, kernel_request_t kernreq,
                                    const eval::eval_context *ectx) const;

    void get_dynamic_type_properties(const std::pair<std::string, nd::callable> **out_properties,
                                     size_t *out_count) const;

    static const type &make()
    {
      static const type bytes_tp(new bytes_type(1), false);
      return *reinterpret_cast<const type *>(&bytes_tp);
    }

    static type make(size_t alignment)
    {
      return type(new bytes_type(alignment), false);
    }
  };

} // namespace dynd::ndt
} // namespace dynd
