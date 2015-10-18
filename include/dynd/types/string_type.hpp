//
// Copyright (C) 2011-15 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//
// The string type uses memory_block references to store
// arbitrarily sized strings.
//

#pragma once

#include <dynd/string.hpp>
#include <dynd/type.hpp>
#include <dynd/typed_data_assign.hpp>
#include <dynd/types/view_type.hpp>
#include <dynd/string_encodings.hpp>

namespace dynd {

struct DYND_API string_type_arrmeta {
  /**
   * A reference to the memory block which contains the string's data.
   * NOTE: This is identical to bytes_type_arrmeta, by design. Maybe
   *       both should become a typedef to a common class?
   */
  memory_block_data *blockref;
};

namespace ndt {

  class DYND_API string_type : public base_string_type {
  public:
    string_type();

    virtual ~string_type();

    string_encoding_t get_encoding() const
    {
      return string_encoding_utf_8;
    }

    /** Alignment of the string data being pointed to. */
    size_t get_target_alignment() const
    {
      return string_encoding_char_size_table[string_encoding_utf_8];
    }

    void get_string_range(const char **out_begin, const char **out_end, const char *arrmeta, const char *data) const;
    void set_from_utf8_string(const char *arrmeta, char *dst, const char *utf8_begin, const char *utf8_end,
                              const eval::eval_context *ectx) const;

    void print_data(std::ostream &o, const char *arrmeta, const char *data) const;

    void print_type(std::ostream &o) const;

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

    size_t make_comparison_kernel(void *ckb, intptr_t ckb_offset, const type &src0_dt, const char *src0_arrmeta,
                                  const type &src1_dt, const char *src1_arrmeta, comparison_type_t comptype,
                                  const eval::eval_context *ectx) const;

    void make_string_iter(dim_iter *out_di, string_encoding_t encoding, const char *arrmeta, const char *data,
                          const memory_block_ptr &ref, intptr_t buffer_max_mem, const eval::eval_context *ectx) const;

    /** Returns type "string" */
    static const type &make()
    {
      static const type string_tp(new string_type(), false);
      return string_tp;
    }
  };

} // namespace dynd::ndt
} // namespace dynd
