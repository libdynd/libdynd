//
// Copyright (C) 2011-15 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//
// The fixed_string type represents a string with
// a particular encoding, stored in a fixed-size
// buffer.
//

#pragma once

#include <dynd/string_encodings.hpp>
#include <dynd/type.hpp>

namespace dynd {
namespace ndt {

  class DYNDT_API fixed_string_type : public base_string_type {
    intptr_t m_stringsize;
    string_encoding_t m_encoding;

  public:
    fixed_string_type(intptr_t stringsize, string_encoding_t encoding);

    intptr_t get_size() const { return m_stringsize; }
    string_encoding_t get_encoding() const { return m_encoding; }

    void get_string_range(const char **out_begin, const char **out_end, const char *arrmeta, const char *data) const;
    void set_from_utf8_string(const char *arrmeta, char *dst, const char *utf8_begin, const char *utf8_end,
                              const eval::eval_context *ectx) const;

    void print_data(std::ostream &o, const char *arrmeta, const char *data) const;

    void print_type(std::ostream &o) const;

    type get_canonical_type() const;

    bool is_lossless_assignment(const type &dst_tp, const type &src_tp) const;

    bool operator==(const base_type &rhs) const;

    void arrmeta_default_construct(char *DYND_UNUSED(arrmeta), bool DYND_UNUSED(blockref_alloc)) const {}
    void arrmeta_copy_construct(char *DYND_UNUSED(dst_arrmeta), const char *DYND_UNUSED(src_arrmeta),
                                const intrusive_ptr<memory_block_data> &DYND_UNUSED(embedded_reference)) const
    {
    }
    void arrmeta_destruct(char *DYND_UNUSED(arrmeta)) const {}
    void arrmeta_debug_print(const char *DYND_UNUSED(arrmeta), std::ostream &DYND_UNUSED(o),
                             const std::string &DYND_UNUSED(indent)) const
    {
    }

    static type make(intptr_t stringsize, string_encoding_t encoding = string_encoding_utf_8)
    {
      return type(new fixed_string_type(stringsize, encoding), false);
    }
  };

} // namespace dynd::ndt
} // namespace dynd
