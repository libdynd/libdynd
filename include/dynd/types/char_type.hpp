//
// Copyright (C) 2011-15 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//
// The char type represents a single character of a specified encoding.
// Its canonical type, datashape "char" is a unicode codepoint stored
// as a 32-bit integer (effectively UTF-32).
//

#pragma once

#include <dynd/string_encodings.hpp>
#include <dynd/type.hpp>

namespace dynd {
namespace ndt {

  class DYNDT_API char_type : public base_type {
    // This encoding can be ascii, latin1, ucs2, or utf32.
    // Not a variable-sized encoding.
    string_encoding_t m_encoding;

  public:
    char_type(string_encoding_t encoding = string_encoding_utf_32);

    string_encoding_t get_encoding() const { return m_encoding; }

    /** Alignment of the string data being pointed to. */
    size_t get_target_alignment() const { return string_encoding_char_size_table[m_encoding]; }

    // Retrieves the character as a unicode code point
    uint32_t get_code_point(const char *data) const;
    // Sets the character as a unicode code point
    void set_code_point(char *out_data, uint32_t cp);

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
  };

} // namespace dynd::ndt
} // namespace dynd
