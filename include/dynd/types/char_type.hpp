//
// Copyright (C) 2011-14 Mark Wiebe, DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//
// The char type represents a single character of a specified encoding.
// Its canonical type, datashape "char" is a unicode codepoint stored
// as a 32-bit integer (effectively UTF-32).
//
#ifndef _DYND__CHAR_TYPE_HPP_
#define _DYND__CHAR_TYPE_HPP_

#include <dynd/type.hpp>
#include <dynd/typed_data_assign.hpp>
#include <dynd/types/view_type.hpp>
#include <dynd/string_encodings.hpp>
#include <dynd/types/static_type_instances.hpp>

namespace dynd {

class char_type : public base_type {
    // This encoding can be ascii, latin1, ucs2, or utf32.
    // Not a variable-sized encoding.
    string_encoding_t m_encoding;

public:
    char_type(string_encoding_t encoding);

    virtual ~char_type();

    string_encoding_t get_encoding() const {
        return m_encoding;
    }

    /** Alignment of the string data being pointed to. */
    size_t get_target_alignment() const {
        return string_encoding_char_size_table[m_encoding];
    }

    // Retrieves the character as a unicode code point
    uint32_t get_code_point(const char *data) const;
    // Sets the character as a unicode code point
    void set_code_point(char *out_data, uint32_t cp);

    void print_data(std::ostream& o, const char *arrmeta, const char *data) const;

    void print_type(std::ostream& o) const;

    ndt::type get_canonical_type() const;

    bool is_lossless_assignment(const ndt::type& dst_tp, const ndt::type& src_tp) const;

    bool operator==(const base_type& rhs) const;

    void arrmeta_default_construct(char *DYND_UNUSED(arrmeta),
                                   intptr_t DYND_UNUSED(ndim),
                                   const intptr_t *DYND_UNUSED(shape),
                                   bool DYND_UNUSED(blockref_alloc)) const
    {
    }
    void arrmeta_copy_construct(char *DYND_UNUSED(dst_arrmeta), const char *DYND_UNUSED(src_arrmeta), memory_block_data *DYND_UNUSED(embedded_reference)) const {
    }
    void arrmeta_destruct(char *DYND_UNUSED(arrmeta)) const {
    }
    void arrmeta_debug_print(const char *DYND_UNUSED(arrmeta), std::ostream& DYND_UNUSED(o), const std::string& DYND_UNUSED(indent)) const {
    }

    size_t make_assignment_kernel(ckernel_builder *ckb, intptr_t ckb_offset,
                                  const ndt::type &dst_tp,
                                  const char *dst_arrmeta,
                                  const ndt::type &src_tp,
                                  const char *src_arrmeta,
                                  kernel_request_t kernreq,
                                  const eval::eval_context *ectx) const;

    size_t make_comparison_kernel(ckernel_builder *ckb, intptr_t ckb_offset,
                                  const ndt::type &src0_dt,
                                  const char *src0_arrmeta,
                                  const ndt::type &src1_dt,
                                  const char *src1_arrmeta,
                                  comparison_type_t comptype,
                                  const eval::eval_context *ectx) const;
};

namespace ndt {
  inline const ndt::type &
  make_char()
  {
    return *reinterpret_cast<const ndt::type *>(&types::char_tp);
  }

  inline ndt::type
  make_char(string_encoding_t encoding)
  {
    return ndt::type(new char_type(encoding), false);
  }
} // namespace ndt

} // namespace dynd

#endif // _DYND__CHAR_TYPE_HPP_
