//
// Copyright (C) 2011-14 Mark Wiebe, DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//
// The string type uses memory_block references to store
// arbitrarily sized strings.
//
#ifndef _DYND__STRING_TYPE_HPP_
#define _DYND__STRING_TYPE_HPP_

#include <dynd/type.hpp>
#include <dynd/typed_data_assign.hpp>
#include <dynd/types/view_type.hpp>
#include <dynd/string_encodings.hpp>
#include <dynd/types/static_type_instances.hpp>

namespace dynd {

struct string_type_arrmeta {
    /**
     * A reference to the memory block which contains the string's data.
     * NOTE: This is identical to bytes_type_arrmeta, by design. Maybe
     *       both should become a typedef to a common class?
     */
    memory_block_data *blockref;
};

struct string_type_data {
    char *begin;
    char *end;
};

class string_type : public base_string_type {
    string_encoding_t m_encoding;

public:
    string_type(string_encoding_t encoding);

    virtual ~string_type();

    inline string_encoding_t get_encoding() const {
        return m_encoding;
    }

    /** Alignment of the string data being pointed to. */
    inline size_t get_target_alignment() const {
        return string_encoding_char_size_table[m_encoding];
    }

    void get_string_range(const char **out_begin, const char **out_end,
                          const char *arrmeta, const char *data) const;
    void set_from_utf8_string(const char *arrmeta, char *dst,
                              const char *utf8_begin, const char *utf8_end,
                              const eval::eval_context *ectx) const;

    void print_data(std::ostream& o, const char *arrmeta, const char *data) const;

    void print_type(std::ostream& o) const;

    bool is_unique_data_owner(const char *arrmeta) const;
    ndt::type get_canonical_type() const;

        void get_shape(intptr_t ndim, intptr_t i, intptr_t *out_shape, const char *arrmeta, const char *data) const;

    bool is_lossless_assignment(const ndt::type& dst_tp, const ndt::type& src_tp) const;

    bool operator==(const base_type& rhs) const;

    void arrmeta_default_construct(char *arrmeta, intptr_t ndim,
                                   const intptr_t *shape,
                                   bool blockref_alloc) const;
    void arrmeta_copy_construct(char *dst_arrmeta, const char *src_arrmeta, memory_block_data *embedded_reference) const;
    void arrmeta_reset_buffers(char *arrmeta) const;
    void arrmeta_finalize_buffers(char *arrmeta) const;
    void arrmeta_destruct(char *arrmeta) const;
    void arrmeta_debug_print(const char *arrmeta, std::ostream& o, const std::string& indent) const;

    size_t make_assignment_kernel(ckernel_builder *ckb, intptr_t ckb_offset,
                                  const ndt::type &dst_tp,
                                  const char *dst_arrmeta,
                                  const ndt::type &src_tp,
                                  const char *src_arrmeta,
                                  kernel_request_t kernreq,
                                  const eval::eval_context *ectx) const;

    size_t make_comparison_kernel(
                    ckernel_builder *ckb, intptr_t ckb_offset,
                    const ndt::type& src0_dt, const char *src0_arrmeta,
                    const ndt::type& src1_dt, const char *src1_arrmeta,
                    comparison_type_t comptype,
                    const eval::eval_context *ectx) const;

    void make_string_iter(dim_iter *out_di, string_encoding_t encoding,
            const char *arrmeta, const char *data,
            const memory_block_ptr& ref,
            intptr_t buffer_max_mem,
            const eval::eval_context *ectx) const;

    nd::array get_option_nafunc() const;
};

namespace ndt {
  /** Returns type "string" */
  inline const ndt::type &make_string()
  {
    return *reinterpret_cast<const ndt::type *>(&types::string_tp);
  }
  /** Returns type "string[<encoding>]" */
  inline ndt::type make_string(string_encoding_t encoding)
  {
    return ndt::type(new string_type(encoding), false);
  }
  /** Returns type "strided * string" */
  inline const ndt::type &make_strided_of_string()
  {
    return *reinterpret_cast<const ndt::type *>(&types::strided_of_string_tp);
  }
} // namespace ndt

} // namespace dynd

#endif // _DYND__STRING_TYPE_HPP_
