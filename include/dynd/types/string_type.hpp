//
// Copyright (C) 2011-13 Mark Wiebe, DyND Developers
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

namespace dynd {

struct string_type_metadata {
    /**
     * A reference to the memory block which contains the string's data.
     * NOTE: This is identical to bytes_type_metadata, by design. Maybe
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

    string_encoding_t get_encoding() const {
        return m_encoding;
    }

    /** Alignment of the string data being pointed to. */
    size_t get_target_alignment() const {
        return string_encoding_char_size_table[m_encoding];
    }

    void get_string_range(const char **out_begin, const char**out_end, const char *metadata, const char *data) const;
    void set_utf8_string(const char *metadata, char *data, assign_error_mode errmode,
                    const char* utf8_begin, const char *utf8_end) const;

    void print_data(std::ostream& o, const char *metadata, const char *data) const;

    void print_type(std::ostream& o) const;

    bool is_unique_data_owner(const char *metadata) const;
    ndt::type get_canonical_type() const;

    void get_shape(size_t ndim, size_t i, intptr_t *out_shape, const char *metadata) const;

    bool is_lossless_assignment(const ndt::type& dst_tp, const ndt::type& src_tp) const;

    bool operator==(const base_type& rhs) const;

    void metadata_default_construct(char *metadata, size_t ndim, const intptr_t* shape) const;
    void metadata_copy_construct(char *dst_metadata, const char *src_metadata, memory_block_data *embedded_reference) const;
    void metadata_reset_buffers(char *metadata) const;
    void metadata_finalize_buffers(char *metadata) const;
    void metadata_destruct(char *metadata) const;
    void metadata_debug_print(const char *metadata, std::ostream& o, const std::string& indent) const;

    size_t make_assignment_kernel(
                    ckernel_builder *out, size_t offset_out,
                    const ndt::type& dst_tp, const char *dst_metadata,
                    const ndt::type& src_tp, const char *src_metadata,
                    kernel_request_t kernreq, assign_error_mode errmode,
                    const eval::eval_context *ectx) const;

    size_t make_comparison_kernel(
                    ckernel_builder *out, size_t offset_out,
                    const ndt::type& src0_dt, const char *src0_metadata,
                    const ndt::type& src1_dt, const char *src1_metadata,
                    comparison_type_t comptype,
                    const eval::eval_context *ectx) const;
};

namespace ndt {
    inline ndt::type make_string(string_encoding_t encoding = string_encoding_utf_8) {
        return ndt::type(new string_type(encoding), false);
    }
} // namespace ndt

} // namespace dynd

#endif // _DYND__STRING_TYPE_HPP_
