//
// Copyright (C) 2011-13, DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//
// The string dtype uses memory_block references to store
// arbitrarily sized strings.
//
#ifndef _DYND__STRING_DTYPE_HPP_
#define _DYND__STRING_DTYPE_HPP_

#include <dynd/dtype.hpp>
#include <dynd/dtype_assign.hpp>
#include <dynd/dtypes/view_dtype.hpp>
#include <dynd/string_encodings.hpp>

namespace dynd {

struct string_dtype_metadata {
    /**
     * A reference to the memory block which contains the string's data.
     * NOTE: This is identical to bytes_dtype_metadata, by design. Maybe
     *       both should become a typedef to a common class?
     */
    memory_block_data *blockref;
};

struct string_dtype_data {
    char *begin;
    char *end;
};

class string_dtype : public base_string_dtype {
    string_encoding_t m_encoding;

public:
    string_dtype(string_encoding_t encoding);

    virtual ~string_dtype();

    string_encoding_t get_encoding() const {
        return m_encoding;
    }

    /** Alignment of the string data being pointed to. */
    size_t get_data_alignment() const {
        return string_encoding_char_size_table[m_encoding];
    }

    void get_string_range(const char **out_begin, const char**out_end, const char *metadata, const char *data) const;
    void set_utf8_string(const char *metadata, char *data, assign_error_mode errmode,
                    const char* utf8_begin, const char *utf8_end) const;

    void print_data(std::ostream& o, const char *metadata, const char *data) const;

    void print_dtype(std::ostream& o) const;

    bool is_unique_data_owner(const char *metadata) const;
    dtype get_canonical_dtype() const;

    void get_shape(size_t i, intptr_t *out_shape) const;
    void get_shape(size_t i, intptr_t *out_shape, const char *metadata) const;

    bool is_lossless_assignment(const dtype& dst_dt, const dtype& src_dt) const;

    bool operator==(const base_dtype& rhs) const;

    void metadata_default_construct(char *metadata, size_t ndim, const intptr_t* shape) const;
    void metadata_copy_construct(char *dst_metadata, const char *src_metadata, memory_block_data *embedded_reference) const;
    void metadata_reset_buffers(char *metadata) const;
    void metadata_finalize_buffers(char *metadata) const;
    void metadata_destruct(char *metadata) const;
    void metadata_debug_print(const char *metadata, std::ostream& o, const std::string& indent) const;

    size_t make_assignment_kernel(
                    hierarchical_kernel *out, size_t offset_out,
                    const dtype& dst_dt, const char *dst_metadata,
                    const dtype& src_dt, const char *src_metadata,
                    kernel_request_t kernreq, assign_error_mode errmode,
                    const eval::eval_context *ectx) const;

    size_t make_comparison_kernel(
                    hierarchical_kernel *out, size_t offset_out,
                    const dtype& src0_dt, const char *src0_metadata,
                    const dtype& src1_dt, const char *src1_metadata,
                    comparison_type_t comptype,
                    const eval::eval_context *ectx) const;
};

inline dtype make_string_dtype(string_encoding_t encoding = string_encoding_utf_8) {
    return dtype(new string_dtype(encoding), false);
}

} // namespace dynd

#endif // _DYND__STRING_DTYPE_HPP_
