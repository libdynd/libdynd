//
// Copyright (C) 2011-14 Mark Wiebe, DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#ifndef _DYND__JSON_TYPE_HPP_
#define _DYND__JSON_TYPE_HPP_

#include <dynd/types/string_type.hpp>

namespace dynd {

// The json type is stored as a string, but limited to
// UTF-8 and is supposed to contain JSON data.
typedef string_type_metadata json_type_metadata;
typedef string_type_data json_type_data;

class json_type : public base_string_type {
public:
    json_type();

    virtual ~json_type();

    string_encoding_t get_encoding() const {
        return string_encoding_utf_8;
    }

    void get_string_range(const char **out_begin, const char**out_end, const char *metadata, const char *data) const;
    void set_utf8_string(const char *metadata, char *data, assign_error_mode errmode,
                    const char* utf8_begin, const char *utf8_end) const;

    void print_data(std::ostream& o, const char *metadata, const char *data) const;

    void print_type(std::ostream& o) const;

    bool is_unique_data_owner(const char *metadata) const;
    ndt::type get_canonical_type() const;

    bool is_lossless_assignment(const ndt::type& dst_tp, const ndt::type& src_tp) const;

    bool operator==(const base_type& rhs) const;

    void metadata_default_construct(char *metadata, intptr_t ndim, const intptr_t* shape) const;
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

    void make_string_iter(dim_iter *out_di, string_encoding_t encoding,
            const char *metadata, const char *data,
            const memory_block_ptr& ref,
            intptr_t buffer_max_mem,
            const eval::eval_context *ectx) const;
};

namespace ndt {
    inline ndt::type make_json() {
        return ndt::type(new json_type(), false);
    }
} // namespace ndt

} // namespace dynd

#endif // _DYND__JSON_TYPE_HPP_

