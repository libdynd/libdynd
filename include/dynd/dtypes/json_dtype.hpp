//
// Copyright (C) 2011-13 Mark Wiebe, DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#ifndef _DYND__JSON_TYPE_HPP_
#define _DYND__JSON_TYPE_HPP_

#include <dynd/dtypes/string_dtype.hpp>

namespace dynd {

// The json dtype is stored as a string, but limited to
// UTF-8 and is supposed to contain JSON data.
typedef string_dtype_metadata json_dtype_metadata;
typedef string_dtype_data json_dtype_data;

class json_dtype : public base_string_dtype {
public:
    json_dtype();

    virtual ~json_dtype();

    string_encoding_t get_encoding() const {
        return string_encoding_utf_8;
    }

    void get_string_range(const char **out_begin, const char**out_end, const char *metadata, const char *data) const;
    void set_utf8_string(const char *metadata, char *data, assign_error_mode errmode,
                    const char* utf8_begin, const char *utf8_end) const;

    void print_data(std::ostream& o, const char *metadata, const char *data) const;

    void print_dtype(std::ostream& o) const;

    bool is_unique_data_owner(const char *metadata) const;
    ndt::type get_canonical_type() const;

    bool is_lossless_assignment(const ndt::type& dst_dt, const ndt::type& src_dt) const;

    bool operator==(const base_dtype& rhs) const;

    void metadata_default_construct(char *metadata, size_t ndim, const intptr_t* shape) const;
    void metadata_copy_construct(char *dst_metadata, const char *src_metadata, memory_block_data *embedded_reference) const;
    void metadata_reset_buffers(char *metadata) const;
    void metadata_finalize_buffers(char *metadata) const;
    void metadata_destruct(char *metadata) const;
    void metadata_debug_print(const char *metadata, std::ostream& o, const std::string& indent) const;

    size_t make_assignment_kernel(
                    hierarchical_kernel *out, size_t offset_out,
                    const ndt::type& dst_dt, const char *dst_metadata,
                    const ndt::type& src_dt, const char *src_metadata,
                    kernel_request_t kernreq, assign_error_mode errmode,
                    const eval::eval_context *ectx) const;
};

inline ndt::type make_json_dtype() {
    return ndt::type(new json_dtype(), false);
}

} // namespace dynd

#endif // _DYND__JSON_TYPE_HPP_

