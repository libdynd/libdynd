//
// Copyright (C) 2011-13, DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#ifndef _DYND__BASE_STRING_DTYPE_HPP_
#define _DYND__BASE_STRING_DTYPE_HPP_

#include <dynd/dtypes/base_dtype.hpp>
#include <dynd/string_encodings.hpp>

namespace dynd {


/**
 * Base class for all string dtypes. If a dtype
 * has kind string_kind, it must be a subclass of
 * base_string_dtype.
 */
class base_string_dtype : public base_dtype {
public:
    inline base_string_dtype(type_id_t type_id, size_t data_size,
                    size_t alignment, flags_type flags, size_t metadata_size)
        : base_dtype(type_id, string_kind, data_size, alignment, flags, metadata_size, 0)
    {}

    virtual ~base_string_dtype();
    /** The encoding used by the string */
    virtual string_encoding_t get_encoding() const = 0;

    /** Retrieves the data range in which a string is stored */
    virtual void get_string_range(const char **out_begin, const char**out_end,
                    const char *metadata, const char *data) const = 0;
    /** Converts a string element into a C++ std::string with a UTF8 encoding */
    std::string get_utf8_string(const char *metadata, const char *data, assign_error_mode errmode) const;
    /** Copies a string with a UTF8 encoding to a string element */
    virtual void set_utf8_string(const char *metadata, char *data, assign_error_mode errmode,
                    const char* utf8_begin, const char *utf8_end) const = 0;
    /** Copies a C++ std::string with a UTF8 encoding to a string element */
    inline void set_utf8_string(const char *metadata, char *data, assign_error_mode errmode,
                    const std::string& utf8_str) const {
        set_utf8_string(metadata, data, errmode, utf8_str.data(), utf8_str.data() + utf8_str.size());
    }

    // String dtypes stop the iterdata chain
    // TODO: Maybe it should be more flexible?
    size_t get_iterdata_size(size_t ndim) const;

    void get_dynamic_dtype_properties(const std::pair<std::string, gfunc::callable> **out_properties, size_t *out_count) const;
};


} // namespace dynd

#endif // _DYND__BASE_STRING_DTYPE_HPP_
