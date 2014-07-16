//
// Copyright (C) 2011-14 Mark Wiebe, DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#ifndef _DYND__BASE_STRING_TYPE_HPP_
#define _DYND__BASE_STRING_TYPE_HPP_

#include <dynd/types/base_type.hpp>
#include <dynd/string_encodings.hpp>
#include <dynd/dim_iter.hpp>

namespace dynd {


/**
 * Base class for all string types. If a type
 * has kind string_kind, it must be a subclass of
 * base_string_type.
 */
class base_string_type : public base_type {
public:
  inline base_string_type(type_id_t type_id, size_t data_size, size_t alignment,
                          flags_type flags, size_t arrmeta_size)
      : base_type(type_id, string_kind, data_size, alignment, flags,
                  arrmeta_size, 0, 0)
    {}

    virtual ~base_string_type();
    /** The encoding used by the string */
    virtual string_encoding_t get_encoding() const = 0;

    /** Retrieves the data range in which a string is stored */
    virtual void get_string_range(const char **out_begin, const char**out_end,
                    const char *arrmeta, const char *data) const = 0;
    /** Converts a string element into a C++ std::string with a UTF8 encoding */
    std::string get_utf8_string(const char *arrmeta, const char *data, assign_error_mode errmode) const;


    /**
     * Creates a dim_iter for processing a single instance of the string.
     *
     * The iterator which is created must be contiguous. The type it is
     * iterating over will either be "char[encoding]" for fixed-size
     * encodings, "bytes[1,1]" for utf8 or "bytes[2,2]" for utf16.
     *
     * The encoding is included as a parameter so that the string type can
     * provide a more optimal iterator, e.g. if the underlying data is in
     * utf8, but the string has a different encoding, a utf8 iterator request
     * can avoid multiple reencodings.
     *
     * \param out_di  The dim_iter to populate. This should point to an
     *                uninitialized dim_iter.
     * \param encoding  The encoding the user of the iterator should see.
     * \param arrmeta  Arrmeta of the string.
     * \param data  Data of the string.
     * \param ref  A reference which holds on to the memory of the string.
     */
    virtual void make_string_iter(
        dim_iter *out_di, string_encoding_t encoding, const char *arrmeta,
        const char *data, const memory_block_ptr &ref,
        intptr_t buffer_max_mem = 65536,
        const eval::eval_context *ectx = &eval::default_eval_context) const = 0;

    // String types stop the iterdata chain
    // TODO: Maybe it should be more flexible?
    size_t get_iterdata_size(intptr_t ndim) const;

    void get_dynamic_type_properties(
                    const std::pair<std::string, gfunc::callable> **out_properties,
                    size_t *out_count) const;

    void get_dynamic_array_functions(
                    const std::pair<std::string, gfunc::callable> **out_functions,
                    size_t *out_count) const;
};


} // namespace dynd

#endif // _DYND__BASE_STRING_TYPE_HPP_
