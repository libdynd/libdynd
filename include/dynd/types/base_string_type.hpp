//
// Copyright (C) 2011-15 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#pragma once

#include <dynd/types/base_type.hpp>
#include <dynd/string_encodings.hpp>

namespace dynd {
namespace ndt {

  /**
   * Base class for all string types. If a type
   * has kind string_kind, it must be a subclass of
   * base_string_type.
   */
  class DYND_API base_string_type : public base_type {
  public:
    base_string_type(type_id_t type_id, type_kind_t type_kind, size_t data_size,
                     size_t alignment, flags_type flags, size_t arrmeta_size)
        : base_type(type_id, type_kind, data_size, alignment, flags,
                    arrmeta_size, 0, 0)
    {
    }

    base_string_type(type_id_t type_id, size_t data_size, size_t alignment,
                     flags_type flags, size_t arrmeta_size)
        : base_type(type_id, string_kind, data_size, alignment, flags,
                    arrmeta_size, 0, 0)
    {
    }

    virtual ~base_string_type();
    /** The encoding used by the string */
    virtual string_encoding_t get_encoding() const = 0;

    /** Retrieves the data range in which a string is stored */
    virtual void get_string_range(const char **out_begin, const char **out_end,
                                  const char *arrmeta,
                                  const char *data) const = 0;
    /** Converts a string element into a C++ std::string with a UTF8 encoding */
    std::string get_utf8_string(const char *arrmeta, const char *data,
                                assign_error_mode errmode) const;

    // String types stop the iterdata chain
    // TODO: Maybe it should be more flexible?
    size_t get_iterdata_size(intptr_t ndim) const;

    void get_dynamic_type_properties(
        const std::pair<std::string, nd::callable> **out_properties,
        size_t *out_count) const;

    void get_dynamic_array_functions(
        const std::pair<std::string, gfunc::callable> **out_functions,
        size_t *out_count) const;
  };

} // namespace dynd::ndt
} // namespace dynd
