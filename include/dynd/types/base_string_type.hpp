//
// Copyright (C) 2011-15 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#pragma once

#include <dynd/string_encodings.hpp>
#include <dynd/types/base_type.hpp>

namespace dynd {
namespace ndt {

  /**
   * Base class for all string types. If a type
   * has kind string_kind, it must be a subclass of
   * base_string_type.
   */
  class DYNDT_API base_string_type : public base_type {
  private:
    const string_encoding_t m_encoding = string_encoding_ascii;

  public:
    base_string_type(type_id_t type_id, size_t data_size, size_t alignment, uint32_t flags, size_t arrmeta_size)
        : base_type(type_id, data_size, alignment, flags, arrmeta_size, 0, 0)
    {
    }

    /** The encoding used by the string */
    virtual string_encoding_t get_encoding() const { return m_encoding; }

    /** Retrieves the data range in which a string is stored */
    virtual void get_string_range(const char **out_begin, const char **out_end, const char *arrmeta,
                                  const char *data) const = 0;
    /** Converts a string element into a C++ std::string with a UTF8 encoding */
    std::string get_utf8_string(const char *arrmeta, const char *data, assign_error_mode errmode) const;

    // String types stop the iterdata chain
    // TODO: Maybe it should be more flexible?
    size_t get_iterdata_size(intptr_t ndim) const;

    std::map<std::string, std::pair<ndt::type, const char *>> get_dynamic_type_properties() const;
  };

} // namespace dynd::ndt
} // namespace dynd
