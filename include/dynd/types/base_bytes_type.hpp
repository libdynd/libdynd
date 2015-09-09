//
// Copyright (C) 2011-15 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#pragma once

#include <dynd/types/base_type.hpp>

namespace dynd {
namespace ndt {

  /**
   * Base class for all bytes types. If a type
   * has kind bytes_kind, it must be a subclass of
   * base_bytes_type.
   */
  class DYND_API base_bytes_type : public base_type {
  public:
    base_bytes_type(type_id_t type_id, type_kind_t kind, size_t data_size,
                    size_t alignment, flags_type flags, size_t arrmeta_size)
        : base_type(type_id, kind, data_size, alignment, flags, arrmeta_size, 0,
                    0)
    {
    }

    virtual ~base_bytes_type();

    /** Retrieves the data range in which a bytes object is stored */
    virtual void get_bytes_range(const char **out_begin, const char **out_end,
                                 const char *arrmeta,
                                 const char *data) const = 0;

    // Bytes types stop the iterdata chain
    // TODO: Maybe it should be more flexible?
    size_t get_iterdata_size(intptr_t ndim) const;
  };

} // namespace dynd::ndt
} // namespace dynd
