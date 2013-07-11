//
// Copyright (C) 2011-13 Mark Wiebe, DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#ifndef _DYND__BASE_BYTES_TYPE_HPP_
#define _DYND__BASE_BYTES_TYPE_HPP_

#include <dynd/dtypes/base_type.hpp>

namespace dynd {


/**
 * Base class for all bytes dtypes. If a dtype
 * has kind bytes_kind, it must be a subclass of
 * base_bytes_type.
 */
class base_bytes_type : public base_type {
public:
    inline base_bytes_type(type_id_t type_id, type_kind_t kind, size_t data_size,
                    size_t alignment, flags_type flags, size_t metadata_size)
        : base_type(type_id, kind, data_size, alignment, flags, metadata_size, 0)
    {}

    virtual ~base_bytes_type();

    /** Retrieves the data range in which a bytes object is stored */
    virtual void get_bytes_range(const char **out_begin, const char**out_end, const char *metadata, const char *data) const = 0;

    // Bytes dtypes stop the iterdata chain
    // TODO: Maybe it should be more flexible?
    size_t get_iterdata_size(size_t ndim) const;
};


} // namespace dynd

#endif // _DYND__BASE_BYTES_TYPE_HPP_
