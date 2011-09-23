//
// Copyright (C) 2011 Mark Wiebe (mwwiebe@gmail.com)
// All rights reserved.
//
// This is unreleased proprietary software.
//
#include <dnd/dtype_casting.hpp>

#include <iostream>//DEBUG
#include <stdexcept>
#include <cstring>

using namespace std;
using namespace dnd;

bool dnd::can_cast_exact(const dtype& dst_dt, const dtype& src_dt)
{
    const extended_dtype *dst_ext, *src_ext;

    if (dst_dt.type_id() != src_dt.type_id() || dst_dt.is_byteswapped() != src_dt.is_byteswapped() ||
            dst_dt.itemsize() != src_dt.itemsize()) {
        return false;
    }

    dst_ext = dst_dt.extended();
    src_ext = src_dt.extended();

    // For 'exact', they must either both or neither be NULL
    if (dst_ext == NULL) {
        return (src_ext == NULL);
    }

    return src_ext->can_cast_exact(dst_dt, src_dt);
}

bool dnd::can_cast_equiv(const dtype& dst_dt, const dtype& src_dt)
{
    const extended_dtype *dst_ext, *src_ext;

    // This is the same as can_cast_exact, except here the byteswapped check is skipped
    if (dst_dt.type_id() != src_dt.type_id() || dst_dt.itemsize() != src_dt.itemsize()) {
        return false;
    }

    dst_ext = dst_dt.extended();
    src_ext = src_dt.extended();

    // For 'equiv', they must either both or neither be NULL
    if (dst_ext == NULL) {
        return (src_ext == NULL);
    }

    return src_ext->can_cast_equiv(dst_dt, src_dt);
}

// Returns true if the destination dtype can represent *all* the values
// of the source dtype, false otherwise. This is used, for example,
// to skip any overflow checks when doing value assignments between differing
// types.
bool dnd::can_cast_lossless(const dtype& dst_dt, const dtype& src_dt)
{
    const extended_dtype *dst_ext, *src_ext;

    dst_ext = dst_dt.extended();
    src_ext = src_dt.extended();

    if (dst_ext == NULL && src_ext == NULL) {
        switch (src_dt.kind()) {
            case generic_kind:
                return true;
            case bool_kind:
                switch (dst_dt.kind()) {
                    case bool_kind:
                    case int_kind:
                    case uint_kind:
                    case float_kind:
                    case complex_kind:
                        return true;
                    case string_kind:
                        return dst_dt.itemsize() > 0;
                    default:
                        break;
                }
                break;
            case int_kind:
                switch (dst_dt.kind()) {
                    case bool_kind:
                        return false;
                    case int_kind:
                        return dst_dt.itemsize() >= src_dt.itemsize();
                    case uint_kind:
                        return false;
                    case float_kind:
                        return dst_dt.itemsize() > src_dt.itemsize();
                    case complex_kind:
                        return dst_dt.itemsize() > 2 * src_dt.itemsize();
                    case string_kind:
                        // Conservative value for 64-bit, could
                        // check speciifically based on the type_id.
                        return dst_dt.itemsize() >= 21;
                    default:
                        break;
                }
                break;
            case uint_kind:
                switch (dst_dt.kind()) {
                    case bool_kind:
                        return false;
                    case int_kind:
                        return dst_dt.itemsize() > src_dt.itemsize();
                    case uint_kind:
                        return dst_dt.itemsize() >= src_dt.itemsize();
                    case float_kind:
                        return dst_dt.itemsize() > src_dt.itemsize();
                    case complex_kind:
                        return dst_dt.itemsize() > 2 * src_dt.itemsize();
                    case string_kind:
                        // Conservative value for 64-bit, could
                        // check speciifically based on the type_id.
                        return dst_dt.itemsize() >= 21;
                    default:
                        break;
                }
                break;
            case float_kind:
                switch (dst_dt.kind()) {
                    case bool_kind:
                    case int_kind:
                    case uint_kind:
                        return false;
                    case float_kind:
                        return dst_dt.itemsize() >= src_dt.itemsize();
                    case complex_kind:
                        return dst_dt.itemsize() >= 2 * src_dt.itemsize();
                    case string_kind:
                        return dst_dt.itemsize() >= 32;
                    default:
                        break;
                }
            case complex_kind:
                switch (dst_dt.kind()) {
                    case bool_kind:
                    case int_kind:
                    case uint_kind:
                    case float_kind:
                        return false;
                    case complex_kind:
                        return dst_dt.itemsize() >= src_dt.itemsize();
                    case string_kind:
                        return dst_dt.itemsize() >= 64;
                    default:
                        break;
                }
            case string_kind:
                switch (dst_dt.kind()) {
                    case bool_kind:
                    case int_kind:
                    case uint_kind:
                    case float_kind:
                    case complex_kind:
                        return false;
                    case string_kind:
                        return src_dt.type_id() == dst_dt.type_id() &&
                                dst_dt.itemsize() >= src_dt.itemsize();
                    default:
                        break;
                }
            default:
                break;
        }

        throw std::runtime_error("unhandled built-in case in can_cast_losslessly");
    }

    // Use the available extended_dtype to check the casting
    if (src_ext != NULL) {
        return src_ext->can_cast_lossless(dst_dt, src_dt);
    }
    else {
        return dst_ext->can_cast_lossless(dst_dt, src_dt);
    }
}

bool dnd::can_cast_same_kind(const dtype& dst_dt, const dtype& src_dt)
{
    const extended_dtype *dst_ext, *src_ext;

    dst_ext = dst_dt.extended();
    src_ext = src_dt.extended();

    // For built-in dtypes, compare the kind directly
    if (dst_ext == NULL && src_ext == NULL) {
        return dst_dt.kind() > src_dt.kind();
    }

    // Use the available extended_dtype to check the casting
    if (src_ext != NULL) {
        return src_ext->can_cast_same_kind(dst_dt, src_dt);
    }
    else {
        return dst_ext->can_cast_same_kind(dst_dt, src_dt);
    }
}

