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

using namespace dnd;

// Returns true if the destination dtype can represent *all* the values
// of the source dtype, false otherwise.
bool dnd::can_cast_losslessly(dtype dst_dt, dtype src_dt)
{
    if (dst_dt.is_trivial() && src_dt.is_trivial()) {
        switch (src_dt.kind_trivial()) {
            case bool_kind:
                switch (dst_dt.kind_trivial()) {
                    case bool_kind:
                    case int_kind:
                    case uint_kind:
                    case float_kind:
                    case complex_kind:
                        return true;
                    case string_kind:
                        return dst_dt.itemsize_trivial() > 0;
                    default:
                        break;
                }
                break;
            case int_kind:
                switch (dst_dt.kind_trivial()) {
                    case int_kind:
                        return dst_dt.itemsize_trivial() >= src_dt.itemsize_trivial();
                    case uint_kind:
                        return false;
                    case float_kind:
                        return dst_dt.itemsize_trivial() > src_dt.itemsize_trivial();
                    case complex_kind:
                        return dst_dt.itemsize_trivial() > 2 * src_dt.itemsize_trivial();
                    case string_kind:
                        // Conservative value for 64-bit, could
                        // check speciifically based on the type_id.
                        return dst_dt.itemsize_trivial() >= 21;
                    default:
                        break;
                }
                break;
            case uint_kind:
                switch (dst_dt.kind_trivial()) {
                    case int_kind:
                        return dst_dt.itemsize_trivial() > src_dt.itemsize_trivial();
                    case uint_kind:
                        return dst_dt.itemsize_trivial() >= src_dt.itemsize_trivial();
                    case float_kind:
                        return dst_dt.itemsize_trivial() > src_dt.itemsize_trivial();
                    case complex_kind:
                        return dst_dt.itemsize_trivial() > 2 * src_dt.itemsize_trivial();
                    case string_kind:
                        // Conservative value for 64-bit, could
                        // check speciifically based on the type_id.
                        return dst_dt.itemsize_trivial() >= 21;
                    default:
                        break;
                }
                break;
            case float_kind:
                switch (dst_dt.kind_trivial()) {
                    case int_kind:
                    case uint_kind:
                        return false;
                    case float_kind:
                        return dst_dt.itemsize_trivial() >= src_dt.itemsize_trivial();
                    case complex_kind:
                        return dst_dt.itemsize_trivial() >= 2 * src_dt.itemsize_trivial();
                    case string_kind:
                        return dst_dt.itemsize_trivial() >= 32;
                    default:
                        break;
                }
            case complex_kind:
                switch (dst_dt.kind_trivial()) {
                    case int_kind:
                    case uint_kind:
                    case float_kind:
                        return false;
                    case complex_kind:
                        return dst_dt.itemsize_trivial() >= src_dt.itemsize_trivial();
                    case string_kind:
                        return dst_dt.itemsize_trivial() >= 64;
                    default:
                        break;
                }
            case string_kind:
                switch (dst_dt.kind_trivial()) {
                    case int_kind:
                    case uint_kind:
                    case float_kind:
                    case complex_kind:
                        return false;
                    case string_kind:
                        return src_dt.type_id_trivial() == dst_dt.type_id_trivial() &&
                                dst_dt.itemsize_trivial() >= src_dt.itemsize_trivial();
                    default:
                        break;
                }
            default:
                break;
        }
    }

    // TODO: Add more rules, mechanism for custom dtypes
    return false;
}
