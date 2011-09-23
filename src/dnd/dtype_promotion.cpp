//
// Copyright (C) 2011 Mark Wiebe (mwwiebe@gmail.com)
// All rights reserved.
//
// This is unreleased proprietary software.
//

#include <stdexcept>

#include <dnd/dtype_promotion.hpp>

using namespace std;
using namespace dnd;

static intptr_t min_strlen_for_builtin_kind(dtype_kind kind)
{
    switch (kind) {
        case bool_kind:
            return 1;
        case int_kind:
        case uint_kind:
            return 24;
        case float_kind:
            return 32;
        case complex_kind:
            return 64;
        default:
            throw runtime_error("cannot get minimum string length for specified kind");
    }
}

dtype dnd::promote_dtypes_arithmetic(const dtype& dt0, const dtype& dt1)
{
    const extended_dtype *dt0_ext, *dt1_ext;
    intptr_t itemsize = 0;

    dt0_ext = dt0.extended();
    dt1_ext = dt1.extended();

    if (dt0_ext == NULL && dt1_ext == NULL) {
        switch (dt0.kind()) {
            case bool_kind:
                switch (dt1.kind()) {
                    case bool_kind:
                        return make_dtype<int>();
                    case int_kind:
                    case uint_kind:
                        return (dt1.itemsize() >= sizeof(int)) ? dt1 : make_dtype<int>();
                    default:
                        return dt1;
                }
            case int_kind:
                switch (dt1.kind()) {
                    case bool_kind:
                        return (dt0.itemsize() >= sizeof(int)) ? dt0 : make_dtype<int>();
                    case int_kind:
                        if (dt0.itemsize() < sizeof(int) && dt1.itemsize() < sizeof(int)) {
                            return make_dtype<int>();
                        } else {
                            return (dt0.itemsize() >= dt1.itemsize()) ? dt0 : dt1;
                        }
                    case uint_kind:
                        if (dt0.itemsize() < sizeof(int) && dt1.itemsize() < sizeof(int)) {
                            return make_dtype<int>();
                        } else {
                            // When the itemsizes are equal, the uint kind wins
                            return (dt0.itemsize() > dt1.itemsize()) ? dt0 : dt1;
                        }
                    case float_kind:
                        // Integer type sizes don't affect float type sizes
                        return dt1;
                    case complex_kind:
                        // Integer type sizes don't affect complex type sizes
                        return dt1;
                    case string_kind:
                        // Presently UTF8 is the only built-in string type
                        itemsize = min_strlen_for_builtin_kind(dt0.kind());
                        if (dt1.itemsize() > itemsize) {
                            itemsize = dt1.itemsize();
                        }
                        return dtype(utf8_type_id, itemsize);
                    default:
                        break;
                }
                break;
            case uint_kind:
                switch (dt1.kind()) {
                    case bool_kind:
                        return (dt0.itemsize() >= sizeof(int)) ? dt0 : make_dtype<int>();
                    case int_kind:
                        if (dt0.itemsize() < sizeof(int) && dt1.itemsize() < sizeof(int)) {
                            return make_dtype<int>();
                        } else {
                            // When the itemsizes are equal, the uint kind wins
                            return (dt0.itemsize() >= dt1.itemsize()) ? dt0 : dt1;
                        }
                    case uint_kind:
                        if (dt0.itemsize() < sizeof(int) && dt1.itemsize() < sizeof(int)) {
                            return make_dtype<int>();
                        } else {
                            return (dt0.itemsize() >= dt1.itemsize()) ? dt0 : dt1;
                        }
                    case float_kind:
                        // Integer type sizes don't affect float type sizes
                        return dt1;
                    case complex_kind:
                        // Integer type sizes don't affect complex type sizes
                        return dt1;
                    case string_kind:
                        // Presently UTF8 is the only built-in string type
                        itemsize = min_strlen_for_builtin_kind(dt0.kind());
                        if (dt1.itemsize() > itemsize) {
                            itemsize = dt1.itemsize();
                        }
                        return dtype(utf8_type_id, itemsize);
                    default:
                        break;
                }
                break;
            case float_kind:
                switch (dt1.kind()) {
                    // Integer type sizes don't affect float type sizes
                    case bool_kind:
                    case int_kind:
                    case uint_kind:
                        return dt0;
                    case float_kind:
                        return (dt0.itemsize() >= dt1.itemsize()) ? dt0 : dt1;
                    case complex_kind:
                        // Float type sizes don't affect complex type sizes
                        return dt1;
                    case string_kind:
                        // Presently UTF8 is the only built-in string type
                        itemsize = min_strlen_for_builtin_kind(dt0.kind());
                        if (dt1.itemsize() > itemsize) {
                            itemsize = dt1.itemsize();
                        }
                        return dtype(utf8_type_id, itemsize);
                    default:
                        break;
                }
                break;
            case complex_kind:
                switch (dt1.kind()) {
                    // Integer and float type sizes don't affect complex type sizes
                    case bool_kind:
                    case int_kind:
                    case uint_kind:
                    case float_kind:
                        return dt0;
                    case complex_kind:
                        return (dt0.itemsize() >= dt1.itemsize()) ? dt0 : dt1;
                    case string_kind:
                        // Presently UTF8 is the only built-in string type
                        itemsize = min_strlen_for_builtin_kind(dt0.kind());
                        if (dt1.itemsize() > itemsize) {
                            itemsize = dt1.itemsize();
                        }
                        return dtype(utf8_type_id, itemsize);
                    default:
                        break;
                }
                break;
            case string_kind:
                switch (dt1.kind()) {
                    case bool_kind:
                    case int_kind:
                    case uint_kind:
                    case float_kind:
                    case complex_kind:
                        // Presently UTF8 is the only built-in string type
                        itemsize = min_strlen_for_builtin_kind(dt1.kind());
                        if (dt0.itemsize() > itemsize) {
                            itemsize = dt0.itemsize();
                        }
                        return dtype(utf8_type_id, itemsize);
                    case string_kind:
                        return (dt0.itemsize() >= dt1.itemsize()) ? dt0 : dt1;
                    default:
                        break;
                }
                break;
            default:
                break;
        }

        throw std::runtime_error("internal error in built-in dtype promotion");
    }

    throw std::runtime_error("type promotion of custom dtypes is not yet supported");
}
