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

static intptr_t min_strlen_for_builtin_kind(dtype_kind_t kind)
{
    switch (kind) {
        case bool_kind:
            return 1;
        case int_kind:
        case uint_kind:
            return 24;
        case real_kind:
            return 32;
        case complex_kind:
            return 64;
        default:
            throw runtime_error("cannot get minimum string length for specified kind");
    }
}

dtype dnd::promote_dtypes_arithmetic(const dtype& dt0, const dtype& dt1)
{
    // Use the value dtypes
    const dtype& dt0_val = dt0.value_dtype();
    const dtype& dt1_val = dt1.value_dtype();

    const extended_dtype *dt0_ext, *dt1_ext;
    uintptr_t itemsize = 0;

    dt0_ext = dt0_val.extended();
    dt1_ext = dt1_val.extended();

    //cout << "Doing type promotion with value types " << dt0_val << " and " << dt1_val << endl;

    if (dt0_ext == NULL && dt1_ext == NULL) {
        switch (dt0_val.kind()) {
            case bool_kind:
                switch (dt1_val.kind()) {
                    case bool_kind:
                        return make_dtype<int>();
                    case int_kind:
                    case uint_kind:
                        return (dt1_val.itemsize() >= sizeof(int)) ? dt1_val
                                                               : make_dtype<int>();
                    default:
                        return dt1_val;
                }
            case int_kind:
                switch (dt1_val.kind()) {
                    case bool_kind:
                        return (dt0_val.itemsize() >= sizeof(int)) ? dt0_val
                                                               : make_dtype<int>();
                    case int_kind:
                        if (dt0_val.itemsize() < sizeof(int) && dt1_val.itemsize() < sizeof(int)) {
                            return make_dtype<int>();
                        } else {
                            return (dt0_val.itemsize() >= dt1_val.itemsize()) ? dt0_val
                                                                              : dt1_val;
                        }
                    case uint_kind:
                        if (dt0_val.itemsize() < sizeof(int) && dt1_val.itemsize() < sizeof(int)) {
                            return make_dtype<int>();
                        } else {
                            // When the itemsizes are equal, the uint kind wins
                            return (dt0_val.itemsize() > dt1_val.itemsize()) ? dt0_val
                                                                             : dt1_val;
                        }
                    case real_kind:
                        // Integer type sizes don't affect float type sizes
                        return dt1_val;
                    case complex_kind:
                        // Integer type sizes don't affect complex type sizes
                        return dt1_val;
                    case string_kind:
                        // Presently UTF8 is the only built-in string type
                        itemsize = min_strlen_for_builtin_kind(dt0_val.kind());
                        if (dt1_val.itemsize() > itemsize) {
                            itemsize = dt1_val.itemsize();
                        }
                        return dtype(utf8_type_id, itemsize);
                    default:
                        break;
                }
                break;
            case uint_kind:
                switch (dt1_val.kind()) {
                    case bool_kind:
                        return (dt0_val.itemsize() >= sizeof(int)) ? dt0_val
                                                               : make_dtype<int>();
                    case int_kind:
                        if (dt0_val.itemsize() < sizeof(int) && dt1_val.itemsize() < sizeof(int)) {
                            return make_dtype<int>();
                        } else {
                            // When the itemsizes are equal, the uint kind wins
                            return (dt0_val.itemsize() >= dt1_val.itemsize()) ? dt0_val
                                                                              : dt1_val;
                        }
                    case uint_kind:
                        if (dt0_val.itemsize() < sizeof(int) && dt1_val.itemsize() < sizeof(int)) {
                            return make_dtype<int>();
                        } else {
                            return (dt0_val.itemsize() >= dt1_val.itemsize()) ? dt0_val
                                                                              : dt1_val;
                        }
                    case real_kind:
                        // Integer type sizes don't affect float type sizes
                        return dt1_val;
                    case complex_kind:
                        // Integer type sizes don't affect complex type sizes
                        return dt1_val;
                    case string_kind:
                        // Presently UTF8 is the only built-in string type
                        itemsize = min_strlen_for_builtin_kind(dt0_val.kind());
                        if (dt1_val.itemsize() > itemsize) {
                            itemsize = dt1_val.itemsize();
                        }
                        return dtype(utf8_type_id, itemsize);
                    default:
                        break;
                }
                break;
            case real_kind:
                switch (dt1_val.kind()) {
                    // Integer type sizes don't affect float type sizes
                    case bool_kind:
                    case int_kind:
                    case uint_kind:
                        return dt0_val;
                    case real_kind:
                        return (dt0_val.itemsize() >= dt1_val.itemsize()) ? dt0_val
                                                                          : dt1_val;
                    case complex_kind:
                        if (dt0_val.type_id() == float64_type_id && dt1_val.type_id() == complex_float32_type_id) {
                            return dtype(complex_float64_type_id);
                        } else {
                            return dt1_val;
                        }
                    case string_kind:
                        // Presently UTF8 is the only built-in string type
                        itemsize = min_strlen_for_builtin_kind(dt0_val.kind());
                        if (dt1_val.itemsize() > itemsize) {
                            itemsize = dt1_val.itemsize();
                        }
                        return dtype(utf8_type_id, itemsize);
                    default:
                        break;
                }
                break;
            case complex_kind:
                switch (dt1_val.kind()) {
                    // Integer and float type sizes don't affect complex type sizes
                    case bool_kind:
                    case int_kind:
                    case uint_kind:
                    case real_kind:
                        if (dt0_val.type_id() == complex_float32_type_id && dt1_val.type_id() == float64_type_id) {
                            return dtype(complex_float64_type_id);
                        } else {
                            return dt0_val;
                        }
                    case complex_kind:
                        return (dt0_val.itemsize() >= dt1_val.itemsize()) ? dt0_val
                                                                          : dt1_val;
                    case string_kind:
                        // Presently UTF8 is the only built-in string type
                        itemsize = min_strlen_for_builtin_kind(dt0_val.kind());
                        if (dt1_val.itemsize() > itemsize) {
                            itemsize = dt1_val.itemsize();
                        }
                        return dtype(utf8_type_id, itemsize);
                    default:
                        break;
                }
                break;
            case string_kind:
                switch (dt1_val.kind()) {
                    case bool_kind:
                    case int_kind:
                    case uint_kind:
                    case real_kind:
                    case complex_kind:
                        // Presently UTF8 is the only built-in string type
                        itemsize = min_strlen_for_builtin_kind(dt1_val.kind());
                        if (dt0_val.itemsize() > itemsize) {
                            itemsize = dt0_val.itemsize();
                        }
                        return dtype(utf8_type_id, itemsize);
                    case string_kind:
                        return (dt0_val.itemsize() >= dt1_val.itemsize()) ? dt0_val
                                                                          : dt1_val;
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
