//
// Copyright (C) 2011-13 Mark Wiebe, DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <stdexcept>

#include <dynd/dtype_promotion.hpp>
#include <dynd/dtypes/string_type.hpp>

using namespace std;
using namespace dynd;

/*
static intptr_t min_strlen_for_builtin_kind(type_kind_t kind)
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
*/

ndt::type dynd::promote_dtypes_arithmetic(const ndt::type& dt0, const ndt::type& dt1)
{
    // Use the value dtypes
    const ndt::type& dt0_val = dt0.value_type();
    const ndt::type& dt1_val = dt1.value_type();

    //cout << "Doing type promotion with value types " << dt0_val << " and " << dt1_val << endl;

    if (dt0_val.is_builtin() && dt1_val.is_builtin()) {
        const size_t int_size = sizeof(int);
        switch (dt0_val.get_kind()) {
            case bool_kind:
                switch (dt1_val.get_kind()) {
                    case bool_kind:
                        return ndt::make_dtype<int>();
                    case int_kind:
                    case uint_kind:
                        return (dt1_val.get_data_size() >= int_size) ? dt1_val
                                                               : ndt::make_dtype<int>();
                    case void_kind:
                        return dt0_val;
                    case real_kind:
                        // The bool type doesn't affect float type sizes, except
                        // require at least float32
                        return dt1_val.unchecked_get_builtin_type_id() != float16_type_id
                                        ? dt1_val : ndt::make_dtype<float>();
                    default:
                        return dt1_val;
                }
            case int_kind:
                switch (dt1_val.get_kind()) {
                    case bool_kind:
                        return (dt0_val.get_data_size() >= int_size) ? dt0_val
                                                               : ndt::make_dtype<int>();
                    case int_kind:
                        if (dt0_val.get_data_size() < int_size && dt1_val.get_data_size() < int_size) {
                            return ndt::make_dtype<int>();
                        } else {
                            return (dt0_val.get_data_size() >= dt1_val.get_data_size()) ? dt0_val
                                                                              : dt1_val;
                        }
                    case uint_kind:
                        if (dt0_val.get_data_size() < int_size && dt1_val.get_data_size() < int_size) {
                            return ndt::make_dtype<int>();
                        } else {
                            // When the element_sizes are equal, the uint kind wins
                            return (dt0_val.get_data_size() > dt1_val.get_data_size()) ? dt0_val
                                                                             : dt1_val;
                        }
                    case real_kind:
                        // Integer type sizes don't affect float type sizes, except
                        // require at least float32
                        return dt1_val.unchecked_get_builtin_type_id() != float16_type_id
                                        ? dt1_val : ndt::make_dtype<float>();
                    case complex_kind:
                        // Integer type sizes don't affect complex type sizes
                        return dt1_val;
                    case void_kind:
                        return dt0_val;
                    default:
                        break;
                }
                break;
            case uint_kind:
                switch (dt1_val.get_kind()) {
                    case bool_kind:
                        return (dt0_val.get_data_size() >= int_size) ? dt0_val
                                                               : ndt::make_dtype<int>();
                    case int_kind:
                        if (dt0_val.get_data_size() < int_size && dt1_val.get_data_size() < int_size) {
                            return ndt::make_dtype<int>();
                        } else {
                            // When the element_sizes are equal, the uint kind wins
                            return (dt0_val.get_data_size() >= dt1_val.get_data_size()) ? dt0_val
                                                                              : dt1_val;
                        }
                    case uint_kind:
                        if (dt0_val.get_data_size() < int_size && dt1_val.get_data_size() < int_size) {
                            return ndt::make_dtype<int>();
                        } else {
                            return (dt0_val.get_data_size() >= dt1_val.get_data_size()) ? dt0_val
                                                                              : dt1_val;
                        }
                    case real_kind:
                        // Integer type sizes don't affect float type sizes, except
                        // require at least float32
                        return dt1_val.unchecked_get_builtin_type_id() != float16_type_id
                                        ? dt1_val : ndt::make_dtype<float>();
                    case complex_kind:
                        // Integer type sizes don't affect complex type sizes
                        return dt1_val;
                    case void_kind:
                        return dt0_val;
                    default:
                        break;
                }
                break;
            case real_kind:
                switch (dt1_val.get_kind()) {
                    // Integer type sizes don't affect float type sizes
                    case bool_kind:
                    case int_kind:
                    case uint_kind:
                        return dt0_val;
                    case real_kind:
                        return ndt::type(max(max(dt0_val.unchecked_get_builtin_type_id(),
                                        dt1_val.unchecked_get_builtin_type_id()), float32_type_id));
                    case complex_kind:
                        if (dt0_val.get_type_id() == float64_type_id && dt1_val.get_type_id() == complex_float32_type_id) {
                            return ndt::type(complex_float64_type_id);
                        } else {
                            return dt1_val;
                        }
                    case void_kind:
                        return dt0_val;
                    default:
                        break;
                }
                break;
            case complex_kind:
                switch (dt1_val.get_kind()) {
                    // Integer and float type sizes don't affect complex type sizes
                    case bool_kind:
                    case int_kind:
                    case uint_kind:
                    case real_kind:
                        if (dt0_val.unchecked_get_builtin_type_id() == complex_float32_type_id &&
                                        dt1_val.unchecked_get_builtin_type_id() == float64_type_id) {
                            return ndt::type(complex_float64_type_id);
                        } else {
                            return dt0_val;
                        }
                    case complex_kind:
                        return (dt0_val.get_data_size() >= dt1_val.get_data_size()) ? dt0_val
                                                                          : dt1_val;
                    case void_kind:
                        return dt0_val;
                    default:
                        break;
                }
                break;
            case void_kind:
                return dt1_val;
            default:
                break;
        }

        stringstream ss;
        ss << "internal error in built-in dtype promotion of " << dt0_val << " and " << dt1_val;
        throw std::runtime_error(ss.str());
    }

    // HACK for getting simple string dtype promotions.
    // TODO: Do this properly in a pluggable manner.
    if ((dt0_val.get_type_id() == string_type_id ||
                    dt0_val.get_type_id() == fixedstring_type_id) &&
                (dt1_val.get_type_id() == string_type_id ||
                    dt1_val.get_type_id() == fixedstring_type_id)) {
        // Always promote to the default utf-8 string (for now, maybe return encoding, etc later?)
        return ndt::make_string();
    }

    // dtype, string -> dtype
    if (dt0_val.get_type_id() == dtype_type_id && dt1_val.get_kind() == string_kind) {
        return dt0_val;
    }
    // string, dtype -> dtype
    if (dt0_val.get_kind() == string_kind && dt1_val.get_type_id() == dtype_type_id) {
        return dt1_val;
    }

    // In general, if one type is void, just return the other type
    if (dt0_val.get_type_id() == void_type_id) {
        return dt1_val;
    } else if (dt1_val.get_type_id() == void_type_id) {
        return dt0_val;
    }

    stringstream ss;
    ss << "type promotion of " << dt0 << " and " << dt1 << " is not yet supported";
    throw std::runtime_error(ss.str());
}
