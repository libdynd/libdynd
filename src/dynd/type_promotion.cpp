//
// Copyright (C) 2011-15 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <stdexcept>
#include <algorithm>

#include <dynd/type_promotion.hpp>
#include <dynd/types/string_type.hpp>
#include <dynd/types/option_type.hpp>
#include <dynd/types/var_dim_type.hpp>

using namespace std;
using namespace dynd;

/*
static intptr_t min_strlen_for_builtin_kind(type_kind_t kind)
{
    switch (kind) {
        case bool_kind:
            return 1;
        case sint_kind:
        case uint_kind:
            return 24;
        case real_kind:
            return 32;
        case complex_kind:
            return 64;
        default:
            throw dynd::type_error("cannot get minimum string length for
specified kind");
    }
}
*/

ndt::type dynd::promote_types_arithmetic(const ndt::type &tp0, const ndt::type &tp1)
{
  // Use the value types
  const ndt::type &tp0_val = tp0.value_type();
  const ndt::type &tp1_val = tp1.value_type();

  // cout << "Doing type promotion with value types " << tp0_val << " and " <<
  // tp1_val << endl;

  if (tp0_val.is_builtin() && tp1_val.is_builtin()) {
    const size_t int_size = sizeof(int);
    if (tp0_val.get_id() == void_id) {
      return tp1_val;
    }
    switch (tp0_val.get_base_id()) {
    case bool_kind_id:
      if (tp1_val.get_id() == void_id) {
        return tp0_val;
      }
      switch (tp1_val.get_base_id()) {
      case bool_kind_id:
        return ndt::make_type<int>();
      case int_kind_id:
      case uint_kind_id:
        return (tp1_val.get_data_size() >= int_size) ? tp1_val : ndt::make_type<int>();
      case float_kind_id:
        // The bool type doesn't affect float type sizes, except
        // require at least float32
        return tp1_val.unchecked_get_builtin_id() != float16_id ? tp1_val : ndt::make_type<float>();
      default:
        return tp1_val;
      }
    case int_kind_id:
      if (tp1_val.get_id() == void_id) {
        return tp0_val;
      }
      switch (tp1_val.get_base_id()) {
      case bool_kind_id:
        return (tp0_val.get_data_size() >= int_size) ? tp0_val : ndt::make_type<int>();
      case int_kind_id:
        if (tp0_val.get_data_size() < int_size && tp1_val.get_data_size() < int_size) {
          return ndt::make_type<int>();
        }
        else {
          return (tp0_val.get_data_size() >= tp1_val.get_data_size()) ? tp0_val : tp1_val;
        }
      case uint_kind_id:
        if (tp0_val.get_data_size() < int_size && tp1_val.get_data_size() < int_size) {
          return ndt::make_type<int>();
        }
        else {
          // When the element_sizes are equal, the uint kind wins
          return (tp0_val.get_data_size() > tp1_val.get_data_size()) ? tp0_val : tp1_val;
        }
      case float_kind_id:
        // Integer type sizes don't affect float type sizes, except
        // require at least float32
        return tp1_val.unchecked_get_builtin_id() != float16_id ? tp1_val : ndt::make_type<float>();
      case complex_kind_id:
        // Integer type sizes don't affect complex type sizes
        return tp1_val;
      default:
        break;
      }
      break;
    case uint_kind_id:
      if (tp1_val.get_id() == void_id) {
        return tp0_val;
      }
      switch (tp1_val.get_base_id()) {
      case bool_kind_id:
        return (tp0_val.get_data_size() >= int_size) ? tp0_val : ndt::make_type<int>();
      case int_kind_id:
        if (tp0_val.get_data_size() < int_size && tp1_val.get_data_size() < int_size) {
          return ndt::make_type<int>();
        }
        else {
          // When the element_sizes are equal, the uint kind wins
          return (tp0_val.get_data_size() >= tp1_val.get_data_size()) ? tp0_val : tp1_val;
        }
      case uint_kind_id:
        if (tp0_val.get_data_size() < int_size && tp1_val.get_data_size() < int_size) {
          return ndt::make_type<int>();
        }
        else {
          return (tp0_val.get_data_size() >= tp1_val.get_data_size()) ? tp0_val : tp1_val;
        }
      case float_kind_id:
        // Integer type sizes don't affect float type sizes, except
        // require at least float32
        return tp1_val.unchecked_get_builtin_id() != float16_id ? tp1_val : ndt::make_type<float>();
      case complex_kind_id:
        // Integer type sizes don't affect complex type sizes
        return tp1_val;
      default:
        break;
      }
      break;
    case float_kind_id:
      if (tp1_val.get_id() == void_id) {
        return tp0_val;
      }
      switch (tp1_val.get_base_id()) {
      // Integer type sizes don't affect float type sizes
      case bool_kind_id:
      case int_kind_id:
      case uint_kind_id:
        return tp0_val;
      case float_kind_id:
        return ndt::type(max(max(tp0_val.unchecked_get_builtin_id(), tp1_val.unchecked_get_builtin_id()), float32_id));
      case complex_kind_id:
        if (tp0_val.get_id() == float64_id && tp1_val.get_id() == complex_float32_id) {
          return ndt::type(complex_float64_id);
        }
        else {
          return tp1_val;
        }
      default:
        break;
      }
      break;
    case complex_kind_id:
      if (tp1_val.get_id() == void_id) {
        return tp0_val;
      }
      switch (tp1_val.get_base_id()) {
      // Integer and float type sizes don't affect complex type sizes
      case bool_kind_id:
      case int_kind_id:
      case uint_kind_id:
      case float_kind_id:
        if (tp0_val.unchecked_get_builtin_id() == complex_float32_id &&
            tp1_val.unchecked_get_builtin_id() == float64_id) {
          return ndt::type(complex_float64_id);
        }
        else {
          return tp0_val;
        }
      case complex_kind_id:
        return (tp0_val.get_data_size() >= tp1_val.get_data_size()) ? tp0_val : tp1_val;
      default:
        break;
      }
      break;
    default:
      break;
    }

    stringstream ss;
    ss << "internal error in built-in dynd type promotion of " << tp0_val << " and " << tp1_val;
    throw dynd::type_error(ss.str());
  }

  // HACK for getting simple string type promotions.
  // TODO: Do this properly in a pluggable manner.
  if ((tp0_val.get_id() == string_id || tp0_val.get_id() == fixed_string_id) &&
      (tp1_val.get_id() == string_id || tp1_val.get_id() == fixed_string_id)) {
    // Always promote to the default utf-8 string (for now, maybe return
    // encoding, etc later?)
    return ndt::make_type<ndt::string_type>();
  }

  // the value underneath the option type promotes
  if (tp0_val.get_id() == option_id) {
    if (tp1_val.get_id() == option_id) {
      return ndt::make_type<ndt::option_type>(
          promote_types_arithmetic(tp0_val.extended<ndt::option_type>()->get_value_type(),
                                   tp1_val.extended<ndt::option_type>()->get_value_type()));
    }
    else {
      return ndt::make_type<ndt::option_type>(
          promote_types_arithmetic(tp0_val.extended<ndt::option_type>()->get_value_type(), tp1_val));
    }
  }
  else if (tp1_val.get_id() == option_id) {
    return ndt::make_type<ndt::option_type>(
        promote_types_arithmetic(tp0_val, tp1_val.extended<ndt::option_type>()->get_value_type()));
  }

  // type, string -> type
  if (tp0_val.get_id() == type_id && tp1_val.get_base_id() == string_kind_id) {
    return tp0_val;
  }
  // string, type -> type
  if (tp0_val.get_base_id() == string_kind_id && tp1_val.get_id() == type_id) {
    return tp1_val;
  }

  // In general, if one type is void, just return the other type
  if (tp0_val.get_id() == void_id) {
    return tp1_val;
  }
  else if (tp1_val.get_id() == void_id) {
    return tp0_val;
  }

  // Promote some dimension types
  if ((tp0_val.get_id() == var_dim_id && tp1_val.get_base_id() == dim_kind_id) ||
      (tp1_val.get_id() == var_dim_id && tp0_val.get_base_id() == dim_kind_id)) {
    return ndt::var_dim_type::make(
        promote_types_arithmetic(tp0_val.extended<ndt::base_dim_type>()->get_element_type(),
                                 tp1_val.extended<ndt::base_dim_type>()->get_element_type()));
  }

  stringstream ss;
  ss << "type promotion of " << tp0 << " and " << tp1 << " is not yet supported";
  throw dynd::type_error(ss.str());
}
