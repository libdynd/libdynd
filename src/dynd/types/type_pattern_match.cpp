//
// Copyright (C) 2011-14 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <dynd/types/type_pattern_match.hpp>
#include <dynd/types/pointer_type.hpp>
#include <dynd/types/base_struct_type.hpp>
#include <dynd/types/option_type.hpp>
#include <dynd/types/base_memory_type.hpp>
#include <dynd/types/fixed_dim_type.hpp>
#include <dynd/types/cfixed_dim_type.hpp>
#include <dynd/types/typevar_type.hpp>
#include <dynd/types/typevar_dim_type.hpp>
#include <dynd/types/pow_dimsym_type.hpp>
#include <dynd/types/ellipsis_dim_type.hpp>
#include <dynd/types/dim_fragment_type.hpp>

using namespace std;
using namespace dynd;

static bool recursive_match(const ndt::type &concrete, const ndt::type &pattern,
                            std::map<nd::string, ndt::type> &typevars)
{
  if (concrete.get_ndim() == 0 && pattern.get_ndim() == 0) {
    // Matching a scalar vs scalar
    if (pattern.get_type_id() == typevar_type_id) {
      ndt::type &tv_type = typevars[pattern.tcast<typevar_type>()->get_name()];
      if (tv_type.is_null()) {
        // This typevar hasn't been seen yet
        tv_type = concrete;
        return true;
      }
      else {
        // Make sure the type matches previous
        // instances of the type var
        return concrete == tv_type;
      }
    }
    else if (concrete.get_type_id() == pattern.get_type_id()) {
      switch (concrete.get_type_id()) {
      case pointer_type_id:
        return recursive_match(
            concrete.tcast<pointer_type>()->get_target_type(),
            pattern.tcast<pointer_type>()->get_target_type(), typevars);
      case struct_type_id:
      case cstruct_type_id:
        if (concrete.tcast<base_struct_type>()->get_field_names().equals_exact(
                pattern.tcast<base_struct_type>()->get_field_names())) {
          // The names are all the same, now match against the
          // types
          size_t field_count =
              concrete.tcast<base_struct_type>()->get_field_count();
          const ndt::type *concrete_fields =
              concrete.tcast<base_struct_type>()->get_field_types_raw();
          const ndt::type *pattern_fields =
              pattern.tcast<base_struct_type>()->get_field_types_raw();
          for (size_t i = 0; i != field_count; ++i) {
            if (!recursive_match(concrete_fields[i], pattern_fields[i],
                                 typevars)) {
              return false;
            }
          }
          return true;
        }
        else {
          return false;
        }
      case tuple_type_id:
      case ctuple_type_id:
        if (concrete.tcast<base_tuple_type>()->get_field_count() ==
            pattern.tcast<base_tuple_type>()->get_field_count()) {
          // Match against the types
          size_t field_count =
              concrete.tcast<base_tuple_type>()->get_field_count();
          const ndt::type *concrete_fields =
              concrete.tcast<base_tuple_type>()->get_field_types_raw();
          const ndt::type *pattern_fields =
              pattern.tcast<base_tuple_type>()->get_field_types_raw();
          for (size_t i = 0; i != field_count; ++i) {
            if (!recursive_match(concrete_fields[i], pattern_fields[i],
                                 typevars)) {
              return false;
            }
          }
          return true;
        }
        else {
          return false;
        }
      case option_type_id:
        return recursive_match(concrete.tcast<option_type>()->get_value_type(),
                               pattern.tcast<option_type>()->get_value_type(),
                               typevars);
      case cuda_host_type_id:
      case cuda_device_type_id:
        return recursive_match(
            concrete.tcast<base_memory_type>()->get_storage_type(),
            pattern.tcast<base_memory_type>()->get_storage_type(), typevars);
      case funcproto_type_id:
        if (concrete.tcast<funcproto_type>()->get_narg() ==
            pattern.tcast<funcproto_type>()->get_narg()) {
          // First match the return type
          if (!recursive_match(
                  concrete.tcast<funcproto_type>()->get_return_type(),
                  pattern.tcast<funcproto_type>()->get_return_type(),
                  typevars)) {
            return false;
          }
          // Then match all the parameters
          size_t param_count = concrete.tcast<funcproto_type>()->get_narg();
          for (size_t i = 0; i != param_count; ++i) {
            if (!recursive_match(
                    concrete.tcast<funcproto_type>()->get_arg_type(i),
                    pattern.tcast<funcproto_type>()->get_arg_type(i),
                    typevars)) {
              return false;
            }
          }
          return true;
        }
        else {
          return false;
        }
      default:
        return pattern == concrete;
      }
    }
    else {
      return false;
    }
  } else {
    // First match the dimensions, then the dtype
    ndt::type concrete_dtype, pattern_dtype;
    if (pattern_match_dims(concrete, pattern, typevars, concrete_dtype,
                           pattern_dtype)) {
      return recursive_match(concrete_dtype, pattern_dtype, typevars);
    } else {
      return false;
    }
  }
}

bool ndt::pattern_match_dims(const ndt::type &concrete,
                             const ndt::type &pattern,
                             std::map<nd::string, ndt::type> &typevars,
                             ndt::type &out_concrete_dtype,
                             ndt::type &out_pattern_dtype)
{
  if (concrete.get_ndim() == 0) {
    if (pattern.get_ndim() == 0) {
      // If both are dtypes already, just return them with as is
      out_concrete_dtype = concrete;
      out_pattern_dtype = pattern;
      return true;
    } else {
      // Matching a scalar vs dimension, only cases which makes sense
      // are an ellipsis_dim or a pow_dimsym
      if (pattern.get_type_id() == ellipsis_dim_type_id) {
        const nd::string &tv_name =
            pattern.tcast<ellipsis_dim_type>()->get_name();
        if (!tv_name.is_null()) {
          ndt::type &tv_type = typevars[tv_name];
          if (tv_type.is_null()) {
            // This typevar hasn't been seen yet, make it
            // be an empty dim fragment
            tv_type = ndt::make_dim_fragment();
          } else {
            // Make sure the type matches previous
            // instances of the type var, which is
            // always ok from the zero dims case
            // because "Dims..." combine
            // with broadcasting rules.
            if (tv_type.get_type_id() != dim_fragment_type_id) {
              // Inconsistent type var usage, previously
              // wasn't a dim fragment
              return false;
            }
          }
        }
        return pattern_match_dims(
            concrete, pattern.tcast<ellipsis_dim_type>()->get_element_type(),
            typevars, out_concrete_dtype, out_pattern_dtype);
      }
      else if (pattern.get_type_id() == pow_dimsym_type_id) {
        if (pattern.tcast<pow_dimsym_type>()->get_element_type().get_ndim() ==
            0) {
          // Look up to see if the exponent typevar is already matched
          ndt::type &tv_type =
              typevars[pattern.tcast<pow_dimsym_type>()->get_exponent()];
          if (tv_type.is_null()) {
            // Fill in the exponent by the number of dimensions left
            tv_type = ndt::make_fixed_dim(0, ndt::make_type<void>());
          }
          else if (tv_type.get_type_id() == fixed_dim_type_id) {
            // Make sure the exponent already seen matches the number of
            // dimensions left in the concrete type
            if (tv_type.tcast<fixed_dim_type>()->get_fixed_dim_size() != 0) {
              return false;
            }
          }
          else {
            // The exponent is always the dim_size inside a fixed_dim_type
            return false;
          }
          return pattern_match_dims(
              concrete, pattern.tcast<pow_dimsym_type>()->get_element_type(),
              typevars, out_concrete_dtype, out_pattern_dtype);
        }
        else {
          return false;
        }
      }
      else {
        return false;
      }
    }
  } else {
    // Matching a dimension vs dimension
    if (concrete.get_type_id() == pattern.get_type_id()) {
      switch (concrete.get_type_id()) {
      case fixed_dimsym_type_id:
      case offset_dim_type_id:
      case var_dim_type_id:
        return pattern_match_dims(
            concrete.tcast<base_dim_type>()->get_element_type(),
            pattern.tcast<base_dim_type>()->get_element_type(), typevars,
            out_concrete_dtype, out_pattern_dtype);
      case fixed_dim_type_id:
        return concrete.tcast<fixed_dim_type>()->get_fixed_dim_size() ==
                   pattern.tcast<fixed_dim_type>()->get_fixed_dim_size() &&
               pattern_match_dims(
                   concrete.tcast<base_dim_type>()->get_element_type(),
                   pattern.tcast<base_dim_type>()->get_element_type(), typevars,
                   out_concrete_dtype, out_pattern_dtype);
      case cfixed_dim_type_id:
        return concrete.tcast<cfixed_dim_type>()->get_fixed_dim_size() ==
                   pattern.tcast<cfixed_dim_type>()->get_fixed_dim_size() &&
               concrete.tcast<cfixed_dim_type>()->get_fixed_stride() ==
                   pattern.tcast<cfixed_dim_type>()->get_fixed_stride() &&
               pattern_match_dims(
                   concrete.tcast<base_dim_type>()->get_element_type(),
                   pattern.tcast<base_dim_type>()->get_element_type(), typevars,
                   out_concrete_dtype, out_pattern_dtype);
      case pow_dimsym_type_id:
        if (pattern_match_dims(
                concrete.tcast<pow_dimsym_type>()->get_base_type(),
                pattern.tcast<pow_dimsym_type>()->get_base_type(), typevars,
                out_concrete_dtype, out_pattern_dtype) &&
            pattern_match_dims(
                concrete.tcast<pow_dimsym_type>()->get_element_type(),
                pattern.tcast<pow_dimsym_type>()->get_element_type(), typevars,
                out_concrete_dtype, out_pattern_dtype)) {
          ndt::type &tv_type =
              typevars[pattern.tcast<pow_dimsym_type>()->get_exponent()];
          if (tv_type.is_null()) {
            // This typevar hasn't been seen yet
            tv_type = ndt::make_typevar_dim(
                pattern.tcast<pow_dimsym_type>()->get_exponent(),
                ndt::make_type<void>());
            return true;
          }
          else {
            // Make sure the type matches previous
            // instances of the type var
            return tv_type.get_type_id() == typevar_dim_type_id &&
                   tv_type.tcast<typevar_dim_type>()->get_name() ==
                       pattern.tcast<pow_dimsym_type>()->get_exponent();
          }
        }
        else {
          return false;
        }
      default:
        break;
      }
      stringstream ss;
      ss << "Type pattern matching between dimension types " << concrete
         << " and " << pattern << " is not yet implemented";
      throw type_error(ss.str());
    }
    else if (pattern.get_type_id() == fixed_dimsym_type_id) {
      // fixed[N] and cfixed[M] matches against fixed (symbolic fixed)
      if (concrete.get_type_id() == fixed_dim_type_id ||
          concrete.get_type_id() == cfixed_dim_type_id) {
        return pattern_match_dims(
            concrete.tcast<base_dim_type>()->get_element_type(),
            pattern.tcast<base_dim_type>()->get_element_type(), typevars,
            out_concrete_dtype, out_pattern_dtype);
      } else {
        return false;
      }
    } else if (pattern.get_type_id() == fixed_dim_type_id) {
      // cfixed[N] matches against fixed[N], and has identical arrmeta
      if (concrete.get_type_id() == cfixed_dim_type_id &&
          pattern.tcast<fixed_dim_type>()->get_fixed_dim_size() ==
              concrete.tcast<cfixed_dim_type>()->get_fixed_dim_size()) {
        return pattern_match_dims(
            concrete.tcast<base_dim_type>()->get_element_type(),
            pattern.tcast<base_dim_type>()->get_element_type(), typevars,
            out_concrete_dtype, out_pattern_dtype);
      } else {
        return false;
      }
    } else if (pattern.get_type_id() == ellipsis_dim_type_id) {
      // Match the number of concrete dimensions required on
      // the left
      if (concrete.get_ndim() >= pattern.get_ndim() - 1) {
        intptr_t matched_ndim = concrete.get_ndim() - pattern.get_ndim() + 1;
        const nd::string &tv_name =
            pattern.tcast<ellipsis_dim_type>()->get_name();
        if (!tv_name.is_null()) {
          ndt::type &tv_type = typevars[tv_name];
          if (tv_type.is_null()) {
            // This typevar hasn't been seen yet, so it's
            // a dim fragment of the given size.
            tv_type = ndt::make_dim_fragment(matched_ndim, concrete);
          } else {
            // Make sure the type matches previous  instances of the type var,
            // which in this case means they should broadcast together.
            if (tv_type.get_type_id() == dim_fragment_type_id) {
              ndt::type result =
                  tv_type.tcast<dim_fragment_type>()->broadcast_with_type(
                      matched_ndim, concrete);
              if (result.is_null()) {
                return false;
              } else {
                tv_type.swap(result);
              }
            } else {
              // Inconsistent type var usage, previously
              // wasn't a dim fragment
              return false;
            }
          }
        }
        return pattern_match_dims(
            concrete.get_type_at_dimension(NULL, matched_ndim),
            pattern.tcast<ellipsis_dim_type>()->get_element_type(), typevars,
            out_concrete_dtype, out_pattern_dtype);
      } else {
        // There are not enough dimensions in the concrete type
        // to match
        return false;
      }
    } else if (pattern.get_type_id() == typevar_dim_type_id) {
      ndt::type &tv_type =
          typevars[pattern.tcast<typevar_dim_type>()->get_name()];
      if (tv_type.is_null()) {
        // This typevar hasn't been seen yet
        tv_type = concrete;
        return pattern_match_dims(
            concrete.get_type_at_dimension(NULL, 1),
            pattern.tcast<typevar_dim_type>()->get_element_type(), typevars,
            out_concrete_dtype, out_pattern_dtype);
      } else {
        // Make sure the type matches previous
        // instances of the type var
        if (concrete.get_type_id() != tv_type.get_type_id()) {
          return false;
        }
        switch (concrete.get_type_id()) {
        case fixed_dim_type_id:
          if (concrete.tcast<fixed_dim_type>()->get_fixed_dim_size() !=
              tv_type.tcast<fixed_dim_type>()->get_fixed_dim_size()) {
            return false;
          }
          break;
        case cfixed_dim_type_id:
          if (concrete.tcast<cfixed_dim_type>()->get_fixed_dim_size() !=
              tv_type.tcast<cfixed_dim_type>()->get_fixed_dim_size()) {
            return false;
          }
          break;
        default:
          break;
        }
        return pattern_match_dims(
            concrete.get_type_at_dimension(NULL, 1),
            pattern.tcast<typevar_dim_type>()->get_element_type(), typevars,
            out_concrete_dtype, out_pattern_dtype);
      }
    } else if (pattern.get_type_id() == pow_dimsym_type_id) {
      // Look up to see if the exponent typevar is already matched
      ndt::type &tv_type =
          typevars[pattern.tcast<pow_dimsym_type>()->get_exponent()];
      intptr_t exponent;
      if (tv_type.is_null()) {
        // Fill in the exponent by the number of dimensions left
        exponent =
            concrete.get_ndim() -
            pattern.tcast<pow_dimsym_type>()->get_element_type().get_ndim();
        tv_type = ndt::make_fixed_dim(exponent, ndt::make_type<void>());
      }
      else if (tv_type.get_type_id() == fixed_dim_type_id) {
        // Make sure the exponent already seen matches the number of
        // dimensions left in the concrete type
        exponent = tv_type.tcast<fixed_dim_type>()->get_fixed_dim_size();
        if (exponent !=
            concrete.get_ndim() -
                pattern.tcast<pow_dimsym_type>()
                    ->get_element_type()
                    .get_ndim()) {
          return false;
        }
      } else {
        // The exponent is always the dim_size inside a fixed_dim_type
        return false;
      }
      // If the exponent is zero, the base doesn't matter, just match the rest
      if (exponent == 0) {
        return pattern_match_dims(
            concrete, pattern.tcast<pow_dimsym_type>()->get_element_type(),
            typevars, out_concrete_dtype, out_pattern_dtype);
      } else if (exponent < 0) {
        return false;
      }
      // Get the base type
      ndt::type base_tp = pattern.tcast<pow_dimsym_type>()->get_base_type();
      if (base_tp.get_type_id() == typevar_dim_type_id) {
        ndt::type &btv_type =
            typevars[base_tp.tcast<typevar_dim_type>()->get_name()];
        if (btv_type.is_null()) {
          // We haven't seen this typevar yet, set it to the concrete's
          // dimension type
          btv_type = concrete;
          base_tp = concrete;
        }
        else if (btv_type.get_ndim() > 0 &&
                 btv_type.get_type_id() != dim_fragment_type_id) {
          // Continue matching after substituting in the typevar for
          // the base type
          base_tp = btv_type;
        }
        else {
          // Doesn't match if the typevar has a dim fragment or dtype in it
          return false;
        }
      }
      // Now make sure the base_tp is repeated the right number of times
      ndt::type concrete_subtype = concrete;
      switch (base_tp.get_type_id()) {
        case fixed_dimsym_type_id:
          for (intptr_t i = 0; i < exponent; ++i) {
            switch (concrete_subtype.get_type_id()) {
            case fixed_dimsym_type_id:
            case fixed_dim_type_id:
            case cfixed_dim_type_id:
              concrete_subtype =
                  concrete_subtype.tcast<base_dim_type>()->get_element_type();
              break;
            default:
              return false;
            }
          }
          break;
        case fixed_dim_type_id: {
          intptr_t dim_size =
              base_tp.tcast<fixed_dim_type>()->get_fixed_dim_size();
          for (intptr_t i = 0; i < exponent; ++i) {
            if (concrete_subtype.get_type_id() == fixed_dim_type_id &&
                concrete_subtype.tcast<fixed_dim_type>()
                        ->get_fixed_dim_size() == dim_size) {
              concrete_subtype =
                  concrete_subtype.tcast<base_dim_type>()->get_element_type();
            } else {
              return false;
            }
          }
          break;
        }
        case var_dim_type_id:
          for (intptr_t i = 0; i < exponent; ++i) {
            if (concrete_subtype.get_type_id() == var_dim_type_id) {
              concrete_subtype =
                  concrete_subtype.tcast<base_dim_type>()->get_element_type();
            }
          }
          break;
        default:
          return false;
      }
      return pattern_match_dims(
          concrete_subtype,
          pattern.tcast<pow_dimsym_type>()->get_element_type(), typevars,
          out_concrete_dtype, out_pattern_dtype);
    }
    else {
      return false;
    }
  }
}

bool ndt::pattern_match(const ndt::type &concrete, const ndt::type &pattern,
                        std::map<nd::string, ndt::type> &typevars)
{
  return recursive_match(concrete, pattern, typevars);
}
