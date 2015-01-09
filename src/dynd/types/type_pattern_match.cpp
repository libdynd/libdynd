//
// Copyright (C) 2011-14 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <dynd/types/type_pattern_match.hpp>
#include <dynd/func/arrfunc.hpp>
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
#include <dynd/types/sym_type_type.hpp>

using namespace std;
using namespace dynd;

// TODO: We need to properly add offsets to the concrete_arrmeta in every
// function below.

static bool recursive_match(const ndt::type &concrete,
                            const char *concrete_arrmeta,
                            const ndt::type &pattern,
                            std::map<nd::string, ndt::type> &typevars)
{
  if (pattern.get_type_id() == sym_type_type_id) {
    return recursive_match(concrete, concrete_arrmeta,
                           pattern.extended<sym_type_type>()->get_sym_type(),
                           typevars);
  } else if (concrete.get_ndim() == 0 && pattern.get_ndim() == 0) {
    // Matching a scalar vs scalar
    if (pattern.get_type_id() == typevar_type_id) {
      ndt::type &tv_type =
          typevars[pattern.extended<typevar_type>()->get_name()];
      if (tv_type.is_null()) {
        // This typevar hasn't been seen yet
        tv_type = concrete;
        return true;
      } else {
        // Make sure the type matches previous
        // instances of the type var
        return concrete == tv_type;
      }
    } else if (concrete.get_type_id() == pattern.get_type_id()) {
      switch (concrete.get_type_id()) {
      case pointer_type_id:
        return recursive_match(
            concrete.extended<pointer_type>()->get_target_type(),
            concrete_arrmeta,
            pattern.extended<pointer_type>()->get_target_type(), typevars);
      case struct_type_id:
      case cstruct_type_id:
        if (concrete.extended<base_struct_type>()
                ->get_field_names()
                .equals_exact(
                    pattern.extended<base_struct_type>()->get_field_names())) {
          // The names are all the same, now match against the
          // types
          size_t field_count =
              concrete.extended<base_struct_type>()->get_field_count();
          const ndt::type *concrete_fields =
              concrete.extended<base_struct_type>()->get_field_types_raw();
          const ndt::type *pattern_fields =
              pattern.extended<base_struct_type>()->get_field_types_raw();
          for (size_t i = 0; i != field_count; ++i) {
            if (!recursive_match(concrete_fields[i], concrete_arrmeta,
                                 pattern_fields[i], typevars)) {
              return false;
            }
          }
          return true;
        } else {
          return false;
        }
      case tuple_type_id:
      case ctuple_type_id: {
        intptr_t concrete_field_count =
            concrete.extended<base_tuple_type>()->get_field_count();
        bool concrete_variadic =
            concrete.extended<base_tuple_type>()->is_variadic();
        intptr_t pattern_field_count =
            pattern.extended<base_tuple_type>()->get_field_count();
        bool pattern_variadic =
            pattern.extended<base_tuple_type>()->is_variadic();
        if ((concrete_field_count == pattern_field_count &&
             !concrete_variadic) ||
            (concrete_field_count >= pattern_field_count && pattern_variadic)) {
          auto concrete_arrmeta_offsets =
              concrete.extended<base_tuple_type>()->get_arrmeta_offsets_raw();
          // Match against the types
          const ndt::type *concrete_fields =
              concrete.extended<base_tuple_type>()->get_field_types_raw();
          const ndt::type *pattern_fields =
              pattern.extended<base_tuple_type>()->get_field_types_raw();
          for (intptr_t i = 0; i != pattern_field_count; ++i) {
            if (!recursive_match(concrete_fields[i],
                                 concrete_arrmeta +
                                     (concrete_arrmeta != NULL
                                          ? concrete_arrmeta_offsets[i]
                                          : 0),
                                 pattern_fields[i], typevars)) {
              return false;
            }
          }
          return true;
        } else {
          return false;
        }
      }
      case option_type_id:
        return recursive_match(
            concrete.extended<option_type>()->get_value_type(),
            concrete_arrmeta, pattern.extended<option_type>()->get_value_type(),
            typevars);
      case cuda_host_type_id:
      case cuda_device_type_id:
        return recursive_match(
            concrete.extended<base_memory_type>()->get_element_type(),
            concrete_arrmeta,
            pattern.extended<base_memory_type>()->get_element_type(), typevars);
      case arrfunc_type_id: {
        // First match the return type
        if (!recursive_match(
                concrete.extended<arrfunc_type>()->get_return_type(),
                concrete_arrmeta,
                pattern.extended<arrfunc_type>()->get_return_type(),
                typevars)) {
          return false;
        }
        // Next match all the positional parameters
        if (!recursive_match(concrete.extended<arrfunc_type>()->get_pos_tuple(),
                             concrete_arrmeta,
                             pattern.extended<arrfunc_type>()->get_pos_tuple(),
                             typevars)) {
          return false;
        }
        // Finally match all the keyword parameters
        if (!recursive_match(
                concrete.extended<arrfunc_type>()->get_kwd_struct(),
                concrete_arrmeta,
                pattern.extended<arrfunc_type>()->get_kwd_struct(), typevars)) {
          return false;
        }
        return true;
      }
      default:
        return pattern == concrete;
      }
    } else if (pattern.get_type_id() == any_sym_type_id) {
      return true;
    } else {
      return false;
    }
  } else {
    // First match the dimensions, then the dtype
    ndt::type concrete_dtype, pattern_dtype;
    if (pattern_match_dims(concrete, concrete_arrmeta, pattern, typevars,
                           concrete_dtype, pattern_dtype)) {
      return recursive_match(concrete_dtype, concrete_arrmeta, pattern_dtype,
                             typevars);
    } else {
      return false;
    }
  }
}

bool ndt::pattern_match_dims(const ndt::type &concrete,
                             const char *concrete_arrmeta,
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
            pattern.extended<ellipsis_dim_type>()->get_name();
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
            concrete, concrete_arrmeta,
            pattern.extended<ellipsis_dim_type>()->get_element_type(), typevars,
            out_concrete_dtype, out_pattern_dtype);
      } else if (pattern.get_type_id() == any_sym_type_id) {
        // "Any" consumes all the dimensions, and remains available to consume
        // the dtype
        out_concrete_dtype = concrete.get_dtype();
        out_pattern_dtype = pattern;
        return true;
      } else if (pattern.get_type_id() == pow_dimsym_type_id) {
        if (pattern.extended<pow_dimsym_type>()
                ->get_element_type()
                .get_ndim() == 0) {
          // Look up to see if the exponent typevar is already matched
          ndt::type &tv_type =
              typevars[pattern.extended<pow_dimsym_type>()->get_exponent()];
          if (tv_type.is_null()) {
            // Fill in the exponent by the number of dimensions left
            tv_type = ndt::make_fixed_dim(0, ndt::make_type<void>());
          } else if (tv_type.get_type_id() == fixed_dim_type_id) {
            // Make sure the exponent already seen matches the number of
            // dimensions left in the concrete type
            if (tv_type.extended<fixed_dim_type>()->get_fixed_dim_size() != 0) {
              return false;
            }
          } else {
            // The exponent is always the dim_size inside a fixed_dim_type
            return false;
          }
          return pattern_match_dims(
              concrete, concrete_arrmeta,
              pattern.extended<pow_dimsym_type>()->get_element_type(), typevars,
              out_concrete_dtype, out_pattern_dtype);
        } else {
          return false;
        }
      } else {
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
            concrete.extended<base_dim_type>()->get_element_type(),
            concrete_arrmeta,
            pattern.extended<base_dim_type>()->get_element_type(), typevars,
            out_concrete_dtype, out_pattern_dtype);
      case fixed_dim_type_id:
        return concrete.extended<fixed_dim_type>()->get_fixed_dim_size() ==
                   pattern.extended<fixed_dim_type>()->get_fixed_dim_size() &&
               pattern_match_dims(
                   concrete.extended<base_dim_type>()->get_element_type(),
                   concrete_arrmeta,
                   pattern.extended<base_dim_type>()->get_element_type(),
                   typevars, out_concrete_dtype, out_pattern_dtype);
      case cfixed_dim_type_id:
        return concrete.extended<cfixed_dim_type>()->get_fixed_dim_size() ==
                   pattern.extended<cfixed_dim_type>()->get_fixed_dim_size() &&
               concrete.extended<cfixed_dim_type>()->get_fixed_stride() ==
                   pattern.extended<cfixed_dim_type>()->get_fixed_stride() &&
               pattern_match_dims(
                   concrete.extended<base_dim_type>()->get_element_type(),
                   concrete_arrmeta,
                   pattern.extended<base_dim_type>()->get_element_type(),
                   typevars, out_concrete_dtype, out_pattern_dtype);
      case ellipsis_dim_type_id:
        return pattern_match_dims(
            concrete.extended<ellipsis_dim_type>()->get_element_type(),
            concrete_arrmeta,
            pattern.extended<ellipsis_dim_type>()->get_element_type(), typevars,
            out_concrete_dtype, out_pattern_dtype);
      case pow_dimsym_type_id:
        if (pattern_match_dims(
                concrete.extended<pow_dimsym_type>()->get_base_type(),
                concrete_arrmeta,
                pattern.extended<pow_dimsym_type>()->get_base_type(), typevars,
                out_concrete_dtype, out_pattern_dtype) &&
            pattern_match_dims(
                concrete.extended<pow_dimsym_type>()->get_element_type(),
                concrete_arrmeta,
                pattern.extended<pow_dimsym_type>()->get_element_type(),
                typevars, out_concrete_dtype, out_pattern_dtype)) {
          ndt::type &tv_type =
              typevars[pattern.extended<pow_dimsym_type>()->get_exponent()];
          if (tv_type.is_null()) {
            // This typevar hasn't been seen yet
            tv_type = ndt::make_typevar_dim(
                pattern.extended<pow_dimsym_type>()->get_exponent(),
                ndt::make_type<void>());
            return true;
          } else {
            // Make sure the type matches previous
            // instances of the type var
            return tv_type.get_type_id() == typevar_dim_type_id &&
                   tv_type.extended<typevar_dim_type>()->get_name() ==
                       pattern.extended<pow_dimsym_type>()->get_exponent();
          }
        } else {
          return false;
        }
      default:
        break;
      }
      stringstream ss;
      ss << "Type pattern matching between dimension types " << concrete
         << " and " << pattern << " is not yet implemented";
      throw type_error(ss.str());
    } else if (pattern.get_type_id() == fixed_dimsym_type_id) {
      // fixed[N] and cfixed[M] matches against fixed (symbolic fixed)
      if (concrete.get_type_id() == fixed_dim_type_id ||
          concrete.get_type_id() == cfixed_dim_type_id) {
        return pattern_match_dims(
            concrete.extended<base_dim_type>()->get_element_type(),
            concrete_arrmeta,
            pattern.extended<base_dim_type>()->get_element_type(), typevars,
            out_concrete_dtype, out_pattern_dtype);
      } else {
        return false;
      }
    } else if (pattern.get_type_id() == cfixed_dim_type_id &&
               concrete_arrmeta != NULL) {
      if (concrete.get_type_id() == fixed_dim_type_id &&
          pattern.extended<cfixed_dim_type>()->get_fixed_dim_size() ==
              concrete.extended<fixed_dim_type>()->get_fixed_dim_size() &&
          pattern.extended<cfixed_dim_type>()->get_fixed_stride() ==
              concrete.extended<fixed_dim_type>()->get_fixed_stride(
                  concrete_arrmeta)) {
        return pattern_match_dims(
            concrete.extended<base_dim_type>()->get_element_type(),
            concrete_arrmeta,
            pattern.extended<base_dim_type>()->get_element_type(), typevars,
            out_concrete_dtype, out_pattern_dtype);
      } else {
        return false;
      }
    } else if (pattern.get_type_id() == fixed_dim_type_id) {
      // cfixed[N] matches against fixed[N], and has identical arrmeta
      if (concrete.get_type_id() == cfixed_dim_type_id &&
          pattern.extended<fixed_dim_type>()->get_fixed_dim_size() ==
              concrete.extended<cfixed_dim_type>()->get_fixed_dim_size()) {
        return pattern_match_dims(
            concrete.extended<base_dim_type>()->get_element_type(),
            concrete_arrmeta,
            pattern.extended<base_dim_type>()->get_element_type(), typevars,
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
            pattern.extended<ellipsis_dim_type>()->get_name();
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
                  tv_type.extended<dim_fragment_type>()->broadcast_with_type(
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
            concrete_arrmeta,
            pattern.extended<ellipsis_dim_type>()->get_element_type(), typevars,
            out_concrete_dtype, out_pattern_dtype);
      } else {
        // There are not enough dimensions in the concrete type
        // to match
        return false;
      }
    } else if (pattern.get_type_id() == typevar_dim_type_id) {
      ndt::type &tv_type =
          typevars[pattern.extended<typevar_dim_type>()->get_name()];
      if (tv_type.is_null()) {
        // This typevar hasn't been seen yet
        tv_type = concrete;
        return pattern_match_dims(
            concrete.get_type_at_dimension(NULL, 1), concrete_arrmeta,
            pattern.extended<typevar_dim_type>()->get_element_type(), typevars,
            out_concrete_dtype, out_pattern_dtype);
      } else {
        // Make sure the type matches previous
        // instances of the type var
        if (concrete.get_type_id() != tv_type.get_type_id()) {
          return false;
        }
        switch (concrete.get_type_id()) {
        case fixed_dim_type_id:
          if (concrete.extended<fixed_dim_type>()->get_fixed_dim_size() !=
              tv_type.extended<fixed_dim_type>()->get_fixed_dim_size()) {
            return false;
          }
          break;
        case cfixed_dim_type_id:
          if (concrete.extended<cfixed_dim_type>()->get_fixed_dim_size() !=
              tv_type.extended<cfixed_dim_type>()->get_fixed_dim_size()) {
            return false;
          }
          break;
        default:
          break;
        }
        return pattern_match_dims(
            concrete.get_type_at_dimension(NULL, 1), concrete_arrmeta,
            pattern.extended<typevar_dim_type>()->get_element_type(), typevars,
            out_concrete_dtype, out_pattern_dtype);
      }
    } else if (pattern.get_type_id() == pow_dimsym_type_id) {
      // Look up to see if the exponent typevar is already matched
      ndt::type &tv_type =
          typevars[pattern.extended<pow_dimsym_type>()->get_exponent()];
      intptr_t exponent;
      if (tv_type.is_null()) {
        // Fill in the exponent by the number of dimensions left
        exponent =
            concrete.get_ndim() -
            pattern.extended<pow_dimsym_type>()->get_element_type().get_ndim();
        tv_type = ndt::make_fixed_dim(exponent, ndt::make_type<void>());
      } else if (tv_type.get_type_id() == fixed_dim_type_id) {
        // Make sure the exponent already seen matches the number of
        // dimensions left in the concrete type
        exponent = tv_type.extended<fixed_dim_type>()->get_fixed_dim_size();
        if (exponent !=
            concrete.get_ndim() -
                pattern.extended<pow_dimsym_type>()
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
            concrete, concrete_arrmeta,
            pattern.extended<pow_dimsym_type>()->get_element_type(), typevars,
            out_concrete_dtype, out_pattern_dtype);
      } else if (exponent < 0) {
        return false;
      }
      // Get the base type
      ndt::type base_tp = pattern.extended<pow_dimsym_type>()->get_base_type();
      if (base_tp.get_type_id() == typevar_dim_type_id) {
        ndt::type &btv_type =
            typevars[base_tp.extended<typevar_dim_type>()->get_name()];
        if (btv_type.is_null()) {
          // We haven't seen this typevar yet, set it to the concrete's
          // dimension type
          btv_type = concrete;
          base_tp = concrete;
        } else if (btv_type.get_ndim() > 0 &&
                   btv_type.get_type_id() != dim_fragment_type_id) {
          // Continue matching after substituting in the typevar for
          // the base type
          base_tp = btv_type;
        } else {
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
                concrete_subtype.extended<base_dim_type>()->get_element_type();
            break;
          default:
            return false;
          }
        }
        break;
      case fixed_dim_type_id: {
        intptr_t dim_size =
            base_tp.extended<fixed_dim_type>()->get_fixed_dim_size();
        for (intptr_t i = 0; i < exponent; ++i) {
          if (concrete_subtype.get_type_id() == fixed_dim_type_id &&
              concrete_subtype.extended<fixed_dim_type>()
                      ->get_fixed_dim_size() == dim_size) {
            concrete_subtype =
                concrete_subtype.extended<base_dim_type>()->get_element_type();
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
                concrete_subtype.extended<base_dim_type>()->get_element_type();
          }
        }
        break;
      default:
        return false;
      }
      return pattern_match_dims(
          concrete_subtype, concrete_arrmeta,
          pattern.extended<pow_dimsym_type>()->get_element_type(), typevars,
          out_concrete_dtype, out_pattern_dtype);
    } else if (pattern.get_type_id() == any_sym_type_id) {
      // "Any" consumes all the dimensions, and remains available to consume
      // the dtype
      out_concrete_dtype = concrete.get_dtype();
      out_pattern_dtype = pattern;
      return true;
    } else {
      return false;
    }
  }
}

bool ndt::pattern_match(const ndt::type &concrete, const char *concrete_arrmeta,
                        const ndt::type &pattern,
                        std::map<nd::string, ndt::type> &typevars)
{
  return recursive_match(concrete, concrete_arrmeta, pattern, typevars);
}
