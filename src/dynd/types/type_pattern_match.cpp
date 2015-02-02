//
// Copyright (C) 2011-15 DyND Developers
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
#include <dynd/types/any_sym_type.hpp>
#include <dynd/types/dim_fragment_type.hpp>
#include <dynd/types/typevar_constructed_type.hpp>

using namespace std;
using namespace dynd;

// TODO: We need to properly add offsets to the concrete_arrmeta in every
// function below.

// if (some other concrete type we understand) {
//   do stuff
// } else {
//  call with pattern type;
//

/*
  if (pattern.get_ndim() == -1) {
    if (pattern.get_type_id() == typevar_constructed_type_id &&
        concrete.get_kind() == memory_kind) {
      ndt::type &tv_type =
          typevars[pattern.extended<typevar_constructed_type>()->get_name()];
      if (tv_type.is_null()) {
        // This typevar hasn't been seen yet
        tv_type = concrete;
      }
      return recursive_match(
          concrete.extended<base_memory_type>()->get_element_type(),
          concrete_arrmeta,
          pattern.extended<typevar_constructed_type>()->get_arg(), typevars);
    }
  } else
*/

/*

if (both scalar) {

}

*/

static bool recursive_match(const ndt::type &concrete,
                            const char *concrete_arrmeta,
                            const ndt::type &pattern,
                            std::map<nd::string, ndt::type> &typevars)
{
  if (concrete.get_ndim() == 0) {
    if (pattern.get_ndim() == 0) {
      // Matching a scalar vs scalar
      if (pattern.get_type_id() == typevar_type_id) {
        return pattern.extended<typevar_type>()->matches(NULL, concrete,
                                                         typevars);
      } else if (concrete.get_type_id() == pattern.get_type_id()) {
        switch (concrete.get_type_id()) {
        case pointer_type_id:
          return concrete.extended<pointer_type>()->matches(concrete_arrmeta,
                                                            pattern, typevars);
        case struct_type_id:
        case cstruct_type_id:
          return concrete.extended<base_struct_type>()->matches(
              concrete_arrmeta, pattern, typevars);
        case tuple_type_id:
        case ctuple_type_id:
          return concrete.extended<base_tuple_type>()->matches(
              concrete_arrmeta, pattern, typevars);
        case option_type_id:
          return concrete.extended<option_type>()->matches(concrete_arrmeta,
                                                           pattern, typevars);
        case cuda_host_type_id:
        case cuda_device_type_id:
          return concrete.extended<base_memory_type>()->matches(
              concrete_arrmeta, pattern, typevars);
        case arrfunc_type_id:
          return concrete.extended<arrfunc_type>()->matches(concrete_arrmeta,
                                                            pattern, typevars);
        default:
          return pattern == concrete;
        }
      } else if (pattern.get_type_id() == any_sym_type_id) {
        return pattern.extended<any_sym_type>()->matches(concrete_arrmeta,
                                                         concrete, typevars);
      } else {
        return false;
      }
    } else {
      // Matching a scalar vs dimension, only cases which makes sense
      // are an ellipsis_dim or a pow_dimsym
      if (pattern.get_type_id() == ellipsis_dim_type_id) {
        return pattern.extended<ellipsis_dim_type>()->matches(
            concrete_arrmeta, concrete, typevars);
      } else if (pattern.get_type_id() == any_sym_type_id) {
        return pattern.extended<any_sym_type>()->matches(concrete_arrmeta,
                                                         concrete, typevars);
      } else if (pattern.get_type_id() == pow_dimsym_type_id) {
        return pattern.extended<pow_dimsym_type>()->matches(concrete_arrmeta,
                                                            concrete, typevars);
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
        return concrete.extended<base_dim_type>()->matches(concrete_arrmeta,
                                                           pattern, typevars);
      case fixed_dim_type_id:
        return concrete.extended<fixed_dim_type>()->matches(concrete_arrmeta,
                                                            pattern, typevars);
      case cfixed_dim_type_id:
        return concrete.extended<cfixed_dim_type>()->matches(concrete_arrmeta,
                                                             pattern, typevars);
      case ellipsis_dim_type_id:
        return concrete.extended<ellipsis_dim_type>()->matches(
            concrete_arrmeta, pattern, typevars);
      case pow_dimsym_type_id:
        return concrete.extended<pow_dimsym_type>()->matches(concrete_arrmeta,
                                                             pattern, typevars);
      default:
        break;
      }
      stringstream ss;
      ss << "Type pattern matching between dimension types " << concrete
         << " and " << pattern << " is not yet implemented";
      throw type_error(ss.str());
    } else if (concrete.get_type_id() == fixed_dim_type_id &&
               pattern.get_type_id() == cfixed_dim_type_id) {
      return concrete.extended<fixed_dim_type>()->matches(concrete_arrmeta,
                                                          pattern, typevars);
    } else if (concrete.get_type_id() == fixed_dim_type_id &&
               pattern.get_type_id() == fixed_dimsym_type_id) {
      return concrete.extended<fixed_dim_type>()->matches(concrete_arrmeta,
                                                          pattern, typevars);
    } else if (concrete.get_type_id() == cfixed_dim_type_id) {
      return concrete.extended<cfixed_dim_type>()->matches(concrete_arrmeta,
                                                           pattern, typevars);
    } else if (pattern.get_type_id() == ellipsis_dim_type_id) {
      return pattern.extended<ellipsis_dim_type>()->matches(concrete_arrmeta,
                                                            concrete, typevars);
    } else if (pattern.get_type_id() == typevar_dim_type_id) {
      return pattern.extended<typevar_dim_type>()->matches(concrete_arrmeta,
                                                           concrete, typevars);
    } else if (pattern.get_type_id() == pow_dimsym_type_id) {
      return pattern.extended<pow_dimsym_type>()->matches(concrete_arrmeta,
                                                          concrete, typevars);
    } else if (pattern.get_type_id() == any_sym_type_id) {
      return pattern.extended<any_sym_type>()->matches(concrete_arrmeta,
                                                       concrete, typevars);
    } else {
      return false;
    }
  }

  return false;
}

bool ndt::pattern_match(const ndt::type &concrete, const char *concrete_arrmeta,
                        const ndt::type &pattern,
                        std::map<nd::string, ndt::type> &typevars)
{
  return recursive_match(concrete, concrete_arrmeta, pattern, typevars);
}
