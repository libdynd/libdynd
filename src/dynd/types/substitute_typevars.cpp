//
// Copyright (C) 2011-16 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <dynd/callable.hpp>
#include <dynd/types/base_memory_type.hpp>
#include <dynd/types/cuda_device_type.hpp>
#include <dynd/types/dim_fragment_type.hpp>
#include <dynd/types/ellipsis_dim_type.hpp>
#include <dynd/types/fixed_dim_type.hpp>
#include <dynd/types/option_type.hpp>
#include <dynd/types/pointer_type.hpp>
#include <dynd/types/pow_dimsym_type.hpp>
#include <dynd/types/struct_type.hpp>
#include <dynd/types/substitute_typevars.hpp>
#include <dynd/types/tuple_type.hpp>
#include <dynd/types/typevar_constructed_type.hpp>
#include <dynd/types/typevar_dim_type.hpp>
#include <dynd/types/typevar_type.hpp>
#include <dynd/types/var_dim_type.hpp>

using namespace std;
using namespace dynd;

/**
 * Substitutes the field types for contiguous array of types
 */
static std::vector<ndt::type> substitute_type_array(const nd::array &type_array,
                                                    const std::map<std::string, ndt::type> &typevars, bool concrete) {
  intptr_t field_count = type_array.get_dim_size();
  const ndt::type *field_types = reinterpret_cast<const ndt::type *>(type_array.cdata());
  std::vector<ndt::type> tmp_field_types(field_count);

  for (intptr_t i = 0; i < field_count; ++i) {
    tmp_field_types[i] = ndt::substitute(field_types[i], typevars, concrete);
  }
  return tmp_field_types;
}

ndt::type ndt::detail::internal_substitute(const ndt::type &pattern, const std::map<std::string, ndt::type> &typevars,
                                           bool concrete) {
  // This function assumes that ``pattern`` is symbolic, so does not
  // have to check types that are always concrete
  switch (pattern.get_id()) {
#ifdef DYND_CUDA
  case cuda_device_id:
    return ndt::make_cuda_device(
        ndt::substitute(pattern.extended<base_memory_type>()->get_element_type(), typevars, concrete));
#endif
  case pointer_id:
    return ndt::pointer_type::make(
        ndt::substitute(pattern.extended<pointer_type>()->get_target_type(), typevars, concrete));
  case fixed_dim_id:
    if (!pattern.extended<base_fixed_dim_type>()->is_sized()) {
      if (!concrete) {
        return ndt::make_fixed_dim_kind(
            ndt::substitute(pattern.extended<base_dim_type>()->get_element_type(), typevars, concrete));
      } else {
        throw invalid_argument("The dynd pattern type includes a symbolic "
                               "'fixed' dimension, which is not concrete as "
                               "requested");
      }
    } else {
      return ndt::make_fixed_dim(
          pattern.extended<fixed_dim_type>()->get_fixed_dim_size(),
          ndt::substitute(pattern.extended<fixed_dim_type>()->get_element_type(), typevars, concrete));
    }
  case var_dim_id:
    return ndt::make_type<ndt::var_dim_type>(
        ndt::substitute(pattern.extended<var_dim_type>()->get_element_type(), typevars, concrete));
  case struct_id:
    return ndt::make_type<ndt::struct_type>(
        pattern.extended<struct_type>()->get_field_names(),
        substitute_type_array(pattern.extended<tuple_type>()->get_field_types(), typevars, concrete));
  case tuple_id: {
    const std::vector<ndt::type> &element_tp =
        substitute_type_array(pattern.extended<tuple_type>()->get_field_types(), typevars, concrete);
    return ndt::make_type<ndt::tuple_type>(element_tp.size(), element_tp.data());
  }
  case option_id:
    return ndt::make_type<ndt::option_type>(
        ndt::substitute(pattern.extended<option_type>()->get_value_type(), typevars, concrete));
  case callable_id:
    return ndt::callable_type::make(
        substitute(pattern.extended<callable_type>()->get_return_type(), typevars, concrete),
        substitute(pattern.extended<callable_type>()->get_pos_tuple(), typevars, concrete),
        substitute(pattern.extended<callable_type>()->get_kwd_struct(), typevars, concrete));
  case typevar_constructed_id: {
    map<std::string, ndt::type>::const_iterator it =
        typevars.find(pattern.extended<typevar_constructed_type>()->get_name());
    if (it->second.get_id() == void_id) {
      return substitute(pattern.extended<typevar_constructed_type>()->get_arg(), typevars, concrete);
    }
#ifdef DYND_CUDA
    if (it->second.get_id() == cuda_device_id) {
      return ndt::make_cuda_device(
          substitute(pattern.extended<typevar_constructed_type>()->get_arg(), typevars, concrete));
    }
#endif
  }
  case typevar_id: {
    map<std::string, ndt::type>::const_iterator it = typevars.find(pattern.extended<typevar_type>()->get_name());
    if (it != typevars.end()) {
      if (it->second.get_ndim() != 0) {
        stringstream ss;
        ss << "The substitution for dynd typevar " << pattern << ", " << it->second
           << ", is a dimension, expected a dtype";
        throw invalid_argument(ss.str());
      }
      if (!concrete || !it->second.is_symbolic()) {
        return it->second;
      } else {
        stringstream ss;
        ss << "The substitution for dynd typevar " << pattern << ", " << it->second << ", is not concrete as required";
        throw invalid_argument(ss.str());
      }
    } else {
      if (concrete) {
        stringstream ss;
        ss << "No substitution type for dynd type var " << pattern << " was available";
        throw invalid_argument(ss.str());
      } else {
        return pattern;
      }
    }
  }
  case typevar_dim_id: {
    map<std::string, ndt::type>::const_iterator it = typevars.find(pattern.extended<typevar_dim_type>()->get_name());
    if (it != typevars.end()) {
      if (it->second.get_ndim() == 0) {
        stringstream ss;
        ss << "The substitution for dynd typevar " << pattern << ", " << it->second
           << ", is a dtype, expected a dimension";
        throw invalid_argument(ss.str());
      }
      if (!concrete || !it->second.is_symbolic()) {
        switch (it->second.get_id()) {
        case fixed_dim_id:
          if (!it->second.extended<base_fixed_dim_type>()->is_sized()) {
            return ndt::make_fixed_dim_kind(
                ndt::substitute(pattern.extended<typevar_dim_type>()->get_element_type(), typevars, concrete));
          } else {
            return ndt::make_fixed_dim(
                it->second.extended<fixed_dim_type>()->get_fixed_dim_size(),
                ndt::substitute(pattern.extended<typevar_dim_type>()->get_element_type(), typevars, concrete));
          }
        case var_dim_id:
          return ndt::make_type<ndt::var_dim_type>(
              ndt::substitute(pattern.extended<typevar_dim_type>()->get_element_type(), typevars, concrete));
        default: {
          stringstream ss;
          ss << "The substitution for dynd typevar " << pattern << ", " << it->second
             << ", is not a substitutable dimension type";
          throw invalid_argument(ss.str());
        }
        }
      } else {
        stringstream ss;
        ss << "The substitution for dynd typevar " << pattern << ", " << it->second << ", is not concrete as required";
        throw invalid_argument(ss.str());
      }
    } else {
      if (concrete) {
        stringstream ss;
        ss << "No substitution type for dynd typevar " << pattern << " was available";
        throw invalid_argument(ss.str());
      } else {
        return ndt::typevar_dim_type::make(
            pattern.extended<typevar_dim_type>()->get_name(),
            ndt::substitute(pattern.extended<typevar_dim_type>()->get_element_type(), typevars, concrete));
      }
    }
  }
  case pow_dimsym_id: {
    // Look up to the exponent typevar
    std::string exponent_name = pattern.extended<pow_dimsym_type>()->get_exponent();
    map<std::string, ndt::type>::const_iterator tv_type = typevars.find(exponent_name);
    intptr_t exponent = -1;
    if (tv_type != typevars.end()) {
      if (tv_type->second.get_id() == fixed_dim_id) {
        exponent = tv_type->second.extended<fixed_dim_type>()->get_fixed_dim_size();
      } else if (tv_type->second.get_id() == typevar_dim_id) {
        // If it's a typevar, substitute the new name in
        exponent_name = tv_type->second.extended<typevar_dim_type>()->get_name();
        if (concrete) {
          stringstream ss;
          ss << "The substitution for dynd typevar " << exponent_name << ", " << tv_type->second
             << ", is not concrete as required";
          throw invalid_argument(ss.str());
        }
      } else {
        stringstream ss;
        ss << "The substitution for dynd typevar " << exponent_name << ", " << tv_type->second
           << ", is not a fixed_dim integer as required";
        throw invalid_argument(ss.str());
      }
    }
    // If the exponent is zero, just substitute the rest of the type
    if (exponent == 0) {
      return ndt::substitute(pattern.extended<pow_dimsym_type>()->get_element_type(), typevars, concrete);
    }
    // Get the base type
    ndt::type base_tp = pattern.extended<pow_dimsym_type>()->get_base_type();
    if (base_tp.get_id() == typevar_dim_id) {
      map<std::string, ndt::type>::const_iterator btv_type =
          typevars.find(base_tp.extended<typevar_dim_type>()->get_name());
      if (btv_type == typevars.end()) {
        // We haven't seen this typevar yet, check if concrete
        // is required
        if (concrete) {
          stringstream ss;
          ss << "No substitution type for dynd typevar " << base_tp << " was available";
          throw invalid_argument(ss.str());
        }
      } else if (btv_type->second.get_ndim() > 0 && btv_type->second.get_id() != dim_fragment_id) {
        // Swap in for the base type
        base_tp = btv_type->second;
      } else {
        stringstream ss;
        ss << "The substitution for dynd typevar " << base_tp << ", " << btv_type->second
           << ", is not a substitutable dimension type";
        throw invalid_argument(ss.str());
      }
    }
    // Substitute the element type, then apply the exponent
    ndt::type result = ndt::substitute(pattern.extended<pow_dimsym_type>()->get_element_type(), typevars, concrete);
    if (exponent == 0) {
      return result;
    } else if (exponent < 0) {
      return ndt::make_pow_dimsym(base_tp, exponent_name, result);
    } else {
      switch (base_tp.get_id()) {
      case fixed_dim_id: {
        if (!base_tp.extended<base_fixed_dim_type>()->is_sized()) {
          if (concrete) {
            stringstream ss;
            ss << "The base for a dimensional power type, 'Fixed ** " << exponent << "', is not concrete as required";
            throw invalid_argument(ss.str());
          }
          for (intptr_t i = 0; i < exponent; ++i) {
            result = ndt::make_fixed_dim_kind(result);
          }
          return result;
        } else {
          intptr_t dim_size = base_tp.extended<fixed_dim_type>()->get_fixed_dim_size();
          for (intptr_t i = 0; i < exponent; ++i) {
            result = ndt::make_fixed_dim(dim_size, result);
          }
          return result;
        }
      }
      case var_dim_id:
        for (intptr_t i = 0; i < exponent; ++i) {
          result = ndt::make_type<ndt::var_dim_type>(result);
        }
        return result;
      case typevar_dim_id: {
        const std::string &tvname = base_tp.extended<typevar_dim_type>()->get_name();
        for (intptr_t i = 0; i < exponent; ++i) {
          result = ndt::typevar_dim_type::make(tvname, result);
        }
        return result;
      }
      default: {
        stringstream ss;
        ss << "Cannot substitute " << base_tp << " as the base of a dynd dimensional power type";
        throw invalid_argument(ss.str());
      }
      }
    }
  }
  case ellipsis_dim_id: {
    const std::string &name = pattern.extended<ellipsis_dim_type>()->get_name();
    if (!name.empty()) {
      map<std::string, ndt::type>::const_iterator it = typevars.find(pattern.extended<typevar_dim_type>()->get_name());
      if (it != typevars.end()) {
        if (it->second.get_id() == dim_fragment_id) {
          return it->second.extended<dim_fragment_type>()->apply_to_dtype(
              ndt::substitute(pattern.extended<ellipsis_dim_type>()->get_element_type(), typevars, concrete));
        } else {
          stringstream ss;
          ss << "The substitution for dynd typevar " << pattern << ", " << it->second
             << ", is not a dim fragment as required";
          throw invalid_argument(ss.str());
        }
      } else {
        if (concrete) {
          stringstream ss;
          ss << "No substitution type for dynd typevar " << pattern << " was available";
          throw invalid_argument(ss.str());
        } else {
          return ndt::make_ellipsis_dim(
              pattern.extended<ellipsis_dim_type>()->get_name(),
              ndt::substitute(pattern.extended<ellipsis_dim_type>()->get_element_type(), typevars, concrete));
        }
      }
    } else {
      throw invalid_argument("Cannot substitute into an unnamed ellipsis typevar");
    }
  }
  case any_kind_id: {
    if (concrete) {
      stringstream ss;
      ss << "The dynd type " << pattern << " is not concrete as required";
      throw invalid_argument(ss.str());
    } else {
      return pattern;
    }
  }
  case scalar_kind_id: {
    if (concrete) {
      stringstream ss;
      ss << "The dynd type " << pattern << " is not concrete as required";
      throw invalid_argument(ss.str());
    } else {
      return pattern;
    }
  }
  default:
    break;
  }

  stringstream ss;
  ss << "Unsupported dynd type \"" << pattern << "\" encountered for substituting typevars";
  throw invalid_argument(ss.str());
}
