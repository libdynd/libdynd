//
// Copyright (C) 2011-16 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <dynd/type_registry.hpp>
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
static std::vector<ndt::type> substitute_type_array(const std::vector<ndt::type> &type_array,
                                                    const std::map<std::string, ndt::type> &typevars, bool concrete) {
  intptr_t field_count = type_array.size();
  std::vector<ndt::type> tmp_field_types(field_count);

  for (intptr_t i = 0; i < field_count; ++i) {
    tmp_field_types[i] = ndt::substitute(type_array[i], typevars, concrete);
  }
  return tmp_field_types;
}

// This substitutes just the types supported by the datashape argument grammar
static void internal_substitute_args(const ndt::type &tp, const char *out_arrmeta, char *out_data,
                                     const char *in_arrmeta, const char *in_data,
                                     const std::map<std::string, ndt::type> &typevars, bool concrete) {
  switch (tp.get_id()) {
  case int64_id:
    *reinterpret_cast<int64_t *>(out_data) = *reinterpret_cast<const int64_t *>(in_data);
    break;
  case string_id:
    *reinterpret_cast<dynd::string *>(out_data) = *reinterpret_cast<const dynd::string *>(in_data);
    break;
  case type_id:
    *reinterpret_cast<ndt::type *>(out_data) =
        ndt::substitute(*reinterpret_cast<const ndt::type *>(in_data), typevars, concrete);
    break;
  case struct_id:
  case tuple_id: {
    intptr_t field_count;
    const ndt::type *field_types;
    const uintptr_t *arrmeta_offsets;
    const uintptr_t *out_data_offsets = reinterpret_cast<const uintptr_t *>(out_arrmeta);
    const uintptr_t *in_data_offsets = reinterpret_cast<const uintptr_t *>(in_arrmeta);
    if (tp.get_id() == tuple_id) {
      field_count = tp.extended<ndt::tuple_type>()->get_field_count();
      field_types = tp.extended<ndt::tuple_type>()->get_field_types_raw();
      arrmeta_offsets = tp.extended<ndt::tuple_type>()->get_arrmeta_offsets_raw();
    } else {
      field_count = tp.extended<ndt::struct_type>()->get_field_count();
      field_types = tp.extended<ndt::struct_type>()->get_field_types_raw();
      arrmeta_offsets = tp.extended<ndt::struct_type>()->get_arrmeta_offsets_raw();
    }
    for (intptr_t i = 0; i < field_count; ++i) {
      internal_substitute_args(field_types[i], out_arrmeta + arrmeta_offsets[i], out_data + out_data_offsets[i],
                               in_arrmeta + arrmeta_offsets[i], in_data + in_data_offsets[i], typevars, concrete);
    }
    break;
  }
  case fixed_dim_id: {
    const ndt::fixed_dim_type *etp = tp.extended<ndt::fixed_dim_type>();
    const ndt::type &el_tp = etp->get_element_type();
    const size_stride_t &out_ss = *reinterpret_cast<const size_stride_t *>(out_arrmeta);
    const size_stride_t &in_ss = *reinterpret_cast<const size_stride_t *>(in_arrmeta);
    out_arrmeta += sizeof(size_stride_t);
    in_arrmeta += sizeof(size_stride_t);
    for (intptr_t i = 0, ie = etp->get_fixed_dim_size(); i != ie; ++i) {
      internal_substitute_args(el_tp, out_arrmeta, out_data, in_arrmeta, in_data, typevars, concrete);
      out_data += out_ss.stride;
      in_data += in_ss.stride;
    }
    break;
  }
  default: {
    stringstream ss;
    ss << "substitute_typevars: type construction args during substitution had invalid type " << tp;
    throw runtime_error(ss.str());
  }
  }
}

static nd::buffer internal_substitute_args(const nd::buffer &args, const std::map<std::string, ndt::type> &typevars,
                                           bool concrete) {
  nd::buffer result = nd::buffer::empty(args.get_type());
  internal_substitute_args(args.get_type(), result->metadata(), result.data(), args->metadata(), args.cdata(), typevars,
                           concrete);
  return result;
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
    return ndt::make_type<ndt::pointer_type>(
        ndt::substitute(pattern.extended<pointer_type>()->get_target_type(), typevars, concrete));
  case fixed_dim_kind_id:
    if (!concrete) {
      return ndt::make_type<ndt::fixed_dim_kind_type>(
          ndt::substitute(pattern.extended<base_dim_type>()->get_element_type(), typevars, concrete));
    } else {
      throw invalid_argument("The dynd pattern type includes a symbolic "
                             "'fixed' dimension, which is not concrete as "
                             "requested");
    }
  case fixed_dim_id:
    return ndt::make_fixed_dim(
        pattern.extended<fixed_dim_type>()->get_fixed_dim_size(),
        ndt::substitute(pattern.extended<fixed_dim_type>()->get_element_type(), typevars, concrete));
  case var_dim_id:
    return ndt::make_type<ndt::var_dim_type>(
        ndt::substitute(pattern.extended<var_dim_type>()->get_element_type(), typevars, concrete));
  case struct_id:
    return ndt::make_type<ndt::struct_type>(
        pattern.extended<struct_type>()->get_field_names(),
        substitute_type_array(pattern.extended<struct_type>()->get_field_types(), typevars, concrete));
  case tuple_id: {
    const std::vector<ndt::type> &element_tp =
        substitute_type_array(pattern.extended<tuple_type>()->get_field_types(), typevars, concrete);
    return ndt::make_type<ndt::tuple_type>(element_tp.size(), element_tp.data());
  }
  case option_id:
    return ndt::make_type<ndt::option_type>(
        ndt::substitute(pattern.extended<option_type>()->get_value_type(), typevars, concrete));
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
        case fixed_dim_kind_id:
          return ndt::make_type<ndt::fixed_dim_kind_type>(
              ndt::substitute(pattern.extended<typevar_dim_type>()->get_element_type(), typevars, concrete));
        case fixed_dim_id:
          return ndt::make_fixed_dim(
              it->second.extended<fixed_dim_type>()->get_fixed_dim_size(),
              ndt::substitute(pattern.extended<typevar_dim_type>()->get_element_type(), typevars, concrete));
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
        return ndt::make_type<ndt::typevar_dim_type>(
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
      return ndt::make_type<ndt::pow_dimsym_type>(base_tp, exponent_name, result);
    } else {
      switch (base_tp.get_id()) {
      case fixed_dim_kind_id: {
        if (concrete) {
          stringstream ss;
          ss << "The base for a dimensional power type, 'Fixed ** " << exponent << "', is not concrete as required";
          throw invalid_argument(ss.str());
        }
        for (intptr_t i = 0; i < exponent; ++i) {
          result = ndt::make_type<ndt::fixed_dim_kind_type>(result);
        }
        return result;
      }
      case fixed_dim_id: {
        intptr_t dim_size = base_tp.extended<fixed_dim_type>()->get_fixed_dim_size();
        for (intptr_t i = 0; i < exponent; ++i) {
          result = ndt::make_fixed_dim(dim_size, result);
        }
        return result;
      }
      case var_dim_id:
        for (intptr_t i = 0; i < exponent; ++i) {
          result = ndt::make_type<ndt::var_dim_type>(result);
        }
        return result;
      case typevar_dim_id: {
        const std::string &tvname = base_tp.extended<typevar_dim_type>()->get_name();
        for (intptr_t i = 0; i < exponent; ++i) {
          result = ndt::make_type<ndt::typevar_dim_type>(tvname, result);
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
          return ndt::make_type<ellipsis_dim_type>(
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
  default: {
    if (pattern.is_builtin()) {
      return pattern;
    } else {
      // Generic code path via type reconstruction. We turn the type into its generic type constructor arguments, do the
      // type variable substitution on them, then reconstruct the type.
      type_id_t tid = pattern.get_id();
      nd::buffer args = pattern.extended()->get_type_constructor_args();
      args = internal_substitute_args(args, typevars, concrete);
      return dynd::detail::infos()[tid].construct_type(tid, args, ndt::type());
    }
  }
  }
}
