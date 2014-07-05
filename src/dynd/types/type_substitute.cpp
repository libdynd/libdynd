//
// Copyright (C) 2011-14 Mark Wiebe, DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <dynd/types/type_substitute.hpp>
#include <dynd/types/pointer_type.hpp>
#include <dynd/types/base_struct_type.hpp>
#include <dynd/types/option_type.hpp>
#include <dynd/types/base_memory_type.hpp>
#include <dynd/types/fixed_dim_type.hpp>
#include <dynd/types/cfixed_dim_type.hpp>
#include <dynd/types/typevar_type.hpp>
#include <dynd/types/typevar_dim_type.hpp>
#include <dynd/types/ellipsis_dim_type.hpp>
#include <dynd/types/dim_fragment_type.hpp>
#include <dynd/types/var_dim_type.hpp>
#include <dynd/types/struct_type.hpp>
#include <dynd/types/cstruct_type.hpp>
#include <dynd/types/tuple_type.hpp>
#include <dynd/types/ctuple_type.hpp>

using namespace std;
using namespace dynd;

/**
 * Substitutes the field types for contiguous array of types
 */
static nd::array
substitute_type_array(const nd::array &type_array,
                      std::map<nd::string, ndt::type> &typevars, bool concrete)
{
  intptr_t field_count = type_array.get_dim_size();
  const ndt::type *field_types =
      reinterpret_cast<const ndt::type *>(type_array.get_readonly_originptr());
  nd::array tmp_field_types(
      nd::typed_empty(1, &field_count, ndt::make_strided_of_type()));
  ndt::type *ftraw =
      reinterpret_cast<ndt::type *>(tmp_field_types.get_readwrite_originptr());
  for (intptr_t i = 0; i < field_count; ++i) {
    ftraw[i] = ndt::substitute(field_types[i], typevars, concrete);
  }
  return tmp_field_types;
}

ndt::type
ndt::detail::internal_substitute(const ndt::type &pattern,
                                 std::map<nd::string, ndt::type> &typevars,
                                 bool concrete)
{
  // This function assumes that ``pattern`` is symbolic, so does not
  // have to check types that are always concrete
  switch (pattern.get_type_id()) {
    case pointer_type_id:
      return ndt::make_pointer(
          ndt::substitute(pattern.tcast<pointer_type>()->get_target_type(),
                          typevars, concrete));
    case strided_dim_type_id:
      return ndt::make_strided_dim(
          ndt::substitute(pattern.tcast<strided_dim_type>()->get_element_type(),
                          typevars, concrete));
    case fixed_dim_type_id:
      return ndt::make_fixed_dim(
          pattern.tcast<fixed_dim_type>()->get_fixed_dim_size(),
          ndt::substitute(pattern.tcast<fixed_dim_type>()->get_element_type(),
                          typevars, concrete));
    case cfixed_dim_type_id:
      return ndt::make_cfixed_dim(
          pattern.tcast<cfixed_dim_type>()->get_fixed_dim_size(),
          ndt::substitute(pattern.tcast<cfixed_dim_type>()->get_element_type(),
                          typevars, concrete));
    case var_dim_type_id:
      return ndt::make_var_dim(
          ndt::substitute(pattern.tcast<var_dim_type>()->get_element_type(),
                          typevars, concrete));
    case struct_type_id:
      return ndt::make_struct(
          pattern.tcast<struct_type>()->get_field_names(),
          substitute_type_array(
              pattern.tcast<base_tuple_type>()->get_field_types(), typevars,
              concrete));
    case cstruct_type_id:
      return ndt::make_cstruct(
          pattern.tcast<cstruct_type>()->get_field_names(),
          substitute_type_array(
              pattern.tcast<base_tuple_type>()->get_field_types(), typevars,
              concrete));
    case tuple_type_id:
      return ndt::make_tuple(
          substitute_type_array(
              pattern.tcast<base_tuple_type>()->get_field_types(), typevars,
              concrete));
    case ctuple_type_id:
      return ndt::make_ctuple(
          substitute_type_array(
              pattern.tcast<base_tuple_type>()->get_field_types(), typevars,
              concrete));
    case option_type_id:
      return ndt::make_option(ndt::substitute(
          pattern.tcast<option_type>()->get_value_type(), typevars, concrete));
    case funcproto_type_id:
      return ndt::make_funcproto(
          substitute_type_array(
              pattern.tcast<funcproto_type>()->get_param_types(), typevars,
              concrete),
          substitute(pattern.tcast<funcproto_type>()->get_return_type(),
                     typevars, concrete));
    case typevar_type_id: {
      map<nd::string, ndt::type>::const_iterator it =
          typevars.find(pattern.tcast<typevar_type>()->get_name());
      if (it != typevars.end()) {
        if (it->second.get_ndim() != 0) {
          stringstream ss;
          ss << "The substitution for dynd type var " << pattern << ", "
             << it->second << ", is a dimension, expected a dtype";
          throw invalid_argument(ss.str());
        }
        if (!concrete || !it->second.is_symbolic()) {
          return it->second;
        } else {
          stringstream ss;
          ss << "The substitution for dynd type var " << pattern << ", "
             << it->second << ", is not concrete as required";
          throw invalid_argument(ss.str());
        }
      } else {
        if (concrete) {
          stringstream ss;
          ss << "No substitution type for dynd type var " << pattern
             << " was available";
          throw invalid_argument(ss.str());
        } else {
          return pattern;
        }
      }
    }
    case typevar_dim_type_id: {
      map<nd::string, ndt::type>::const_iterator it =
          typevars.find(pattern.tcast<typevar_dim_type>()->get_name());
      if (it != typevars.end()) {
        if (it->second.get_ndim() == 0) {
          stringstream ss;
          ss << "The substitution for dynd type var " << pattern << ", "
             << it->second << ", is a dtype, expected a dimension";
          throw invalid_argument(ss.str());
        }
        if (!concrete || !it->second.is_symbolic()) {
          switch (it->second.get_type_id()) {
          case strided_dim_type_id:
            return ndt::make_strided_dim(ndt::substitute(
                pattern.tcast<typevar_dim_type>()->get_element_type(), typevars,
                concrete));
          case fixed_dim_type_id:
            return ndt::make_fixed_dim(
                it->second.tcast<fixed_dim_type>()->get_fixed_dim_size(),
                ndt::substitute(
                    pattern.tcast<typevar_dim_type>()->get_element_type(),
                    typevars, concrete));
          case cfixed_dim_type_id:
            return ndt::make_cfixed_dim(
                it->second.tcast<cfixed_dim_type>()->get_fixed_dim_size(),
                ndt::substitute(
                    pattern.tcast<typevar_dim_type>()->get_element_type(),
                    typevars, concrete));
          case var_dim_type_id:
            return ndt::make_var_dim(ndt::substitute(
                pattern.tcast<typevar_dim_type>()->get_element_type(), typevars,
                concrete));
          default: {
            stringstream ss;
            ss << "The substitution for dynd type var " << pattern << ", "
               << it->second << ", is not a substitutable dimension type";
            throw invalid_argument(ss.str());
          }
          }
        } else {
          stringstream ss;
          ss << "The substitution for dynd type var " << pattern << ", "
             << it->second << ", is not concrete as required";
          throw invalid_argument(ss.str());
        }
      } else {
        if (concrete) {
          stringstream ss;
          ss << "No substitution type for dynd type var " << pattern
             << " was available";
          throw invalid_argument(ss.str());
        } else {
          return ndt::make_typevar_dim(
              pattern.tcast<typevar_dim_type>()->get_name(),
              ndt::substitute(
                  pattern.tcast<typevar_dim_type>()->get_element_type(),
                  typevars, concrete));
        }
      }
    }
    case ellipsis_dim_type_id: {
      const nd::string &name = pattern.tcast<ellipsis_dim_type>()->get_name();
      if (!name.is_null()) {
        map<nd::string, ndt::type>::const_iterator it =
            typevars.find(pattern.tcast<typevar_dim_type>()->get_name());
        if (it != typevars.end()) {
          if (it->second.get_type_id() == dim_fragment_type_id) {
            return it->second.tcast<dim_fragment_type>()->apply_to_dtype(
                ndt::substitute(
                    pattern.tcast<ellipsis_dim_type>()->get_element_type(),
                    typevars, concrete));
          } else {
            stringstream ss;
            ss << "The substitution for dynd type var " << pattern << ", "
               << it->second << ", is not a dim fragment as required";
            throw invalid_argument(ss.str());
          }
        } else {
          if (concrete) {
            stringstream ss;
            ss << "No substitution type for dynd type var " << pattern
               << " was available";
            throw invalid_argument(ss.str());
          } else {
            return ndt::make_ellipsis_dim(
                pattern.tcast<ellipsis_dim_type>()->get_name(),
                ndt::substitute(
                    pattern.tcast<ellipsis_dim_type>()->get_element_type(),
                    typevars, concrete));
          }
        }
      } else {
        throw invalid_argument(
            "Cannot substitute into an unnamed ellipsis typevar");
      }
    }
    default:
      break;
  }

  stringstream ss;
  ss << "Unsupported dynd type " << pattern
     << " encountered for substituting typevars";
  throw invalid_argument(ss.str());
}
