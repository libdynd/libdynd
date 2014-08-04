//
// Copyright (C) 2011-14 Mark Wiebe, DyND Developers
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
#include <dynd/types/ellipsis_dim_type.hpp>
#include <dynd/types/dim_fragment_type.hpp>

using namespace std;
using namespace dynd;

static bool recursive_match(const ndt::type& concrete, const ndt::type &pattern,
                std::map<nd::string, ndt::type> &typevars)
{
    if (!pattern.is_symbolic()) {
        // If both are concrete, it's just an equality check
        return concrete == pattern;
    }

    if (concrete.get_ndim() == 0) {
        if (pattern.get_ndim() == 0) {
            // Matching a scalar vs scalar
            if (pattern.get_type_id() == typevar_type_id) {
                ndt::type &tv_type =
                    typevars[pattern.tcast<typevar_type>()->get_name()];
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
                            concrete.tcast<pointer_type>()->get_target_type(),
                            pattern.tcast<pointer_type>()->get_target_type(),
                            typevars);
                    case struct_type_id:
                    case cstruct_type_id:
                        if (concrete.tcast<base_struct_type>()
                                 ->get_field_names()
                                 .equals_exact(pattern.tcast<base_struct_type>()
                                                   ->get_field_names())) {
                            // The names are all the same, now match against the
                            // types
                            size_t field_count =
                                concrete.tcast<base_struct_type>()
                                    ->get_field_count();
                            const ndt::type *concrete_fields =
                                concrete.tcast<base_struct_type>()
                                    ->get_field_types_raw();
                            const ndt::type *pattern_fields =
                                pattern.tcast<base_struct_type>()
                                    ->get_field_types_raw();
                            for (size_t i = 0; i != field_count; ++i) {
                                if (!recursive_match(concrete_fields[i],
                                                     pattern_fields[i],
                                                     typevars)) {
                                    return false;
                                }
                            }
                            return true;
                        } else {
                            return false;
                        }
                    case tuple_type_id:
                    case ctuple_type_id:
                        if (concrete.tcast<base_tuple_type>()
                                ->get_field_count() ==
                            pattern.tcast<base_tuple_type>()
                                ->get_field_count()) {
                            // Match against the types
                            size_t field_count =
                                concrete.tcast<base_tuple_type>()
                                    ->get_field_count();
                            const ndt::type *concrete_fields =
                                concrete.tcast<base_tuple_type>()
                                    ->get_field_types_raw();
                            const ndt::type *pattern_fields =
                                pattern.tcast<base_tuple_type>()
                                    ->get_field_types_raw();
                            for (size_t i = 0; i != field_count; ++i) {
                                if (!recursive_match(concrete_fields[i],
                                                     pattern_fields[i],
                                                     typevars)) {
                                    return false;
                                }
                            }
                            return true;
                        } else {
                            return false;
                        }
                    case option_type_id:
                        return recursive_match(
                            concrete.tcast<option_type>()->get_value_type(),
                            pattern.tcast<option_type>()->get_value_type(),
                            typevars);
                    case cuda_host_type_id:
                    case cuda_device_type_id:
                        return recursive_match(
                            concrete.tcast<base_memory_type>()
                                ->get_storage_type(),
                            pattern.tcast<base_memory_type>()
                                ->get_storage_type(),
                            typevars);
                    case funcproto_type_id:
                        if (concrete.tcast<funcproto_type>()
                                ->get_param_count() ==
                            pattern.tcast<funcproto_type>()
                                ->get_param_count()) {
                            // First match the return type
                            if (!recursive_match(
                                    concrete.tcast<funcproto_type>()
                                        ->get_return_type(),
                                    pattern.tcast<funcproto_type>()
                                        ->get_return_type(),
                                    typevars)) {
                                return false;
                            }
                            // Then match all the parameters
                            size_t param_count =
                                concrete.tcast<funcproto_type>()
                                    ->get_param_count();
                            for (size_t i = 0; i != param_count; ++i) {
                                if (!recursive_match(
                                        concrete.tcast<funcproto_type>()
                                            ->get_param_type(i),
                                        pattern.tcast<funcproto_type>()
                                            ->get_param_type(i),
                                        typevars)) {
                                    return false;
                                }
                            }
                            return true;
                        } else {
                            return false;
                        }
                    default:
                        return false;
                }
            } else {
                return false;
            }
        } else {
            // Matching a scalar vs dimension, only case which makes sense
            // is an ellipsis_dim
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
                return recursive_match(
                    concrete,
                    pattern.tcast<ellipsis_dim_type>()->get_element_type(),
                    typevars);
            } else {
                return false;
            }
        }
    } else {
        // Matching a dimension vs dimension
        if (concrete.get_type_id() == pattern.get_type_id()) {
            switch (concrete.get_type_id()) {
                case strided_dim_type_id:
                case offset_dim_type_id:
                case var_dim_type_id:
                    return recursive_match(
                        concrete.tcast<base_dim_type>()
                            ->get_element_type(),
                        pattern.tcast<base_dim_type>()
                            ->get_element_type(),
                        typevars);
                case fixed_dim_type_id:
                    return concrete.tcast<fixed_dim_type>()
                                   ->get_fixed_dim_size() ==
                               pattern.tcast<fixed_dim_type>()
                                   ->get_fixed_dim_size() &&
                           recursive_match(
                               concrete.tcast<base_dim_type>()
                                   ->get_element_type(),
                               pattern.tcast<base_dim_type>()
                                   ->get_element_type(),
                               typevars);
                case cfixed_dim_type_id:
                    return concrete.tcast<cfixed_dim_type>()
                                   ->get_fixed_dim_size() ==
                               pattern.tcast<cfixed_dim_type>()
                                   ->get_fixed_dim_size() &&
                           concrete.tcast<cfixed_dim_type>()
                                   ->get_fixed_stride() ==
                               pattern.tcast<cfixed_dim_type>()
                                   ->get_fixed_stride() &&
                           recursive_match(
                               concrete.tcast<base_dim_type>()
                                   ->get_element_type(),
                               pattern.tcast<base_dim_type>()
                                   ->get_element_type(),
                               typevars);
                default:
                    break;
            }
            stringstream ss;
            ss << "Type pattern matching between dimension types " << concrete
               << " and " << pattern << " is not yet implemented";
            throw type_error(ss.str());
        } else if (pattern.get_type_id() == ellipsis_dim_type_id) {
            // Match the number of concrete dimensions required on
            // the left
            if (concrete.get_ndim() >= pattern.get_ndim() - 1) {
                intptr_t matched_ndim =
                    concrete.get_ndim() - pattern.get_ndim() + 1;
                const nd::string &tv_name =
                    pattern.tcast<ellipsis_dim_type>()->get_name();
                if (!tv_name.is_null()) {
                    ndt::type &tv_type = typevars[tv_name];
                    if (tv_type.is_null()) {
                        // This typevar hasn't been seen yet, so it's
                        // a dim fragment of the given size.
                        tv_type =
                            ndt::make_dim_fragment(matched_ndim, concrete);
                    } else {
                        // Make sure the type matches previous
                        // instances of the type var, which in
                        // this case means they should broadcast
                        // together.
                        if (tv_type.get_type_id() == dim_fragment_type_id) {
                            ndt::type result =
                                tv_type.tcast<dim_fragment_type>()
                                    ->broadcast_with_type(matched_ndim,
                                                          concrete);
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
                return recursive_match(
                    concrete.get_type_at_dimension(NULL, matched_ndim),
                    pattern.tcast<ellipsis_dim_type>()->get_element_type(),
                    typevars);
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
                return recursive_match(
                    concrete.get_type_at_dimension(NULL, 1),
                    pattern.tcast<typevar_dim_type>()->get_element_type(),
                    typevars);
            } else {
                // Make sure the type matches previous
                // instances of the type var
                if (concrete.get_type_id() != tv_type.get_type_id()) {
                    return false;
                }
                switch (concrete.get_type_id()) {
                    case fixed_dim_type_id:
                        if (concrete.tcast<fixed_dim_type>()
                                ->get_fixed_dim_size() !=
                            tv_type.tcast<fixed_dim_type>()
                                ->get_fixed_dim_size()) {
                            return false;
                        }
                        break;
                    case cfixed_dim_type_id:
                        if (concrete.tcast<cfixed_dim_type>()
                                ->get_fixed_dim_size() !=
                            tv_type.tcast<cfixed_dim_type>()
                                ->get_fixed_dim_size()) {
                            return false;
                        }
                        break;
                    default:
                        break;
                }
                return recursive_match(
                    concrete.get_type_at_dimension(NULL, 1),
                    pattern.tcast<typevar_dim_type>()->get_element_type(),
                    typevars);
            }
        } else {
            return false;
        }
    }
}

bool ndt::pattern_match(const ndt::type &concrete,
                             const ndt::type &pattern,
                             std::map<nd::string, ndt::type> &typevars)
{
    // Don't allow symbols in the LHS
    if (concrete.is_symbolic()) {
        stringstream ss;
        ss << "Expected a concrete type for matching, got symbolic type "
           << concrete;
        throw type_error(ss.str());
    }

    return recursive_match(concrete, pattern, typevars);
}
