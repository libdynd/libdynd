//
// Copyright (C) 2011-16 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <dynd/assignment.hpp>
#include <dynd/callables/assign_callable.hpp>
#include <dynd/callables/assign_dispatch_callable.hpp>
#include <dynd/callables/copy_callable.hpp>
#include <dynd/functional.hpp>
#include <dynd/kernels/assignment_kernels.hpp>
#include <dynd/types/any_kind_type.hpp>

using namespace std;
using namespace dynd;

namespace {

template <typename VariadicType, template <typename, typename, VariadicType...> class T>
struct DYND_API _bind {
  template <typename Type0, typename Type1>
  using type = T<Type0, Type1>;
};

nd::callable make_assign() {
  typedef type_sequence<bool1, int8_t, int16_t, int32_t, int64_t, int128, uint8_t, uint16_t, uint32_t, uint64_t,
                        uint128, float, double, dynd::complex<float>, dynd::complex<double>>
      numeric_types;

  ndt::type self_tp = ndt::make_type<ndt::callable_type>(
      ndt::make_type<ndt::any_kind_type>(), {ndt::make_type<ndt::any_kind_type>()},
      {{ndt::make_type<ndt::option_type>(ndt::make_type<assign_error_mode>()), "error_mode"}});

  auto dispatcher =
      nd::callable::make_all<_bind<assign_error_mode, nd::assign_callable>::type, numeric_types, numeric_types>();
  dispatcher.insert({{string_id, string_id}, nd::make_callable<nd::assign_callable<dynd::string, dynd::string>>()});
  dispatcher.insert({{bytes_id, bytes_id}, nd::make_callable<nd::assign_callable<dynd::bytes, dynd::bytes>>()});
  dispatcher.insert({{fixed_bytes_id, fixed_bytes_id},
                     nd::make_callable<nd::assign_callable<ndt::fixed_bytes_type, ndt::fixed_bytes_type>>()});
  dispatcher.insert(
      {{char_id, char_id}, nd::make_callable<nd::assign_callable<ndt::fixed_string_type, ndt::fixed_string_type>>()});
  dispatcher.insert({{char_id, fixed_string_id},
                     nd::make_callable<nd::assign_callable<ndt::fixed_string_type, ndt::fixed_string_type>>()});
  dispatcher.insert({{char_id, string_id}, nd::make_callable<nd::assign_callable<char, dynd::string>>()});
  dispatcher.insert({{{adapt_id, any_kind_id}, nd::make_callable<nd::adapt_assign_to_callable>()},
                     {{any_kind_id, adapt_id}, nd::make_callable<nd::adapt_assign_from_callable>()},
                     {{adapt_id, adapt_id}, nd::make_callable<nd::adapt_assign_from_callable>()}});
  dispatcher.insert({{fixed_string_id, char_id},
                     nd::make_callable<nd::assign_callable<ndt::fixed_string_type, ndt::fixed_string_type>>()});
  dispatcher.insert({{string_id, char_id}, nd::make_callable<nd::assign_callable<dynd::string, char>>()});
  dispatcher.insert({{type_id, type_id}, nd::make_callable<nd::assign_callable<ndt::type, ndt::type>>()});
  dispatcher.insert({{string_id, int32_id}, nd::make_callable<nd::string_to_int_assign_callable<int32_t>>()});
  dispatcher.insert({{fixed_string_id, fixed_string_id},
                     nd::make_callable<nd::assign_callable<ndt::fixed_string_type, ndt::fixed_string_type>>()});
  dispatcher.insert(
      {{fixed_string_id, string_id}, nd::make_callable<nd::assign_callable<ndt::fixed_string_type, dynd::string>>()});
  //  dispatcher.insert({{fixed_string_id, uint8_id}, callable::make<assignment_kernel<fixed_string_id, uint8_id>>()});
  //  dispatcher.insert({{fixed_string_id, uint16_id}, callable::make<assignment_kernel<fixed_string_id,
  //  uint16_id>>()});
  // dispatcher.insert({{fixed_string_id, uint32_id}, callable::make<assignment_kernel<fixed_string_id, uint32_id>>()});
  //  dispatcher.insert({{fixed_string_id, uint64_id}, callable::make<assignment_kernel<fixed_string_id,
  //  uint64_id>>()});
  // dispatcher.insert({{fixed_string_id, uint128_id}, callable::make<assignment_kernel<fixed_string_id,
  // uint128_id>>()});
  dispatcher.insert({{int32_id, fixed_string_id}, nd::make_callable<nd::int_to_string_assign_callable<int32_t>>()});
  dispatcher.insert({{string_id, string_id}, nd::make_callable<nd::assign_callable<dynd::string, dynd::string>>()});
  dispatcher.insert(
      {{string_id, fixed_string_id}, nd::make_callable<nd::assign_callable<dynd::string, ndt::fixed_string_type>>()});
  dispatcher.insert({{bool_id, string_id}, nd::make_callable<nd::assign_callable<bool1, dynd::string>>()});
  dispatcher.insert(
      {{{scalar_kind_id, option_id}, nd::make_callable<nd::option_to_value_callable>()},
       {{option_id, option_id}, nd::make_callable<nd::assign_callable<ndt::option_type, ndt::option_type>>()},
       {{option_id, scalar_kind_id}, nd::make_callable<nd::assignment_option_callable>()}});
  dispatcher.insert({{option_id, string_id}, nd::make_callable<nd::assign_callable<ndt::option_type, dynd::string>>()});
  dispatcher.insert(
      {{option_id, float64_id}, nd::make_callable<nd::assign_callable<ndt::option_type, ndt::float_kind_type>>()});
  dispatcher.insert({{string_id, type_id}, nd::make_callable<nd::assign_callable<dynd::string, ndt::type>>()});
  dispatcher.insert({{type_id, string_id}, nd::make_callable<nd::assign_callable<ndt::type, dynd::string>>()});
  dispatcher.insert(
      {{pointer_id, pointer_id}, nd::make_callable<nd::assign_callable<ndt::pointer_type, ndt::pointer_type>>()});
  dispatcher.insert({{int8_id, string_id}, nd::make_callable<nd::int_to_string_assign_callable<int8_t>>()});
  dispatcher.insert({{int16_id, string_id}, nd::make_callable<nd::int_to_string_assign_callable<int16_t>>()});
  dispatcher.insert({{int32_id, string_id}, nd::make_callable<nd::int_to_string_assign_callable<int32_t>>()});
  dispatcher.insert({{int64_id, string_id}, nd::make_callable<nd::int_to_string_assign_callable<int64_t>>()});
  //  dispatcher.insert({{uint8_id, string_id}, callable::make<assignment_kernel<uint8_id, string_id>>()});
  //  dispatcher.insert({{uint16_id, string_id}, callable::make<assignment_kernel<uint16_id, string_id>>()});
  // dispatcher.insert({{uint32_id, string_id}, callable::make<assignment_kernel<uint32_id, string_id>>()});
  // dispatcher.insert({{uint64_id, string_id}, callable::make<assignment_kernel<uint64_id, string_id>>()});
  dispatcher.insert({{float32_id, string_id}, nd::make_callable<nd::assign_callable<float, dynd::string>>()});
  dispatcher.insert({{float64_id, string_id}, nd::make_callable<nd::assign_callable<double, dynd::string>>()});
  dispatcher.insert({{tuple_id, tuple_id}, nd::make_callable<nd::assign_callable<ndt::tuple_type, ndt::tuple_type>>()});
  dispatcher.insert(
      {{struct_id, int32_id}, nd::make_callable<nd::assign_callable<ndt::struct_type, ndt::struct_type>>()});
  dispatcher.insert(
      {{struct_id, struct_id}, nd::make_callable<nd::assign_callable<ndt::struct_type, ndt::struct_type>>()});
  dispatcher.insert({{scalar_kind_id, dim_kind_id}, nd::get_elwise()});
  dispatcher.insert({{dim_kind_id, scalar_kind_id}, nd::get_elwise()});
  dispatcher.insert({{dim_kind_id, dim_kind_id}, nd::get_elwise()});

  return nd::make_callable<nd::assign_dispatch_callable>(self_tp, dispatcher);
}

} // anonymous namespace

DYND_API nd::callable nd::assign = make_assign();

DYND_API nd::callable nd::copy = nd::make_callable<nd::copy_callable>();
