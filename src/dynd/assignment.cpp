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

template <typename VariadicType, template <type_id_t, type_id_t, VariadicType...> class T>
struct DYND_API _bind {
  template <type_id_t TypeID0, type_id_t TypeID1>
  using type = T<TypeID0, TypeID1>;
};

nd::callable make_assign() {
  typedef type_id_sequence<bool_id, int8_id, int16_id, int32_id, int64_id, int128_id, uint8_id, uint16_id, uint32_id,
                           uint64_id, uint128_id, float32_id, float64_id, complex_float32_id, complex_float64_id>
      numeric_ids;

  ndt::type self_tp =
      ndt::callable_type::make(ndt::any_kind_type::make(), {ndt::any_kind_type::make()},
                               {{ndt::make_type<ndt::option_type>(ndt::make_type<assign_error_mode>()), "error_mode"}});

  auto dispatcher =
      nd::callable::new_make_all<_bind<assign_error_mode, nd::assign_callable>::type, numeric_ids, numeric_ids>();
  dispatcher.insert({{string_id, string_id}, nd::make_callable<nd::assign_callable<string_id, string_id>>()});
  dispatcher.insert({{bytes_id, bytes_id}, nd::make_callable<nd::assign_callable<bytes_id, bytes_id>>()});
  dispatcher.insert(
      {{fixed_bytes_id, fixed_bytes_id}, nd::make_callable<nd::assign_callable<fixed_bytes_id, fixed_bytes_id>>()});
  dispatcher.insert({{char_id, char_id}, nd::make_callable<nd::assign_callable<fixed_string_id, fixed_string_id>>()});
  dispatcher.insert(
      {{char_id, fixed_string_id}, nd::make_callable<nd::assign_callable<fixed_string_id, fixed_string_id>>()});
  dispatcher.insert({{char_id, string_id}, nd::make_callable<nd::assign_callable<char_id, string_id>>()});
  dispatcher.insert({{{adapt_id, any_kind_id}, nd::make_callable<nd::adapt_assign_to_callable>()},
                     {{any_kind_id, adapt_id}, nd::make_callable<nd::adapt_assign_from_callable>()},
                     {{adapt_id, adapt_id}, nd::make_callable<nd::adapt_assign_from_callable>()}});
  dispatcher.insert(
      {{fixed_string_id, char_id}, nd::make_callable<nd::assign_callable<fixed_string_id, fixed_string_id>>()});
  dispatcher.insert({{string_id, char_id}, nd::make_callable<nd::assign_callable<string_id, char_id>>()});
  dispatcher.insert({{type_id, type_id}, nd::make_callable<nd::assign_callable<type_id, type_id>>()});
  dispatcher.insert({{string_id, int32_id}, nd::make_callable<nd::string_to_int_assign_callable<int32_id>>()});
  dispatcher.insert(
      {{fixed_string_id, fixed_string_id}, nd::make_callable<nd::assign_callable<fixed_string_id, fixed_string_id>>()});
  dispatcher.insert(
      {{fixed_string_id, string_id}, nd::make_callable<nd::assign_callable<fixed_string_id, string_id>>()});
  //  dispatcher.insert({{fixed_string_id, uint8_id}, callable::make<assignment_kernel<fixed_string_id, uint8_id>>()});
  //  dispatcher.insert({{fixed_string_id, uint16_id}, callable::make<assignment_kernel<fixed_string_id,
  //  uint16_id>>()});
  // dispatcher.insert({{fixed_string_id, uint32_id}, callable::make<assignment_kernel<fixed_string_id, uint32_id>>()});
  //  dispatcher.insert({{fixed_string_id, uint64_id}, callable::make<assignment_kernel<fixed_string_id,
  //  uint64_id>>()});
  // dispatcher.insert({{fixed_string_id, uint128_id}, callable::make<assignment_kernel<fixed_string_id,
  // uint128_id>>()});
  dispatcher.insert({{int32_id, fixed_string_id}, nd::make_callable<nd::int_to_string_assign_callable<int32_id>>()});
  dispatcher.insert({{string_id, string_id}, nd::make_callable<nd::assign_callable<string_id, string_id>>()});
  dispatcher.insert(
      {{string_id, fixed_string_id}, nd::make_callable<nd::assign_callable<string_id, fixed_string_id>>()});
  dispatcher.insert({{bool_id, string_id}, nd::make_callable<nd::assign_callable<bool_id, string_id>>()});
  dispatcher.insert({{{scalar_kind_id, option_id}, nd::make_callable<nd::option_to_value_callable>()},
                     {{option_id, option_id}, nd::make_callable<nd::assign_callable<option_id, option_id>>()},
                     {{option_id, scalar_kind_id}, nd::make_callable<nd::assignment_option_callable>()}});
  dispatcher.insert({{option_id, string_id}, nd::make_callable<nd::assign_callable<option_id, string_id>>()});
  dispatcher.insert({{option_id, float64_id}, nd::make_callable<nd::assign_callable<option_id, float_kind_id>>()});
  dispatcher.insert({{string_id, type_id}, nd::make_callable<nd::assign_callable<string_id, type_id>>()});
  dispatcher.insert({{type_id, string_id}, nd::make_callable<nd::assign_callable<type_id, string_id>>()});
  dispatcher.insert({{pointer_id, pointer_id}, nd::make_callable<nd::assign_callable<pointer_id, pointer_id>>()});
  dispatcher.insert({{int8_id, string_id}, nd::make_callable<nd::int_to_string_assign_callable<int8_id>>()});
  dispatcher.insert({{int16_id, string_id}, nd::make_callable<nd::int_to_string_assign_callable<int16_id>>()});
  dispatcher.insert({{int32_id, string_id}, nd::make_callable<nd::int_to_string_assign_callable<int32_id>>()});
  dispatcher.insert({{int64_id, string_id}, nd::make_callable<nd::int_to_string_assign_callable<int64_id>>()});
  //  dispatcher.insert({{uint8_id, string_id}, callable::make<assignment_kernel<uint8_id, string_id>>()});
  //  dispatcher.insert({{uint16_id, string_id}, callable::make<assignment_kernel<uint16_id, string_id>>()});
  // dispatcher.insert({{uint32_id, string_id}, callable::make<assignment_kernel<uint32_id, string_id>>()});
  // dispatcher.insert({{uint64_id, string_id}, callable::make<assignment_kernel<uint64_id, string_id>>()});
  dispatcher.insert({{float32_id, string_id}, nd::make_callable<nd::assign_callable<float32_id, string_id>>()});
  dispatcher.insert({{float64_id, string_id}, nd::make_callable<nd::assign_callable<float64_id, string_id>>()});
  dispatcher.insert({{tuple_id, tuple_id}, nd::make_callable<nd::assign_callable<tuple_id, tuple_id>>()});
  dispatcher.insert({{struct_id, int32_id}, nd::make_callable<nd::assign_callable<struct_id, struct_id>>()});
  dispatcher.insert({{struct_id, struct_id}, nd::make_callable<nd::assign_callable<struct_id, struct_id>>()});
  dispatcher.insert({{scalar_kind_id, dim_kind_id}, nd::get_elwise()});
  dispatcher.insert({{dim_kind_id, scalar_kind_id}, nd::get_elwise()});
  dispatcher.insert({{dim_kind_id, dim_kind_id}, nd::get_elwise()});

  return nd::make_callable<nd::assign_dispatch_callable>(self_tp, dispatcher);
}

} // anonymous namespace

DYND_API nd::callable nd::assign = make_assign();

DYND_API nd::callable nd::copy = nd::make_callable<nd::copy_callable>();
