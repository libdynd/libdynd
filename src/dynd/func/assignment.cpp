//
// Copyright (C) 2011-15 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <dynd/func/assignment.hpp>
#include <dynd/callables/assign_callable.hpp>
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

} // anonymous namespace

DYND_API nd::callable nd::assign::make()
{
  typedef type_id_sequence<bool_id, int8_id, int16_id, int32_id, int64_id, int128_id, uint8_id, uint16_id, uint32_id,
                           uint64_id, uint128_id, float32_id, float64_id, complex_float32_id,
                           complex_float64_id> numeric_ids;

  ndt::type self_tp = ndt::callable_type::make(ndt::any_kind_type::make(), {ndt::any_kind_type::make()}, {"error_mode"},
                                               {ndt::make_type<ndt::option_type>(ndt::make_type<assign_error_mode>())});

  auto dispatcher =
      callable::new_make_all<_bind<assign_error_mode, assignment_kernel>::type, numeric_ids, numeric_ids>();
  dispatcher.insert({{string_id, string_id}, callable::make<assignment_kernel<string_id, string_id>>()});
  dispatcher.insert({{bytes_id, bytes_id}, callable::make<assignment_kernel<bytes_id, bytes_id>>()});
  dispatcher.insert(
      {{fixed_bytes_id, fixed_bytes_id}, callable::make<assignment_kernel<fixed_bytes_id, fixed_bytes_id>>()});
  dispatcher.insert({{char_id, char_id}, callable::make<assignment_kernel<char_id, char_id>>()});
  dispatcher.insert({{char_id, fixed_string_id}, callable::make<assignment_kernel<char_id, fixed_string_id>>()});
  dispatcher.insert({{char_id, string_id}, callable::make<assignment_kernel<char_id, string_id>>()});
  dispatcher.insert(
      {{{adapt_id, any_kind_id}, nd::callable::make<detail::adapt_assign_to_kernel>(ndt::type("(Any) -> Any"))},
       {{any_kind_id, adapt_id}, nd::callable::make<detail::adapt_assign_from_kernel>(ndt::type("(Any) -> Any"))},
       {{adapt_id, adapt_id}, nd::callable::make<detail::adapt_assign_from_kernel>(ndt::type("(Any) -> Any"))}});
  dispatcher.insert({{fixed_string_id, char_id}, callable::make<assignment_kernel<fixed_string_id, char_id>>()});
  dispatcher.insert({{string_id, char_id}, callable::make<assignment_kernel<string_id, char_id>>()});
  dispatcher.insert({{type_id, type_id}, callable::make<assignment_kernel<type_id, type_id>>()});
  dispatcher.insert({{string_id, int32_id}, callable::make<assignment_kernel<string_id, int32_id>>()});
  dispatcher.insert(
      {{fixed_string_id, fixed_string_id}, callable::make<assignment_kernel<fixed_string_id, fixed_string_id>>()});
  dispatcher.insert({{fixed_string_id, string_id}, callable::make<assignment_kernel<fixed_string_id, string_id>>()});
  dispatcher.insert({{fixed_string_id, uint8_id}, callable::make<assignment_kernel<fixed_string_id, uint8_id>>()});
  dispatcher.insert({{fixed_string_id, uint16_id}, callable::make<assignment_kernel<fixed_string_id, uint16_id>>()});
  dispatcher.insert({{fixed_string_id, uint32_id}, callable::make<assignment_kernel<fixed_string_id, uint32_id>>()});
  dispatcher.insert({{fixed_string_id, uint64_id}, callable::make<assignment_kernel<fixed_string_id, uint64_id>>()});
  dispatcher.insert({{fixed_string_id, uint128_id}, callable::make<assignment_kernel<fixed_string_id, uint128_id>>()});
  dispatcher.insert({{int32_id, fixed_string_id}, callable::make<assignment_kernel<int32_id, fixed_string_id>>()});
  dispatcher.insert({{string_id, string_id}, callable::make<assignment_kernel<string_id, string_id>>()});
  dispatcher.insert({{string_id, fixed_string_id}, callable::make<assignment_kernel<string_id, fixed_string_id>>()});
  dispatcher.insert({{bool_id, string_id}, callable::make<assignment_kernel<bool_id, string_id>>()});
  dispatcher.insert(
      {{{scalar_kind_id, option_id}, callable::make<option_to_value_ck>(ndt::type("(?Any) -> Any"))},
       {{option_id, option_id}, callable::make<assignment_kernel<option_id, option_id>>()},
       {{option_id, scalar_kind_id}, callable::make<detail::assignment_option_kernel>(ndt::type("(Any) -> ?Any"))}});
  dispatcher.insert({{option_id, string_id}, callable::make<assignment_kernel<option_id, string_id>>()});
  dispatcher.insert({{option_id, float64_id}, callable::make<assignment_kernel<option_id, float64_id>>()});
  dispatcher.insert({{string_id, type_id}, callable::make<assignment_kernel<string_id, type_id>>()});
  dispatcher.insert({{type_id, string_id}, callable::make<assignment_kernel<type_id, string_id>>()});
  dispatcher.insert({{pointer_id, pointer_id}, callable::make<assignment_kernel<pointer_id, pointer_id>>()});
  dispatcher.insert({{int8_id, string_id}, callable::make<assignment_kernel<int8_id, string_id>>()});
  dispatcher.insert({{int16_id, string_id}, callable::make<assignment_kernel<int16_id, string_id>>()});
  dispatcher.insert({{int32_id, string_id}, callable::make<assignment_kernel<int32_id, string_id>>()});
  dispatcher.insert({{int64_id, string_id}, callable::make<assignment_kernel<int64_id, string_id>>()});
  dispatcher.insert({{uint8_id, string_id}, callable::make<assignment_kernel<uint8_id, string_id>>()});
  dispatcher.insert({{uint16_id, string_id}, callable::make<assignment_kernel<uint16_id, string_id>>()});
  dispatcher.insert({{uint32_id, string_id}, callable::make<assignment_kernel<uint32_id, string_id>>()});
  dispatcher.insert({{uint64_id, string_id}, callable::make<assignment_kernel<uint64_id, string_id>>()});
  dispatcher.insert({{float32_id, string_id}, callable::make<assignment_kernel<float32_id, string_id>>()});
  dispatcher.insert({{float64_id, string_id}, callable::make<assignment_kernel<float64_id, string_id>>()});
  dispatcher.insert({{tuple_id, tuple_id}, callable::make<assignment_kernel<tuple_id, tuple_id>>()});
  dispatcher.insert({{struct_id, int32_id}, callable::make<assignment_kernel<struct_id, struct_id>>()});
  dispatcher.insert({{struct_id, struct_id}, callable::make<assignment_kernel<struct_id, struct_id>>()});
  dispatcher.insert(
      {{scalar_kind_id, dim_kind_id}, nd::functional::elwise(nd::functional::call<assign>(ndt::type("(Any) -> Any")))});
  dispatcher.insert(
      {{dim_kind_id, scalar_kind_id}, nd::functional::elwise(nd::functional::call<assign>(ndt::type("(Any) -> Any")))});
  dispatcher.insert(
      {{dim_kind_id, dim_kind_id}, nd::functional::elwise(nd::functional::call<assign>(ndt::type("(Any) -> Any")))});

  return make_callable<assign_dispatch_callable>(self_tp, std::make_shared<dynd::dispatcher<callable>>(dispatcher));
}

DYND_DEFAULT_DECLFUNC_GET(nd::assign)

DYND_API struct nd::assign nd::assign;

void dynd::make_assignment_kernel(nd::kernel_builder *ckb, const ndt::type &dst_tp, const char *dst_arrmeta,
                                  const ndt::type &src_tp, const char *src_arrmeta, kernel_request_t kernreq,
                                  const eval::eval_context *ectx)
{
  nd::array error_mode = ectx->errmode;
  nd::assign::get()->instantiate(nd::assign::get()->static_data(), NULL, ckb, dst_tp, dst_arrmeta, 1, &src_tp,
                                 &src_arrmeta, kernreq, 1, &error_mode, std::map<std::string, ndt::type>());
}

void dynd::make_pod_typed_data_assignment_kernel(nd::kernel_builder *ckb, size_t data_size,
                                                 size_t DYND_UNUSED(data_alignment), kernel_request_t kernreq)
{
  // Aligned specialization tables
  switch (data_size) {
  case 1:
    ckb->emplace_back<nd::trivial_copy_kernel<1>>(kernreq);
    break;
  case 2:
    ckb->emplace_back<nd::trivial_copy_kernel<2>>(kernreq);
    break;
  case 4:
    ckb->emplace_back<nd::trivial_copy_kernel<4>>(kernreq);
    break;
  case 8:
    ckb->emplace_back<nd::trivial_copy_kernel<8>>(kernreq);
    break;
  default:
    ckb->emplace_back<nd::unaligned_copy_ck>(kernreq, data_size);
    break;
  }
}
