//
// Copyright (C) 2011-15 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <dynd/func/assignment.hpp>
#include <dynd/functional.hpp>
#include <dynd/kernels/assignment_kernels.hpp>
#include <dynd/types/any_kind_type.hpp>

using namespace std;
using namespace dynd;

DYND_API nd::callable nd::assign::make()
{
  typedef type_id_sequence<bool_type_id, int8_type_id, int16_type_id, int32_type_id, int64_type_id, int128_type_id,
                           uint8_type_id, uint16_type_id, uint32_type_id, uint64_type_id, uint128_type_id,
                           float32_type_id, float64_type_id, complex_float32_type_id,
                           complex_float64_type_id> numeric_type_ids;

  ndt::type self_tp = ndt::callable_type::make(ndt::any_kind_type::make(), {ndt::any_kind_type::make()}, {"error_mode"},
                                               {ndt::make_type<ndt::option_type>(ndt::make_type<assign_error_mode>())});

  map<std::array<type_id_t, 2>, callable> children =
      callable::make_all<_bind<assign_error_mode, assignment_kernel>::type, numeric_type_ids, numeric_type_ids>();
  children[{{date_type_id, date_type_id}}] = callable::make<assignment_kernel<date_type_id, date_type_id>>();
  children[{{date_type_id, string_type_id}}] = callable::make<assignment_kernel<date_type_id, string_type_id>>();
  children[{{date_type_id, fixed_string_type_id}}] = callable::make<assignment_kernel<date_type_id, string_type_id>>();
  children[{{string_type_id, date_type_id}}] = callable::make<assignment_kernel<string_type_id, date_type_id>>();
  children[{{string_type_id, string_type_id}}] = callable::make<assignment_kernel<string_type_id, string_type_id>>();
  children[{{bytes_type_id, bytes_type_id}}] = callable::make<assignment_kernel<bytes_type_id, bytes_type_id>>();
  children[{{fixed_bytes_type_id, fixed_bytes_type_id}}] =
      callable::make<assignment_kernel<fixed_bytes_type_id, fixed_bytes_type_id>>();
  children[{{char_type_id, char_type_id}}] = callable::make<assignment_kernel<char_type_id, char_type_id>>();
  children[{{char_type_id, fixed_string_type_id}}] =
      callable::make<assignment_kernel<char_type_id, fixed_string_type_id>>();
  children[{{char_type_id, string_type_id}}] = callable::make<assignment_kernel<char_type_id, string_type_id>>();
  children[{{fixed_string_type_id, char_type_id}}] =
      callable::make<assignment_kernel<fixed_string_type_id, char_type_id>>();
  children[{{string_type_id, char_type_id}}] = callable::make<assignment_kernel<string_type_id, char_type_id>>();
  children[{{type_type_id, type_type_id}}] = callable::make<assignment_kernel<type_type_id, type_type_id>>();
  children[{{time_type_id, string_type_id}}] = callable::make<assignment_kernel<time_type_id, string_type_id>>();
  children[{{string_type_id, time_type_id}}] = callable::make<assignment_kernel<string_type_id, time_type_id>>();
  children[{{string_type_id, int32_type_id}}] = callable::make<assignment_kernel<string_type_id, int32_type_id>>();
  children[{{datetime_type_id, string_type_id}}] =
      callable::make<assignment_kernel<datetime_type_id, string_type_id>>();
  children[{{string_type_id, datetime_type_id}}] =
      callable::make<assignment_kernel<string_type_id, datetime_type_id>>();
  children[{{datetime_type_id, datetime_type_id}}] =
      callable::make<assignment_kernel<datetime_type_id, datetime_type_id>>();
  children[{{fixed_string_type_id, fixed_string_type_id}}] =
      callable::make<assignment_kernel<fixed_string_type_id, fixed_string_type_id>>();
  children[{{fixed_string_type_id, string_type_id}}] =
      callable::make<assignment_kernel<fixed_string_type_id, string_type_id>>();
  children[{{fixed_string_type_id, uint8_type_id}}] =
      callable::make<assignment_kernel<fixed_string_type_id, uint8_type_id>>();
  children[{{fixed_string_type_id, uint16_type_id}}] =
      callable::make<assignment_kernel<fixed_string_type_id, uint16_type_id>>();
  children[{{fixed_string_type_id, uint32_type_id}}] =
      callable::make<assignment_kernel<fixed_string_type_id, uint32_type_id>>();
  children[{{fixed_string_type_id, uint64_type_id}}] =
      callable::make<assignment_kernel<fixed_string_type_id, uint64_type_id>>();
  children[{{fixed_string_type_id, uint128_type_id}}] =
      callable::make<assignment_kernel<fixed_string_type_id, uint128_type_id>>();
  children[{{int32_type_id, fixed_string_type_id}}] =
      callable::make<assignment_kernel<int32_type_id, fixed_string_type_id>>();
  children[{{string_type_id, string_type_id}}] = callable::make<assignment_kernel<string_type_id, string_type_id>>();
  children[{{string_type_id, fixed_string_type_id}}] =
      callable::make<assignment_kernel<string_type_id, fixed_string_type_id>>();
  children[{{bool_type_id, string_type_id}}] = callable::make<assignment_kernel<bool_type_id, string_type_id>>();
  children[{{option_type_id, option_type_id}}] =
      callable::make<detail::assignment_option_kernel>(ndt::type("(?Any) -> ?Any"));
  for (type_id_t tp_id : {int32_type_id, string_type_id, float64_type_id, bool_type_id, int8_type_id, uint32_type_id}) {
    children[{{tp_id, option_type_id}}] = callable::make<detail::assignment_option_kernel>(ndt::type("(?Any) -> ?Any"));
    children[{{option_type_id, tp_id}}] = callable::make<detail::assignment_option_kernel>(ndt::type("(?Any) -> ?Any"));
  }
  children[{{string_type_id, type_type_id}}] = callable::make<assignment_kernel<string_type_id, type_type_id>>();
  children[{{type_type_id, string_type_id}}] = callable::make<assignment_kernel<type_type_id, string_type_id>>();
  children[{{pointer_type_id, pointer_type_id}}] =
      callable::make<assignment_kernel<pointer_type_id, pointer_type_id>>();
  children[{{int8_type_id, string_type_id}}] = callable::make<assignment_kernel<int8_type_id, string_type_id>>();
  children[{{int16_type_id, string_type_id}}] = callable::make<assignment_kernel<int16_type_id, string_type_id>>();
  children[{{int32_type_id, string_type_id}}] = callable::make<assignment_kernel<int32_type_id, string_type_id>>();
  children[{{int64_type_id, string_type_id}}] = callable::make<assignment_kernel<int64_type_id, string_type_id>>();
  children[{{uint8_type_id, string_type_id}}] = callable::make<assignment_kernel<uint8_type_id, string_type_id>>();
  children[{{uint16_type_id, string_type_id}}] = callable::make<assignment_kernel<uint16_type_id, string_type_id>>();
  children[{{uint32_type_id, string_type_id}}] = callable::make<assignment_kernel<uint32_type_id, string_type_id>>();
  children[{{uint64_type_id, string_type_id}}] = callable::make<assignment_kernel<uint64_type_id, string_type_id>>();
  children[{{float32_type_id, string_type_id}}] = callable::make<assignment_kernel<float32_type_id, string_type_id>>();
  children[{{float64_type_id, string_type_id}}] = callable::make<assignment_kernel<float64_type_id, string_type_id>>();
  children[{{tuple_type_id, tuple_type_id}}] = callable::make<assignment_kernel<tuple_type_id, tuple_type_id>>();
  children[{{struct_type_id, int32_type_id}}] = callable::make<assignment_kernel<struct_type_id, struct_type_id>>();
  children[{{struct_type_id, struct_type_id}}] = callable::make<assignment_kernel<struct_type_id, struct_type_id>>();
  for (type_id_t tp_id : {bool_type_id, int8_type_id, int16_type_id, int32_type_id, int64_type_id, int128_type_id,
                          uint8_type_id, uint16_type_id, uint32_type_id, uint64_type_id, uint128_type_id,
                          float32_type_id, float64_type_id, fixed_dim_type_id, type_type_id}) {
    children[{{tp_id, var_dim_type_id}}] =
        nd::functional::elwise(nd::functional::call<assign>(ndt::type("(Any) -> Any")));
    children[{{var_dim_type_id, tp_id}}] =
        nd::functional::elwise(nd::functional::call<assign>(ndt::type("(Any) -> Any")));
  }
  children[{{var_dim_type_id, var_dim_type_id}}] =
      nd::functional::elwise(nd::functional::call<assign>(ndt::type("(Any) -> Any")));
  for (type_id_t tp_id : {bool_type_id, int8_type_id, int16_type_id, int32_type_id, int64_type_id, int128_type_id,
                          uint8_type_id, uint16_type_id, uint32_type_id, uint64_type_id, uint128_type_id,
                          float32_type_id, float64_type_id, type_type_id}) {
    children[{{tp_id, fixed_dim_type_id}}] =
        nd::functional::elwise(nd::functional::call<assign>(ndt::type("(Any) -> Any")));
    children[{{fixed_dim_type_id, tp_id}}] =
        nd::functional::elwise(nd::functional::call<assign>(ndt::type("(Any) -> Any")));
  }
  children[{{fixed_dim_type_id, fixed_dim_type_id}}] =
      nd::functional::elwise(nd::functional::call<assign>(ndt::type("(Any) -> Any")));

  children[{{adapt_type_id, int16_type_id}}] =
      nd::callable::make<detail::adapt_assign_to_kernel>(ndt::type("(Any) -> Any"));
  children[{{adapt_type_id, int32_type_id}}] =
      nd::callable::make<detail::adapt_assign_to_kernel>(ndt::type("(Any) -> Any"));
  children[{{adapt_type_id, int64_type_id}}] =
      nd::callable::make<detail::adapt_assign_to_kernel>(ndt::type("(Any) -> Any"));
  children[{{adapt_type_id, float32_type_id}}] =
      nd::callable::make<detail::adapt_assign_to_kernel>(ndt::type("(Any) -> Any"));
  children[{{adapt_type_id, float64_type_id}}] =
      nd::callable::make<detail::adapt_assign_to_kernel>(ndt::type("(Any) -> Any"));
  children[{{adapt_type_id, complex_float32_type_id}}] =
      nd::callable::make<detail::adapt_assign_to_kernel>(ndt::type("(Any) -> Any"));
  children[{{adapt_type_id, complex_float64_type_id}}] =
      nd::callable::make<detail::adapt_assign_to_kernel>(ndt::type("(Any) -> Any"));
  children[{{adapt_type_id, date_type_id}}] =
      nd::callable::make<detail::adapt_assign_to_kernel>(ndt::type("(Any) -> Any"));

  children[{{int32_type_id, adapt_type_id}}] =
      nd::callable::make<detail::adapt_assign_from_kernel>(ndt::type("(Any) -> Any"));
  children[{{struct_type_id, adapt_type_id}}] =
      nd::callable::make<detail::adapt_assign_from_kernel>(ndt::type("(Any) -> Any"));

  children[{{fixed_bytes_type_id, view_type_id}}] =
      callable::make<assignment_kernel<fixed_bytes_type_id, view_type_id>>();
  children[{{date_type_id, expr_type_id}}] = callable::make<assignment_kernel<date_type_id, expr_type_id>>();
  children[{{option_type_id, convert_type_id}}] = callable::make<assignment_kernel<option_type_id, convert_type_id>>();
  children[{{string_type_id, convert_type_id}}] = callable::make<assignment_kernel<string_type_id, convert_type_id>>();
  children[{{string_type_id, expr_type_id}}] = callable::make<assignment_kernel<string_type_id, expr_type_id>>();
  children[{{datetime_type_id, convert_type_id}}] =
      callable::make<assignment_kernel<datetime_type_id, convert_type_id>>();
  children[{{fixed_string_type_id, convert_type_id}}] =
      callable::make<assignment_kernel<fixed_string_type_id, convert_type_id>>();
  children[{{convert_type_id, convert_type_id}}] =
      callable::make<assignment_kernel<convert_type_id, convert_type_id>>();
  children[{{int16_type_id, view_type_id}}] = callable::make<assignment_kernel<int16_type_id, view_type_id>>();
  children[{{int32_type_id, view_type_id}}] = callable::make<assignment_kernel<int32_type_id, view_type_id>>();
  children[{{int64_type_id, view_type_id}}] = callable::make<assignment_kernel<int64_type_id, view_type_id>>();
  children[{{categorical_type_id, convert_type_id}}] =
      callable::make<assignment_kernel<categorical_type_id, convert_type_id>>();
  children[{{string_type_id, convert_type_id}}] = callable::make<assignment_kernel<string_type_id, convert_type_id>>();
  children[{{bool_type_id, convert_type_id}}] = callable::make<assignment_kernel<bool_type_id, convert_type_id>>();
  children[{{type_type_id, convert_type_id}}] = callable::make<assignment_kernel<type_type_id, convert_type_id>>();
  children[{{convert_type_id, convert_type_id}}] =
      callable::make<assignment_kernel<convert_type_id, convert_type_id>>();
  children[{{float64_type_id, convert_type_id}}] =
      callable::make<assignment_kernel<float64_type_id, convert_type_id>>();
  children[{{int32_type_id, convert_type_id}}] = callable::make<assignment_kernel<int32_type_id, convert_type_id>>();
  children[{{int8_type_id, convert_type_id}}] = callable::make<assignment_kernel<int8_type_id, convert_type_id>>();
  children[{{float32_type_id, convert_type_id}}] =
      callable::make<assignment_kernel<float32_type_id, convert_type_id>>();
  children[{{int64_type_id, convert_type_id}}] = callable::make<assignment_kernel<int64_type_id, convert_type_id>>();
  children[{{int16_type_id, convert_type_id}}] = callable::make<assignment_kernel<int16_type_id, convert_type_id>>();
  children[{{convert_type_id, float64_type_id}}] =
      callable::make<assignment_kernel<convert_type_id, float64_type_id>>();
  children[{{convert_type_id, int8_type_id}}] = callable::make<assignment_kernel<convert_type_id, int8_type_id>>();
  children[{{convert_type_id, int16_type_id}}] = callable::make<assignment_kernel<convert_type_id, int16_type_id>>();
  children[{{convert_type_id, int32_type_id}}] = callable::make<assignment_kernel<convert_type_id, int32_type_id>>();
  children[{{convert_type_id, int64_type_id}}] = callable::make<assignment_kernel<convert_type_id, int64_type_id>>();
  children[{{convert_type_id, uint8_type_id}}] = callable::make<assignment_kernel<convert_type_id, uint8_type_id>>();
  children[{{convert_type_id, uint16_type_id}}] = callable::make<assignment_kernel<convert_type_id, uint16_type_id>>();
  children[{{convert_type_id, uint32_type_id}}] = callable::make<assignment_kernel<convert_type_id, uint32_type_id>>();
  children[{{convert_type_id, uint64_type_id}}] = callable::make<assignment_kernel<convert_type_id, uint64_type_id>>();
  children[{{view_type_id, int8_type_id}}] = callable::make<assignment_kernel<view_type_id, int8_type_id>>();
  children[{{view_type_id, int16_type_id}}] = callable::make<assignment_kernel<view_type_id, int16_type_id>>();
  children[{{view_type_id, int32_type_id}}] = callable::make<assignment_kernel<view_type_id, int32_type_id>>();
  children[{{view_type_id, int64_type_id}}] = callable::make<assignment_kernel<view_type_id, int64_type_id>>();
  children[{{view_type_id, uint8_type_id}}] = callable::make<assignment_kernel<view_type_id, uint8_type_id>>();
  children[{{view_type_id, uint16_type_id}}] = callable::make<assignment_kernel<view_type_id, uint16_type_id>>();
  children[{{view_type_id, uint32_type_id}}] = callable::make<assignment_kernel<view_type_id, uint32_type_id>>();
  children[{{view_type_id, uint64_type_id}}] = callable::make<assignment_kernel<view_type_id, uint64_type_id>>();
  children[{{uint8_type_id, view_type_id}}] = callable::make<assignment_kernel<uint8_type_id, view_type_id>>();
  children[{{uint16_type_id, view_type_id}}] = callable::make<assignment_kernel<uint16_type_id, view_type_id>>();
  children[{{uint32_type_id, view_type_id}}] = callable::make<assignment_kernel<uint32_type_id, view_type_id>>();
  children[{{uint64_type_id, view_type_id}}] = callable::make<assignment_kernel<uint64_type_id, view_type_id>>();
  children[{{uint8_type_id, convert_type_id}}] = callable::make<assignment_kernel<uint8_type_id, convert_type_id>>();
  children[{{uint16_type_id, convert_type_id}}] = callable::make<assignment_kernel<uint16_type_id, convert_type_id>>();
  children[{{uint32_type_id, convert_type_id}}] = callable::make<assignment_kernel<uint32_type_id, convert_type_id>>();
  children[{{uint64_type_id, convert_type_id}}] = callable::make<assignment_kernel<uint64_type_id, convert_type_id>>();
  children[{{fixed_dim_type_id, convert_type_id}}] =
      callable::make<assignment_kernel<fixed_dim_type_id, convert_type_id>>();
  children[{{struct_type_id, convert_type_id}}] = callable::make<assignment_kernel<struct_type_id, convert_type_id>>();
  children[{{view_type_id, int32_type_id}}] = callable::make<assignment_kernel<view_type_id, int32_type_id>>();
  children[{{view_type_id, int64_type_id}}] = callable::make<assignment_kernel<view_type_id, int64_type_id>>();
  children[{{view_type_id, view_type_id}}] = callable::make<assignment_kernel<view_type_id, view_type_id>>();
  children[{{complex_float32_type_id, convert_type_id}}] =
      callable::make<assignment_kernel<complex_float32_type_id, convert_type_id>>();
  children[{{complex_float64_type_id, convert_type_id}}] =
      callable::make<assignment_kernel<complex_float64_type_id, convert_type_id>>();
  children[{{time_type_id, convert_type_id}}] = callable::make<assignment_kernel<time_type_id, convert_type_id>>();
  children[{{date_type_id, convert_type_id}}] = callable::make<assignment_kernel<date_type_id, convert_type_id>>();

  return functional::dispatch(self_tp, [children](const ndt::type &dst_tp, intptr_t DYND_UNUSED(nsrc),
                                                  const ndt::type *src_tp) mutable -> callable & {
    callable &child = children[{{dst_tp.get_type_id(), src_tp[0].get_type_id()}}];
    if (child.is_null()) {
      //      throw std::runtime_error("assignment error");
    }
    return child;
  });
}

DYND_API struct nd::assign nd::assign;

void dynd::make_assignment_kernel(nd::kernel_builder *ckb, const ndt::type &dst_tp, const char *dst_arrmeta,
                                  const ndt::type &src_tp, const char *src_arrmeta, kernel_request_t kernreq,
                                  const eval::eval_context *ectx)
{
  nd::array error_mode = ectx->errmode;
  nd::assign::get()->instantiate(nd::assign::get()->static_data(), NULL, ckb, ckb->m_size, dst_tp, dst_arrmeta, 1,
                                 &src_tp, &src_arrmeta, kernreq, 1, &error_mode, std::map<std::string, ndt::type>());
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
