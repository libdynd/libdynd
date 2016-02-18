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

  map<std::array<type_id_t, 2>, callable> children =
      callable::make_all<_bind<assign_error_mode, assignment_kernel>::type, numeric_ids, numeric_ids>();
  children[{{string_id, string_id}}] = callable::make<assignment_kernel<string_id, string_id>>();
  children[{{bytes_id, bytes_id}}] = callable::make<assignment_kernel<bytes_id, bytes_id>>();
  children[{{fixed_bytes_id, fixed_bytes_id}}] = callable::make<assignment_kernel<fixed_bytes_id, fixed_bytes_id>>();
  children[{{char_id, char_id}}] = callable::make<assignment_kernel<char_id, char_id>>();
  children[{{char_id, fixed_string_id}}] = callable::make<assignment_kernel<char_id, fixed_string_id>>();
  children[{{char_id, string_id}}] = callable::make<assignment_kernel<char_id, string_id>>();
  children[{{fixed_string_id, char_id}}] = callable::make<assignment_kernel<fixed_string_id, char_id>>();
  children[{{string_id, char_id}}] = callable::make<assignment_kernel<string_id, char_id>>();
  children[{{type_id, type_id}}] = callable::make<assignment_kernel<type_id, type_id>>();
  children[{{string_id, int32_id}}] = callable::make<assignment_kernel<string_id, int32_id>>();
  children[{{fixed_string_id, fixed_string_id}}] =
      callable::make<assignment_kernel<fixed_string_id, fixed_string_id>>();
  children[{{fixed_string_id, string_id}}] = callable::make<assignment_kernel<fixed_string_id, string_id>>();
  children[{{fixed_string_id, uint8_id}}] = callable::make<assignment_kernel<fixed_string_id, uint8_id>>();
  children[{{fixed_string_id, uint16_id}}] = callable::make<assignment_kernel<fixed_string_id, uint16_id>>();
  children[{{fixed_string_id, uint32_id}}] = callable::make<assignment_kernel<fixed_string_id, uint32_id>>();
  children[{{fixed_string_id, uint64_id}}] = callable::make<assignment_kernel<fixed_string_id, uint64_id>>();
  children[{{fixed_string_id, uint128_id}}] = callable::make<assignment_kernel<fixed_string_id, uint128_id>>();
  children[{{int32_id, fixed_string_id}}] = callable::make<assignment_kernel<int32_id, fixed_string_id>>();
  children[{{string_id, string_id}}] = callable::make<assignment_kernel<string_id, string_id>>();
  children[{{string_id, fixed_string_id}}] = callable::make<assignment_kernel<string_id, fixed_string_id>>();
  children[{{bool_id, string_id}}] = callable::make<assignment_kernel<bool_id, string_id>>();
  children[{{option_id, option_id}}] = callable::make<assignment_kernel<option_id, option_id>>();
  children[{{int32_id, option_id}}] = callable::make<option_to_value_ck>(ndt::type("(?Any) -> Any"));
  children[{{string_id, option_id}}] = callable::make<option_to_value_ck>(ndt::type("(?Any) -> Any"));
  children[{{float64_id, option_id}}] = callable::make<option_to_value_ck>(ndt::type("(?Any) -> Any"));
  children[{{bool_id, option_id}}] = callable::make<option_to_value_ck>(ndt::type("(?Any) -> Any"));
  children[{{int8_id, option_id}}] = callable::make<option_to_value_ck>(ndt::type("(?Any) -> Any"));
  children[{{uint32_id, option_id}}] = callable::make<option_to_value_ck>(ndt::type("(?Any) -> Any"));
  children[{{option_id, int32_id}}] = callable::make<detail::assignment_option_kernel>(ndt::type("(Any) -> ?Any"));
  children[{{option_id, string_id}}] = callable::make<assignment_kernel<option_id, string_id>>();
  children[{{option_id, float64_id}}] = callable::make<assignment_kernel<option_id, float64_id>>();
  children[{{option_id, bool_id}}] = callable::make<detail::assignment_option_kernel>(ndt::type("(Any) -> ?Any"));
  children[{{option_id, int8_id}}] = callable::make<detail::assignment_option_kernel>(ndt::type("(Any) -> ?Any"));
  children[{{option_id, uint32_id}}] = callable::make<detail::assignment_option_kernel>(ndt::type("(Any) -> ?Any"));
  children[{{string_id, type_id}}] = callable::make<assignment_kernel<string_id, type_id>>();
  children[{{type_id, string_id}}] = callable::make<assignment_kernel<type_id, string_id>>();
  children[{{pointer_id, pointer_id}}] = callable::make<assignment_kernel<pointer_id, pointer_id>>();
  children[{{int8_id, string_id}}] = callable::make<assignment_kernel<int8_id, string_id>>();
  children[{{int16_id, string_id}}] = callable::make<assignment_kernel<int16_id, string_id>>();
  children[{{int32_id, string_id}}] = callable::make<assignment_kernel<int32_id, string_id>>();
  children[{{int64_id, string_id}}] = callable::make<assignment_kernel<int64_id, string_id>>();
  children[{{uint8_id, string_id}}] = callable::make<assignment_kernel<uint8_id, string_id>>();
  children[{{uint16_id, string_id}}] = callable::make<assignment_kernel<uint16_id, string_id>>();
  children[{{uint32_id, string_id}}] = callable::make<assignment_kernel<uint32_id, string_id>>();
  children[{{uint64_id, string_id}}] = callable::make<assignment_kernel<uint64_id, string_id>>();
  children[{{float32_id, string_id}}] = callable::make<assignment_kernel<float32_id, string_id>>();
  children[{{float64_id, string_id}}] = callable::make<assignment_kernel<float64_id, string_id>>();
  children[{{tuple_id, tuple_id}}] = callable::make<assignment_kernel<tuple_id, tuple_id>>();
  children[{{struct_id, int32_id}}] = callable::make<assignment_kernel<struct_id, struct_id>>();
  children[{{struct_id, struct_id}}] = callable::make<assignment_kernel<struct_id, struct_id>>();
  for (type_id_t tp_id : {bool_id, int8_id, int16_id, int32_id, int64_id, int128_id, uint8_id, uint16_id, uint32_id,
                          uint64_id, uint128_id, float32_id, float64_id, fixed_dim_id, type_id}) {
    children[{{tp_id, var_dim_id}}] = nd::functional::elwise(nd::functional::call<assign>(ndt::type("(Any) -> Any")));
    children[{{var_dim_id, tp_id}}] = nd::functional::elwise(nd::functional::call<assign>(ndt::type("(Any) -> Any")));
  }
  children[{{var_dim_id, var_dim_id}}] =
      nd::functional::elwise(nd::functional::call<assign>(ndt::type("(Any) -> Any")));
  for (type_id_t tp_id : {bool_id, int8_id, int16_id, int32_id, int64_id, int128_id, uint8_id, uint16_id, uint32_id,
                          uint64_id, uint128_id, float32_id, float64_id, type_id}) {
    children[{{tp_id, fixed_dim_id}}] = nd::functional::elwise(nd::functional::call<assign>(ndt::type("(Any) -> Any")));
    children[{{fixed_dim_id, tp_id}}] = nd::functional::elwise(nd::functional::call<assign>(ndt::type("(Any) -> Any")));
  }
  children[{{fixed_dim_id, fixed_dim_id}}] =
      nd::functional::elwise(nd::functional::call<assign>(ndt::type("(Any) -> Any")));

  children[{{adapt_id, int16_id}}] = nd::callable::make<detail::adapt_assign_to_kernel>(ndt::type("(Any) -> Any"));
  children[{{adapt_id, int32_id}}] = nd::callable::make<detail::adapt_assign_to_kernel>(ndt::type("(Any) -> Any"));
  children[{{adapt_id, int64_id}}] = nd::callable::make<detail::adapt_assign_to_kernel>(ndt::type("(Any) -> Any"));
  children[{{adapt_id, float32_id}}] = nd::callable::make<detail::adapt_assign_to_kernel>(ndt::type("(Any) -> Any"));
  children[{{adapt_id, float64_id}}] = nd::callable::make<detail::adapt_assign_to_kernel>(ndt::type("(Any) -> Any"));
  children[{{adapt_id, complex_float32_id}}] =
      nd::callable::make<detail::adapt_assign_to_kernel>(ndt::type("(Any) -> Any"));
  children[{{adapt_id, complex_float64_id}}] =
      nd::callable::make<detail::adapt_assign_to_kernel>(ndt::type("(Any) -> Any"));

  children[{{int32_id, adapt_id}}] = nd::callable::make<detail::adapt_assign_from_kernel>(ndt::type("(Any) -> Any"));
  children[{{struct_id, adapt_id}}] = nd::callable::make<detail::adapt_assign_from_kernel>(ndt::type("(Any) -> Any"));

  return functional::dispatch(self_tp, [children](const ndt::type &dst_tp, intptr_t DYND_UNUSED(nsrc),
                                                  const ndt::type *src_tp) mutable -> callable & {
    callable &child = children[{{dst_tp.get_id(), src_tp[0].get_id()}}];
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
