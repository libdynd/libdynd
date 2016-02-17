//
// Copyright (C) 2011-15 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <dynd/functional.hpp>
#include <dynd/option.hpp>
#include <dynd/kernels/assign_na_kernel.hpp>
#include <dynd/kernels/is_na_kernel.hpp>

using namespace std;
using namespace dynd;

DYND_API nd::callable nd::assign_na::make()
{
  typedef type_id_sequence<bool_id, int8_id, int16_id, int32_id, int64_id, int128_id, float32_id, float64_id,
                           complex_float32_id, complex_float64_id, void_id, string_id, fixed_dim_id, date_id,
                           datetime_id> type_ids;

  std::map<type_id_t, callable> children = callable::make_all<assign_na_kernel, type_ids>();
  children[uint32_id] = callable::make<assign_na_kernel<uint32_id>>();
  std::array<callable, 2> dim_children;

  auto t = ndt::type("() -> ?Any");
  callable self = functional::call<assign_na>(t);

  for (auto tp_id : {fixed_dim_id, var_dim_id}) {
    dim_children[tp_id - fixed_dim_id] = functional::elwise(self);
  }

  return functional::dispatch(t, [children, dim_children](const ndt::type &dst_tp, intptr_t DYND_UNUSED(nsrc),
                                                          const ndt::type *DYND_UNUSED(src_tp)) mutable -> callable & {
    callable *child = nullptr;
    if (dst_tp.get_id() == option_id) {
      child = &children[dst_tp.extended<ndt::option_type>()->get_value_type().get_id()];
    }
    else
      child = &dim_children[dst_tp.get_id() - fixed_dim_id];

    if (child->is_null()) {
      throw std::runtime_error("no child found");
    }

    return *child;
  });
}

DYND_API struct nd::assign_na nd::assign_na;

DYND_API nd::callable nd::is_na::make()
{
  typedef type_id_sequence<bool_id, int8_id, int16_id, int32_id, int64_id, int128_id, uint32_id, float32_id, float64_id,
                           complex_float32_id, complex_float64_id, void_id, string_id, fixed_dim_id, date_id,
                           datetime_id> type_ids;

  std::map<type_id_t, callable> children = callable::make_all<is_na_kernel, type_ids>();
  std::array<callable, 2> dim_children;

  callable self = functional::call<is_na>(ndt::type("(Any) -> Any"));

  for (auto tp_id : {fixed_dim_id, var_dim_id}) {
    dim_children[tp_id - fixed_dim_id] = functional::elwise(self);
  }

  return functional::dispatch(ndt::type("(Any) -> Any"),
                              [children, dim_children](const ndt::type &DYND_UNUSED(dst_tp), intptr_t DYND_UNUSED(nsrc),
                                                       const ndt::type *src_tp) mutable -> callable & {
                                callable *child = nullptr;
                                if (src_tp[0].get_id() == option_id)
                                  child = &children[src_tp[0].extended<ndt::option_type>()->get_value_type().get_id()];
                                else
                                  child = &dim_children[src_tp[0].get_id() - fixed_dim_id];

                                if (child->is_null()) {
                                  throw std::runtime_error("no child found");
                                }

                                return *child;
                              });
}

DYND_API struct nd::is_na nd::is_na;
