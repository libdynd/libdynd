//
// Copyright (C) 2011-15 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <dynd/functional.hpp>
#include <dynd/callables/assign_na_dispatch_callable.hpp>
#include <dynd/callables/is_na_dispatch_callable.hpp>
#include <dynd/callables/assign_na_callable.hpp>
#include <dynd/callables/is_na_callable.hpp>
#include <dynd/option.hpp>

using namespace std;
using namespace dynd;

DYND_API nd::callable nd::assign_na::make()
{
  typedef type_id_sequence<bool_id, int8_id, int16_id, int32_id, int64_id, int128_id, float32_id, float64_id,
                           complex_float32_id, complex_float64_id, void_id, bytes_id, string_id, fixed_dim_id> type_ids;

  dispatcher<callable> dispatcher = callable::new_make_all<assign_na_callable, type_ids>();
  dispatcher.insert({{uint32_id}, callable::make<assign_na_callable<uint32_id>>()});
  dynd::dispatcher<callable> dim_dispatcher;

  auto t = ndt::type("() -> ?Any");
  callable self = functional::call<assign_na>(t);

  for (auto tp_id : {fixed_dim_id, var_dim_id}) {
    dim_dispatcher.insert({{tp_id}, functional::elwise(self)});
  }

  return make_callable<assign_na_dispatch_callable>(t, dispatcher, dim_dispatcher);
}

DYND_DEFAULT_DECLFUNC_GET(nd::assign_na)

DYND_API struct nd::assign_na nd::assign_na;

void nd::old_assign_na(const ndt::type &option_tp, const char *arrmeta, char *data)
{
  const ndt::type &value_tp = option_tp.extended<ndt::option_type>()->get_value_type();
  if (value_tp.is_builtin()) {
    assign_na_builtin(value_tp.get_id(), data);
  }
  else {
    nd::kernel_builder ckb;
    nd::callable &af = nd::assign_na::get();
    af.get()->instantiate(af->static_data(), NULL, &ckb, option_tp, arrmeta, 0, NULL, NULL, kernel_request_single, 0,
                          NULL, std::map<std::string, ndt::type>());
    nd::kernel_prefix *ckp = ckb.get();
    ckp->get_function<kernel_single_t>()(ckp, data, NULL);
  }
}

DYND_API nd::callable nd::is_na::make()
{
  typedef type_id_sequence<bool_id, int8_id, int16_id, int32_id, int64_id, int128_id, uint32_id, float32_id, float64_id,
                           complex_float32_id, complex_float64_id, void_id, bytes_id, string_id, fixed_dim_id> type_ids;

  dispatcher<callable> dispatcher = callable::new_make_all<is_na_callable, type_ids>();
  dynd::dispatcher<callable> dim_dispatcher;

  callable self = functional::call<is_na>(ndt::type("(Any) -> Any"));

  for (auto tp_id : {fixed_dim_id, var_dim_id}) {
    dim_dispatcher.insert({{tp_id}, functional::elwise(self)});
  }

  return make_callable<is_na_dispatch_callable>(ndt::type("(Any) -> Any"), dispatcher, dim_dispatcher);
}

DYND_DEFAULT_DECLFUNC_GET(nd::is_na)

DYND_API struct nd::is_na nd::is_na;

bool nd::old_is_avail(const ndt::type &option_tp, const char *arrmeta, const char *data)
{
  const ndt::type value_tp = option_tp.extended<ndt::option_type>()->get_value_type();
  if (value_tp.is_builtin()) {
    return is_avail_builtin(value_tp.get_id(), data);
  }
  else {
    nd::kernel_builder ckb;
    nd::callable &af = nd::is_na::get();
    ndt::type src_tp[1] = {option_tp};
    af.get()->instantiate(af->static_data(), NULL, &ckb, ndt::make_type<bool1>(), NULL, 1, src_tp, &arrmeta,
                          kernel_request_single, 0, NULL, std::map<std::string, ndt::type>());
    nd::kernel_prefix *ckp = ckb.get();
    char result;
    ckp->get_function<kernel_single_t>()(ckp, &result, const_cast<char **>(&data));
    return result == 0;
  }
}

void nd::set_option_from_utf8_string(const ndt::type &option_tp, const char *arrmeta, char *data,
                                     const char *utf8_begin, const char *utf8_end, const eval::eval_context *ectx)
{
  const ndt::type value_tp = option_tp.extended<ndt::option_type>()->get_value_type();
  if (value_tp.get_base_id() != string_kind_id && parse_na(utf8_begin, utf8_end)) {
    nd::old_assign_na(option_tp, arrmeta, data);
  }
  else {
    if (value_tp.is_builtin()) {
      if (value_tp.unchecked_get_builtin_id() == bool_id) {
        *reinterpret_cast<bool1 *>(data) = parse<bool>(utf8_begin, utf8_end);
      }
      else {
        string_to_number(data, value_tp.unchecked_get_builtin_id(), utf8_begin, utf8_end, ectx->errmode);
      }
    }
    else {
      value_tp.extended()->set_from_utf8_string(arrmeta, data, utf8_begin, utf8_end, ectx);
    }
  }
}
