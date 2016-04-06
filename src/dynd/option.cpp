//
// Copyright (C) 2011-16 DyND Developers
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

namespace {

nd::callable make_assign_na() {
  typedef type_id_sequence<bool_id, int8_id, int16_id, int32_id, int64_id, int128_id, float32_id, float64_id,
                           complex_float32_id, complex_float64_id, void_id, bytes_id, string_id, fixed_dim_id> type_ids;

  dispatcher<nd::callable> dispatcher = nd::callable::new_make_all<nd::assign_na_callable, type_ids>();
  dispatcher.insert({{uint32_id}, nd::make_callable<nd::assign_na_callable<uint32_id>>()});
  dynd::dispatcher<nd::callable> dim_dispatcher;

  auto t = ndt::type("() -> ?Any");
  nd::callable self = nd::functional::call<nd::assign_na>(t);

  for (auto tp_id : {fixed_dim_id, var_dim_id}) {
    dim_dispatcher.insert({{tp_id}, nd::functional::elwise(self)});
  }

  return nd::make_callable<nd::assign_na_dispatch_callable>(t, dispatcher, dim_dispatcher);
}

nd::callable make_is_na() {
  typedef type_id_sequence<bool_id, int8_id, int16_id, int32_id, int64_id, int128_id, uint32_id, float32_id, float64_id,
                           complex_float32_id, complex_float64_id, void_id, bytes_id, string_id, fixed_dim_id> type_ids;

  dispatcher<nd::callable> dispatcher = nd::callable::new_make_all<nd::is_na_callable, type_ids>();
  dynd::dispatcher<nd::callable> dim_dispatcher;

  nd::callable self = nd::functional::call<nd::is_na>(ndt::type("(Any) -> Any"));

  for (auto tp_id : {fixed_dim_id, var_dim_id}) {
    dim_dispatcher.insert({{tp_id}, nd::functional::elwise(self)});
  }

  return nd::make_callable<nd::is_na_dispatch_callable>(ndt::type("(Any) -> Any"), dispatcher, dim_dispatcher);
}

} // unnamed namespace

DYND_API nd::callable nd::assign_na = make_assign_na();
DYND_API nd::callable nd::is_na = make_is_na();

void nd::old_assign_na(const ndt::type &option_tp, const char *arrmeta, char *data) {
  const ndt::type &value_tp = option_tp.extended<ndt::option_type>()->get_value_type();
  if (value_tp.is_builtin()) {
    assign_na_builtin(value_tp.get_id(), data);
  } else {
    assign_na->call(option_tp, arrmeta, data, 0, nullptr, nullptr, nullptr, 0, nullptr,
                    std::map<std::string, ndt::type>());
  }
}

bool nd::old_is_avail(const ndt::type &option_tp, const char *arrmeta, const char *data) {

  const ndt::type value_tp = option_tp.extended<ndt::option_type>()->get_value_type();
  if (value_tp.is_builtin()) {
    return is_avail_builtin(value_tp.get_id(), data);
  } else {
    ndt::type src_tp[1] = {option_tp};
    char result;
    is_na->call(ndt::make_type<bool1>(), nullptr, &result, 1, src_tp, &arrmeta, const_cast<char **>(&data), 0, nullptr,
                std::map<std::string, ndt::type>());
    return result == 0;
  }
}

void nd::set_option_from_utf8_string(const ndt::type &option_tp, const char *arrmeta, char *data,
                                     const char *utf8_begin, const char *utf8_end, const eval::eval_context *ectx) {
  const ndt::type value_tp = option_tp.extended<ndt::option_type>()->get_value_type();
  if (value_tp.get_base_id() != string_kind_id && parse_na(utf8_begin, utf8_end)) {
    nd::old_assign_na(option_tp, arrmeta, data);
  } else {
    if (value_tp.is_builtin()) {
      if (value_tp.unchecked_get_builtin_id() == bool_id) {
        *reinterpret_cast<bool1 *>(data) = parse<bool>(utf8_begin, utf8_end);
      } else {
        string_to_number(data, value_tp.unchecked_get_builtin_id(), utf8_begin, utf8_end, ectx->errmode);
      }
    } else {
      value_tp.extended()->set_from_utf8_string(arrmeta, data, utf8_begin, utf8_end, ectx);
    }
  }
}
