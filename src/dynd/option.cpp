//
// Copyright (C) 2011-16 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <dynd/callables/assign_na_callable.hpp>
#include <dynd/callables/is_na_callable.hpp>
#include <dynd/callables/multidispatch_callable.hpp>
#include <dynd/functional.hpp>
#include <dynd/option.hpp>

using namespace std;
using namespace dynd;

namespace {

static std::vector<ndt::type> assign_na_func_ptr(const ndt::type &dst_tp, size_t DYND_UNUSED(nsrc),
                                                 const ndt::type *DYND_UNUSED(src_tp)) {
  return {dst_tp};
}

static std::vector<ndt::type> is_na_func_ptr(const ndt::type &DYND_UNUSED(dst_tp), size_t DYND_UNUSED(nsrc),
                                             const ndt::type *src_tp) {
  return {src_tp[0]};
}

nd::callable make_assign_na() {
  auto children = nd::callable::make_all<
      nd::assign_na_callable,
      type_sequence<bool, int8_t, int16_t, int32_t, int64_t, int128, uint32_t, float, double, dynd::complex<float>,
                    dynd::complex<double>, void, dynd::bytes, dynd::string, ndt::fixed_dim_kind_type>>(
      assign_na_func_ptr);
  children.insert(nd::get_elwise(ndt::type("() -> Fixed * Any")));
  children.insert(nd::get_elwise(ndt::type("() -> var * Any")));

  return nd::make_callable<nd::multidispatch_callable<1>>(ndt::type("() -> ?Any"), children);
}

nd::callable make_is_na() {
  dispatcher<1, nd::callable> dispatcher = nd::callable::make_all<
      nd::is_na_callable,
      type_sequence<bool, int8_t, int16_t, int32_t, int64_t, int128, uint32_t, float, double, dynd::complex<float>,
                    dynd::complex<double>, void, dynd::bytes, dynd::string, ndt::fixed_dim_kind_type>>(is_na_func_ptr);
  dispatcher.insert(nd::get_elwise(ndt::type("(Fixed * Any) -> Fixed * Any")));
  dispatcher.insert(nd::get_elwise(ndt::type("(var * Any) -> var * Any")));

  return nd::make_callable<nd::multidispatch_callable<1>>(ndt::type("(Any) -> Any"), dispatcher);
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
