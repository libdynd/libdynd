//
// Copyright (C) 2011-16 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <dynd/callables/assign_na_callable.hpp>
#include <dynd/callables/is_na_callable.hpp>
#include <dynd/callables/is_na_dispatch_callable.hpp>
#include <dynd/functional.hpp>
#include <dynd/option.hpp>

using namespace std;
using namespace dynd;

namespace {

static std::vector<ndt::type> assign_na_func_ptr(const ndt::type &dst_tp, size_t DYND_UNUSED(nsrc),
                                                 const ndt::type *DYND_UNUSED(src_tp)) {
  if (dst_tp.get_id() == option_id) {
    return {dst_tp.extended<ndt::option_type>()->get_value_type()};
  }

  return {};
}

static std::vector<ndt::type> is_na_func_ptr(const ndt::type &DYND_UNUSED(dst_tp), size_t DYND_UNUSED(nsrc),
                                             const ndt::type *src_tp) {
  if (src_tp[0].get_id() == option_id) {
    return {src_tp[0].extended<ndt::option_type>()->get_value_type()};
  } else {
    return {src_tp[0]};
  }
}

template <std::vector<ndt::type> (*Func)(const ndt::type &, size_t, const ndt::type *)>
nd::callable make_assign_na() {
  class dispatch_callable : public nd::base_dispatch_callable {
    dispatcher<Func, 1, nd::callable> m_dispatcher;

  public:
    dispatch_callable()
        : base_dispatch_callable(ndt::type("() -> ?Any")),
          m_dispatcher(
              nd::callable::make_all<Func, nd::assign_na_callable,
                                     type_sequence<bool, int8_t, int16_t, int32_t, int64_t, int128, uint32_t, float,
                                                   double, dynd::complex<float>, dynd::complex<double>, void,
                                                   dynd::bytes, dynd::string, ndt::fixed_dim_kind_type>>()) {}

    const nd::callable &specialize(const ndt::type &dst_tp, intptr_t DYND_UNUSED(nsrc),
                                   const ndt::type *DYND_UNUSED(src_tp)) {
      if (dst_tp.get_id() == option_id) {
        const ndt::type &dst_value_tp = dst_tp.extended<ndt::option_type>()->get_value_type();
        return m_dispatcher(dst_value_tp);
      }

      return nd::elwise;
    }
  };

  return nd::make_callable<dispatch_callable>();
}

nd::callable make_is_na() {
  dispatcher<is_na_func_ptr, 1, nd::callable> dispatcher = nd::callable::make_all<
      is_na_func_ptr, nd::is_na_callable,
      type_sequence<bool, int8_t, int16_t, int32_t, int64_t, int128, uint32_t, float, double, dynd::complex<float>,
                    dynd::complex<double>, void, dynd::bytes, dynd::string, ndt::fixed_dim_kind_type>>();
  dynd::dispatcher<is_na_func_ptr, 1, nd::callable> dim_dispatcher;

  for (auto tp_id : {ndt::make_type<ndt::fixed_dim_kind_type>(), ndt::make_type<ndt::var_dim_type>()}) {
    dim_dispatcher.insert({{tp_id}, nd::get_elwise()});
  }

  return nd::make_callable<nd::is_na_dispatch_callable<is_na_func_ptr>>(ndt::type("(Any) -> Any"), dispatcher,
                                                                        dim_dispatcher);
}

} // unnamed namespace

DYND_API nd::callable nd::assign_na = make_assign_na<assign_na_func_ptr>();
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
