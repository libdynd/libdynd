//
// Copyright (C) 2011-15 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <dynd/functional.hpp>
#include <dynd/kernels/assign_na_kernel.hpp>
#include <dynd/kernels/is_na_kernel.hpp>
#include <dynd/option.hpp>

using namespace std;
using namespace dynd;

DYND_API nd::callable nd::assign_na::make()
{
  typedef type_id_sequence<bool_id, int8_id, int16_id, int32_id, int64_id, int128_id, float32_id, float64_id,
                           complex_float32_id, complex_float64_id, void_id, bytes_id, string_id, fixed_dim_id> type_ids;

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

DYND_DEFAULT_DECLFUNC_GET(nd::assign_na)

DYND_API struct nd::assign_na nd::assign_na;

void nd::old_assign_na(const ndt::type &option_tp, const char *arrmeta, char *data)
{
  const ndt::type &value_tp = option_tp.extended<ndt::option_type>()->get_value_type();
  if (value_tp.is_builtin()) {
    switch (value_tp.get_id()) {
    // Just use the known value assignments for these builtins
    case bool_id:
      *data = 2;
      return;
    case int8_id:
      *reinterpret_cast<int8_t *>(data) = DYND_INT8_NA;
      return;
    case int16_id:
      *reinterpret_cast<int16_t *>(data) = DYND_INT16_NA;
      return;
    case int32_id:
      *reinterpret_cast<int32_t *>(data) = DYND_INT32_NA;
      return;
    case int64_id:
      *reinterpret_cast<int64_t *>(data) = DYND_INT64_NA;
      return;
    case int128_id:
      *reinterpret_cast<int128 *>(data) = DYND_INT128_NA;
      return;
    case float32_id:
      *reinterpret_cast<uint32_t *>(data) = DYND_FLOAT32_NA_AS_UINT;
      return;
    case float64_id:
      *reinterpret_cast<uint64_t *>(data) = DYND_FLOAT64_NA_AS_UINT;
      return;
    case complex_float32_id:
      reinterpret_cast<uint32_t *>(data)[0] = DYND_FLOAT32_NA_AS_UINT;
      reinterpret_cast<uint32_t *>(data)[1] = DYND_FLOAT32_NA_AS_UINT;
      return;
    case complex_float64_id:
      reinterpret_cast<uint64_t *>(data)[0] = DYND_FLOAT64_NA_AS_UINT;
      reinterpret_cast<uint64_t *>(data)[1] = DYND_FLOAT64_NA_AS_UINT;
      return;
    default:
      break;
    }
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

DYND_DEFAULT_DECLFUNC_GET(nd::is_na)

DYND_API struct nd::is_na nd::is_na;

bool nd::old_is_avail(const ndt::type &option_tp, const char *arrmeta, const char *data)
{
  const ndt::type value_tp = option_tp.extended<ndt::option_type>()->get_value_type();
  if (value_tp.is_builtin()) {
    switch (value_tp.get_id()) {
    // Just use the known value assignments for these builtins
    case bool_id:
      return *reinterpret_cast<const unsigned char *>(data) <= 1;
    case int8_id:
      return *reinterpret_cast<const int8_t *>(data) != DYND_INT8_NA;
    case int16_id:
      return *reinterpret_cast<const int16_t *>(data) != DYND_INT16_NA;
    case int32_id:
      return *reinterpret_cast<const int32_t *>(data) != DYND_INT32_NA;
    case uint32_id:
      return *reinterpret_cast<const uint32_t *>(data) != DYND_UINT32_NA;
    case int64_id:
      return *reinterpret_cast<const int64_t *>(data) != DYND_INT64_NA;
    case int128_id:
      return *reinterpret_cast<const int128 *>(data) != DYND_INT128_NA;
    case float32_id:
      return !isnan(*reinterpret_cast<const float *>(data));
    case float64_id:
      return !isnan(*reinterpret_cast<const double *>(data));
    case complex_float32_id:
      return reinterpret_cast<const uint32_t *>(data)[0] != DYND_FLOAT32_NA_AS_UINT ||
             reinterpret_cast<const uint32_t *>(data)[1] != DYND_FLOAT32_NA_AS_UINT;
    case complex_float64_id:
      return reinterpret_cast<const uint64_t *>(data)[0] != DYND_FLOAT64_NA_AS_UINT ||
             reinterpret_cast<const uint64_t *>(data)[1] != DYND_FLOAT64_NA_AS_UINT;
    default:
      return false;
    }
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
