//
// Copyright (C) 2011-15 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <memory>

#include <dynd/callables/base_callable.hpp>

using namespace std;
using namespace dynd;

nd::array nd::base_callable::operator()(ndt::type &dst_tp, intptr_t nsrc, const ndt::type *src_tp,
                                        const char *const *src_arrmeta, char *const *src_data, intptr_t nkwd,
                                        const array *kwds, const std::map<std::string, ndt::type> &tp_vars)
{
  // Allocate, then initialize, the data
  char *data = data_init(static_data(), dst_tp, nsrc, src_tp, nkwd, kwds, tp_vars);

  // Resolve the destination type
  if (dst_tp.is_symbolic()) {
    if (resolve_dst_type == NULL) {
      throw std::runtime_error("dst_tp is symbolic, but resolve_dst_type is NULL");
    }

    resolve_dst_type(static_data(), data, dst_tp, nsrc, src_tp, nkwd, kwds, tp_vars);
  }

  // Allocate the destination array
  array dst = empty(dst_tp);

  // Generate and evaluate the ckernel
  ckernel_builder<kernel_request_host> ckb;
  instantiate(static_data(), data, &ckb, 0, dst_tp, dst.get()->metadata(), nsrc, src_tp, src_arrmeta,
              kernel_request_single, &eval::default_eval_context, nkwd, kwds, tp_vars);
  expr_single_t fn = ckb.get()->get_function<expr_single_t>();
  fn(ckb.get(), dst.data(), src_data);

  return dst;
}

nd::array nd::base_callable::operator()(ndt::type &dst_tp, intptr_t nsrc, const ndt::type *src_tp,
                                        const char *const *src_arrmeta, array *const *src_data, intptr_t nkwd,
                                        const array *kwds, const std::map<std::string, ndt::type> &tp_vars)
{
  // Allocate, then initialize, the data
  char *data = data_init(static_data(), dst_tp, nsrc, src_tp, nkwd, kwds, tp_vars);

  // Resolve the destination type
  if (dst_tp.is_symbolic()) {
    if (resolve_dst_type == NULL) {
      throw std::runtime_error("dst_tp is symbolic, but resolve_dst_type is NULL");
    }

    resolve_dst_type(static_data(), data, dst_tp, nsrc, src_tp, nkwd, kwds, tp_vars);
  }

  // Allocate the destination array
  array dst = empty(dst_tp);

  // Generate and evaluate the ckernel
  ckernel_builder<kernel_request_host> ckb;
  instantiate(static_data(), data, &ckb, 0, dst_tp, dst.get()->metadata(), nsrc, src_tp, src_arrmeta, kernreq,
              &eval::default_eval_context, nkwd, kwds, tp_vars);
  expr_metadata_single_t fn = ckb.get()->get_function<expr_metadata_single_t>();
  fn(ckb.get(), &dst, src_data);

  return dst;
}

void nd::base_callable::operator()(const ndt::type &dst_tp, const char *dst_arrmeta, char *dst_data, intptr_t nsrc,
                                   const ndt::type *src_tp, const char *const *src_arrmeta, char *const *src_data,
                                   intptr_t nkwd, const array *kwds, const std::map<std::string, ndt::type> &tp_vars)
{
  char *data = data_init(static_data(), dst_tp, nsrc, src_tp, nkwd, kwds, tp_vars);

  // Generate and evaluate the ckernel
  ckernel_builder<kernel_request_host> ckb;
  instantiate(static_data(), data, &ckb, 0, dst_tp, dst_arrmeta, nsrc, src_tp, src_arrmeta, kernel_request_single,
              &eval::default_eval_context, nkwd, kwds, tp_vars);
  expr_single_t fn = ckb.get()->get_function<expr_single_t>();
  fn(ckb.get(), dst_data, src_data);
}

void nd::base_callable::operator()(const ndt::type &DYND_UNUSED(dst_tp), const char *DYND_UNUSED(dst_arrmeta),
                                   char *DYND_UNUSED(dst_data), intptr_t DYND_UNUSED(nsrc),
                                   const ndt::type *DYND_UNUSED(src_tp), const char *const *DYND_UNUSED(src_arrmeta),
                                   array *const *DYND_UNUSED(src_data), intptr_t DYND_UNUSED(nkwd),
                                   const array *DYND_UNUSED(kwds),
                                   const std::map<std::string, ndt::type> &DYND_UNUSED(tp_vars))
{
  throw std::runtime_error("view callables are not fully implemented yet");
}
