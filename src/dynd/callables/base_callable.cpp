//
// Copyright (C) 2011-15 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <dynd/callables/base_callable.hpp>
#include <dynd/callables/call_stack.hpp>

using namespace std;
using namespace dynd;

nd::base_callable::~base_callable() {}

nd::array nd::base_callable::call(ndt::type &dst_tp, intptr_t nsrc, const ndt::type *src_tp,
                                  const char *const *src_arrmeta, char *const *src_data, intptr_t nkwd,
                                  const array *kwds, const std::map<std::string, ndt::type> &tp_vars)
{
  // Allocate, then initialize, the data
  char *data = data_init(dst_tp, nsrc, src_tp, nkwd, kwds, tp_vars);

  // Resolve the destination type
  if (dst_tp.is_symbolic()) {
    resolve_dst_type(data, dst_tp, nsrc, src_tp, nkwd, kwds, tp_vars);
  }

  // Allocate the destination array
  array dst = alloc(&dst_tp);

  // Generate and evaluate the ckernel
  kernel_builder ckb;
  instantiate(data, &ckb, dst_tp, dst.get()->metadata(), nsrc, src_tp, src_arrmeta, kernel_request_single, nkwd, kwds,
              tp_vars);
  kernel_single_t fn = ckb.get()->get_function<kernel_single_t>();
  fn(ckb.get(), dst.data(), src_data);

  return dst;
}

nd::array nd::base_callable::call(ndt::type &dst_tp, intptr_t nsrc, const ndt::type *src_tp,
                                  const char *const *src_arrmeta, const array *src_data, intptr_t nkwd,
                                  const array *kwds, const std::map<std::string, ndt::type> &tp_vars)
{
  if (m_new_style) {
    call_stack s;
    s.push_back(callable(this, true), dst_tp, nsrc, src_tp, kernel_request_call);
    new_resolve(s, nkwd, kwds, tp_vars);

    // Allocate the destination array
    array dst = empty(dst_tp);

    kernel_builder ckb;
    for (auto frame : s) {
      frame.func->new_instantiate(frame.data, &ckb, frame.dst_tp, dst->metadata(), frame.nsrc, frame.src_tp.data(),
                                  src_arrmeta, frame.kernreq, nkwd, kwds);
    }

    kernel_call_t fn = ckb.get()->get_function<kernel_call_t>();
    fn(ckb.get(), &dst, src_data);

    return dst;
  }

  // Allocate, then initialize, the data
  char *data = data_init(dst_tp, nsrc, src_tp, nkwd, kwds, tp_vars);

  // Resolve the destination type
  if (dst_tp.is_symbolic()) {
    resolve_dst_type(data, dst_tp, nsrc, src_tp, nkwd, kwds, tp_vars);
  }

  // Allocate the destination array
  array dst = empty(dst_tp);

  // Generate and evaluate the ckernel
  kernel_builder ckb;
  instantiate(data, &ckb, dst_tp, dst->metadata(), nsrc, src_tp, src_arrmeta, kernel_request_call, nkwd, kwds, tp_vars);
  kernel_call_t fn = ckb.get()->get_function<kernel_call_t>();
  fn(ckb.get(), &dst, src_data);

  return dst;
}

void nd::base_callable::call(const ndt::type &dst_tp, const char *dst_arrmeta, char *dst_data, intptr_t nsrc,
                             const ndt::type *src_tp, const char *const *src_arrmeta, char *const *src_data,
                             intptr_t nkwd, const array *kwds, const std::map<std::string, ndt::type> &tp_vars)
{
  char *data = data_init(dst_tp, nsrc, src_tp, nkwd, kwds, tp_vars);

  // Generate and evaluate the ckernel
  kernel_builder ckb;
  instantiate(data, &ckb, dst_tp, dst_arrmeta, nsrc, src_tp, src_arrmeta, kernel_request_single, nkwd, kwds, tp_vars);
  kernel_single_t fn = ckb.get()->get_function<kernel_single_t>();
  fn(ckb.get(), dst_data, src_data);
}

void nd::base_callable::call(const ndt::type &dst_tp, const char *dst_arrmeta, array *dst, intptr_t nsrc,
                             const ndt::type *src_tp, const char *const *src_arrmeta, const array *src, intptr_t nkwd,
                             const array *kwds, const std::map<std::string, ndt::type> &tp_vars)
{
  char *data = data_init(dst_tp, nsrc, src_tp, nkwd, kwds, tp_vars);

  // Generate and evaluate the ckernel
  kernel_builder ckb;
  instantiate(data, &ckb, dst_tp, dst_arrmeta, nsrc, src_tp, src_arrmeta, kernel_request_call, nkwd, kwds, tp_vars);
  kernel_call_t fn = ckb.get()->get_function<kernel_call_t>();
  fn(ckb.get(), dst, src);
}
