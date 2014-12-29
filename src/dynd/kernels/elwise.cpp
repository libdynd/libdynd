#include <dynd/kernels/elwise.hpp>
#include <dynd/func/elwise.hpp>

size_t dynd::kernels::make_lifted_expr_ckernel(
    const arrfunc_type_data *child, const arrfunc_type *child_tp, void *ckb,
    intptr_t ckb_offset, const ndt::type &dst_tp, const char *dst_arrmeta,
    const ndt::type *src_tp, const char *const *src_arrmeta,
    dynd::kernel_request_t kernreq, const eval::eval_context *ectx,
    const nd::array &kwds)
{
  return nd::elwise.instantiate(child, child_tp, ckb, ckb_offset, dst_tp,
                                dst_arrmeta, src_tp, src_arrmeta, kernreq, ectx,
                                kwds);
}