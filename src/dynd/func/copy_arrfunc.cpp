//
// Copyright (C) 2011-14 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <dynd/func/copy_arrfunc.hpp>
#include <dynd/kernels/assignment_kernels.hpp>

using namespace std;
using namespace dynd;

static intptr_t
instantiate_copy(const arrfunc_old_type_data *DYND_UNUSED(af_self),
                 dynd::ckernel_builder *ckb, intptr_t ckb_offset,
                 const ndt::type &dst_tp, const char *dst_arrmeta,
                 const ndt::type *src_tp, const char *const *src_arrmeta,
                 kernel_request_t kernreq, const eval::eval_context *ectx,
                 const nd::array &DYND_UNUSED(args), const nd::array &DYND_UNUSED(kwds))
{
  if (dst_tp.is_builtin()) {
    if (src_tp[0].is_builtin()) {
      if (dst_tp.extended() == src_tp[0].extended()) {
        return make_pod_typed_data_assignment_kernel(
            ckb, ckb_offset,
            detail::builtin_data_sizes[dst_tp.unchecked_get_builtin_type_id()],
            detail::builtin_data_alignments
                [dst_tp.unchecked_get_builtin_type_id()],
            kernreq);
      } else {
        return make_builtin_type_assignment_kernel(
            ckb, ckb_offset, dst_tp.get_type_id(), src_tp[0].get_type_id(),
            kernreq, ectx->errmode);
      }
    } else {
      return src_tp[0].extended()->make_assignment_kernel(
          ckb, ckb_offset, dst_tp, dst_arrmeta, src_tp[0], src_arrmeta[0],
          kernreq, ectx);
    }
  } else {
    return dst_tp.extended()->make_assignment_kernel(
        ckb, ckb_offset, dst_tp, dst_arrmeta, src_tp[0], src_arrmeta[0],
        kernreq, ectx);
  }
}

static int resolve_dst_copy_type(const arrfunc_old_type_data *DYND_UNUSED(self),
                                 intptr_t nsrc, const ndt::type *src_tp,
                                 int throw_on_error, ndt::type &out_dst_tp,
                                 const nd::array &DYND_UNUSED(args), const nd::array &DYND_UNUSED(kwds))
{
  if (nsrc != 1) {
    if (throw_on_error) {
      stringstream ss;
      ss << "arrfunc 'copy' expected 1 argument, got " << nsrc;
      throw std::invalid_argument(ss.str());
    } else {
      return 0;
    }
  }
  out_dst_tp = src_tp[0].get_canonical_type();
  return 1;
}

static void make_copy_arrfunc(arrfunc_old_type_data *out_af)
{
  out_af->free_func = NULL;
  out_af->func_proto = ndt::type("(A... * S) -> B... * T");
  out_af->resolve_dst_type = &resolve_dst_copy_type;
  out_af->instantiate = &instantiate_copy;
}

static nd::arrfunc make_copy_arrfunc_instance()
{
  nd::array af = nd::empty(ndt::make_arrfunc());
  make_copy_arrfunc(
      reinterpret_cast<arrfunc_old_type_data *>(af.get_readwrite_originptr()));
  af.flag_as_immutable();
  return af;
}

const nd::arrfunc& dynd::make_copy_arrfunc()
{
  static nd::arrfunc af = make_copy_arrfunc_instance();
  return af;
}

const nd::arrfunc& dynd::make_broadcast_copy_arrfunc()
{
  throw runtime_error("TODO: distinguish copy and broadcast_copy");
}
