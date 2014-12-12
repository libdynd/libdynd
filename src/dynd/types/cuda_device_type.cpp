//
// Copyright (C) 2011-14 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <dynd/types/cuda_device_type.hpp>
#include <dynd/kernels/assignment_kernels.hpp>
#include <dynd/kernels/make_lifted_ckernel.hpp>
#include <dynd/func/arrfunc.hpp>

using namespace std;
using namespace dynd;

#ifdef DYND_CUDA

cuda_device_type::cuda_device_type(const ndt::type &element_tp)
    : base_memory_type(cuda_device_type_id, element_tp,
                       element_tp.get_data_size(),
                       get_cuda_device_data_alignment(element_tp.get_dtype()),
                       0, element_tp.get_flags() | type_flag_not_host_readable)
{
}

cuda_device_type::~cuda_device_type() {}

void cuda_device_type::print_data(std::ostream &o, const char *arrmeta,
                                  const char *data) const
{
  nd::array a = nd::empty(m_element_tp);
  typed_data_assign(a.get_type(), a.get_arrmeta(), a.get_readwrite_originptr(),
                    ndt::type(this, true), arrmeta, data);

  if (m_element_tp.is_builtin()) {
    print_builtin_scalar(m_element_tp.get_type_id(), o,
                         a.get_readonly_originptr());
  } else {
    m_element_tp.extended()->print_data(o, a.get_arrmeta(),
                                        a.get_readonly_originptr());
  }
}

void cuda_device_type::print_type(ostream &o) const
{
  o << "cuda_device[" << m_element_tp << "]";
}

bool cuda_device_type::operator==(const base_type &rhs) const
{
  if (this == &rhs) {
    return true;
  } else if (rhs.get_type_id() != cuda_device_type_id) {
    return false;
  } else {
    const cuda_device_type *tp = static_cast<const cuda_device_type *>(&rhs);
    return m_element_tp == tp->m_element_tp;
  }
}

ndt::type
cuda_device_type::with_replaced_storage_type(const ndt::type &element_tp) const
{
  return make_cuda_device(element_tp);
}

void cuda_device_type::data_alloc(char **data, size_t size) const
{
  throw_if_not_cuda_success(cudaMalloc(data, size));
}

void cuda_device_type::data_zeroinit(char *data, size_t size) const
{
  throw_if_not_cuda_success(cudaMemset(data, 0, size));
}

void cuda_device_type::data_free(char *data) const
{
  throw_if_not_cuda_success(cudaFree(data));
}

intptr_t cuda_device_type::make_assignment_kernel(
    const arrfunc_type_data *self, const arrfunc_type *af_tp, void *ckb,
    intptr_t ckb_offset, const ndt::type &dst_tp, const char *dst_arrmeta,
    const ndt::type &src_tp, const char *src_arrmeta, kernel_request_t kernreq,
    const eval::eval_context *ectx, const nd::array &kwds) const
{
  //    std::cout << "(cuda_device_type::make_assignment_kernel) dst_tp = " <<
  //  dst_tp << std::endl;
  // std::cout << "(cuda_device_type::make_assignment_kernel) src_tp = " <<
  // src_tp << std::endl;

  /*
    bool dst_cuda_device, src_cuda_device;
    if (this == dst_tp.extended()) {
      dst_cuda_device = true;
      src_cuda_device = src_tp.get_type_id() == cuda_device_type_id;
    } else if (this == src_tp.extended()) {
      dst_cuda_device = dst_tp.get_type_id() == cuda_device_type_id;
      src_cuda_device = true;
    } else {

    }
  */

  if (dst_tp.get_ndim() >= src_tp.get_ndim() && dst_tp.get_ndim() > 0) {
    arrfunc_type_data child = arrfunc_type_data(
        &make_cuda_builtin_type_assignment_kernel, NULL, NULL, NULL);
    ndt::type child_tp = ndt::make_funcproto(src_tp.storage_type().get_dtype(),
                                             dst_tp.storage_type().get_dtype());
    intptr_t src_ndim = src_tp.get_ndim();
    return make_lifted_expr_ckernel(&child, child_tp.extended<arrfunc_type>(),
                                    ckb, ckb_offset, dst_tp.get_ndim(), dst_tp,
                                    dst_arrmeta, &src_ndim, &src_tp,
                                    &src_arrmeta, kernreq, ectx);
  }

  return make_cuda_builtin_type_assignment_kernel(
      self, af_tp, ckb, ckb_offset, dst_tp, dst_arrmeta, &src_tp, &src_arrmeta,
      kernreq, ectx, kwds);
}

#endif // DYND_CUDA