//
// Copyright (C) 2011-14 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <dynd/types/cuda_host_type.hpp>
#include <dynd/kernels/assignment_kernels.hpp>

using namespace std;
using namespace dynd;

#ifdef DYND_CUDA

cuda_host_type::cuda_host_type(const ndt::type &element_tp,
                               unsigned int cuda_host_flags)
    : base_memory_type(cuda_host_type_id, element_tp,
                       element_tp.get_data_size(),
                       get_cuda_device_data_alignment(element_tp.get_dtype()),
                       0, element_tp.get_flags()),
      m_cuda_host_flags(cuda_host_flags)
{
}

cuda_host_type::~cuda_host_type() {}

void cuda_host_type::print_type(std::ostream &o) const
{
  o << "cuda_host[" << m_element_tp << "]";
}

bool cuda_host_type::operator==(const base_type &rhs) const
{
  if (this == &rhs) {
    return true;
  } else if (rhs.get_type_id() != cuda_host_type_id) {
    return false;
  } else {
    const cuda_host_type *tp = static_cast<const cuda_host_type *>(&rhs);
    return m_element_tp == tp->m_element_tp &&
           m_cuda_host_flags == tp->get_cuda_host_flags();
  }
}

ndt::type
cuda_host_type::with_replaced_storage_type(const ndt::type &element_tp) const
{
  return make_cuda_host(element_tp, m_cuda_host_flags);
}

void cuda_host_type::data_alloc(char **data, size_t size) const
{
  throw_if_not_cuda_success(cudaHostAlloc(data, size, m_cuda_host_flags));
}

void cuda_host_type::data_zeroinit(char *data, size_t size) const
{
  memset(data, 0, size);
}

void cuda_host_type::data_free(char *data) const
{
  throw_if_not_cuda_success(cudaFreeHost(data));
}

intptr_t cuda_host_type::make_assignment_kernel(
    const arrfunc_type_data *self, const arrfunc_type *af_tp, void *ckb,
    intptr_t ckb_offset, const ndt::type &dst_tp, const char *dst_arrmeta,
    const ndt::type &src_tp, const char *src_arrmeta, kernel_request_t kernreq,
    const eval::eval_context *ectx, const nd::array &kwds) const
{
  //  std::cout << "(cuda_host_type::make_assignment_kernel) dst_tp = " <<
  //  dst_tp << std::endl;
  //  std::cout << "(cuda_host_type::make_assignment_kernel) src_tp = " <<
  //  src_tp << std::endl;

  if (this == dst_tp.extended()) {
    return ::make_assignment_kernel(self, af_tp, ckb, ckb_offset,
                                    dst_tp.without_memory_type(), dst_arrmeta,
                                    src_tp, src_arrmeta, kernreq, ectx, kwds);
  } else if (this == src_tp.extended()) {
    return ::make_assignment_kernel(self, af_tp, ckb, ckb_offset, dst_tp,
                                    dst_arrmeta, src_tp.without_memory_type(),
                                    src_arrmeta, kernreq, ectx, kwds);
  } else {
    stringstream ss;
    ss << "Cannot assign from " << src_tp << " to " << dst_tp;
    throw dynd::type_error(ss.str());
  }
}

#endif // DYND_CUDA
