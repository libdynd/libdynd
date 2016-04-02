//
// Copyright (C) 2011-16 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <dynd/types/cuda_host_type.hpp>

using namespace std;
using namespace dynd;

#ifdef DYND_CUDA

cuda_host_type::cuda_host_type(const ndt::type &element_tp, unsigned int cuda_host_flags)
    : base_memory_type(cuda_host_id, element_tp, element_tp.get_data_size(), get_cuda_device_data_alignment(element_tp),
                       0, element_tp.get_flags()),
      m_cuda_host_flags(cuda_host_flags)
{
}

cuda_host_type::~cuda_host_type() {}

void cuda_host_type::print_type(std::ostream &o) const { o << "cuda_host[" << m_element_tp << "]"; }

bool cuda_host_type::operator==(const base_type &rhs) const
{
  if (this == &rhs) {
    return true;
  }
  else if (rhs.get_id() != cuda_host_id) {
    return false;
  }
  else {
    const cuda_host_type *tp = static_cast<const cuda_host_type *>(&rhs);
    return m_element_tp == tp->m_element_tp && m_cuda_host_flags == tp->get_cuda_host_flags();
  }
}

ndt::type cuda_host_type::with_replaced_storage_type(const ndt::type &element_tp) const
{
  return make_cuda_host(element_tp, m_cuda_host_flags);
}

void cuda_host_type::data_alloc(char **data, size_t size) const
{
  cuda_throw_if_not_success(cudaHostAlloc(data, size, m_cuda_host_flags));
}

void cuda_host_type::data_zeroinit(char *data, size_t size) const { memset(data, 0, size); }

void cuda_host_type::data_free(char *data) const { cuda_throw_if_not_success(cudaFreeHost(data)); }

#endif // DYND_CUDA
