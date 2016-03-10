//
// Copyright (C) 2011-15 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <dynd/types/cuda_device_type.hpp>

using namespace std;
using namespace dynd;

#ifdef DYND_CUDA

cuda_device_type::cuda_device_type(const ndt::type &element_tp)
    : base_memory_type(cuda_device_id, element_tp, element_tp.get_data_size(),
                       get_cuda_device_data_alignment(element_tp), 0,
                       element_tp.get_flags() | type_flag_not_host_readable)
{
}

cuda_device_type::~cuda_device_type() {}

void cuda_device_type::print_data(std::ostream &o, const char *arrmeta, const char *data) const
{
  nd::array a = nd::empty(m_element_tp);

  if (m_element_tp.is_builtin()) {
    print_builtin_scalar(m_element_tp.get_id(), o, a.get_readonly_originptr());
  }
  else {
    m_element_tp.extended()->print_data(o, a.get_arrmeta(), a.get_readonly_originptr());
  }
}

void cuda_device_type::print_type(ostream &o) const { o << "cuda_device[" << m_element_tp << "]"; }

bool cuda_device_type::operator==(const base_type &rhs) const
{
  if (this == &rhs) {
    return true;
  }
  else if (rhs.get_id() != cuda_device_id) {
    return false;
  }
  else {
    const cuda_device_type *tp = static_cast<const cuda_device_type *>(&rhs);
    return m_element_tp == tp->m_element_tp;
  }
}

ndt::type cuda_device_type::with_replaced_storage_type(const ndt::type &element_tp) const
{
  return make_cuda_device(element_tp);
}

void cuda_device_type::data_alloc(char **data, size_t size) const { cuda_throw_if_not_success(cudaMalloc(data, size)); }

void cuda_device_type::data_zeroinit(char *data, size_t size) const
{
  cuda_throw_if_not_success(cudaMemset(data, 0, size));
}

void cuda_device_type::data_free(char *data) const { cuda_throw_if_not_success(cudaFree(data)); }

size_t dynd::get_cuda_device_data_alignment(const ndt::type &tp)
{
  if (tp.is_symbolic()) {
    return 0;
  }

  const ndt::type &dtp = tp.without_memory_type().get_dtype();
  if (dtp.is_builtin()) {
    return dtp.get_data_size();
  }
  else {
    // TODO: Return the data size of the largest built-in component
    return 0;
  }
}

#endif // DYND_CUDA
