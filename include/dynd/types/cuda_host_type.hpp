//
// Copyright (C) 2011-15 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#pragma once

#include <dynd/types/base_memory_type.hpp>
#include <dynd/types/cuda_device_type.hpp>

namespace dynd {

#ifdef DYND_CUDA

class DYNDT_API cuda_host_type : public base_memory_type {
  unsigned int m_cuda_host_flags;

public:
  cuda_host_type(const ndt::type &element_tp, unsigned int cuda_host_flags = cudaHostAllocDefault);

  unsigned int get_cuda_host_flags() const { return m_cuda_host_flags; }

  void print_type(std::ostream &o) const;

  bool operator==(const base_type &rhs) const;

  ndt::type with_replaced_storage_type(const ndt::type &element_tp) const;

  void data_alloc(char **data, size_t size) const;
  void data_zeroinit(char *data, size_t size) const;
  void data_free(char *data) const;
};

namespace ndt {
  inline ndt::type make_cuda_host(const ndt::type &element_tp, unsigned int cuda_host_flags = cudaHostAllocDefault)
  {
    return ndt::type(new cuda_host_type(element_tp, cuda_host_flags), false);
  }
} // namespace ndt

#endif

} // namespace dynd
