//
// Copyright (C) 2011-16 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <dynd/buffer.hpp>
#include <dynd/parse_util.hpp>
#include <dynd/types/cuda_host_type.hpp>
#include <dynd/types/datashape_parser.hpp>

using namespace std;
using namespace dynd;

#if defined(DYND_CUDA)
#define CUDA_HOST_ALLOC_DEFAULT (cudaHostAllocDefault)
#else
#define CUDA_HOST_ALLOC_DEFAULT 0
#endif

ndt::cuda_host_type::cuda_host_type(type_id_t id, const ndt::type &element_tp)
    : cuda_host_type(id, element_tp, CUDA_HOST_ALLOC_DEFAULT) {}

ndt::cuda_host_type::cuda_host_type(type_id_t id, const ndt::type &element_tp, unsigned int cuda_host_flags)
    : base_memory_type(id, element_tp, element_tp.get_data_size(), get_cuda_device_data_alignment(element_tp), 0,
                       element_tp.get_flags()),
      m_cuda_host_flags(cuda_host_flags) {}

void ndt::cuda_host_type::print_type(std::ostream &o) const { o << "cuda_host[" << m_element_tp << "]"; }

bool ndt::cuda_host_type::operator==(const base_type &rhs) const {
  if (this == &rhs) {
    return true;
  } else if (rhs.get_id() != cuda_host_id) {
    return false;
  } else {
    const cuda_host_type *tp = static_cast<const cuda_host_type *>(&rhs);
    return m_element_tp == tp->m_element_tp && m_cuda_host_flags == tp->get_cuda_host_flags();
  }
}

ndt::type ndt::cuda_host_type::with_replaced_storage_type(const ndt::type &element_tp) const {
  return ndt::make_type<cuda_host_type>(element_tp, m_cuda_host_flags);
}

void ndt::cuda_host_type::data_alloc(char **DYND_UNUSED(data), size_t DYND_UNUSED(size)) const {
#if defined(DYND_CUDA)
  cuda_throw_if_not_success(cudaHostAlloc(data, size, m_cuda_host_flags));
#else
  throw runtime_error("CUDA memory allocation is not available");
#endif
}

void ndt::cuda_host_type::data_zeroinit(char *data, size_t size) const { memset(data, 0, size); }

void ndt::cuda_host_type::data_free(char *DYND_UNUSED(data)) const {
#if defined(DYND_CUDA)
  cuda_throw_if_not_success(cudaFreeHost(data));
#else
  throw runtime_error("CUDA memory allocation is not available");
#endif
}

// cuda_host_type : cuda_host[storage_type]
ndt::type ndt::cuda_host_type::parse_type_args(type_id_t DYND_UNUSED(id), const char *&rbegin,
                                               const char *DYND_UNUSED(end),
                                               std::map<std::string, ndt::type> &DYND_UNUSED(symtable)) {
#ifdef DYND_CUDA
  const char *begin = rbegin;
  if (datashape::parse_token(begin, end, '[')) {
    ndt::type tp = datashape::parse(begin, end, symtable);
    if (tp.is_null()) {
      throw datashape::internal_parse_error(begin, "expected a type parameter");
    }
    if (!datashape::parse_token(begin, end, ']')) {
      throw datashape::internal_parse_error(begin, "expected closing ']'");
    }
    rbegin = begin;
    return ndt::make_type<cuda_host_type>(tp);
  }
#else
  throw datashape::internal_parse_error(rbegin, "cuda_device type is not available");
#endif // DYND_CUDA
}