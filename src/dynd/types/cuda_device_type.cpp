//
// Copyright (C) 2011-16 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <dynd/buffer.hpp>
#include <dynd/parse_util.hpp>
#include <dynd/types/cuda_device_type.hpp>
#include <dynd/types/datashape_parser.hpp>

using namespace std;
using namespace dynd;

ndt::cuda_device_type::cuda_device_type(type_id_t id, const ndt::type &element_tp)
    : base_memory_type(id, element_tp, element_tp.get_data_size(), get_cuda_device_data_alignment(element_tp), 0,
                       element_tp.get_flags() | type_flag_not_host_readable) {}

void ndt::cuda_device_type::print_data(std::ostream &o, const char *arrmeta, const char *data) const {
  if (m_element_tp.is_builtin()) {
    print_builtin_scalar(m_element_tp.get_id(), o, data);
  } else {
    m_element_tp.extended()->print_data(o, arrmeta, data);
  }
}

void ndt::cuda_device_type::print_type(ostream &o) const { o << "cuda_device[" << m_element_tp << "]"; }

bool ndt::cuda_device_type::operator==(const base_type &rhs) const {
  if (this == &rhs) {
    return true;
  } else if (rhs.get_id() != cuda_device_id) {
    return false;
  } else {
    const cuda_device_type *tp = static_cast<const cuda_device_type *>(&rhs);
    return m_element_tp == tp->m_element_tp;
  }
}

ndt::type ndt::cuda_device_type::with_replaced_storage_type(const ndt::type &element_tp) const {
  return ndt::make_type<cuda_device_type>(element_tp);
}

void ndt::cuda_device_type::data_alloc(char **DYND_UNUSED(data), size_t DYND_UNUSED(size)) const {
#if defined(DYND_CUDA)
  cuda_throw_if_not_success(cudaMalloc(data, size));
#else
  throw runtime_error("CUDA memory allocation is not available");
#endif
}

void ndt::cuda_device_type::data_zeroinit(char *DYND_UNUSED(data), size_t DYND_UNUSED(size)) const {
#if defined(DYND_CUDA)
  cuda_throw_if_not_success(cudaMemset(data, 0, size));
#else
  throw runtime_error("CUDA memset is not available");
#endif
}

void ndt::cuda_device_type::data_free(char *DYND_UNUSED(data)) const {
#if defined(DYND_CUDA)
  cuda_throw_if_not_success(cudaFree(data));
#else
  throw runtime_error("CUDA memory allocation is not available");
#endif
}

size_t dynd::ndt::get_cuda_device_data_alignment(const ndt::type &tp) {
  if (tp.is_symbolic()) {
    return 0;
  }

  const ndt::type &dtp = tp.without_memory_type().get_dtype();
  if (dtp.is_builtin()) {
    return dtp.get_data_size();
  } else {
    // TODO: Return the data size of the largest built-in component
    return 0;
  }
}

// cuda_device_type : cuda_device[storage_type]
ndt::type ndt::cuda_device_type::parse_type_args(type_id_t DYND_UNUSED(id), const char *&rbegin,
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
    return ndt::make_type<cuda_device_type>(tp);
  } else {
    throw datashape::internal_parse_error(begin, "expected opening '['");
  }
#else
  throw datashape::internal_parse_error(rbegin, "cuda_device type is not available");
#endif // DYND_CUDA
}