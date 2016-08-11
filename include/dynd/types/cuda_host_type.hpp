//
// Copyright (C) 2011-16 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#pragma once

#include <dynd/types/base_memory_type.hpp>
#include <dynd/types/cuda_device_type.hpp>

namespace dynd {
namespace ndt {

  class DYNDT_API cuda_host_type : public base_memory_type {
    unsigned int m_cuda_host_flags;

  public:
    cuda_host_type(type_id_t id, const ndt::type &element_tp);
    cuda_host_type(type_id_t id, const ndt::type &element_tp, unsigned int cuda_host_flags);

    unsigned int get_cuda_host_flags() const { return m_cuda_host_flags; }

    void print_type(std::ostream &o) const;

    bool operator==(const base_type &rhs) const;

    ndt::type with_replaced_storage_type(const ndt::type &element_tp) const;

    void data_alloc(char **data, size_t size) const;
    void data_zeroinit(char *data, size_t size) const;
    void data_free(char *data) const;

    static ndt::type parse_type_args(type_id_t id, const char *&begin, const char *end,
                                     std::map<std::string, ndt::type> &symtable);
  };

  template <>
  struct id_of<cuda_host_type> : std::integral_constant<type_id_t, cuda_host_id> {};
} // namespace ndt
} // namespace dynd
