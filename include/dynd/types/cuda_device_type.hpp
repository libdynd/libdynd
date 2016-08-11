//
// Copyright (C) 2011-16 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#pragma once

#include <dynd/types/base_memory_type.hpp>

namespace dynd {
namespace ndt {

  class DYNDT_API cuda_device_type : public base_memory_type {
  public:
    cuda_device_type(type_id_t id, const ndt::type &element_tp);

    void print_data(std::ostream &o, const char *arrmeta, const char *data) const;

    void print_type(std::ostream &o) const;

    bool operator==(const base_type &rhs) const;

    ndt::type with_replaced_storage_type(const ndt::type &element_tp) const;

    void data_alloc(char **data, size_t size) const;
    void data_zeroinit(char *data, size_t size) const;
    void data_free(char *data) const;

    static ndt::type parse_type_args(type_id_t id, const char *&begin, const char *end,
                                     std::map<std::string, ndt::type> &symtable);
  };

  DYNDT_API size_t get_cuda_device_data_alignment(const ndt::type &tp);

  template <>
  struct id_of<cuda_device_type> : std::integral_constant<type_id_t, cuda_device_id> {};
} // namespace ndt
} // namespace dynd
