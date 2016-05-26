//
// Copyright (C) 2011-16 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#pragma once

#include <dynd/type.hpp>
#include <dynd/types/base_bytes_type.hpp>

namespace dynd {
namespace ndt {

  class DYNDT_API fixed_bytes_type : public base_bytes_type {
  public:
    fixed_bytes_type(type_id_t new_id, intptr_t data_size, intptr_t data_alignment)
        : base_bytes_type(new_id, fixed_bytes_id, data_size, data_alignment, type_flag_none, 0) {
      if (data_alignment > data_size) {
        std::stringstream ss;
        ss << "Cannot make a bytes[" << data_size << ", align=";
        ss << data_alignment << "] type, its alignment is greater than its size";
        throw std::runtime_error(ss.str());
      }
      if (data_alignment != 1 && data_alignment != 2 && data_alignment != 4 && data_alignment != 8 &&
          data_alignment != 16) {
        std::stringstream ss;
        ss << "Cannot make a bytes[" << data_size << ", align=";
        ss << data_alignment << "] type, its alignment is not a small power of two";
        throw std::runtime_error(ss.str());
      }
      if ((data_size & (data_alignment - 1)) != 0) {
        std::stringstream ss;
        ss << "Cannot make a fixed_bytes[" << data_size << ", align=";
        ss << data_alignment << "] type, its alignment does not divide into its element size";
        throw std::runtime_error(ss.str());
      }
    }

    void print_data(std::ostream &o, const char *arrmeta, const char *data) const;

    void print_type(std::ostream &o) const;

    void get_bytes_range(const char **out_begin, const char **out_end, const char *arrmeta, const char *data) const;

    bool is_lossless_assignment(const type &dst_tp, const type &src_tp) const;

    bool operator==(const base_type &rhs) const;

    void arrmeta_default_construct(char *DYND_UNUSED(arrmeta), bool DYND_UNUSED(blockref_alloc)) const {}
    void arrmeta_copy_construct(char *DYND_UNUSED(dst_arrmeta), const char *DYND_UNUSED(src_arrmeta),
                                const nd::memory_block &DYND_UNUSED(embedded_reference)) const {}
    void arrmeta_destruct(char *DYND_UNUSED(arrmeta)) const {}
    void arrmeta_debug_print(const char *DYND_UNUSED(arrmeta), std::ostream &DYND_UNUSED(o),
                             const std::string &DYND_UNUSED(indent)) const {}
  };

  template <>
  struct id_of<fixed_bytes_type> : std::integral_constant<type_id_t, fixed_bytes_id> {};

} // namespace dynd::ndt
} // namespace dynd
