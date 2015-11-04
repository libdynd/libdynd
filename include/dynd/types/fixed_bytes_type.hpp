//
// Copyright (C) 2011-15 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#pragma once

#include <dynd/type.hpp>
#include <dynd/typed_data_assign.hpp>
#include <dynd/types/base_bytes_type.hpp>
#include <dynd/types/view_type.hpp>

namespace dynd {
namespace ndt {

  class DYND_API fixed_bytes_type : public base_bytes_type {
  public:
    fixed_bytes_type(intptr_t element_size, intptr_t alignment);

    virtual ~fixed_bytes_type();

    void print_data(std::ostream &o, const char *arrmeta,
                    const char *data) const;

    void print_type(std::ostream &o) const;

    void get_bytes_range(const char **out_begin, const char **out_end,
                         const char *arrmeta, const char *data) const;

    bool is_lossless_assignment(const type &dst_tp, const type &src_tp) const;

    bool operator==(const base_type &rhs) const;

    void arrmeta_default_construct(char *DYND_UNUSED(arrmeta),
                                   bool DYND_UNUSED(blockref_alloc)) const
    {
    }
    void arrmeta_copy_construct(
        char *DYND_UNUSED(dst_arrmeta), const char *DYND_UNUSED(src_arrmeta),
        const intrusive_ptr<memory_block_data> &DYND_UNUSED(embedded_reference)) const
    {
    }
    void arrmeta_destruct(char *DYND_UNUSED(arrmeta)) const {}
    void arrmeta_debug_print(const char *DYND_UNUSED(arrmeta),
                             std::ostream &DYND_UNUSED(o),
                             const std::string &DYND_UNUSED(indent)) const
    {
    }

    intptr_t make_assignment_kernel(void *ckb, intptr_t ckb_offset,
                                    const type &dst_tp, const char *dst_arrmeta,
                                    const type &src_tp, const char *src_arrmeta,
                                    kernel_request_t kernreq,
                                    const eval::eval_context *ectx) const;
  };

  /**
   * Creates a bytes<size, alignment> type, for representing
   * raw, uninterpreted bytes.
   */
  inline type make_fixed_bytes(intptr_t element_size, intptr_t alignment)
  {
    return type(new fixed_bytes_type(element_size, alignment), false);
  }

} // namespace dynd::ndt
} // namespace dynd
