//
// Copyright (C) 2011-15 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#pragma once

#include <dynd/array.hpp>

namespace dynd {
namespace ndt {

  /**
   * The ndarrayarg type contains nd::array references
   * that are borrowed, the reference is owned by someone
   * else. This type is here to support dynamic function
   * call parameter passing without introducing a general
   * nd::array type which would necessitate the addition
   * of cycle collection.
   */
  class DYND_API ndarrayarg_type : public base_type {
  public:
    ndarrayarg_type()
        : base_type(ndarrayarg_type_id, dynamic_kind,
                    sizeof(memory_block_data *), sizeof(memory_block_data *),
                    type_flag_zeroinit, 0, 0, 0)
    {
    }

    virtual ~ndarrayarg_type() {}

    void print_data(std::ostream &o, const char *arrmeta,
                    const char *data) const;

    void print_type(std::ostream &o) const;

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

    size_t make_comparison_kernel(void *ckb, intptr_t ckb_offset,
                                  const type &src0_dt, const char *src0_arrmeta,
                                  const type &src1_dt, const char *src1_arrmeta,
                                  comparison_type_t comptype,
                                  const eval::eval_context *ectx) const;
  };

  inline const type &make_ndarrayarg()
  {
    static const type ndarrayarg_tp(new ndarrayarg_type(), false);
    return ndarrayarg_tp;
  }

} // namespace dynd::ndt
} // namespace dynd
