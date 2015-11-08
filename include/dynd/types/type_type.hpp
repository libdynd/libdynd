//
// Copyright (C) 2011-15 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#pragma once

#include <dynd/type.hpp>
#include <dynd/typed_data_assign.hpp>

namespace dynd {

typedef ndt::type type_type_data;

namespace ndt {

  /**
   * A dynd type whose nd::array instances themselves contain
   * dynd types.
   */
  class DYND_API type_type : public base_type {
    type m_pattern_tp;

  public:
    type_type();

    type_type(const type &pattern_tp);

    virtual ~type_type();

    const type &get_pattern_type() const { return m_pattern_tp; }

    void print_data(std::ostream &o, const char *arrmeta,
                    const char *data) const;

    void print_type(std::ostream &o) const;

    bool operator==(const base_type &rhs) const;

    void arrmeta_default_construct(char *arrmeta, bool blockref_alloc) const;
    void arrmeta_copy_construct(char *dst_arrmeta, const char *src_arrmeta,
                                const intrusive_ptr<memory_block_data> &embedded_reference) const;
    void arrmeta_reset_buffers(char *arrmeta) const;
    void arrmeta_finalize_buffers(char *arrmeta) const;
    void arrmeta_destruct(char *arrmeta) const;
    void arrmeta_debug_print(const char *DYND_UNUSED(arrmeta),
                             std::ostream &DYND_UNUSED(o),
                             const std::string &DYND_UNUSED(indent)) const
    {
    }

    void data_destruct(const char *arrmeta, char *data) const;
    void data_destruct_strided(const char *arrmeta, char *data, intptr_t stride,
                               size_t count) const;

    intptr_t make_assignment_kernel(void *ckb, intptr_t ckb_offset,
                                    const type &dst_tp, const char *dst_arrmeta,
                                    const type &src_tp, const char *src_arrmeta,
                                    kernel_request_t kernreq,
                                    const eval::eval_context *ectx) const;
  };

  /** Returns type "type" */
  inline const type &make_type()
  {
    static const type type_tp(new type_type(), false);
    return type_tp;
  }

  inline type make_type(const type &pattern_tp)
  {
    return type(new type_type(pattern_tp), false);
  }

} // namespace dynd::ndt
} // namespace dynd
