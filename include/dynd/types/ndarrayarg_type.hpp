//
// Copyright (C) 2011-14 Mark Wiebe, DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

/**
 * The ndarrayarg type contains nd::array references
 * that are borrowed, the reference is owned by someone
 * else. This type is here to support dynamic function
 * call parameter passing without introducing a general
 * nd::array type which would necessitate the addition
 * of cycle collection.
 */

#ifndef _DYND__NDARRAYARG_TYPE_HPP_
#define _DYND__NDARRAYARG_TYPE_HPP_

#include <dynd/array.hpp>

namespace dynd {

class ndarrayarg_type : public base_type {
public:
  ndarrayarg_type()
      : base_type(ndarrayarg_type_id, dynamic_kind, sizeof(memory_block_data *),
                  sizeof(memory_block_data *), type_flag_zeroinit, 0, 0, 0)
    {
    }

    virtual ~ndarrayarg_type() {}

    void print_data(std::ostream& o, const char *arrmeta, const char *data) const;

    void print_type(std::ostream& o) const;

    bool is_lossless_assignment(const ndt::type& dst_tp, const ndt::type& src_tp) const;

    bool operator==(const base_type& rhs) const;

    void arrmeta_default_construct(char *DYND_UNUSED(arrmeta),
                                    intptr_t DYND_UNUSED(ndim),
                                    const intptr_t *DYND_UNUSED(shape)) const
    {
    }
    void arrmeta_copy_construct(
        char *DYND_UNUSED(dst_arrmeta), const char *DYND_UNUSED(src_arrmeta),
        memory_block_data *DYND_UNUSED(embedded_reference)) const
    {
    }
    void arrmeta_destruct(char *DYND_UNUSED(arrmeta)) const {}
    void arrmeta_debug_print(const char *DYND_UNUSED(arrmeta),
                              std::ostream &DYND_UNUSED(o),
                              const std::string &DYND_UNUSED(indent)) const
    {
    }

    size_t make_assignment_kernel(ckernel_builder *ckb, intptr_t ckb_offset,
                                  const ndt::type &dst_tp,
                                  const char *dst_arrmeta,
                                  const ndt::type &src_tp,
                                  const char *src_arrmeta,
                                  kernel_request_t kernreq,
                                  const eval::eval_context *ectx) const;

    size_t make_comparison_kernel(ckernel_builder *ckb, intptr_t ckb_offset,
                                  const ndt::type &src0_dt,
                                  const char *src0_arrmeta,
                                  const ndt::type &src1_dt,
                                  const char *src1_arrmeta,
                                  comparison_type_t comptype,
                                  const eval::eval_context *ectx) const;
};

namespace ndt {
    ndt::type make_ndarrayarg();
} // namespace ndt

} // namespace dynd

#endif // _DYND__NDARRAYARG_TYPE_HPP_
 
