//
// Copyright (C) 2011-14 Mark Wiebe, DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#ifndef _DYND__TYPE_TYPE_HPP_
#define _DYND__TYPE_TYPE_HPP_

#include <dynd/type.hpp>
#include <dynd/typed_data_assign.hpp>
#include <dynd/types/static_type_instances.hpp>

namespace dynd {

struct type_type_data {
    const base_type *tp;
};

/**
 * A dynd type whose nd::array instances themselves contain
 * dynd types.
 */
class type_type : public base_type {
public:
    type_type();

    virtual ~type_type();

    void print_data(std::ostream& o, const char *arrmeta, const char *data) const;

    void print_type(std::ostream& o) const;

    bool operator==(const base_type& rhs) const;

    void arrmeta_default_construct(char *arrmeta, intptr_t ndim,
                                   const intptr_t *shape,
                                   bool blockref_alloc) const;
    void arrmeta_copy_construct(char *dst_arrmeta, const char *src_arrmeta, memory_block_data *embedded_reference) const;
    void arrmeta_reset_buffers(char *arrmeta) const;
    void arrmeta_finalize_buffers(char *arrmeta) const;
    void arrmeta_destruct(char *arrmeta) const;
    void arrmeta_debug_print(const char *DYND_UNUSED(arrmeta),
                              std::ostream &DYND_UNUSED(o),
                              const std::string &DYND_UNUSED(indent)) const
    {
    }

    void data_destruct(const char *arrmeta, char *data) const;
    void data_destruct_strided(const char *arrmeta, char *data,
                    intptr_t stride, size_t count) const;

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
  /** Returns type "type" */
  inline const ndt::type &make_type()
  {
    return *reinterpret_cast<const ndt::type *>(&types::type_tp);
  }
  /** Returns type "strided * type" */
  inline const ndt::type &make_strided_of_type()
  {
    return *reinterpret_cast<const ndt::type *>(&types::strided_of_type_tp);
  }
} // namespace ndt

} // namespace dynd

#endif // _DYND__TYPE_TYPE_HPP_
