//
// Copyright (C) 2011-14 Mark Wiebe, DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#ifndef _DYND__ARRFUNC_TYPE_HPP_
#define _DYND__ARRFUNC_TYPE_HPP_

#include <dynd/type.hpp>
#include <dynd/types/static_type_instances.hpp>

namespace dynd {

/**
 * A dynd type whose nd::array instances contain
 * arrfunc objects.
 */
class arrfunc_type : public base_type {
public:
    arrfunc_type();

    virtual ~arrfunc_type();

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

    void get_dynamic_array_properties(
        const std::pair<std::string, gfunc::callable> **out_properties,
        size_t *out_count) const;
    void get_dynamic_array_functions(
        const std::pair<std::string, gfunc::callable> **out_functions,
        size_t *out_count) const;
};

namespace ndt {
  inline const ndt::type &make_arrfunc()
  {
    return *reinterpret_cast<const ndt::type *>(&types::arrfunc_tp);
  }
} // namespace ndt

} // namespace dynd

#endif // _DYND__ARRFUNC_TYPE_HPP_
