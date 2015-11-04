//
// Copyright (C) 2011-15 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#pragma once

#include <dynd/types/base_type.hpp>
#include <dynd/type.hpp>

namespace dynd {
namespace ndt {

  class DYND_API c_contiguous_type : public base_type {
    type m_child_tp;

  public:
    c_contiguous_type(const type &child_tp);

    const type &get_child_type() const
    {
      return m_child_tp;
    }

    void print_data(std::ostream &o, const char *arrmeta, const char *data) const;

    void print_type(std::ostream &o) const;

    void get_shape(intptr_t ndim, intptr_t i, intptr_t *out_shape, const char *arrmeta, const char *data) const;

    type apply_linear_index(intptr_t nindices, const irange *indices, size_t current_i, const type &root_tp,
                            bool leading_dimension) const;

    intptr_t apply_linear_index(intptr_t nindices, const irange *indices, const char *arrmeta, const type &result_type,
                                char *out_arrmeta, const intrusive_ptr<memory_block_data> &embedded_reference, size_t current_i,
                                const type &root_tp, bool leading_dimension, char **inout_data,
                                intrusive_ptr<memory_block_data> &inout_dataref) const;

    type at_single(intptr_t i0, const char **inout_arrmeta, const char **inout_data) const;

    type get_type_at_dimension(char **inout_arrmeta, intptr_t i, intptr_t total_ndim = 0) const;

    bool is_c_contiguous(const char *arrmeta) const;

    bool operator==(const base_type &rhs) const;

    virtual void arrmeta_default_construct(char *arrmeta, bool blockref_alloc) const;
    virtual void arrmeta_copy_construct(char *dst_arrmeta, const char *src_arrmeta,
                                        const intrusive_ptr<memory_block_data> &embedded_reference) const;
    virtual void arrmeta_destruct(char *arrmeta) const;

    virtual bool match(const char *arrmeta, const type &candidate_tp, const char *candidate_arrmeta,
                       std::map<std::string, type> &tp_vars) const;

    static type make(const type &child_tp)
    {
      return type(new c_contiguous_type(child_tp), false);
    }
  };

} // namespace dynd::ndt
} // namespace dynd
