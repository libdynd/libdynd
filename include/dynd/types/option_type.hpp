//
// Copyright (C) 2011-15 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

/**
 * The option type represents data which may or may not be there.
 */

#pragma once

#include <dynd/type.hpp>
#include <dynd/func/callable.hpp>
#include <dynd/func/option.hpp>

namespace dynd {

#define DYND_BOOL_NA (2)
#define DYND_INT8_NA (std::numeric_limits<int8_t>::min())
#define DYND_INT16_NA (std::numeric_limits<int16_t>::min())
#define DYND_INT32_NA (std::numeric_limits<int32_t>::min())
#define DYND_INT64_NA (std::numeric_limits<int64_t>::min())
#define DYND_INT128_NA (std::numeric_limits<int128>::min())
#define DYND_FLOAT16_NA_AS_UINT (0x7e0au)
#define DYND_FLOAT32_NA_AS_UINT (0x7f8007a2U)
#define DYND_FLOAT64_NA_AS_UINT (0x7ff00000000007a2ULL)

namespace ndt {

  class DYND_API option_type : public base_type {
    type m_value_tp;

  public:
    option_type(const type &value_tp);

    virtual ~option_type();

    size_t get_default_data_size() const
    {
      return m_value_tp.get_default_data_size();
    }

    void get_vars(std::unordered_set<std::string> &vars) const;

    const type &get_value_type() const
    {
      return m_value_tp.value_type();
    }

    /** Assigns NA to one value */
    void assign_na(const char *arrmeta, char *data, const eval::eval_context *ectx) const;

    /** Returns true if the value is available */
    bool is_avail(const char *arrmeta, const char *data, const eval::eval_context *ectx) const;

    nd::callable &get_is_avail() const
    {
      return nd::is_avail::get_child(m_value_tp);
    }

    nd::callable &get_assign_na() const
    {
      return nd::assign_na_decl::get_child(m_value_tp);
    }

    void print_data(std::ostream &o, const char *arrmeta, const char *data) const;

    void print_type(std::ostream &o) const;

    bool is_expression() const;
    bool is_unique_data_owner(const char *arrmeta) const;
    void transform_child_types(type_transform_fn_t transform_fn, intptr_t arrmeta_offset, void *extra,
                               type &out_transformed_tp, bool &out_was_transformed) const;
    type get_canonical_type() const;

    void set_from_utf8_string(const char *arrmeta, char *data, const char *utf8_begin, const char *utf8_end,
                              const eval::eval_context *ectx) const;

    type get_type_at_dimension(char **inout_arrmeta, intptr_t i, intptr_t total_ndim = 0) const;

    bool is_lossless_assignment(const type &dst_tp, const type &src_tp) const;

    bool operator==(const base_type &rhs) const;

    void arrmeta_default_construct(char *arrmeta, bool blockref_alloc) const;
    void arrmeta_copy_construct(char *dst_arrmeta, const char *src_arrmeta,
                                const intrusive_ptr<memory_block_data> &embedded_reference) const;
    void arrmeta_reset_buffers(char *arrmeta) const;
    void arrmeta_finalize_buffers(char *arrmeta) const;
    void arrmeta_destruct(char *arrmeta) const;
    void arrmeta_debug_print(const char *arrmeta, std::ostream &o, const std::string &indent) const;

    void data_destruct(const char *arrmeta, char *data) const;
    void data_destruct_strided(const char *arrmeta, char *data, intptr_t stride, size_t count) const;

    intptr_t make_assignment_kernel(void *ckb, intptr_t ckb_offset, const type &dst_tp, const char *dst_arrmeta,
                                    const type &src_tp, const char *src_arrmeta, kernel_request_t kernreq,
                                    const eval::eval_context *ectx) const;

    bool match(const char *arrmeta, const type &candidate_tp, const char *candidate_arrmeta,
               std::map<std::string, type> &tp_vars) const;

    void get_dynamic_type_properties(const std::pair<std::string, nd::callable> **out_properties,
                                     size_t *out_count) const;

    static type make(const type &value_tp);
  };

} // namespace dynd::ndt
} // namespace dynd
