//
// Copyright (C) 2011-15 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

/**
 * The option type represents data which may or may not be there.
 */

#pragma once

#include <dynd/type.hpp>
#include <dynd/func/arrfunc.hpp>

namespace dynd {

#define DYND_BOOL_NA (2)
#define DYND_INT8_NA (std::numeric_limits<int8_t>::min())
#define DYND_INT16_NA (std::numeric_limits<int16_t>::min())
#define DYND_INT32_NA (std::numeric_limits<int32_t>::min())
#define DYND_INT64_NA (std::numeric_limits<int64_t>::min())
#define DYND_INT128_NA (std::numeric_limits<dynd_int128>::min())
#define DYND_FLOAT16_NA_AS_UINT (0x7e0au)
#define DYND_FLOAT32_NA_AS_UINT (0x7f8007a2U)
#define DYND_FLOAT64_NA_AS_UINT (0x7ff00000000007a2ULL)

class option_type : public base_type {
  ndt::type m_value_tp;
  /**
   * An array with type
   *  c{
   *      is_avail: arrfunc,  # (option[T]) -> bool
   *      assign_na: arrfunc, # () -> option[T]
   *  }
   * with functions which can classify whether values
   * are available, and assign the NA (not available)
   * value.
   */
  nd::array m_nafunc;

public:
  option_type(const ndt::type &value_tp);

  virtual ~option_type();

  size_t get_default_data_size() const {
    return m_value_tp.get_default_data_size();
  }

  /** Returns the type that m_nafunc has */
  static const ndt::type &make_nafunc_type();

  const ndt::type &get_value_type() const { return m_value_tp.value_type(); }

  const nd::array &get_nafunc() const { return m_nafunc; }

  /** Assigns NA to one value */
  void assign_na(const char *arrmeta, char *data,
                 const eval::eval_context *ectx) const;

  /** Returns true if the value is available */
  bool is_avail(const char *arrmeta, const char *data,
                const eval::eval_context *ectx) const;

  const arrfunc_type_data *get_is_avail_arrfunc() const
  {
    return reinterpret_cast<const arrfunc_type_data *>(
        m_nafunc.get_readonly_originptr());
  }

  const arrfunc_type *get_is_avail_arrfunc_type() const
  {
    return m_nafunc.get_type()
        .extended<base_tuple_type>()
        ->get_field_type(0)
        .extended<arrfunc_type>();
  }

  const arrfunc_type_data *get_assign_na_arrfunc() const
  {
    return reinterpret_cast<const arrfunc_type_data *>(
               m_nafunc.get_readonly_originptr()) +
           1;
  }

  const arrfunc_type *get_assign_na_arrfunc_type() const
  {
    return m_nafunc.get_type()
        .extended<base_tuple_type>()
        ->get_field_type(1)
        .extended<arrfunc_type>();
  }

  void print_data(std::ostream &o, const char *arrmeta, const char *data) const;

  void print_type(std::ostream &o) const;

  bool is_expression() const;
  bool is_unique_data_owner(const char *arrmeta) const;
  void transform_child_types(type_transform_fn_t transform_fn,
                             intptr_t arrmeta_offset, void *extra,
                             ndt::type &out_transformed_tp,
                             bool &out_was_transformed) const;
  ndt::type get_canonical_type() const;

  void set_from_utf8_string(const char *arrmeta, char *data,
                            const char *utf8_begin, const char *utf8_end,
                            const eval::eval_context *ectx) const;

  ndt::type get_type_at_dimension(char **inout_arrmeta, intptr_t i, intptr_t total_ndim = 0) const;

  bool is_lossless_assignment(const ndt::type &dst_tp,
                              const ndt::type &src_tp) const;

  bool operator==(const base_type &rhs) const;

  void arrmeta_default_construct(char *arrmeta, bool blockref_alloc) const;
  void arrmeta_copy_construct(char *dst_arrmeta, const char *src_arrmeta,
                              memory_block_data *embedded_reference) const;
  void arrmeta_reset_buffers(char *arrmeta) const;
  void arrmeta_finalize_buffers(char *arrmeta) const;
  void arrmeta_destruct(char *arrmeta) const;
  void arrmeta_debug_print(const char *arrmeta, std::ostream &o,
                           const std::string &indent) const;

  intptr_t make_assignment_kernel(
      const arrfunc_type_data *self, const arrfunc_type *af_tp, void *ckb,
      intptr_t ckb_offset, const ndt::type &dst_tp, const char *dst_arrmeta,
      const ndt::type &src_tp, const char *src_arrmeta,
      kernel_request_t kernreq, const eval::eval_context *ectx,
      const nd::array &kwds) const;

  bool matches(const char *arrmeta, const ndt::type &other,
               std::map<nd::string, ndt::type> &tp_vars) const;

  void get_dynamic_type_properties(
      const std::pair<std::string, gfunc::callable> **out_properties,
      size_t *out_count) const;
};

namespace ndt {
  ndt::type make_option(const ndt::type &value_tp);

  template <typename Tnative>
  inline ndt::type make_option()
  {
    return make_option(ndt::make_type<Tnative>());
  }
} // namespace ndt

} // namespace dynd
