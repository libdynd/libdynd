//
// Copyright (C) 2011-15 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#pragma once

#include <dynd/type.hpp>

#define DYND_BOOL_NA (2)
#define DYND_INT8_NA (std::numeric_limits<int8_t>::min())
#define DYND_INT16_NA (std::numeric_limits<int16_t>::min())
#define DYND_INT32_NA (std::numeric_limits<int32_t>::min())
#define DYND_UINT32_NA (std::numeric_limits<uint32_t>::max())
#define DYND_INT64_NA (std::numeric_limits<int64_t>::min())
#define DYND_INT128_NA (std::numeric_limits<int128>::min())
#define DYND_FLOAT16_NA_AS_UINT (0x7e0au)
#define DYND_FLOAT32_NA_AS_UINT (0x7f8007a2U)
#define DYND_FLOAT64_NA_AS_UINT (0x7ff00000000007a2ULL)

namespace dynd {

template <typename ValueType>
struct option {
  ValueType value;

  option() : value(ndt::traits<ValueType>::na()) {}

  option(ValueType value) : value(value) {}
};

template <typename ValueType>
option<ValueType> opt()
{
  return option<ValueType>();
}

template <typename ValueType>
option<ValueType> opt(ValueType value)
{
  return option<ValueType>(value);
}

DYNDT_API void assign_na_builtin(type_id_t value_id, char *data);
DYNDT_API bool is_avail_builtin(type_id_t value_id, const char *data);

namespace ndt {

  /**
   * The option type represents data which may or may not be there.
   */
  class DYNDT_API option_type : public base_type {
    type m_value_tp;

  public:
    option_type(const type &value_tp);

    size_t get_default_data_size() const { return m_value_tp.get_default_data_size(); }

    void get_vars(std::unordered_set<std::string> &vars) const;

    const type &get_value_type() const { return m_value_tp.value_type(); }

    void print_type(std::ostream &o) const;
    void print_data(std::ostream &o, const char *arrmeta, const char *data) const;

    bool is_expression() const;
    bool is_unique_data_owner(const char *arrmeta) const;
    void transform_child_types(type_transform_fn_t transform_fn, intptr_t arrmeta_offset, void *extra,
                               type &out_transformed_tp, bool &out_was_transformed) const;
    type get_canonical_type() const;

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

    bool match(const type &candidate_tp, std::map<std::string, type> &tp_vars) const;

    std::map<std::string, std::pair<ndt::type, const char *>> get_dynamic_type_properties() const;
  };

  template <typename ValueType>
  struct traits<option<ValueType>> {
    static const bool is_same_layout = true;

    static type equivalent() { return make_type<option_type>(make_type<ValueType>()); }
  };

} // namespace dynd::ndt
} // namespace dynd
