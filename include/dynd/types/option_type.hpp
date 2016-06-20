//
// Copyright (C) 2011-16 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#pragma once

#include <dynd/type.hpp>
#include <dynd/types/any_kind_type.hpp>

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
class optional {
  ValueType m_value;

public:
  optional(ValueType value) : m_value(value) {}

  optional() : optional(ndt::traits<ValueType>::na()) {}

  void assign_na() { m_value = ndt::traits<ValueType>::na(); }

  bool is_na() const { return m_value == ndt::traits<ValueType>::na(); }

  const ValueType &value() const { return m_value; }

  optional<ValueType> &operator=(ValueType value) {
    m_value = value;
    return *this;
  }
};

template <typename ValueType>
std::ostream &operator<<(std::ostream &o, const optional<ValueType> &rhs) {
  if (rhs.is_na()) {
    return o << "NA";
  }

  return o << rhs.value();
}

template <typename ValueType>
optional<ValueType> opt() {
  return optional<ValueType>();
}

template <typename ValueType>
optional<ValueType> opt(ValueType value) {
  return optional<ValueType>(value);
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
    typedef any_kind_type base;

    option_type(type_id_t id, const type &value_tp = make_type<any_kind_type>())
        : base_type(id, value_tp.get_data_size(), value_tp.get_data_alignment(),
                    value_tp.get_flags() & (type_flags_value_inherited | type_flags_operand_inherited),
                    value_tp.get_arrmeta_size(), value_tp.get_ndim(), 0),
          m_value_tp(value_tp) {
      if (value_tp.get_id() == option_id) {
        std::stringstream ss;
        ss << "Cannot construct an option type out of " << value_tp << ", it is already an option type";
        throw type_error(ss.str());
      }
    }

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
                                const nd::memory_block &embedded_reference) const;
    void arrmeta_reset_buffers(char *arrmeta) const;
    void arrmeta_finalize_buffers(char *arrmeta) const;
    void arrmeta_destruct(char *arrmeta) const;
    void arrmeta_debug_print(const char *arrmeta, std::ostream &o, const std::string &indent) const;

    void data_destruct(const char *arrmeta, char *data) const;
    void data_destruct_strided(const char *arrmeta, char *data, intptr_t stride, size_t count) const;

    bool match(const type &candidate_tp, std::map<std::string, type> &tp_vars) const;

    std::map<std::string, std::pair<ndt::type, const char *>> get_dynamic_type_properties() const;
  };

  template <>
  struct id_of<option_type> : std::integral_constant<type_id_t, option_id> {};

  template <typename ValueType>
  struct traits<optional<ValueType>> {
    static const size_t ndim = 0;
    static const size_t metadata_size = 0;

    static const bool is_same_layout = true;

    static type equivalent() { return make_type<option_type>(make_type<ValueType>()); }

    static void metadata_copy_construct(char *DYND_UNUSED(dst), const char *DYND_UNUSED(src)) {}
  };

} // namespace dynd::ndt
} // namespace dynd
