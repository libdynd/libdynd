//
// Copyright (C) 2011-14 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#pragma once

#include <vector>
#include <string>

#include <dynd/array.hpp>
#include <dynd/types/fixed_dim_type.hpp>
#include <dynd/types/fixed_dimsym_type.hpp>
#include <dynd/types/string_type.hpp>

namespace dynd {

class arrfunc_type : public base_type {
  ndt::type m_return_type;
  nd::array m_arg_types;
  nd::array m_arg_names;

public:
  arrfunc_type(const ndt::type &ret_type, const nd::array &arg_types, const nd::array &arg_names);

  virtual ~arrfunc_type() {}

  const string_type_data& get_arg_name_raw(intptr_t i) const {
    return unchecked_fixed_dim_get<string_type_data>(m_arg_names, i);
  }

  const nd::array& get_arg_types() const {
    return m_arg_types;
  }

  const ndt::type *get_arg_types_raw() const {
    return reinterpret_cast<const ndt::type *>(m_arg_types.get_readonly_originptr());
  }
  const ndt::type &get_arg_type(intptr_t i) const {
    return get_arg_types_raw()[i];
  }

  intptr_t get_arg_index(const std::string& arg_name) const {
    return get_arg_index(arg_name.data(), arg_name.data() + arg_name.size());
  }
  intptr_t get_arg_index(const char *arg_name_begin, const char *arg_name_end) const;

  const nd::array &get_arg_names() const {
    return m_arg_names;
  }

  intptr_t get_npos() const {
    return get_narg() - get_nkwd();
  }

  intptr_t get_nsrc() const {
    return get_npos();
  }

  intptr_t get_nkwd() const {
    if (m_arg_names.is_null()) {
      return 0;
    }

    return m_arg_names.get_dim_size();
  }

  intptr_t get_narg() const {
    if (m_arg_types.is_null()) {
      return 0;
    }

    return m_arg_types.get_dim_size();
  }

  const ndt::type& get_return_type() const {
    return m_return_type;
  }

  void print_data(std::ostream &o, const char *arrmeta, const char *data) const;

  void print_type(std::ostream &o) const;

  void transform_child_types(type_transform_fn_t transform_fn,
                             intptr_t arrmeta_offset, void *extra,
                             ndt::type &out_transformed_tp,
                             bool &out_was_transformed) const;
  ndt::type get_canonical_type() const;

  ndt::type apply_linear_index(intptr_t nindices, const irange *indices,
                               size_t current_i, const ndt::type &root_tp,
                               bool leading_dimension) const;
  intptr_t apply_linear_index(intptr_t nindices, const irange *indices,
                              const char *arrmeta, const ndt::type &result_tp,
                              char *out_arrmeta,
                              memory_block_data *embedded_reference,
                              size_t current_i, const ndt::type &root_tp,
                              bool leading_dimension, char **inout_data,
                              memory_block_data **inout_dataref) const;

  bool is_lossless_assignment(const ndt::type &dst_tp,
                              const ndt::type &src_tp) const;

  bool operator==(const base_type &rhs) const;

  void arrmeta_default_construct(char *arrmeta, bool blockref_alloc) const;
  void arrmeta_copy_construct(char *dst_arrmeta, const char *src_arrmeta,
                              memory_block_data *embedded_reference) const;
  void arrmeta_reset_buffers(char *arrmeta) const;
  void arrmeta_finalize_buffers(char *arrmeta) const;
  void arrmeta_destruct(char *arrmeta) const;

  void data_destruct(const char *arrmeta, char *data) const;
  void data_destruct_strided(const char *arrmeta, char *data, intptr_t stride,
                             size_t count) const;

  size_t make_assignment_kernel(
      ckernel_builder *ckb, intptr_t ckb_offset, const ndt::type &dst_tp,
      const char *dst_arrmeta, const ndt::type &src_tp, const char *src_arrmeta,
      kernel_request_t kernreq, const eval::eval_context *ectx) const;

  void get_dynamic_type_properties(
      const std::pair<std::string, gfunc::callable> **out_properties,
      size_t *out_count) const;
  void get_dynamic_array_functions(
      const std::pair<std::string, gfunc::callable> **out_functions,
      size_t *out_count) const;
}; // class arrfunc_type

namespace ndt { namespace detail {

template <typename funcproto_type>
struct funcproto_factory;

template <typename R>
struct funcproto_factory<R ()> {
  static ndt::type make()
  {
    nd::array arg_tp = nd::empty(0, ndt::make_type());
    arg_tp.flag_as_immutable();
    return make_funcproto(arg_tp, make_type<R>());
  }
};

template <typename R, typename... A>
struct funcproto_factory<R (A...)> {
  static ndt::type make()
  {
    ndt::type arg_tp[sizeof...(A)] = {make_type<typename std::remove_cv<typename std::remove_reference<A>::type>::type>()...};
    return make_funcproto(arg_tp, make_type<R>());
  }

  template <typename... T>
  static ndt::type make(const T &... names)
  {
    const char *raw_names[] = {names...};

    ndt::type arg_tp[sizeof...(A)] = {make_type<typename std::remove_cv<typename std::remove_reference<A>::type>::type>()...};
    return make_funcproto(arg_tp, make_type<R>(), raw_names);
  }
};

} // namespace ndt::detail

/** Makes a funcproto type with the specified types */
inline ndt::type make_funcproto(const nd::array &arg_types,
                                const ndt::type &return_type,
                                const nd::array &arg_names)
{
    return ndt::type(
        new arrfunc_type(return_type, arg_types, arg_names), false);
}

inline ndt::type make_funcproto(const nd::array &arg_types,
                                const ndt::type &return_type)
{
  return make_funcproto(arg_types, return_type, nd::array());
}

/** Makes a funcproto type with the specified types */
inline ndt::type make_funcproto(intptr_t narg,
                                const ndt::type *arg_types,
                                const ndt::type &return_type)
{
    nd::array tmp = nd::empty(narg, ndt::make_type());
    ndt::type *tmp_vals =
        reinterpret_cast<ndt::type *>(tmp.get_readwrite_originptr());
    for (intptr_t i = 0; i != narg; ++i) {
        tmp_vals[i] = arg_types[i];
    }
    tmp.flag_as_immutable();
    return make_funcproto(tmp, return_type);
}

/** Makes a unary funcproto type with the specified types */
inline ndt::type make_funcproto(const ndt::type& single_arg_type,
                                const ndt::type &return_type)
{
    ndt::type arg_types[1] = {single_arg_type};
    return make_funcproto(arg_types, return_type);
}

/** Makes a funcproto type from the C++ function type */
template <typename funcproto_type, typename... T>
ndt::type make_funcproto(const T &... names) {
    return detail::funcproto_factory<funcproto_type>::make(names...);
}

ndt::type make_generic_funcproto(intptr_t nargs);

} // namespace ndt

} // namespace dynd
