//
// Copyright (C) 2011-15 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#pragma once

#include <vector>
#include <string>

#include <dynd/array.hpp>
#include <dynd/types/cuda_device_type.hpp>
#include <dynd/types/fixed_dim_type.hpp>
#include <dynd/types/fixed_dimsym_type.hpp>
#include <dynd/types/string_type.hpp>
#include <dynd/types/tuple_type.hpp>
#include <dynd/types/struct_type.hpp>

namespace dynd {

class arrfunc_type : public base_type {
  ndt::type m_return_type;
  // Always a tuple type containing the types for positional args
  ndt::type m_pos_tuple;
  // Always a struct type containing the names and types for keyword args
  ndt::type m_kwd_struct;

  // Indices of the optional args
  std::vector<intptr_t> m_opt_kwd_indices;

public:
  arrfunc_type(const ndt::type &ret_type);

  arrfunc_type(const ndt::type &pos_types, const ndt::type &ret_type);

  arrfunc_type(const ndt::type &pos_types, const ndt::type &kwd_types,
               const ndt::type &ret_type);

  virtual ~arrfunc_type() {}

  const string_type_data &get_kwd_name_raw(intptr_t i) const
  {
    return m_kwd_struct.extended<struct_type>()->get_field_name_raw(i);
  }

  const ndt::type &get_return_type() const { return m_return_type; }

  const ndt::type &get_pos_tuple() const { return m_pos_tuple; }

  const nd::array &get_pos_types() const
  {
    return m_pos_tuple.extended<tuple_type>()->get_field_types();
  }

  bool is_pos_variadic() const
  {
    return m_pos_tuple.extended<tuple_type>()->is_variadic();
  }

  const ndt::type &get_kwd_struct() const { return m_kwd_struct; }

  const nd::array &get_kwd_types() const
  {
    return m_kwd_struct.extended<struct_type>()->get_field_types();
  }

  const nd::array &get_kwd_names() const
  {
    return m_kwd_struct.extended<struct_type>()->get_field_names();
  }

  const ndt::type *get_pos_types_raw() const
  {
    return m_pos_tuple.extended<tuple_type>()->get_field_types_raw();
  }

  const ndt::type &get_pos_type(intptr_t i) const
  {
    return m_pos_tuple.extended<tuple_type>()->get_field_type(i);
  }

  const ndt::type &get_kwd_type(intptr_t i) const
  {
    return m_kwd_struct.extended<struct_type>()->get_field_type(i);
  }

  intptr_t get_kwd_index(const std::string &arg_name) const
  {
    return m_kwd_struct.extended<struct_type>()->get_field_index(arg_name);
  }

  bool has_kwd(const std::string &name) const {
    return get_kwd_index(name) != -1;
  }

  const std::vector<intptr_t> &get_option_kwd_indices() const
  {
    return m_opt_kwd_indices;
  }

  /** Returns the number of arguments, both positional and keyword. */
  intptr_t get_narg() const { return get_npos() + get_nkwd(); }

  /** Returns the number of positional arguments. */
  intptr_t get_npos() const
  {
    return m_pos_tuple.extended<tuple_type>()->get_field_count();
  }

  /** Returns the number of keyword arguments. */
  intptr_t get_nkwd() const
  {
    return m_kwd_struct.extended<tuple_type>()->get_field_count();
  }

  /** Returns the number of optional arguments. */
  intptr_t get_nopt() const { return m_opt_kwd_indices.size(); }

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
  void get_dynamic_array_functions(
      const std::pair<std::string, gfunc::callable> **out_functions,
      size_t *out_count) const;
}; // class arrfunc_type

namespace ndt {
  namespace detail {

    template <kernel_request_t kernreq, typename funcproto_type>
    struct funcproto_factory;

    template <typename R>
    struct funcproto_factory<kernel_request_host, R()> {
      static ndt::type make()
      {
        nd::array arg_tp = nd::empty(0, ndt::make_type());
        arg_tp.flag_as_immutable();
        return make_arrfunc(ndt::make_tuple(arg_tp), make_type<R>());
      }
    };

    template <typename R>
    struct funcproto_factory<kernel_request_cuda_device, R()> {
      static ndt::type make()
      {
        nd::array arg_tp = nd::empty(0, ndt::make_type());
        arg_tp.flag_as_immutable();
        return make_arrfunc(ndt::make_tuple(arg_tp),
                            make_cuda_device(make_type<R>()));
      }
    };

    template <typename R, typename... A>
    struct funcproto_factory<kernel_request_host, R(A...)> {
      static ndt::type make()
      {
        ndt::type arg_tp[sizeof...(A)] = {make_type<typename std::remove_cv<
            typename std::remove_reference<A>::type>::type>()...};
        return make_arrfunc(ndt::make_tuple(arg_tp), make_type<R>());
      }

      template <typename... T>
      static ndt::type make(T &&... names)
      {
        const char *raw_names[] = {names...};

        ndt::type arg_tp[sizeof...(A)] = {make_type<typename std::remove_cv<
            typename std::remove_reference<A>::type>::type>()...};
        return make_arrfunc(
            ndt::make_tuple(nd::array(arg_tp, sizeof...(A) - sizeof...(T))),
            ndt::make_struct(raw_names,
                             nd::array(arg_tp + (sizeof...(A) - sizeof...(T)),
                                       sizeof...(T))),
            make_type<R>());
      }
    };

    template <typename R, typename... A>
    struct funcproto_factory<kernel_request_cuda_device, R(A...)> {
      static ndt::type make()
      {
        ndt::type arg_tp[sizeof...(A)] = {
            make_cuda_device(make_type<typename std::remove_cv<
                typename std::remove_reference<A>::type>::type>())...};
        return make_arrfunc(ndt::make_tuple(arg_tp),
                            make_cuda_device(make_type<R>()));
      }

      template <typename... T>
      static ndt::type make(T &&... names)
      {
        const char *raw_names[] = {names...};

        ndt::type arg_tp[sizeof...(A)] = {
            make_cuda_device(make_type<typename std::remove_cv<
                typename std::remove_reference<A>::type>::type>())...};
        return make_arrfunc(
            ndt::make_tuple(nd::array(arg_tp, sizeof...(A) - sizeof...(T))),
            ndt::make_struct(raw_names,
                             nd::array(arg_tp + (sizeof...(A) - sizeof...(T)),
                                       sizeof...(T))),
            make_cuda_device(make_type<R>()));
      }
    };

  } // namespace ndt::detail

  /** Makes an arrfunc type with both positional and keyword arguments */
  inline ndt::type make_arrfunc(const ndt::type &pos_tuple,
                                const ndt::type &kwd_struct,
                                const ndt::type &return_type)
  {
    return ndt::type(new arrfunc_type(pos_tuple, kwd_struct, return_type),
                     false);
  }

  /** Makes an arrfunc type with both positional and keyword arguments */
  inline ndt::type make_arrfunc(const nd::array &pos_types,
                                const nd::array &kwd_names,
                                const nd::array &kwd_types,
                                const ndt::type &return_type)
  {
    return ndt::type(new arrfunc_type(ndt::make_tuple(pos_types),
                                      ndt::make_struct(kwd_names, kwd_types),
                                      return_type),
                     false);
  }

  /** Makes an arrfunc type with just positional arguments */
  inline ndt::type make_arrfunc(const ndt::type &pos_tuple,
                                const ndt::type &return_type)
  {
    return ndt::type(new arrfunc_type(pos_tuple, return_type), false);
  }

  /** Makes a funcproto type with the specified types */
  inline ndt::type make_arrfunc(intptr_t narg, const ndt::type *arg_types,
                                const ndt::type &return_type)
  {
    nd::array tmp = nd::empty(narg, ndt::make_type());
    ndt::type *tmp_vals =
        reinterpret_cast<ndt::type *>(tmp.get_readwrite_originptr());
    for (intptr_t i = 0; i != narg; ++i) {
      tmp_vals[i] = arg_types[i];
    }
    tmp.flag_as_immutable();
    return make_arrfunc(ndt::make_tuple(tmp), return_type);
  }

  /** Makes a funcproto type from the C++ function type */
  template <kernel_request_t kernreq, typename funcproto_type, typename... T>
  ndt::type make_arrfunc(T &&... names)
  {
    return detail::funcproto_factory<kernreq, funcproto_type>::make(
        std::forward<T>(names)...);
  }

  template <typename funcproto_type, typename... T>
  ndt::type make_arrfunc(T &&... names)
  {
    return make_arrfunc<kernel_request_host, funcproto_type>(
        std::forward<T>(names)...);
  }

  ndt::type make_generic_funcproto(intptr_t nargs);

} // namespace ndt

} // namespace dynd
