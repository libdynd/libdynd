//
// Copyright (C) 2011-15 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#pragma once

#include <vector>
#include <string>

#include <dynd/array.hpp>
#include <dynd/callables/base_callable.hpp>
#include <dynd/types/cuda_device_type.hpp>
#include <dynd/types/fixed_dim_type.hpp>
#include <dynd/types/fixed_dim_kind_type.hpp>
#include <dynd/types/string_type.hpp>
#include <dynd/types/tuple_type.hpp>
#include <dynd/types/struct_type.hpp>
#include <dynd/types/typevar_constructed_type.hpp>

namespace dynd {
namespace ndt {

  class DYND_API callable_type : public base_type {
    type m_return_type;
    // Always a tuple type containing the types for positional args
    type m_pos_tuple;
    // Always a struct type containing the names and types for keyword args
    type m_kwd_struct;

    // Indices of the optional args
    std::vector<intptr_t> m_opt_kwd_indices;

    struct get_pos_types_kernel;

  public:
    typedef nd::base_callable data_type;

    callable_type(const type &ret_type, const type &pos_types, const type &kwd_types);

    virtual ~callable_type()
    {
    }

    const string &get_kwd_name_raw(intptr_t i) const
    {
      return m_kwd_struct.extended<struct_type>()->get_field_name_raw(i);
    }

    const type &get_return_type() const
    {
      return m_return_type;
    }

    const type &get_pos_tuple() const
    {
      return m_pos_tuple;
    }

    const nd::array &get_pos_types() const
    {
      return m_pos_tuple.extended<tuple_type>()->get_field_types();
    }

    bool is_pos_variadic() const
    {
      return m_pos_tuple.extended<tuple_type>()->is_variadic();
    }

    const type &get_kwd_struct() const
    {
      return m_kwd_struct;
    }

    const nd::array &get_kwd_types() const
    {
      return m_kwd_struct.extended<struct_type>()->get_field_types();
    }

    const nd::array &get_kwd_names() const
    {
      return m_kwd_struct.extended<struct_type>()->get_field_names();
    }

    const type *get_pos_types_raw() const
    {
      return m_pos_tuple.extended<tuple_type>()->get_field_types_raw();
    }

    const type &get_pos_type(intptr_t i) const
    {
      if (i == -1) {
        return get_return_type();
      }

      return m_pos_tuple.extended<tuple_type>()->get_field_type(i);
    }

    const type &get_kwd_type(intptr_t i) const
    {
      return m_kwd_struct.extended<struct_type>()->get_field_type(i);
    }

    std::string get_kwd_name(intptr_t i) const
    {
      return m_kwd_struct.extended<struct_type>()->get_field_name(i);
    }

    intptr_t get_kwd_index(const std::string &arg_name) const
    {
      return m_kwd_struct.extended<struct_type>()->get_field_index(arg_name);
    }

    void get_vars(std::unordered_set<std::string> &vars) const;

    bool has_kwd(const std::string &name) const
    {
      return get_kwd_index(name) != -1;
    }

    const std::vector<intptr_t> &get_option_kwd_indices() const
    {
      return m_opt_kwd_indices;
    }

    /** Returns the number of arguments, both positional and keyword. */
    intptr_t get_narg() const
    {
      return get_npos() + get_nkwd();
    }

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
    intptr_t get_nopt() const
    {
      return m_opt_kwd_indices.size();
    }

    void print_data(std::ostream &o, const char *arrmeta, const char *data) const;

    void print_type(std::ostream &o) const;

    void transform_child_types(type_transform_fn_t transform_fn, intptr_t arrmeta_offset, void *extra,
                               type &out_transformed_tp, bool &out_was_transformed) const;
    type get_canonical_type() const;

    type apply_linear_index(intptr_t nindices, const irange *indices, size_t current_i, const type &root_tp,
                            bool leading_dimension) const;
    intptr_t apply_linear_index(intptr_t nindices, const irange *indices, const char *arrmeta, const type &result_tp,
                                char *out_arrmeta, const intrusive_ptr<memory_block_data> &embedded_reference,
                                size_t current_i, const type &root_tp, bool leading_dimension, char **inout_data,
                                intrusive_ptr<memory_block_data> &inout_dataref) const;

    bool is_lossless_assignment(const type &dst_tp, const type &src_tp) const;

    bool operator==(const base_type &rhs) const;

    void arrmeta_default_construct(char *arrmeta, bool blockref_alloc) const;
    void arrmeta_copy_construct(char *dst_arrmeta, const char *src_arrmeta,
                                const intrusive_ptr<memory_block_data> &embedded_reference) const;
    void arrmeta_reset_buffers(char *arrmeta) const;
    void arrmeta_finalize_buffers(char *arrmeta) const;
    void arrmeta_destruct(char *arrmeta) const;

    void data_destruct(const char *arrmeta, char *data) const;
    void data_destruct_strided(const char *arrmeta, char *data, intptr_t stride, size_t count) const;

    intptr_t make_assignment_kernel(void *ckb, intptr_t ckb_offset, const type &dst_tp, const char *dst_arrmeta,
                                    const type &src_tp, const char *src_arrmeta, kernel_request_t kernreq,
                                    const eval::eval_context *ectx) const;

    bool match(const char *arrmeta, const type &candidate_tp, const char *candidate_arrmeta,
               std::map<std::string, type> &tp_vars) const;

    void get_dynamic_type_properties(const std::pair<std::string, nd::callable> **out_properties,
                                     size_t *out_count) const;
    void get_dynamic_array_functions(const std::pair<std::string, gfunc::callable> **out_functions,
                                     size_t *out_count) const;

    /** Makes an callable type with both positional and keyword arguments */
    static type make(const type &ret_tp, const type &tuple_tp, const type &struct_tp)
    {
      return type(new callable_type(ret_tp, tuple_tp, struct_tp), false);
    }

    /** Makes an callable type with both positional and keyword arguments */
    static type make(const type &ret_tp, const nd::array &pos_tp, const nd::array &kwd_names, const nd::array &kwd_tp)
    {
      return make(ret_tp, tuple_type::make(pos_tp), struct_type::make(kwd_names, kwd_tp));
    }

    /** Makes an callable type with just positional arguments */
    static type make(const type &ret_tp, const type &tuple_tp)
    {
      if (tuple_tp.get_type_id() != tuple_type_id) {
        return make(ret_tp, tuple_type::make({tuple_tp}), struct_type::make());
      }

      return make(ret_tp, tuple_tp, struct_type::make(tuple_tp.extended<base_tuple_type>()->is_variadic()));
    }

    /** Makes an callable type with just positional arguments */
    static type make(const type &ret_tp, const nd::array &pos_tp)
    {
      return make(ret_tp, tuple_type::make(pos_tp), struct_type::make());
    }

    /** Makes an callable type with no arguments */
    static type make(const type &ret_tp)
    {
      return make(ret_tp, tuple_type::make(), struct_type::make());
    }
  };

  template <typename R>
  struct type::equivalent<R()> {
    static type make()
    {
      return callable_type::make(type::make<R>());
    }
  };

  template <typename R, typename A0, typename... A>
  struct type::equivalent<R(A0, A...)> {
    static type make()
    {
      return callable_type::make(type::make<R>(), {type::make<A0>(), type::make<A>()...});
    }

    template <typename... T>
    static type make(const T &... names)
    {
      type tp[1 + sizeof...(A)] = {type::make<A0>(), type::make<A>()...};

      return callable_type::make(type::make<R>(), nd::array(tp, 1 + sizeof...(A) - sizeof...(T)), {names...},
                                 nd::array(tp + (1 + sizeof...(A) - sizeof...(T)), sizeof...(T)));
    }
  };

  DYND_API type make_generic_funcproto(intptr_t nargs);

} // namespace dynd::ndt
} // namespace dynd
