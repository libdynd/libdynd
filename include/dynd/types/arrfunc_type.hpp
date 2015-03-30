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
#include <dynd/types/typevar_constructed_type.hpp>

namespace dynd {

class arrfunc_type_data;

/**
 * Function prototype for instantiating a ckernel from an
 * arrfunc. To use this function, the
 * caller should first allocate a `ckernel_builder` instance,
 * either from C++ normally or by reserving appropriately aligned/sized
 * data and calling the C function constructor dynd provides. When the
 * data types of the kernel require arrmeta, such as for 'strided'
 * or 'var' dimension types, the arrmeta must be provided as well.
 *
 * \param self  The arrfunc.
 * \param self_tp  The function prototype of the arrfunc.
 * \param ckb  A ckernel_builder instance where the kernel is placed.
 * \param ckb_offset  The offset into the output ckernel_builder `ckb`
 *                    where the kernel should be placed.
 * \param dst_tp  The destination type of the ckernel to generate. This may be
 *                different from the one in the function prototype, but must
 *                match its pattern.
 * \param dst_arrmeta  The destination arrmeta.
 * \param nsrc  The number of source arrays.
 * \param src_tp  An array of the source types of the ckernel to generate. These
 *                may be different from the ones in the function prototype, but
 *                must match the patterns.
 * \param src_arrmeta  An array of dynd arrmeta pointers,
 *                     corresponding to the source types.
 * \param kernreq  What kind of C function prototype the resulting ckernel
 *                 should follow. Defined by the enum with kernel_request_*
 *                 values.
 * \param ectx  The evaluation context.
 * \param kwds  A struct array of named auxiliary arguments.
 *
 * \returns  The offset into ``ckb`` immediately after the instantiated ckernel.
 */
typedef intptr_t (*arrfunc_instantiate_t)(
    const arrfunc_type_data *self, const arrfunc_type *self_tp, char *data,
    void *ckb, intptr_t ckb_offset, const ndt::type &dst_tp,
    const char *dst_arrmeta, intptr_t nsrc, const ndt::type *src_tp,
    const char *const *src_arrmeta, kernel_request_t kernreq,
    const eval::eval_context *ectx, const nd::array &kwds,
    const std::map<nd::string, ndt::type> &tp_vars);

/**
 * Resolves the destination type for this arrfunc based on the types
 * of the source parameters.
 *
 * \param self  The arrfunc.
 * \param af_tp  The function prototype of the arrfunc.
 * \param nsrc  The number of source parameters.
 * \param src_tp  An array of the source types.
 * \param throw_on_error  If true, should throw when there's an error, if
 *                        false, should return 0 when there's an error.
 * \param out_dst_tp  To be filled with the destination type.
 *
 * \returns  True on success, false on error (if throw_on_error was false).
 */
typedef int (*arrfunc_resolve_dst_type_t)(
    const arrfunc_type_data *self, const arrfunc_type *af_tp, char *data,
    intptr_t nsrc, const ndt::type *src_tp, int throw_on_error, ndt::type &out_dst_tp,
    const nd::array &kwds, const std::map<nd::string, ndt::type> &tp_vars);

/**
 * Resolves any missing keyword arguments for this arrfunc based on
 * the types of the positional arguments and the available keywords arguments.
 *
 * \param self    The arrfunc.
 * \param self_tp The function prototype of the arrfunc.
 * \param nsrc    The number of positional arguments.
 * \param src_tp  An array of the source types.
 * \param kwds    An array of the.
 */
typedef void (*arrfunc_resolve_option_values_t)(
    const arrfunc_type_data *self, const arrfunc_type *self_tp,
    char *data, intptr_t nsrc,
    const ndt::type *src_tp, nd::array &kwds,
    const std::map<nd::string, ndt::type> &tp_vars);

/**
 * A function which deallocates the memory behind data_ptr after
 * freeing any additional resources it might contain.
 */
typedef void (*arrfunc_free_t)(arrfunc_type_data *self);

typedef ndt::type (*arrfunc_make_type_t)();

template <typename T>
void destroy_wrapper(arrfunc_type_data *self);

/**
 * This is a struct designed for interoperability at
 * the C ABI level. It contains enough information
 * to pass arrfuncs from one library to another
 * with no dependencies between them.
 *
 * The arrfunc can produce a ckernel with with a few
 * variations, like choosing between a single
 * operation and a strided operation, or constructing
 * with different array arrmeta.
 */
class arrfunc_type_data {
  // non-copyable
  arrfunc_type_data(const arrfunc_type_data &) = delete;

public:
  /**
   * Some memory for the arrfunc to use. If this is not
   * enough space to hold all the data by value, should allocate
   * space on the heap, and free it when free is called.
   *
   * On 32-bit platforms, if the size changes, it may be
   * necessary to use
   * char data[4 * 8 + ((sizeof(void *) == 4) ? 4 : 0)];
   * to ensure the total struct size is divisible by 64 bits.
   */
  char data[4 * 8 + ((sizeof(void *) == 4) ? 4 : 0)];

  arrfunc_instantiate_t instantiate;
  arrfunc_resolve_option_values_t resolve_option_values;
  arrfunc_resolve_dst_type_t resolve_dst_type;
  arrfunc_free_t free;
  const size_t size;

  static const size_t data_size = 64;

  arrfunc_type_data()
      : instantiate(NULL), resolve_option_values(NULL), resolve_dst_type(NULL),
        free(NULL), size(1)
  {
    static_assert((sizeof(arrfunc_type_data) & 7) == 0,
                       "arrfunc_type_data must have size divisible by 8");
  }

  arrfunc_type_data(arrfunc_instantiate_t instantiate,
                    arrfunc_resolve_option_values_t resolve_option_values,
                    arrfunc_resolve_dst_type_t resolve_dst_type, size_t size = 1)
      : instantiate(instantiate), resolve_option_values(resolve_option_values),
        resolve_dst_type(resolve_dst_type), size(size)
  {
  }

  arrfunc_type_data(arrfunc_instantiate_t instantiate,
                    arrfunc_resolve_option_values_t resolve_option_values,
                    arrfunc_resolve_dst_type_t resolve_dst_type,
                    arrfunc_free_t free, size_t size = 1)
      : instantiate(instantiate), resolve_option_values(resolve_option_values),
        resolve_dst_type(resolve_dst_type), free(free), size(size)
  {
  }

  template <typename T>
  arrfunc_type_data(T &&data, arrfunc_instantiate_t instantiate,
                    arrfunc_resolve_option_values_t resolve_option_values,
                    arrfunc_resolve_dst_type_t resolve_dst_type,
                    arrfunc_free_t free = NULL, size_t size = 1)
      : instantiate(instantiate), resolve_option_values(resolve_option_values),
        resolve_dst_type(resolve_dst_type),
        free(free == NULL
                 ? &destroy_wrapper<typename std::remove_reference<T>::type>
                 : free), size(size)
  {
    new (this->data)(typename std::remove_reference<T>::type)(
        std::forward<T>(data));
  }

  template <typename T>
  arrfunc_type_data(T *data, arrfunc_instantiate_t instantiate,
                    arrfunc_resolve_option_values_t resolve_option_values,
                    arrfunc_resolve_dst_type_t resolve_dst_type,
                    arrfunc_free_t free = NULL, size_t size = 1)
      : instantiate(instantiate), resolve_option_values(resolve_option_values),
        resolve_dst_type(resolve_dst_type), free(free), size(size)
  {
    new (this->data) (T*)(data);
  }

  ~arrfunc_type_data()
  {
    // Call the free function, if it exists
    if (free != NULL) {
      free(this);
    }
  }

  /**
   * Helper function to reinterpret the data as the specified type.
   */
  template <typename T>
  T *get_data_as()
  {
    if (sizeof(T) > sizeof(data)) {
      throw std::runtime_error("data does not fit");
    }
    if ((int)scalar_align_of<T>::value >
        (int)scalar_align_of<uint64_t>::value) {
      throw std::runtime_error("data requires stronger alignment");
    }
    return reinterpret_cast<T *>(data);
  }
  template <typename T>
  const T *get_data_as() const
  {
    if (sizeof(T) > sizeof(data)) {
      throw std::runtime_error("data does not fit");
    }
    if ((int)scalar_align_of<T>::value >
        (int)scalar_align_of<uint64_t>::value) {
      throw std::runtime_error("data requires stronger alignment");
    }
    return reinterpret_cast<const T *>(data);
  }
};

template <typename T>
void destroy_wrapper(arrfunc_type_data *self)
{
  self->get_data_as<T>()->~T();
}

template <typename T>
void delete_wrapper(arrfunc_type_data *self)
{
  delete *self->get_data_as<T *>();
}

template <typename T, void (*free)(void *) = &std::free>
void free_wrapper(arrfunc_type_data *self)
{
  free(*self->get_data_as<T *>());
}

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

  std::string get_kwd_name(intptr_t i) const
  {
    return m_kwd_struct.extended<struct_type>()->get_field_name(i);
  }

  intptr_t get_kwd_index(const std::string &arg_name) const
  {
    return m_kwd_struct.extended<struct_type>()->get_field_index(arg_name);
  }

  void get_vars(std::unordered_set<std::string> &vars) const;

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

/*
  bool matches(intptr_t j, const ndt::type &actual_tp,
               std::map<nd::string, ndt::type> &typevars) const
  {
    ndt::type expected_tp = get_kwd_type(j);
    if (expected_tp.get_type_id() == option_type_id) {
      expected_tp = expected_tp.p("value_type").as<ndt::type>();
    }
    if (!actual_tp.value_type().matches(expected_tp, typevars)) {
      std::stringstream ss;
      ss << "keyword \"" << get_kwd_name(j) << "\" does not match, ";
      ss << "arrfunc expected " << expected_tp << " but passed " << actual_tp;
      throw std::invalid_argument(ss.str());
    }
    return true;
  }
*/

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

  bool match(const char *arrmeta, const ndt::type &candidate_tp,
             const char *candidate_arrmeta,
             std::map<nd::string, ndt::type> &tp_vars) const;

  void get_dynamic_type_properties(
      const std::pair<std::string, gfunc::callable> **out_properties,
      size_t *out_count) const;
  void get_dynamic_array_functions(
      const std::pair<std::string, gfunc::callable> **out_functions,
      size_t *out_count) const;
}; // class arrfunc_type

namespace ndt {
  template <kernel_request_t kernreq, typename funcproto_type>
  struct as_arrfunc_type;

  template <typename R>
  struct as_arrfunc_type<kernel_request_host, R()> {
    static ndt::type make()
    {
      nd::array arg_tp = nd::empty(0, ndt::make_type());
      arg_tp.flag_as_immutable();
      return make_arrfunc(ndt::make_tuple(arg_tp), make_type<R>());
    }
  };

  template <typename R, typename... A>
  struct as_arrfunc_type<kernel_request_host, R(A...)> {
    static type make()
    {
      type tp[sizeof...(A)] = {make_type<typename std::remove_cv<
          typename std::remove_reference<A>::type>::type>()...};
      return make_arrfunc(ndt::make_tuple(tp), make_type<R>());
    }

    template <typename... T>
    static type make(T &&... names)
    {
      type tp[sizeof...(A)] = {make_type<typename std::remove_cv<
          typename std::remove_reference<A>::type>::type>()...};

      return make_arrfunc(
          make_tuple(nd::array(tp, sizeof...(A) - sizeof...(T))),
          make_struct({names...}, nd::array(tp + (sizeof...(A) - sizeof...(T)),
                                            sizeof...(T))),
          make_type<R>());
    }
  };

#ifdef DYND_CUDA

  template <typename R, typename... A>
  struct as_arrfunc_type<kernel_request_cuda_device, R(A...)> {
    static ndt::type make()
    {
      ndt::type ret_tp = make_type<R>();
      if (ret_tp.get_kind() != void_kind) {
        ret_tp = make_cuda_device(ret_tp);
      }

      ndt::type arg_tp[sizeof...(A)] = {
          make_cuda_device(make_type<typename std::remove_cv<
              typename std::remove_reference<A>::type>::type>())...};
      return make_arrfunc(ndt::make_tuple(arg_tp),
                          ret_tp);
    }

    template <typename... T>
    static ndt::type make(T &&... names)
    {
      ndt::type ret_tp = make_type<R>();
      if (ret_tp.get_kind() != void_kind) {
        ret_tp = make_cuda_device(ret_tp);
      }

      ndt::type arg_tp[sizeof...(A)] = {
          make_cuda_device(make_type<typename std::remove_cv<
              typename std::remove_reference<A>::type>::type>())...};
      return make_arrfunc(
          ndt::make_tuple(nd::array(arg_tp, sizeof...(A) - sizeof...(T))),
          ndt::make_struct(
              {names...},
              nd::array(arg_tp + (sizeof...(A) - sizeof...(T)), sizeof...(T))),
          ret_tp);
    }
  };

  template <typename R>
  struct as_arrfunc_type<kernel_request_cuda_device, R()> {
    static ndt::type make()
    {
      nd::array arg_tp = nd::empty(0, ndt::make_type());
      arg_tp.flag_as_immutable();
      return make_arrfunc(ndt::make_tuple(arg_tp),
                          make_cuda_device(make_type<R>()));
    }
  };

#endif

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
  type make_arrfunc(T &&... names)
  {
    return as_arrfunc_type<kernreq, funcproto_type>::make(
        std::forward<T>(names)...);
  }

  template <typename funcproto_type, typename... T>
  type make_arrfunc(T &&... names)
  {
    return make_arrfunc<kernel_request_host, funcproto_type>(
        std::forward<T>(names)...);
  }

  ndt::type make_generic_funcproto(intptr_t nargs);

} // namespace ndt

} // namespace dynd
