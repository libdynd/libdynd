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
#include <dynd/types/fixed_dim_kind_type.hpp>
#include <dynd/types/string_type.hpp>
#include <dynd/types/tuple_type.hpp>
#include <dynd/types/struct_type.hpp>
#include <dynd/types/typevar_constructed_type.hpp>

namespace dynd {

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
typedef void (*arrfunc_data_init_t)(
    char *static_data, size_t data_size, char *data, const ndt::type &dst_tp,
    intptr_t nsrc, const ndt::type *src_tp, const nd::array &kwds,
    const std::map<nd::string, ndt::type> &tp_vars);

/**
 * Resolves the destination type for this arrfunc based on the types
 * of the source parameters.
 *
 * \param self  The arrfunc.
 * \param af_tp  The function prototype of the arrfunc.
 * \param dst_tp  To be filled with the destination type.
 * \param nsrc  The number of source parameters.
 * \param src_tp  An array of the source types.
 */
typedef void (*arrfunc_resolve_dst_type_t)(
    char *static_data, size_t data_size, char *data, ndt::type &dst_tp,
    intptr_t nsrc, const ndt::type *src_tp, const nd::array &kwds,
    const std::map<nd::string, ndt::type> &tp_vars);

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
    char *static_data, size_t data_size, char *data, void *ckb,
    intptr_t ckb_offset, const ndt::type &dst_tp, const char *dst_arrmeta,
    intptr_t nsrc, const ndt::type *src_tp, const char *const *src_arrmeta,
    kernel_request_t kernreq, const eval::eval_context *ectx,
    const nd::array &kwds, const std::map<nd::string, ndt::type> &tp_vars);

/**
 * A function which deallocates the memory behind data_ptr after
 * freeing any additional resources it might contain.
 */
typedef void (*arrfunc_static_data_free_t)(char *static_data);

typedef ndt::type (*arrfunc_make_type_t)();

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
struct arrfunc_type_data {
  /**
   * On 32-bit platforms, if the size changes, it may be
   * necessary to use
   * char data[4 * 8 + ((sizeof(void *) == 4) ? 4 : 0)];
   * to ensure the total struct size is divisible by 64 bits.
   */
  static const std::size_t static_data_size =
      4 * 8 + ((sizeof(void *) == 4) ? 4 : 0);

  char static_data[static_data_size];
  std::size_t data_size;
  arrfunc_data_init_t data_init;
  arrfunc_resolve_dst_type_t resolve_dst_type;
  arrfunc_instantiate_t instantiate;
  arrfunc_static_data_free_t static_data_free;

  arrfunc_type_data()
      : data_size(0), data_init(NULL), resolve_dst_type(NULL),
        instantiate(NULL), static_data_free(NULL)
  {
  }

  arrfunc_type_data(std::size_t data_size, arrfunc_data_init_t data_init,
                    arrfunc_resolve_dst_type_t resolve_dst_type,
                    arrfunc_instantiate_t instantiate)
      : data_size(data_size), data_init(data_init),
        resolve_dst_type(resolve_dst_type), instantiate(instantiate),
        static_data_free(NULL)
  {
  }

  template <typename T>
  arrfunc_type_data(T &&static_data, std::size_t data_size,
                    arrfunc_data_init_t data_init,
                    arrfunc_resolve_dst_type_t resolve_dst_type,
                    arrfunc_instantiate_t instantiate)
      : data_size(data_size), data_init(data_init),
        resolve_dst_type(resolve_dst_type), instantiate(instantiate),
        static_data_free(
            &static_data_destroy<typename std::remove_reference<T>::type>)
  {
    typedef typename std::remove_reference<T>::type static_data_type;
    static_assert(sizeof(static_data_type) <= static_data_size,
                  "static data does not fit");
    static_assert(scalar_align_of<static_data_type>::value <=
                      scalar_align_of<std::uint64_t>::value,
                  "static data requires stronger alignment");
    new (this->static_data)(static_data_type)(std::forward<T>(static_data));
  }

  // non-copyable
  arrfunc_type_data(const arrfunc_type_data &) = delete;

  ~arrfunc_type_data()
  {
    // Call the static_data_free function, if it exists
    if (static_data_free != NULL) {
      static_data_free(static_data);
    }
  }

  nd::array operator()(ndt::type &dst_tp, intptr_t nsrc,
                       const ndt::type *src_tp, const char *const *src_arrmeta,
                       char *const *src_data, const nd::array &kwds,
                       const std::map<nd::string, ndt::type> &tp_vars);

  void operator()(const ndt::type &dst_tp, const char *dst_arrmeta,
                  char *dst_data, intptr_t nsrc, const ndt::type *src_tp,
                  const char *const *src_arrmeta, char *const *src_data,
                  const nd::array &kwds,
                  const std::map<nd::string, ndt::type> &tp_vars);

  template <typename StaticDataType>
  static void static_data_destroy(char *static_data)
  {
    reinterpret_cast<StaticDataType *>(static_data)->~StaticDataType();
  }
};

static_assert((sizeof(arrfunc_type_data) & 7) == 0,
              "arrfunc_type_data must have size divisible by 8");

namespace ndt {

  class arrfunc_type : public base_type {
    type m_return_type;
    // Always a tuple type containing the types for positional args
    type m_pos_tuple;
    // Always a struct type containing the names and types for keyword args
    type m_kwd_struct;

    // Indices of the optional args
    std::vector<intptr_t> m_opt_kwd_indices;

  public:
    arrfunc_type(const type &ret_type);

    arrfunc_type(const type &ret_type, const type &pos_types);

    arrfunc_type(const type &ret_type, const type &pos_types,
                 const type &kwd_types);

    virtual ~arrfunc_type() {}

    const string_type_data &get_kwd_name_raw(intptr_t i) const
    {
      return m_kwd_struct.extended<struct_type>()->get_field_name_raw(i);
    }

    const type &get_return_type() const { return m_return_type; }

    const type &get_pos_tuple() const { return m_pos_tuple; }

    const nd::array &get_pos_types() const
    {
      return m_pos_tuple.extended<tuple_type>()->get_field_types();
    }

    bool is_pos_variadic() const
    {
      return m_pos_tuple.extended<tuple_type>()->is_variadic();
    }

    const type &get_kwd_struct() const { return m_kwd_struct; }

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
      bool matches(intptr_t j, const type &actual_tp,
                   std::map<nd::string, type> &typevars) const
      {
        type expected_tp = get_kwd_type(j);
        if (expected_tp.get_type_id() == option_type_id) {
          expected_tp = expected_tp.p("value_type").as<type>();
        }
        if (!actual_tp.value_type().matches(expected_tp, typevars)) {
          std::stringstream ss;
          ss << "keyword \"" << get_kwd_name(j) << "\" does not match, ";
          ss << "arrfunc expected " << expected_tp << " but passed " <<
      actual_tp;
          throw std::invalid_argument(ss.str());
        }
        return true;
      }
    */

    /** Returns the number of optional arguments. */
    intptr_t get_nopt() const { return m_opt_kwd_indices.size(); }

    void print_data(std::ostream &o, const char *arrmeta,
                    const char *data) const;

    void print_type(std::ostream &o) const;

    void transform_child_types(type_transform_fn_t transform_fn,
                               intptr_t arrmeta_offset, void *extra,
                               type &out_transformed_tp,
                               bool &out_was_transformed) const;
    type get_canonical_type() const;

    type apply_linear_index(intptr_t nindices, const irange *indices,
                            size_t current_i, const type &root_tp,
                            bool leading_dimension) const;
    intptr_t apply_linear_index(intptr_t nindices, const irange *indices,
                                const char *arrmeta, const type &result_tp,
                                char *out_arrmeta,
                                memory_block_data *embedded_reference,
                                size_t current_i, const type &root_tp,
                                bool leading_dimension, char **inout_data,
                                memory_block_data **inout_dataref) const;

    bool is_lossless_assignment(const type &dst_tp, const type &src_tp) const;

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

    intptr_t make_assignment_kernel(void *ckb, intptr_t ckb_offset,
                                    const type &dst_tp, const char *dst_arrmeta,
                                    const type &src_tp, const char *src_arrmeta,
                                    kernel_request_t kernreq,
                                    const eval::eval_context *ectx) const;

    bool match(const char *arrmeta, const type &candidate_tp,
               const char *candidate_arrmeta,
               std::map<nd::string, type> &tp_vars) const;

    void get_dynamic_type_properties(
        const std::pair<std::string, gfunc::callable> **out_properties,
        size_t *out_count) const;
    void get_dynamic_array_functions(
        const std::pair<std::string, gfunc::callable> **out_functions,
        size_t *out_count) const;

    //    static type make(const nd::array &pos_tp, const type &ret_tp);

    /** Makes an arrfunc type with both positional and keyword arguments */
    static type make(const type &return_type, const type &pos_tuple,
                     const type &kwd_struct)
    {
      return type(new arrfunc_type(return_type, pos_tuple, kwd_struct), false);
    }

    /** Makes an arrfunc type with both positional and keyword arguments */
    static type make(const type &return_type, const nd::array &pos_types,
                     const nd::array &kwd_names, const nd::array &kwd_types)
    {
      return type(new arrfunc_type(return_type, tuple_type::make(pos_types),
                                   struct_type::make(kwd_names, kwd_types)),
                  false);
    }

    /** Makes an arrfunc type with just positional arguments */
    static type make(const type &return_type, const type &pos_tuple)
    {
      return type(new arrfunc_type(return_type, pos_tuple), false);
    }

    /** Makes a funcproto type with the specified types */
    static type make(const type &return_type, intptr_t narg,
                     const type *arg_types)
    {
      nd::array tmp = nd::empty(narg, make_type());
      type *tmp_vals = reinterpret_cast<type *>(tmp.get_readwrite_originptr());
      for (intptr_t i = 0; i != narg; ++i) {
        tmp_vals[i] = arg_types[i];
      }
      tmp.flag_as_immutable();
      return make(return_type, tuple_type::make(tmp));
    }
  };

  template <typename R>
  struct type::equivalent<R()> {
    static type make()
    {
      nd::array arg_tp = nd::empty(0, make_type());
      arg_tp.flag_as_immutable();
      return arrfunc_type::make(type::make<R>(), tuple_type::make(arg_tp));
    }
  };

  template <typename R, typename... A>
  struct type::equivalent<R(A...)> {
    static type make()
    {
      return arrfunc_type::make(type::make<R>(),
                                tuple_type::make({type::make<A>()...}));
    }

    template <typename... T>
    static type make(T &&... names)
    {
      type tp[sizeof...(A)] = {type::make<A>()...};

      return arrfunc_type::make(
          type::make<R>(),
          tuple_type::make(nd::array(tp, sizeof...(A) - sizeof...(T))),
          struct_type::make(
              {names...},
              nd::array(tp + (sizeof...(A) - sizeof...(T)), sizeof...(T))));
    }
  };

  type make_generic_funcproto(intptr_t nargs);

} // namespace dynd::ndt
} // namespace dynd