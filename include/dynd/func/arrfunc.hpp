//
// Copyright (C) 2011-14 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#pragma once

#include <dynd/config.hpp>
#include <dynd/eval/eval_context.hpp>
#include <dynd/types/base_type.hpp>
#include <dynd/types/arrfunc_type.hpp>
#include <dynd/types/arrfunc_old_type.hpp>
#include <dynd/kernels/ckernel_builder.hpp>
#include <dynd/types/struct_type.hpp>
#include <dynd/types/type_pattern_match.hpp>
#include <dynd/types/substitute_typevars.hpp>

namespace dynd {

namespace ndt {
  ndt::type make_option(const ndt::type &value_tp);
}

struct arrfunc_type_data;

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
 * \param af_tp  The function prototype of the arrfunc.
 * \param ckb  A ckernel_builder instance where the kernel is placed.
 * \param ckb_offset  The offset into the output ckernel_builder `ckb`
 *                    where the kernel should be placed.
 * \param dst_tp  The destination type of the ckernel to generate. This may be
 *                different from the one in the function prototype, but must
 *                match its pattern.
 * \param dst_arrmeta  The destination arrmeta.
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
    const arrfunc_type_data *self, const arrfunc_type *af_tp, void *ckb,
    intptr_t ckb_offset, const ndt::type &dst_tp, const char *dst_arrmeta,
    const ndt::type *src_tp, const char *const *src_arrmeta,
    kernel_request_t kernreq, const eval::eval_context *ectx,
    const nd::array &kwds);

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
    const arrfunc_type_data *self, const arrfunc_type *af_tp, intptr_t nsrc,
    const ndt::type *src_tp, int throw_on_error, ndt::type &out_dst_tp,
    const nd::array &kwds);

typedef void (*arrfunc_resolve_option_types_t)(const arrfunc_type_data *self,
                                               const arrfunc_type *af_tp,
                                               intptr_t nsrc,
                                               const ndt::type *src_tp,
                                               nd::array &kwds);

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
   * Some memory for the arrfunc to use. If this is not
   * enough space to hold all the data by value, should allocate
   * space on the heap, and free it when free_func is called.
   *
   * On 32-bit platforms, if the size changes, it may be
   * necessary to use
   * char data[4 * 8 + ((sizeof(void *) == 4) ? 4 : 0)];
   * to ensure the total struct size is divisible by 64.
   */
  char data[4 * 8 + ((sizeof(void *) == 4) ? 4 : 0)];
  /**
   * The function which instantiates a ckernel. See the documentation
   * for the function typedef for more details.
   */
  arrfunc_instantiate_t instantiate;
  arrfunc_resolve_option_types_t resolve_option_types;
  arrfunc_resolve_dst_type_t resolve_dst_type;

  /**
   * A function which deallocates the memory behind data_ptr after
   * freeing any additional resources it might contain.
   */
  void (*free_func)(arrfunc_type_data *self_data_ptr);

  // Default to all NULL, so the destructor works correctly
  arrfunc_type_data() : instantiate(0), free_func(0)
  {
    DYND_STATIC_ASSERT((sizeof(arrfunc_type_data) & 7) == 0,
                       "arrfunc_type_data must have size divisible by 8");
  }

  // If it contains an arrfunc, free it
  ~arrfunc_type_data()
  {
    if (free_func) {
      free_func(this);
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

namespace nd {
  inline array
  make_arrfunc(ndt::type af_tp, arrfunc_instantiate_t instantiate,
               arrfunc_resolve_option_types_t resolve_option_types = NULL)
  {
    array af = empty(af_tp);
    arrfunc_type_data *out_af =
        reinterpret_cast<arrfunc_type_data *>(af.get_readwrite_originptr());
    out_af->instantiate = instantiate;
    out_af->resolve_option_types = resolve_option_types;
    af.flag_as_immutable();

    return af;
  }

  inline nd::array forward_as_array(const nd::array &DYND_UNUSED(names),
                                    const nd::array &DYND_UNUSED(types),
                                    const std::tuple<> &DYND_UNUSED(vals),
                                    const intptr_t *DYND_UNUSED(perm) = NULL)
  {
    return nd::array();
  }

  template <typename... A>
  nd::array forward_as_array(const nd::array &names, const nd::array &types,
                             const std::tuple<A...> &vals,
                             const intptr_t *perm = NULL)
  {
    typedef typename make_index_sequence<0, sizeof...(A)>::type I;

    nd::array res = nd::empty_shell(ndt::make_struct(names, types));
    struct_type::fill_default_data_offsets(
        res.get_dim_size(),
        reinterpret_cast<const ndt::type *>(types.get_readonly_originptr()),
        reinterpret_cast<uintptr_t *>(res.get_arrmeta()));
    nd::index_proxy<I>::template forward_as_array(
        reinterpret_cast<const ndt::type *>(types.get_readonly_originptr()),
        res.get_arrmeta(),
        res.get_type().extended<base_struct_type>()->get_arrmeta_offsets_raw(),
        res.get_readwrite_originptr(),
        res.get_type().extended<base_struct_type>()->get_data_offsets(
            res.get_arrmeta()),
        vals, perm);

    return res;
  }

  namespace detail {
    template <typename... T>
    class kwds;

    template <>
    class kwds<> {
      std::tuple<> m_vals;

    public:
      const char *get_name(intptr_t DYND_UNUSED(i)) const
      {
        throw std::runtime_error("");
      }

      ndt::type get_type(intptr_t DYND_UNUSED(i)) const
      {
        throw std::runtime_error("");
      }

      array get_names() const { return array(); }

      //      array get_types() const { return array(); }

      const std::tuple<> &get_vals() const { return m_vals; }
    };

    template <typename... T>
    class kwds {
      const char *m_names[sizeof...(T)];
      ndt::type m_types[sizeof...(T)];
      std::tuple<T...> m_vals;

#if !(defined(_MSC_VER) && _MSC_VER == 1800)
    public:
      kwds(const typename as_<T, const char *>::type &... names, T &&... vals)
          : m_names{names...}, m_vals(std::forward<T>(vals)...)
      {
        typedef typename make_index_sequence<0, sizeof...(T)>::type I;

        ndt::index_proxy<I>::template get_types(m_types, get_vals());
      }
#else
      template <size_t I>
      void assign_names()
      {
      }

      template <size_t I, typename T0, typename... T>
      void assign_names(T0 &&t0, T &&... t)
      {
        m_names[I] = t0;
        assign_names<I + 1>(std::forward<T>(t)...);
      }

    public:
      kwds(const typename as_<T, const char *>::type &... names, T &&... vals)
          : m_vals(std::forward<T>(vals)...)
      {
        typedef typename make_index_sequence<0, sizeof...(T)>::type I;

        assign_names<0>(names...);
        ndt::index_proxy<I>::template get_types(m_types, get_vals());
      }
#endif

      /*
        const char *(&get_names() const)[sizeof...(T)] {
          return nd::make_strided_string_array(sizeof...(T), m_names);
        }
      */

      nd::array get_names() const
      {
        return nd::make_strided_string_array(const_cast<const char **>(m_names),
                                             sizeof...(T));
      }

      const char *get_name(intptr_t i) const { return m_names[i]; }

      ndt::type get_type(intptr_t i) const { return m_types[i]; }

      //      ndt::type (&get_types() const)[sizeof...(T)] { return m_types; }

      const std::tuple<T...> &get_vals() const { return m_vals; }
    };

    /*
        template <typename... T>
        using kwds_for = typename instantiate<
            detail::kwds,
            typename take<typename make_index_sequence<1, sizeof...(T),
       2>::type,
                          type_sequence<T...>>::type>::type;
    */
  }
} // namespace nd

inline nd::detail::kwds<> kwds() { return nd::detail::kwds<>(); }

template <typename... T>
typename instantiate<
    nd::detail::kwds,
    typename take<typename make_index_sequence<1, sizeof...(T), 2>::type,
                  type_sequence<T...>>::type>::type
kwds(T &&... args)
{
  // Sequence of even integers, for extracting the keyword names
  typedef typename make_index_sequence<0, sizeof...(T), 2>::type I;
  // Sequence of odd integers, for extracting the values
  typedef typename make_index_sequence<1, sizeof...(T), 2>::type J;
  // Sequence of evens followed by odds
  typedef typename concatenate<I, J>::type IJ;
  // Type sequence of the values' types
  typedef typename take<J, type_sequence<T...>>::type ValuesTypes;
  // The kwds<...> type instantiated with the values' types
  typedef typename instantiate<nd::detail::kwds, ValuesTypes>::type KwdsType;

  return index_proxy<IJ>::template make<KwdsType>(std::forward<T>(args)...);
}

namespace nd {
  /**
   * Holds a single instance of an arrfunc in an immutable nd::array,
   * providing some more direct convenient interface.
   */
  class arrfunc {
    nd::array m_value;

  public:
    arrfunc() : m_value() {}
    arrfunc(const arrfunc &rhs) : m_value(rhs.m_value) {}
    /**
      * Constructor from an nd::array. Validates that the input
      * has "arrfunc" type and is immutable.
      */
    arrfunc(const nd::array &rhs);

    arrfunc &operator=(const arrfunc &rhs)
    {
      m_value = rhs.m_value;
      return *this;
    }

    bool is_null() const { return m_value.is_null(); }

    const arrfunc_type_data *get() const
    {
      return !m_value.is_null() ? reinterpret_cast<const arrfunc_type_data *>(
                                      m_value.get_readonly_originptr())
                                : NULL;
    }

    const arrfunc_type *get_type() const
    {
      return !m_value.is_null() ? m_value.get_type().extended<arrfunc_type>()
                                : NULL;
    }

    const ndt::type &get_array_type() const { return m_value.get_type(); }

    operator nd::array() const { return m_value; }

    void swap(nd::arrfunc &rhs) { m_value.swap(rhs.m_value); }

    template <typename... T>
    void set_as_option(arrfunc_resolve_option_types_t resolve_option_types,
                       T &&... names)
    {
      intptr_t missing[sizeof...(T)] = {get_type()->get_arg_index(names)...};

      nd::array tp = get_type()->get_arg_types().eval_copy();
      for (size_t i = 0; i < sizeof...(T); ++i) {
        tp(missing[i]).val_assign(
            ndt::make_option(tp(missing[i]).template as<ndt::type>()));
      }

      m_value =
          make_arrfunc(ndt::make_funcproto(tp, get_type()->get_return_type(),
                                           get_type()->get_arg_names()),
                       get()->instantiate, resolve_option_types);
    }

    template <typename... K>
    ndt::type resolve(intptr_t nsrc, const ndt::type *src_tp,
                      const detail::kwds<K...> &kwds,
                      array &kwds_as_array) const
    {
      const arrfunc_type_data *af = get();
      const arrfunc_type *af_tp = m_value.get_type().extended<arrfunc_type>();

      if (af->resolve_dst_type != NULL) {
        kwds_as_array = forward_as_array(
            kwds.get_names(), ndt::get_forward_types(kwds.get_vals()),
            kwds.get_vals());

        ndt::type dst_tp;
        af->resolve_dst_type(af, af_tp, nsrc, src_tp, true, dst_tp,
                             kwds_as_array);

        return dst_tp;
      }

      if (nsrc != af_tp->get_npos()) {
        std::stringstream ss;
        ss << "arrfunc expected " << af_tp->get_npos()
           << " parameters, but received " << nsrc;
        throw std::invalid_argument(ss.str());
      }
      const ndt::type *param_types = af_tp->get_arg_types_raw();
      std::map<nd::string, ndt::type> typevars;
      for (intptr_t i = 0; i != nsrc; ++i) {
        if (!ndt::pattern_match(src_tp[i].value_type(), param_types[i],
                                typevars)) {
          std::stringstream ss;
          ss << "parameter " << (i + 1) << " to arrfunc does not match, ";
          ss << "expected " << param_types[i] << ", received " << src_tp[i];
          throw std::invalid_argument(ss.str());
        }
      }

      nd::array kwd_tp2 = nd::empty(af_tp->get_nkwd(), ndt::make_type());

      ndt::type *kwd_tp =
          reinterpret_cast<ndt::type *>(kwd_tp2.get_readwrite_originptr());

      std::vector<intptr_t> available(sizeof...(K));
      std::vector<intptr_t> missing(af_tp->get_nkwd() - sizeof...(K));

      std::cout << af_tp->get_nkwd() << std::endl;

      for (intptr_t i = 0; i != sizeof...(K); i++) {
        available[i] = af_tp->get_arg_index(kwds.get_name(i));
        ndt::type tp = kwds.get_type(i);
        ndt::type expected = param_types[available[i]];
        if (expected.get_type_id() == option_type_id) {
          expected = expected.p("value_type").as<ndt::type>();
        }
        if (!ndt::pattern_match(tp.value_type(), expected, typevars)) {
          std::stringstream ss;
          ss << "keyword parameter \"" << kwds.get_name(i)
             << "\" to arrfunc does not match, ";
          ss << "expected " << param_types[available[i]] << ", received " << tp;
          throw std::invalid_argument(ss.str());
        }
        kwd_tp[available[i] - af_tp->get_npos()] = tp;
        available[i] -= af_tp->get_npos();
      }
      for (size_t i = 0; i < available.size(); ++i) {
        std::cout << "available[i] = " << available[i] << std::endl;
      }



      ndt::get_forward_types(kwd_tp2, kwds.get_vals(),
                             available.empty() ? NULL : available.data());

      std::cout << kwd_tp2 << std::endl;
      kwds_as_array =
          forward_as_array(af_tp->get_arg_names(), kwd_tp2, kwds.get_vals(),
                           available.empty() ? NULL : available.data());
      std::cout << kwds_as_array << std::endl;

      return ndt::substitute(af_tp->get_return_type(), typevars, true);
    }

    /** Implements the general call operator */
    template <typename... K>
    array call(intptr_t narg, const nd::array *args,
               const detail::kwds<K...> &kwds,
               const eval::eval_context *ectx) const
    {
      const arrfunc_type_data *af = get();
      const arrfunc_type *af_tp = m_value.get_type().extended<arrfunc_type>();

      std::vector<ndt::type> arg_tp(narg);
      for (intptr_t i = 0; i < narg; ++i) {
        arg_tp[i] = args[i].get_type();
      }

      std::vector<const char *> src_arrmeta(af_tp->get_npos());
      for (intptr_t i = 0; i < af_tp->get_npos(); ++i) {
        src_arrmeta[i] = args[i].get_arrmeta();
      }
      std::vector<char *> src_data(af_tp->get_npos());
      for (intptr_t i = 0; i < af_tp->get_npos(); ++i) {
        src_data[i] = const_cast<char *>(args[i].get_readonly_originptr());
      }

      nd::array kwds_as_array;

      // Resolve the destination type
      ndt::type dst_tp =
          resolve(af_tp->get_npos(), af_tp->get_npos() ? &arg_tp[0] : NULL,
                  kwds, kwds_as_array);

      // Construct the destination array
      nd::array res = nd::empty(dst_tp);

      // Generate and evaluate the ckernel
      ckernel_builder<kernel_request_host> ckb;
      af->instantiate(af, af_tp, &ckb, 0, dst_tp, res.get_arrmeta(), &arg_tp[0],
                      &src_arrmeta[0], kernel_request_single, ectx,
                      kwds_as_array);
      expr_single_t fn = ckb.get()->get_function<expr_single_t>();
      fn(res.get_readwrite_originptr(), src_data.empty() ? NULL : &src_data[0],
         ckb.get());
      return res;
    }

    nd::array call(intptr_t arg_count, const nd::array *args,
                   const eval::eval_context *ectx) const
    {
      return call(arg_count, args, kwds(), ectx);
    }

    /** Convenience call operators */
    template <typename... K>
    nd::array operator()(const detail::kwds<K...> &kwds_ = kwds()) const
    {
      return call(0, NULL, kwds_, &eval::default_eval_context);
    }
    template <typename... K>
    nd::array operator()(const nd::array &a0,
                         const detail::kwds<K...> &kwds_ = kwds()) const
    {
      return call(1, &a0, kwds_, &eval::default_eval_context);
    }
    template <typename... K>
    nd::array operator()(const nd::array &a0, const nd::array &a1,
                         const detail::kwds<K...> &kwds_ = kwds()) const
    {
      nd::array args[2] = {a0, a1};
      return call(2, args, kwds_, &eval::default_eval_context);
    }
    template <typename... K>
    nd::array operator()(const nd::array &a0, const nd::array &a1,
                         const nd::array &a2,
                         const detail::kwds<K...> &kwds_ = kwds()) const
    {
      nd::array args[3] = {a0, a1, a2};
      return call(3, args, kwds_, &eval::default_eval_context);
    }

    /** Implements the general call operator with output parameter */
    template <typename... K>
    void call_out(intptr_t narg, const nd::array *args,
                  const detail::kwds<K...> &DYND_UNUSED(kwds),
                  const nd::array &out, const eval::eval_context *ectx) const
    {
      const arrfunc_type_data *af = get();
      const arrfunc_type *af_tp = m_value.get_type().extended<arrfunc_type>();

      std::vector<ndt::type> arg_tp(narg);
      for (intptr_t i = 0; i < narg; ++i) {
        arg_tp[i] = args[i].get_type();
      }

      std::vector<const char *> src_arrmeta(af_tp->get_npos());
      for (intptr_t i = 0; i < af_tp->get_npos(); ++i) {
        src_arrmeta[i] = args[i].get_arrmeta();
      }
      std::vector<char *> src_data(af_tp->get_npos());
      for (intptr_t i = 0; i < af_tp->get_npos(); ++i) {
        src_data[i] = const_cast<char *>(args[i].get_readonly_originptr());
      }

      // Generate and evaluate the ckernel
      ckernel_builder<kernel_request_host> ckb;
      af->instantiate(af, af_tp, &ckb, 0, out.get_type(), out.get_arrmeta(),
                      &arg_tp[0], &src_arrmeta[0], kernel_request_single, ectx,
                      array());
      expr_single_t fn = ckb.get()->get_function<expr_single_t>();
      fn(out.get_readwrite_originptr(), src_data.empty() ? NULL : &src_data[0],
         ckb.get());
    }
    void call_out(intptr_t arg_count, const nd::array *args,
                  const nd::array &out, const eval::eval_context *ectx) const
    {
      call_out(arg_count, args, kwds(), out, ectx);
    }

    /** Convenience call operators with output parameter */
    void call_out(const nd::array &out) const
    {
      call_out(0, NULL, out, &eval::default_eval_context);
    }
    template <typename... K>
    void call_out(const nd::array &a0, const nd::array &out,
                  const detail::kwds<K...> &kwds_ = kwds()) const
    {
      call_out(1, &a0, kwds_, out, &eval::default_eval_context);
    }
    template <typename... K>
    void call_out(const nd::array &a0, const nd::array &a1,
                  const nd::array &out,
                  const detail::kwds<K...> &kwds_ = kwds()) const
    {
      nd::array args[2] = {a0, a1};
      call_out(2, args, kwds_, out, &eval::default_eval_context);
    }
    template <typename... K>
    void call_out(const nd::array &a0, const nd::array &a1, const nd::array &a2,
                  const nd::array &out,
                  const detail::kwds<K...> &kwds_ = kwds()) const
    {
      nd::array args[3] = {a0, a1, a2};
      call_out(3, args, kwds_, out, &eval::default_eval_context);
    }
    template <typename... K>
    void call_out(const nd::array &a0, const nd::array &a1, const nd::array &a2,
                  const nd::array &a3, nd::array &out,
                  const detail::kwds<K...> &kwds_ = kwds()) const
    {
      nd::array args[4] = {a0, a1, a2, a3};
      call_out(4, args, kwds_, out, &eval::default_eval_context);
    }
  };

  /**
   * This is a helper class for creating static nd::arrfunc instances
   * whose lifetime is managed by init/cleanup functions. When declared
   * as a global static variable, because it is a POD type, this will begin with
   * the value NULL. It can generally be treated just like an nd::arrfunc,
   * though
   * its internals are not protected from meddling.
   */
  struct pod_arrfunc {
    memory_block_data *m_memblock;

    operator const nd::arrfunc &()
    {
      return *reinterpret_cast<const nd::arrfunc *>(&m_memblock);
    }

    const arrfunc_type_data *get() const
    {
      return reinterpret_cast<const nd::arrfunc *>(&m_memblock)->get();
    }

    const arrfunc_type *get_type() const
    {
      return reinterpret_cast<const nd::arrfunc *>(&m_memblock)->get_type();
    }

    void init(const nd::arrfunc &rhs)
    {
      m_memblock = nd::array(rhs).get_memblock().get();
      memory_block_incref(m_memblock);
    }

    void cleanup()
    {
      if (m_memblock) {
        memory_block_decref(m_memblock);
        m_memblock = NULL;
      }
    }
  };

  template <typename T>
  arrfunc make_arrfunc(ndt::type af_tp, arrfunc_instantiate_t instantiate,
                       const T &data)
  {
    array af = empty(af_tp);
    arrfunc_type_data *out_af =
        reinterpret_cast<arrfunc_type_data *>(af.get_readwrite_originptr());
    out_af->instantiate = instantiate;
    *out_af->get_data_as<T>() = data;
    af.flag_as_immutable();

    return af;
  }

} // namespace nd

/**
 * Creates an arrfunc which does the assignment from
 * data of src_tp to dst_tp.
 *
 * \param dst_tp  The type of the destination.
 * \param src_tp  The type of the source.
 * \param errmode  The error mode to use for the assignment.
 */
nd::arrfunc make_arrfunc_from_assignment(const ndt::type &dst_tp,
                                         const ndt::type &src_tp,
                                         assign_error_mode errmode);

/**
 * Creates an arrfunc which does the assignment from
 * data of `tp` to its property `propname`
 *
 * \param tp  The type of the source.
 * \param propname  The name of the property.
 */
nd::arrfunc make_arrfunc_from_property(const ndt::type &tp,
                                       const std::string &propname);

} // namespace dynd

#include <dynd/types/option_type.hpp>