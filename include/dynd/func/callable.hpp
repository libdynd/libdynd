//
// Copyright (C) 2011-15 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#pragma once

#include <memory>

#include <dynd/config.hpp>
#include <dynd/eval/eval_context.hpp>
#include <dynd/types/base_type.hpp>
#include <dynd/types/callable_type.hpp>
#include <dynd/kernels/ckernel_builder.hpp>
#include <dynd/kernels/base_kernel.hpp>
#include <dynd/types/struct_type.hpp>
#include <dynd/types/substitute_typevars.hpp>
#include <dynd/types/type_type.hpp>
#include <dynd/callables/static_data_callable.hpp>

namespace dynd {

/**
 * TODO: This `as_array` metafunction should either go somewhere better (this
 *       file is for callable), or be in a detail:: namespace.
 */
template <typename T>
struct as_array {
  typedef nd::array type;
};

template <>
struct as_array<nd::array> {
  typedef nd::array type;
};

template <>
struct as_array<const nd::array> {
  typedef const nd::array type;
};

template <>
struct as_array<const nd::array &> {
  typedef const nd::array &type;
};

template <>
struct as_array<nd::array &> {
  typedef nd::array &type;
};

namespace nd {
  namespace detail {

    /**
     * Presently, there are some specially treated keyword arguments in
     * arrfuncs. The "dst_tp" keyword argument always tells the desired
     * output type, and the "dst" keyword argument always provides an
     * output array.
     */
    template <typename T>
    bool is_special_kwd(const ndt::callable_type *DYND_UNUSED(self_tp), const std::string &DYND_UNUSED(name),
                        const T &DYND_UNUSED(value), std::map<std::string, ndt::type> &DYND_UNUSED(tp_vars))
    {
      return false;
    }

    inline bool is_special_kwd(const ndt::callable_type *DYND_UNUSED(self_tp), array &dst, const std::string &name,
                               const ndt::type &value)
    {
      if (name == "dst_tp") {
        dst = nd::empty(value);
        return true;
      }

      return false;
    }

    inline bool is_special_kwd(const ndt::callable_type *DYND_UNUSED(self_tp), array &dst, const std::string &name,
                               const nd::array &value)
    {
      if (name == "dst_tp") {
        dst = nd::empty(value.as<ndt::type>());
        return true;
      }
      else if (name == "dst") {
        dst = value;
        return true;
      }

      return false;
    }

    template <typename T>
    void check_name(const ndt::callable_type *af_tp, array &dst, const std::string &name, const T &value,
                    bool &has_dst_tp, ndt::type *kwd_tp, std::vector<intptr_t> &available)
    {
      intptr_t j = af_tp->get_kwd_index(name);
      if (j == -1) {
        if (is_special_kwd(af_tp, dst, name, value)) {
          has_dst_tp = true;
        }
        else {
          std::stringstream ss;
          ss << "passed an unexpected keyword \"" << name << "\" to callable with type " << ndt::type(af_tp, true);
          throw std::invalid_argument(ss.str());
        }
      }
      else {
        ndt::type &actual_tp = kwd_tp[j];
        if (!actual_tp.is_null()) {
          std::stringstream ss;
          ss << "callable passed keyword \"" << name << "\" more than once";
          throw std::invalid_argument(ss.str());
        }
        actual_tp = ndt::type_of(value);
      }
      available.push_back(j);
    }

    DYND_API void fill_missing_values(const ndt::type *tp, char *arrmeta, const uintptr_t *arrmeta_offsets, char *data,
                                      const uintptr_t *data_offsets,
                                      std::vector<nd::array> &DYND_UNUSED(kwds_as_vector),
                                      const std::vector<intptr_t> &missing);

    DYND_API void check_narg(const ndt::callable_type *af_tp, intptr_t npos);

    DYND_API void check_arg(const ndt::callable_type *af_tp, intptr_t i, const ndt::type &actual_tp,
                            const char *actual_arrmeta, std::map<std::string, ndt::type> &tp_vars);

    DYND_API void check_nkwd(const ndt::callable_type *af_tp, const std::vector<intptr_t> &available,
                             const std::vector<intptr_t> &missing);

    DYND_API void validate_kwd_types(const ndt::callable_type *af_tp, std::vector<ndt::type> &kwd_tp,
                                     const std::vector<intptr_t> &available, const std::vector<intptr_t> &missing,
                                     std::map<std::string, ndt::type> &tp_vars);

    inline void set_data(char *&data, array &value) { data = const_cast<char *>(value.cdata()); }

    inline void set_data(char *&data, const array &value) { data = const_cast<char *>(value.cdata()); }

    inline void set_data(array *&data, array &value) { data = &value; }

    inline void set_data(array *&data, const array &value) { data = const_cast<array *>(&value); }

    /** A holder class for the keyword arguments */
    template <typename... K>
    class kwds;

    /** The case of no keyword arguments being provided */
    template <>
    class kwds<> {
      void fill_values(const ndt::type *tp, char *arrmeta, const uintptr_t *arrmeta_offsets, char *data,
                       const uintptr_t *data_offsets, std::vector<nd::array> &kwds_as_vector,
                       const std::vector<intptr_t> &DYND_UNUSED(available), const std::vector<intptr_t> &missing) const
      {
        fill_missing_values(tp, arrmeta, arrmeta_offsets, data, data_offsets, kwds_as_vector, missing);
      }

    public:
      void validate_names(const ndt::callable_type *af_tp, array &DYND_UNUSED(dst),
                          std::vector<ndt::type> &DYND_UNUSED(tp), std::vector<intptr_t> &available,
                          std::vector<intptr_t> &missing) const
      {
        // No keywords provided, so all are missing
        for (intptr_t j : af_tp->get_option_kwd_indices()) {
          missing.push_back(j);
        }

        check_nkwd(af_tp, available, missing);
      }

      /** Converts the keyword args + filled in defaults into an nd::array */
      array as_array(const ndt::type &tp, std::vector<nd::array> &kwds_as_vector,
                     const std::vector<intptr_t> &available, const std::vector<intptr_t> &missing) const
      {
        array res = empty_shell(tp);
        ndt::struct_type::fill_default_data_offsets(res.get_dim_size(),
                                                    tp.extended<ndt::base_struct_type>()->get_field_types_raw(),
                                                    reinterpret_cast<uintptr_t *>(res.get()->metadata()));

        char *arrmeta = res.get()->metadata();
        const uintptr_t *arrmeta_offsets = res.get_type().extended<ndt::base_struct_type>()->get_arrmeta_offsets_raw();
        char *data = res.data();
        const uintptr_t *data_offsets =
            res.get_type().extended<ndt::base_struct_type>()->get_data_offsets(res.get()->metadata());

        fill_values(tp.extended<ndt::base_struct_type>()->get_field_types_raw(), arrmeta, arrmeta_offsets, data,
                    data_offsets, kwds_as_vector, available, missing);

        return res;
      }
    };

    template <typename... K>
    class kwds {
      const char *m_names[sizeof...(K)];
      std::tuple<K...> m_values;

      struct {
        kwds *self;

        template <size_t I>
        void on_each(typename as_<K, const char *>::type... names)
        {
          self->m_names[I] = get<I>(names...);
        }

        void operator()(typename as_<K, const char *>::type... names)
        {
          typedef make_index_sequence<sizeof...(K)> I;
          for_each<I>(*this, names...);
        }
      } set_names;

      struct {
        kwds *self;

        template <size_t I>
        void on_each(const ndt::type *tp, char *arrmeta, const uintptr_t *arrmeta_offsets, char *data,
                     const uintptr_t *data_offsets, std::vector<nd::array> &kwds_as_vector,
                     const std::vector<intptr_t> &available) const
        {
          intptr_t j = available[I];
          if (j != -1) {
            nd::forward_as_array(tp[j], arrmeta + arrmeta_offsets[j], data + data_offsets[j],
                                 std::get<I>(self->m_values));
            kwds_as_vector[j] = nd::array(std::get<I>(self->m_values));
          }
        }

        void operator()(const ndt::type *tp, char *arrmeta, const uintptr_t *arrmeta_offsets, char *data,
                        const uintptr_t *data_offsets, std::vector<nd::array> &kwds_as_vector,
                        const std::vector<intptr_t> &available) const
        {
          typedef make_index_sequence<sizeof...(K)> I;
          for_each<I>(*this, tp, arrmeta, arrmeta_offsets, data, data_offsets, kwds_as_vector, available);
        }
      } fill_available_values;

      void fill_values(const ndt::type *tp, char *arrmeta, const uintptr_t *arrmeta_offsets, char *data,
                       const uintptr_t *data_offsets, std::vector<nd::array> &kwds_as_vector,
                       const std::vector<intptr_t> &available, const std::vector<intptr_t> &missing) const
      {
        fill_available_values(tp, arrmeta, arrmeta_offsets, data, data_offsets, kwds_as_vector, available);
        fill_missing_values(tp, arrmeta, arrmeta_offsets, data, data_offsets, kwds_as_vector, missing);
      }

    public:
      kwds(typename as_<K, const char *>::type... names, K &&... values) : m_values(std::forward<K>(values)...)
      {
        set_names.self = this;
        validate_names.self = this;
        fill_available_values.self = this;

        set_names(names...);
      }

      struct {
        kwds *self;

        template <size_t I>
        void on_each(const ndt::callable_type *af_tp, array &dst, bool &has_dst_tp, std::vector<ndt::type> &kwd_tp,
                     std::vector<intptr_t> &available) const
        {
          check_name(af_tp, dst, self->m_names[I], std::get<I>(self->m_values), has_dst_tp, kwd_tp.data(), available);
        }

        void operator()(const ndt::callable_type *af_tp, array &dst, std::vector<ndt::type> &tp,
                        std::vector<intptr_t> &available, std::vector<intptr_t> &missing) const
        {
          bool has_dst_tp = false;

          typedef make_index_sequence<sizeof...(K)> I;
          for_each<I>(*this, af_tp, dst, has_dst_tp, tp, available);

          intptr_t nkwd = sizeof...(K);
          if (has_dst_tp) {
            nkwd--;
          }

          for (intptr_t j : af_tp->get_option_kwd_indices()) {
            if (tp[j].is_null()) {
              missing.push_back(j);
            }
          }

          check_nkwd(af_tp, available, missing);
        }
      } validate_names;

      array as_array(const ndt::type &tp, std::vector<nd::array> &kwds_as_vector,
                     const std::vector<intptr_t> &available, const std::vector<intptr_t> &missing) const
      {
        array res = empty_shell(tp);
        ndt::struct_type::fill_default_data_offsets(res.get_dim_size(),
                                                    tp.extended<ndt::base_struct_type>()->get_field_types_raw(),
                                                    reinterpret_cast<uintptr_t *>(res.get()->metadata()));

        fill_values(tp.extended<ndt::base_struct_type>()->get_field_types_raw(), res.get()->metadata(),
                    res.get_type().extended<ndt::base_struct_type>()->get_arrmeta_offsets_raw(), res.data(),
                    res.get_type().extended<ndt::base_struct_type>()->get_data_offsets(res.get()->metadata()),
                    kwds_as_vector, available, missing);

        return res;
      }
    };

    template <>
    class kwds<intptr_t, const char *const *, array *> {
      intptr_t m_size;
      const char *const *m_names;
      array *m_values;

      void fill_available_values(const ndt::type *tp, char *arrmeta, const uintptr_t *arrmeta_offsets, char *data,
                                 const uintptr_t *data_offsets, std::vector<nd::array> &kwds_as_vector,
                                 const std::vector<intptr_t> &available) const
      {
        for (intptr_t i = 0; i < m_size; ++i) {
          intptr_t j = available[i];
          if (j != -1) {
            nd::forward_as_array(tp[j], arrmeta + arrmeta_offsets[j], data + data_offsets[j], this->m_values[i]);
            kwds_as_vector[j] = this->m_values[i];
          }
        }
      }

      void fill_values(const ndt::type *tp, char *arrmeta, const uintptr_t *arrmeta_offsets, char *data,
                       const uintptr_t *data_offsets, std::vector<nd::array> &kwds_as_vector,
                       const std::vector<intptr_t> &available, const std::vector<intptr_t> &missing) const
      {
        fill_available_values(tp, arrmeta, arrmeta_offsets, data, data_offsets, kwds_as_vector, available);
        fill_missing_values(tp, arrmeta, arrmeta_offsets, data, data_offsets, kwds_as_vector, missing);
      }

    public:
      kwds(intptr_t size, const char *const *names, array *values) : m_size(size), m_names(names), m_values(values) {}

      void validate_names(const ndt::callable_type *af_tp, array &dst, std::vector<ndt::type> &kwd_tp,
                          std::vector<intptr_t> &available, std::vector<intptr_t> &missing) const
      {
        bool has_dst_tp = false;

        for (intptr_t i = 0; i < m_size; ++i) {
          check_name(af_tp, dst, m_names[i], m_values[i], has_dst_tp, kwd_tp.data(), available);
        }

        intptr_t nkwd = m_size;
        if (has_dst_tp) {
          nkwd--;
        }

        for (intptr_t j : af_tp->get_option_kwd_indices()) {
          if (kwd_tp[j].is_null()) {
            missing.push_back(j);
          }
        }

        check_nkwd(af_tp, available, missing);
      }

      array as_array(const ndt::type &tp, std::vector<nd::array> &kwds_as_vector,
                     const std::vector<intptr_t> &available, const std::vector<intptr_t> &missing) const
      {
        array res = empty_shell(tp);
        auto field_count = tp.extended<ndt::base_struct_type>()->get_field_count();
        auto field_types = tp.extended<ndt::base_struct_type>()->get_field_types_raw();
        ndt::struct_type::fill_default_data_offsets(field_count, field_types,
                                                    reinterpret_cast<uintptr_t *>(res.get()->metadata()));

        fill_values(field_types, res.get()->metadata(),
                    res.get_type().extended<ndt::base_struct_type>()->get_arrmeta_offsets_raw(), res.data(),
                    res.get_type().extended<ndt::base_struct_type>()->get_data_offsets(res.get()->metadata()),
                    kwds_as_vector, available, missing);

        return res;
      }
    };

    template <typename... T>
    struct is_variadic_kwds {
      enum { value = true };
    };

    template <typename T0, typename T1, typename T2>
    struct is_variadic_kwds<T0, T1, T2> {
      enum {
        value = !std::is_convertible<T0, intptr_t>::value || !std::is_convertible<T1, const char *const *>::value ||
                !std::is_convertible<T2, nd::array *>::value
      };
    };

    template <typename... T>
    struct as_kwds {
      typedef typename instantiate<
          nd::detail::kwds,
          typename dynd::take<type_sequence<T...>, make_index_sequence<1, sizeof...(T), 2>>::type>::type type;
    };
  } // namespace dynd::nd::detail

  /**
   * A function to provide keyword arguments to an callable. The arguments
   * must alternate between the keyword name and the argument value.
   *
   *   callable af = <some callable>;
   *   af(arr1, arr2, kwds("firstkwarg", kwval1, "second", kwval2));
   */
  template <typename... T>
  typename std::enable_if<detail::is_variadic_kwds<T...>::value, typename detail::as_kwds<T...>::type>::type
  kwds(T &&... t)
  {
    // Sequence of even integers, for extracting the keyword names
    typedef make_index_sequence<0, sizeof...(T), 2> I;
    // Sequence of odd integers, for extracting the keyword values
    typedef make_index_sequence<1, sizeof...(T), 2> J;

    return make_with<typename join<I, J>::type, decltype(kwds(std::forward<T>(t)...))>(std::forward<T>(t)...);
  }

  /**
   * A special way to provide the keyword argument as an array of
   * names and an array of nd::array values.
   */
  template <typename... T>
  typename std::enable_if<!detail::is_variadic_kwds<T...>::value,
                          detail::kwds<intptr_t, const char *const *, array *>>::type
  kwds(T &&... t)
  {
    return detail::kwds<intptr_t, const char *const *, array *>(std::forward<T>(t)...);
  }

  /**
   * Empty keyword args.
   */
  inline detail::kwds<> kwds() { return detail::kwds<>(); }

} // namespace dynd::nd

using nd::kwds;

namespace nd {
  namespace detail {

    DYND_HAS(data_size);
    DYND_HAS(data_init);
    DYND_HAS(resolve_dst_type);
    DYND_HAS(instantiate);
    DYND_HAS(static_data_free);

    DYND_GET(static_data_free, callable_static_data_free_t, NULL);

    template <typename KernelType>
    typename std::enable_if<!has_member_single<KernelType, void(array *, array *const *)>::value,
                            kernel_request_t>::type
    get_kernreq()
    {
      return kernel_request_single;
    }

    template <typename KernelType>
    typename std::enable_if<has_member_single<KernelType, void(array *, array *const *)>::value, kernel_request_t>::type
    get_kernreq()
    {
      return kernel_request_array;
    }

    template <typename KernelType>
    typename std::enable_if<has_member_single<KernelType, void(char *, char *const *)>::value, single_t>::type
    get_single_t()
    {
      return single_t(&KernelType::single_wrapper::func, KernelType::single_wrapper::ir);
    }

    template <typename KernelType>
    typename std::enable_if<!has_member_single<KernelType, void(char *, char *const *)>::value, kernel_targets_t>::type
    get_targets()
    {
      return kernel_targets_t{NULL, NULL, NULL};
    }

    template <typename KernelType>
    typename std::enable_if<has_member_single<KernelType, void(char *, char *const *)>::value, kernel_targets_t>::type
    get_targets()
    {
      return kernel_targets_t{reinterpret_cast<void *>(static_cast<void (*)(ckernel_prefix *, char *, char *const *)>(
                                  KernelType::single_wrapper)),
                              NULL, reinterpret_cast<void *>(KernelType::strided_wrapper)};
    }

    template <typename KernelType>
    typename std::enable_if<has_member_single<KernelType, void(char *, char *const *)>::value,
                            const volatile char *>::type
    get_ir()
    {
      return KernelType::ir;
    }

    template <typename KernelType>
    typename std::enable_if<!has_member_single<KernelType, void(char *, char *const *)>::value,
                            const volatile char *>::type
    get_ir()
    {
      return NULL;
    }

    template <typename KernelType>
    typename std::enable_if<std::is_same<decltype(&KernelType::instantiate), callable_instantiate_t>::value,
                            callable_instantiate_t>::type
    get_instantiate()
    {
      return &KernelType::instantiate;
    }

    template <typename KernelType>
    typename std::enable_if<!has_data_init<KernelType>::value, callable_data_init_t>::type get_data_init()
    {
      return NULL;
    }

    template <typename KernelType>
    typename std::enable_if<has_data_init<KernelType>::value &&
                                std::is_same<decltype(&KernelType::data_init), callable_data_init_t>::value,
                            callable_data_init_t>::type
    get_data_init()
    {
      return &KernelType::data_init;
    }

    template <typename KernelType>
    typename std::enable_if<!has_resolve_dst_type<KernelType>::value, callable_resolve_dst_type_t>::type
    get_resolve_dst_type()
    {
      return NULL;
    }

    template <typename KernelType>
    typename std::enable_if<
        has_resolve_dst_type<KernelType>::value &&
            std::is_same<decltype(&KernelType::resolve_dst_type), callable_resolve_dst_type_t>::value,
        callable_resolve_dst_type_t>::type
    get_resolve_dst_type()
    {
      return &KernelType::resolve_dst_type;
    }

    template <template <type_id_t...> class KernelType>
    struct make_all;

  } // namespace dynd::nd::detail

  /**
   * Holds a single instance of an callable in an nd::array,
   * providing some more direct convenient interface.
   */
  class DYND_API callable : public intrusive_ptr<base_callable> {
    template <typename DataType, typename... A>
    class args;

    template <typename... A>
    struct has_kwds;

  public:
    using intrusive_ptr<base_callable>::intrusive_ptr;

    callable() = default;

    callable(const ndt::type &self_tp, expr_single_t single, expr_strided_t strided)
        : intrusive_ptr<base_callable>(
              new (sizeof(kernel_targets_t)) base_callable(
                  self_tp, kernel_targets_t{reinterpret_cast<void *>(single), NULL, reinterpret_cast<void *>(strided)}),
              true)
    {
    }

    callable(const ndt::type &self_tp, kernel_request_t kernreq, kernel_targets_t targets, const volatile char *ir,
             callable_data_init_t data_init, callable_resolve_dst_type_t resolve_dst_type,
             callable_instantiate_t instantiate)
        : intrusive_ptr<base_callable>(
              new base_callable(self_tp, kernreq, targets, ir, data_init, resolve_dst_type, instantiate), true)
    {
    }

    template <typename T>
    callable(const ndt::type &self_tp, kernel_request_t kernreq, kernel_targets_t targets, const volatile char *ir,
             T &&static_data, callable_data_init_t data_init, callable_resolve_dst_type_t resolve_dst_type,
             callable_instantiate_t instantiate)
        : intrusive_ptr<base_callable>(new static_data_callable<T>(self_tp, kernreq, targets, ir, data_init,
                                                                   resolve_dst_type, instantiate,
                                                                   std::forward<T>(static_data)),
                                       true)

    {
    }

    bool is_null() const { return get() == NULL; }

    callable_property get_flags() const { return right_associative; }

    const ndt::callable_type *get_type() const
    {
      if (get() == NULL) {
        return NULL;
      }

      return get()->tp.extended<ndt::callable_type>();
    }

    const ndt::type &get_array_type() const { return get()->tp; }

    const ndt::type &get_ret_type() const { return get_type()->get_return_type(); }

    std::intptr_t get_narg() const { return get_type()->get_npos(); }

    const ndt::type &get_arg_type(std::intptr_t i) const { return get_type()->get_pos_type(i); }

    const array &get_arg_types() const { return get_type()->get_pos_types(); }

    /** Implements the general call operator which returns an array */
    template <typename ArgsType, typename KwdsType>
    array call(const ArgsType &args, const KwdsType &kwds, std::map<std::string, ndt::type> &tp_vars)
    {
      const ndt::callable_type *self_tp = get_type();

      array dst;

      // ...
      std::vector<ndt::type> kwd_tp(self_tp->get_nkwd());
      std::vector<intptr_t> available, missing;
      kwds.validate_names(self_tp, dst, kwd_tp, available, missing);

      // Validate the destination type, if it was provided
      if (!dst.is_null()) {
        if (!self_tp->get_return_type().match(NULL, dst.get_type(), dst.get()->metadata(), tp_vars)) {
          std::stringstream ss;
          ss << "provided \"dst\" type " << dst.get_type() << " does not match callable return type "
             << self_tp->get_return_type();
          throw std::invalid_argument(ss.str());
        }
      }

      // Validate the keyword arguments, and does substitutions to make
      // them concrete
      detail::validate_kwd_types(self_tp, kwd_tp, available, missing, tp_vars);

      // ...
      std::vector<nd::array> kwds_as_vector(available.size() + missing.size());
      array kwds_as_array =
          kwds.as_array(ndt::struct_type::make(self_tp->get_kwd_names(), kwd_tp), kwds_as_vector, available, missing);

      ndt::type dst_tp;
      if (dst.is_null()) {
        dst_tp = self_tp->get_return_type();
        return (*get())(dst_tp, args.size(), args.types(), args.arrmeta(), args.data(), kwds_as_vector.size(),
                        kwds_as_vector.data(), tp_vars);
      }

      dst_tp = dst.get_type();
      (*get())(dst_tp, dst.get()->metadata(), dst.data(), args.size(), args.types(), args.arrmeta(), args.data(),
               kwds_as_vector.size(), kwds_as_vector.data(), tp_vars);
      return dst;
    }

    /**
    * operator()(kwds<...>(...))
    */
    template <template <typename...> class ArgsType, typename AT0, typename... K>
    array _call(detail::kwds<K...> &&k)
    {
      std::map<std::string, ndt::type> tp_vars;
      return call(ArgsType<AT0>(tp_vars, get_type()), std::forward<detail::kwds<K...>>(k), tp_vars);
    }

    /**
     * operator()(a0, a1, ..., an, kwds<...>(...))
     */
    template <template <typename...> class ArgsType, typename AT0, typename... T>
    typename std::enable_if<sizeof...(T) != 3, array>::type _call(T &&... a)
    {
      std::map<std::string, ndt::type> tp_vars;

      typedef typename instantiate<ArgsType, typename to<type_sequence<AT0, T...>, sizeof...(T)>::type>::type args_type;
      typedef make_index_sequence<sizeof...(T) + 1> I;
      return call(make_with<I, args_type>(tp_vars, get_type(), std::forward<T>(a)...),
                  dynd::get<sizeof...(T)-1>(std::forward<T>(a)...), tp_vars);
    }

    template <template <typename...> class ArgsType, typename AT0, typename A0, typename A1, typename... K>
    typename std::enable_if<!std::is_convertible<A0 &&, size_t>::value || !std::is_convertible<A1 &&, array *>::value,
                            array>::type
    _call(A0 &&a0, A1 &&a1, const detail::kwds<K...> &kwds)
    {
      std::map<std::string, ndt::type> tp_vars;
      return call(
          ArgsType<AT0, array, array>(tp_vars, get_type(), array(std::forward<A0>(a0)), array(std::forward<A1>(a1))),
          kwds, tp_vars);
    }

    template <template <typename...> class ArgsType, typename AT0, typename A0, typename A1, typename... K>
    typename std::enable_if<std::is_convertible<A0 &&, size_t>::value && std::is_convertible<A1 &&, array *>::value,
                            array>::type
    _call(A0 &&a0, A1 &&a1, const detail::kwds<K...> &kwds)
    {
      std::map<std::string, ndt::type> tp_vars;
      return call(ArgsType<AT0, size_t, array *>(tp_vars, get_type(), std::forward<A0>(a0), std::forward<A1>(a1)), kwds,
                  tp_vars);
    }

    template <typename... A>
    typename std::enable_if<has_kwds<A...>::value, array>::type operator()(A &&... a)
    {
      if (get()->kernreq == kernel_request_single) {
        return _call<args, char *>(std::forward<A>(a)...);
      }

      return _call<args, array *>(std::forward<A>(a)...);
    }

    template <typename... A>
    typename std::enable_if<!has_kwds<A...>::value, array>::type operator()(A &&... a)
    {
      return (*this)(std::forward<A>(a)..., kwds());
    }

    template <typename KernelType>
    static typename std::enable_if<
        ndt::type::has_equivalent<KernelType>::value && detail::has_data_size<KernelType>::value, callable>::type
    make()
    {
      return callable(ndt::type::equivalent<KernelType>::make(), detail::get_kernreq<KernelType>(),
                      detail::get_targets<KernelType>(), detail::get_ir<KernelType>(),
                      detail::get_data_init<KernelType>(), detail::get_resolve_dst_type<KernelType>(),
                      detail::get_instantiate<KernelType>());
    }

    template <typename KernelType, typename StaticDataType>
    static typename std::enable_if<
        ndt::type::has_equivalent<KernelType>::value && detail::has_data_size<KernelType>::value, callable>::type
    make(StaticDataType &&static_data)
    {
      return callable(ndt::type::equivalent<KernelType>::make(), detail::get_kernreq<KernelType>(),
                      detail::get_targets<KernelType>(), detail::get_ir<KernelType>(),
                      std::forward<StaticDataType>(static_data), detail::get_data_init<KernelType>(),
                      detail::get_resolve_dst_type<KernelType>(), detail::get_instantiate<KernelType>());
    }

    template <typename KernelType>
    static typename std::enable_if<
        ndt::type::has_equivalent<KernelType>::value && !detail::has_data_size<KernelType>::value, callable>::type
    make()
    {
      return callable(ndt::type::equivalent<KernelType>::make(), detail::get_kernreq<KernelType>(),
                      detail::get_targets<KernelType>(), detail::get_ir<KernelType>(),
                      detail::get_data_init<KernelType>(), detail::get_resolve_dst_type<KernelType>(),
                      detail::get_instantiate<KernelType>());
    }

    template <typename KernelType, typename StaticDataType>
    static typename std::enable_if<
        ndt::type::has_equivalent<KernelType>::value && !detail::has_data_size<KernelType>::value, callable>::type
    make(StaticDataType &&static_data)
    {
      return callable(ndt::type::equivalent<KernelType>::make(), detail::get_kernreq<KernelType>(),
                      detail::get_targets<KernelType>(), detail::get_ir<KernelType>(),
                      std::forward<StaticDataType>(static_data), detail::get_data_init<KernelType>(),
                      detail::get_resolve_dst_type<KernelType>(), detail::get_instantiate<KernelType>());
    }

    template <typename KernelType>
    static typename std::enable_if<
        !ndt::type::has_equivalent<KernelType>::value && detail::has_data_size<KernelType>::value, callable>::type
    make(const ndt::type &self_tp)
    {
      return callable(self_tp, detail::get_kernreq<KernelType>(), detail::get_targets<KernelType>(),
                      detail::get_ir<KernelType>(), detail::get_data_init<KernelType>(),
                      detail::get_resolve_dst_type<KernelType>(), detail::get_instantiate<KernelType>());
    }

    template <typename KernelType, typename StaticDataType>
    static typename std::enable_if<
        !ndt::type::has_equivalent<KernelType>::value && detail::has_data_size<KernelType>::value, callable>::type
    make(const ndt::type &self_tp, StaticDataType &&static_data)
    {
      return callable(self_tp, detail::get_kernreq<KernelType>(), detail::get_targets<KernelType>(),
                      detail::get_ir<KernelType>(), std::forward<StaticDataType>(static_data),
                      detail::get_data_init<KernelType>(), detail::get_resolve_dst_type<KernelType>(),
                      detail::get_instantiate<KernelType>());
    }

    template <typename KernelType>
    static typename std::enable_if<
        !ndt::type::has_equivalent<KernelType>::value && !detail::has_data_size<KernelType>::value, callable>::type
    make(const ndt::type &self_tp)
    {
      return callable(self_tp, detail::get_kernreq<KernelType>(), detail::get_targets<KernelType>(),
                      detail::get_ir<KernelType>(), detail::get_data_init<KernelType>(),
                      detail::get_resolve_dst_type<KernelType>(), detail::get_instantiate<KernelType>());
    }

    template <typename KernelType, typename StaticDataType>
    static typename std::enable_if<
        !ndt::type::has_equivalent<KernelType>::value && !detail::has_data_size<KernelType>::value, callable>::type
    make(const ndt::type &self_tp, StaticDataType &&static_data)
    {
      return callable(self_tp, detail::get_kernreq<KernelType>(), detail::get_targets<KernelType>(),
                      detail::get_ir<KernelType>(), std::forward<StaticDataType>(static_data),
                      detail::get_data_init<KernelType>(), detail::get_resolve_dst_type<KernelType>(),
                      detail::get_instantiate<KernelType>());
    }

    template <template <int> class CKT, typename T>
    static callable make(const ndt::type &self_tp, T &&data)
    {
      switch (self_tp.extended<ndt::callable_type>()->get_npos()) {
      case 0:
        return make<CKT<0>>(self_tp, std::forward<T>(data));
      case 1:
        return make<CKT<1>>(self_tp, std::forward<T>(data));
      case 2:
        return make<CKT<2>>(self_tp, std::forward<T>(data));
      case 3:
        return make<CKT<3>>(self_tp, std::forward<T>(data));
      case 4:
        return make<CKT<4>>(self_tp, std::forward<T>(data));
      case 5:
        return make<CKT<5>>(self_tp, std::forward<T>(data));
      case 6:
        return make<CKT<6>>(self_tp, std::forward<T>(data));
      case 7:
        return make<CKT<7>>(self_tp, std::forward<T>(data));
      default:
        throw std::runtime_error("callable with nsrc > 7 not implemented yet");
      }
    }

    template <template <type_id_t> class KernelType, typename I0, typename... A>
    static std::map<type_id_t, callable> make_all(A &&... a)
    {
      std::map<type_id_t, callable> callables;
      for_each<I0>(detail::make_all<KernelType>(), callables, std::forward<A>(a)...);

      return callables;
    }

    template <template <type_id_t, type_id_t, type_id_t...> class KernelType, typename I0, typename I1, typename... I,
              typename... A>
    static std::map<std::array<type_id_t, 2 + sizeof...(I)>, callable> make_all(A &&... a)
    {
      std::map<std::array<type_id_t, 2 + sizeof...(I)>, callable> callables;
      for_each<typename outer<I0, I1, I...>::type>(detail::make_all<KernelType>(), callables, std::forward<A>(a)...);

      return callables;
    }
  };

  inline std::ostream &operator<<(std::ostream &o, const callable &rhs)
  {
    return o << "<callable <" << rhs.get()->tp << "> at " << reinterpret_cast<const void *>(rhs.get()) << ">";
  }

  template <typename DataType>
  class callable::args<DataType> {
  public:
    args(std::map<std::string, ndt::type> &DYND_UNUSED(tp_vars), const ndt::callable_type *self_tp)
    {
      detail::check_narg(self_tp, 0);
    }

    size_t size() const { return 0; }

    const ndt::type *types() const { return NULL; }

    const char *const *arrmeta() const { return NULL; }

    DataType const *data() const { return NULL; }
  };

  /** A holder class for the array arguments */
  template <typename DataType, typename... A>
  class callable::args {
    struct init {
      template <size_t I>
      void on_each(const args *self, const ndt::callable_type *af_tp, ndt::type *src_tp, const char **src_arrmeta,
                   DataType *src_data, std::map<std::string, ndt::type> &tp_vars) const
      {
        auto &value = std::get<I>(self->m_values);
        const ndt::type &tp = ndt::type::make<decltype(value)>(value);
        const char *arrmeta = value.get()->metadata();

        detail::check_arg(af_tp, I, tp, arrmeta, tp_vars);

        src_tp[I] = tp;
        src_arrmeta[I] = arrmeta;
        detail::set_data(src_data[I], value);
      }
    };

    std::tuple<typename as_array<A>::type...> m_values;
    ndt::type m_tp[sizeof...(A)];
    const char *m_arrmeta[sizeof...(A)];
    DataType m_data[sizeof...(A)];

  public:
    args(std::map<std::string, ndt::type> &tp_vars, const ndt::callable_type *self_tp, A &&... a)
        : m_values(std::forward<A>(a)...)
    {
      detail::check_narg(self_tp, sizeof...(A));

      typedef make_index_sequence<sizeof...(A)> I;
      for_each<I>(init(), this, self_tp, m_tp, m_arrmeta, m_data, tp_vars);
    }

    size_t size() const { return sizeof...(A); }

    const ndt::type *types() const { return m_tp; }

    const char *const *arrmeta() const { return m_arrmeta; }

    DataType const *data() const { return m_data; }
  };

  /** A way to pass a run-time array of array arguments */
  template <typename DataType>
  class callable::args<DataType, size_t, array *> {
    size_t m_size;
    std::vector<ndt::type> m_tp;
    std::vector<const char *> m_arrmeta;
    std::vector<DataType> m_data;

  public:
    args(std::map<std::string, ndt::type> &tp_vars, const ndt::callable_type *self_tp, size_t size, array *values)
        : m_size(size), m_tp(m_size), m_arrmeta(m_size), m_data(m_size)
    {
      detail::check_narg(self_tp, m_size);

      for (std::size_t i = 0; i < m_size; ++i) {
        array &value = values[i];
        const char *arrmeta = value.get()->metadata();
        const ndt::type &tp = value.get_type();

        detail::check_arg(self_tp, i, tp, arrmeta, tp_vars);

        m_tp[i] = tp;
        m_arrmeta[i] = arrmeta;
        detail::set_data(m_data[i], value);
      }
    }

    size_t size() const { return m_size; }

    const ndt::type *types() const { return m_tp.data(); }

    const char *const *arrmeta() const { return m_arrmeta.data(); }

    DataType const *data() const { return m_data.data(); }
  };

  template <>
  struct callable::has_kwds<> {
    static const bool value = false;
  };

  template <typename A0, typename... A>
  struct callable::has_kwds<A0, A...> {
    static const bool value = is_instance<detail::kwds, typename std::decay<A0>::type>::value || has_kwds<A...>::value;
  };

  namespace detail {

    template <template <type_id_t...> class KernelType, typename S>
    struct apply;

    template <template <type_id_t...> class KernelType, type_id_t... I>
    struct apply<KernelType, type_id_sequence<I...>> {
      typedef KernelType<I...> type;
    };

    template <template <type_id_t...> class KernelType>
    struct make_all {
      template <type_id_t TypeID, typename... A>
      void on_each(std::map<type_id_t, callable> &callables, A &&... a) const
      {
        callables[TypeID] = callable::make<KernelType<TypeID>>(std::forward<A>(a)...);
      }

      template <typename TypeIDSequence, typename... A>
      void on_each(std::map<std::array<type_id_t, TypeIDSequence::size2()>, callable> &callables, A &&... a) const
      {
        callables[i2a<TypeIDSequence>()] =
            callable::make<typename apply<KernelType, TypeIDSequence>::type>(std::forward<A>(a)...);
      }
    };

  } // namespace dynd::nd::detail

  template <typename FuncType>
  struct declfunc {
    operator callable &() { return get(); }

    operator const callable &() const { return get(); }

    template <typename... A>
    array operator()(A &&... a)
    {
      return get()(std::forward<A>(a)...);
    }

    static callable &get()
    {
      static callable self = FuncType::make();
      return self;
    }
  };

  template <typename FuncType>
  std::ostream &operator<<(std::ostream &o, const declfunc<FuncType> &rhs)
  {
    o << static_cast<const callable &>(rhs);

    return o;
  }

} // namespace nd

/**
 * Creates an callable which does the assignment from
 * data of src_tp to dst_tp.
 *
 * \param dst_tp  The type of the destination.
 * \param src_tp  The type of the source.
 * \param errmode  The error mode to use for the assignment.
 */
DYND_API nd::callable make_callable_from_assignment(const ndt::type &dst_tp, const ndt::type &src_tp,
                                                    assign_error_mode errmode);

/**
 * Creates an callable which does the assignment from
 * data of `tp` to its property `propname`
 *
 * \param tp  The type of the source.
 * \param propname  The name of the property.
 */
DYND_API nd::callable make_callable_from_property(const ndt::type &tp, const std::string &propname);

} // namespace dynd
