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
#include <dynd/types/substitute_typevars.hpp>
#include <dynd/types/option_type.hpp>
#include <dynd/types/type_type.hpp>
#include <dynd/callables/static_data_callable.hpp>

namespace dynd {
namespace nd {
  namespace detail {

    /**
     * Presently, there are some specially treated keyword arguments in
     * arrfuncs. The "dst_tp" keyword argument always tells the desired
     * output type, and the "dst" keyword argument always provides an
     * output array.
     */
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

    DYND_API void check_narg(const ndt::callable_type *af_tp, intptr_t narg);

    DYND_API void check_arg(const ndt::callable_type *af_tp, intptr_t i, const ndt::type &actual_tp,
                            const char *actual_arrmeta, std::map<std::string, ndt::type> &tp_vars);

    inline void set_data(char *&data, array &value) { data = const_cast<char *>(value.cdata()); }

    inline void set_data(char *&data, const array &value) { data = const_cast<char *>(value.cdata()); }

    inline void set_data(array *&data, array &value) { data = &value; }

    inline void set_data(array *&data, const array &value) { data = const_cast<array *>(&value); }

    /** A holder class for the keyword arguments */
    template <typename... K>
    struct kwds {
      static const size_t size = sizeof...(K);
      const char *m_names[sizeof...(K)];
      array values[sizeof...(K)];

      kwds(typename as_<K, const char *>::type... names, K &&... values)
          : m_names{names...}, values{std::forward<K>(values)...}
      {
      }
    };

    /** The case of no keyword arguments being provided */
    template <>
    struct kwds<> {
      static const size_t size = 0;
      const char *m_names[1];
      array values[1];
    };

    template <>
    struct kwds<intptr_t, const char *const *, array *> {
      size_t size;
      const char *const *m_names;
      array *values;

      kwds(intptr_t size, const char *const *names, array *values) : size(size), m_names(names), values(values) {}
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
    struct args;

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

      intptr_t narg = args.size;

      // ...
      intptr_t nkwd = args.size - self_tp->get_npos();
      if (!self_tp->is_kwd_variadic() && nkwd > self_tp->get_nkwd()) {
        throw std::invalid_argument("too many extra positional arguments");
      }

      std::vector<array> kwds_as_vector(nkwd + self_tp->get_nkwd());
      for (intptr_t i = 0; i < nkwd; ++i) {
        kwds_as_vector[i] = args.values[self_tp->get_npos() + i];
        --narg;
      }

      for (size_t i = 0; i < kwds.size; ++i) {
        intptr_t j = self_tp->get_kwd_index(kwds.m_names[i]);
        if (j == -1) {
          if (detail::is_special_kwd(self_tp, dst, kwds.m_names[i], kwds.values[i])) {
          }
          else {
            std::stringstream ss;
            ss << "passed an unexpected keyword \"" << kwds.m_names[i] << "\" to callable with type " << get()->tp;
            throw std::invalid_argument(ss.str());
          }
        }
        else {
          array &value = kwds_as_vector[j];
          if (!value.is_null()) {
            std::stringstream ss;
            ss << "callable passed keyword \"" << kwds.m_names[i] << "\" more than once";
            throw std::invalid_argument(ss.str());
          }
          value = kwds.values[i];

          ndt::type expected_tp = self_tp->get_kwd_type(j);
          if (expected_tp.get_type_id() == option_type_id) {
            expected_tp = expected_tp.p("value_type").as<ndt::type>();
          }

          const ndt::type &actual_tp = value.get_type();
          if (!expected_tp.match(actual_tp.value_type(), tp_vars)) {
            std::stringstream ss;
            ss << "keyword \"" << self_tp->get_kwd_name(j) << "\" does not match, ";
            ss << "callable expected " << expected_tp << " but passed " << actual_tp;
            throw std::invalid_argument(ss.str());
          }
          ++nkwd;
        }
      }

      // Validate the destination type, if it was provided
      if (!dst.is_null()) {
        if (!self_tp->get_return_type().match(NULL, dst.get_type(), dst.get()->metadata(), tp_vars)) {
          std::stringstream ss;
          ss << "provided \"dst\" type " << dst.get_type() << " does not match callable return type "
             << self_tp->get_return_type();
          throw std::invalid_argument(ss.str());
        }
      }

      for (intptr_t j : self_tp->get_option_kwd_indices()) {
        if (kwds_as_vector[j].is_null()) {
          ndt::type actual_tp = ndt::substitute(self_tp->get_kwd_type(j), tp_vars, false);
          if (actual_tp.is_symbolic()) {
            actual_tp = ndt::option_type::make(ndt::type::make<void>());
          }
          kwds_as_vector[j] = empty(actual_tp);
          kwds_as_vector[j].assign_na();
          ++nkwd;
        }
      }

      if (nkwd < self_tp->get_nkwd()) {
        std::stringstream ss;
        // TODO: Provide the missing keyword parameter names in this error
        //       message
        ss << "callable requires keyword parameters that were not provided. "
              "callable signature "
           << get()->tp;
        throw std::invalid_argument(ss.str());
      }

      ndt::type dst_tp;
      if (dst.is_null()) {
        dst_tp = self_tp->get_return_type();
        return (*get())(dst_tp, narg, args.tp, args.arrmeta, args.data(), nkwd, kwds_as_vector.data(), tp_vars);
      }

      dst_tp = dst.get_type();
      (*get())(dst_tp, dst->metadata(), dst.data(), narg, args.tp, args.arrmeta, args.data(), nkwd,
               kwds_as_vector.data(), tp_vars);
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

  template <typename CallableType, typename... ArgTypes>
  callable make_callable(ArgTypes &&... args)
  {
    return callable(new CallableType(std::forward<ArgTypes>(args)...), true);
  }

  inline std::ostream &operator<<(std::ostream &o, const callable &rhs)
  {
    return o << "<callable <" << rhs.get()->tp << "> at " << reinterpret_cast<const void *>(rhs.get()) << ">";
  }

  template <typename DataType>
  struct callable::args<DataType> {
    static const size_t size = 0;
    array *values;
    ndt::type *tp;
    const char *const *arrmeta;

    args(std::map<std::string, ndt::type> &DYND_UNUSED(tp_vars), const ndt::callable_type *self_tp)
        : values(nullptr), tp(nullptr), arrmeta(nullptr)
    {
      detail::check_narg(self_tp, 0);
    }

    DataType const *data() const { return NULL; }
  };

  /** A holder class for the array arguments */
  template <typename DataType, typename... A>
  struct callable::args {
    static const size_t size = sizeof...(A);
    array values[sizeof...(A)];
    ndt::type tp[sizeof...(A)];
    const char *arrmeta[sizeof...(A)];
    DataType m_data[sizeof...(A)];

    args(std::map<std::string, ndt::type> &tp_vars, const ndt::callable_type *self_tp, A &&... a)
        : values{std::forward<A>(a)...}
    {
      if (!self_tp->is_pos_variadic() && (static_cast<intptr_t>(sizeof...(A)) < self_tp->get_npos())) {
        std::stringstream ss;
        ss << "callable expected " << self_tp->get_npos() << " positional arguments, but received " << sizeof...(A);
        throw std::invalid_argument(ss.str());
      }

      for (intptr_t i = 0; i < (self_tp->is_pos_variadic() ? static_cast<intptr_t>(sizeof...(A)) : self_tp->get_npos());
           ++i) {
        detail::check_arg(self_tp, i, values[i]->tp, values[i]->metadata(), tp_vars);

        tp[i] = values[i]->tp;
        arrmeta[i] = values[i]->metadata();
        detail::set_data(m_data[i], values[i]);
      }
    }

    DataType const *data() const { return m_data; }
  };

  /** A way to pass a run-time array of array arguments */
  template <typename DataType>
  struct callable::args<DataType, size_t, array *> {
    size_t size;
    array *values;
    ndt::type *tp;
    const char **arrmeta;
    std::vector<DataType> m_data;

    args(std::map<std::string, ndt::type> &tp_vars, const ndt::callable_type *self_tp, size_t size, array *values)
        : size(size), values(values), tp(new ndt::type[size]), arrmeta(new const char *[size]), m_data(size)
    {
      detail::check_narg(self_tp, size);

      for (std::size_t i = 0; i < size; ++i) {
        detail::check_arg(self_tp, i, values[i]->tp, values[i]->metadata(), tp_vars);

        tp[i] = values[i]->tp;
        arrmeta[i] = values[i]->metadata();
        detail::set_data(m_data[i], values[i]);
      }
    }

    ~args()
    {
      delete[] tp;
      delete[] arrmeta;
    }

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
  std::ostream &operator<<(std::ostream &o, const declfunc<FuncType> &DYND_UNUSED(rhs))
  {
    return o << FuncType::get();
  }

  template <typename... ArgTypes>
  array array::f(const char *name, ArgTypes &&... args)
  {
    callable f = find_dynamic_function(name);
    return f(*this, std::forward<ArgTypes>(args)...);
  }

  template <typename... ArgTypes>
  array array::f(const char *name, ArgTypes &&... args) const
  {
    callable f = find_dynamic_function(name);
    return f(*this, std::forward<ArgTypes>(args)...);
  }

} // namespace dynd::nd

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
