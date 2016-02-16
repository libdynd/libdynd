//
// Copyright (C) 2011-15 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#pragma once

#include <memory>

#include <dynd/config.hpp>
#include <dynd/kernels/apply_function_kernel.hpp>
#include <dynd/kernels/apply_callable_kernel.hpp>
#include <dynd/kernels/apply_member_function_kernel.hpp>
#include <dynd/kernels/construct_then_apply_callable_kernel.hpp>
#include <dynd/types/callable_type.hpp>
#include <dynd/types/substitute_typevars.hpp>
#include <dynd/types/option_type.hpp>
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

    DYND_HAS(resolve_dst_type);

    template <typename KernelType>
    kernel_targets_t get_targets()
    {
      return kernel_targets_t{reinterpret_cast<void *>(static_cast<void (*)(kernel_prefix *, char *, char *const *)>(
                                  KernelType::single_wrapper)),
                              NULL, reinterpret_cast<void *>(KernelType::strided_wrapper)};
    }

    template <typename KernelType>
    const volatile char *get_ir()
    {
      return KernelType::ir;
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

  typedef array callable_arg_t;
  typedef std::pair<const char *, array> callable_kwd_t;

  /**
   * Holds a single instance of an callable in an nd::array,
   * providing some more direct convenient interface.
   */
  class DYND_API callable : public intrusive_ptr<base_callable> {
  public:
    using intrusive_ptr<base_callable>::intrusive_ptr;

    callable() = default;

    callable(const ndt::type &self_tp, kernel_single_t single, kernel_strided_t strided)
        : intrusive_ptr<base_callable>(
              new (sizeof(kernel_targets_t)) base_callable(
                  self_tp, kernel_targets_t{reinterpret_cast<void *>(single), NULL, reinterpret_cast<void *>(strided)}),
              true)
    {
    }

    callable(const ndt::type &self_tp, kernel_targets_t targets, const volatile char *ir,
             callable_data_init_t data_init, callable_resolve_dst_type_t resolve_dst_type,
             callable_instantiate_t instantiate)
        : intrusive_ptr<base_callable>(
              new base_callable(self_tp, targets, ir, data_init, resolve_dst_type, instantiate), true)
    {
    }

    template <typename T>
    callable(const ndt::type &self_tp, kernel_targets_t targets, const volatile char *ir, T &&static_data,
             callable_data_init_t data_init, callable_resolve_dst_type_t resolve_dst_type,
             callable_instantiate_t instantiate)
        : intrusive_ptr<base_callable>(
              new static_data_callable<typename std::remove_reference<T>::type>(
                  self_tp, targets, ir, data_init, resolve_dst_type, instantiate, std::forward<T>(static_data)),
              true)
    {
    }

    template <typename CallableType, typename... T, typename = std::enable_if_t<all_char_string_params<T...>::value>>
    explicit callable(CallableType f, T &&... names);

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

    const std::vector<ndt::type> &get_arg_types() const { return get_type()->get_pos_types(); }

    const callable &get_overload(const ndt::type &ret_tp, intptr_t narg, const ndt::type *arg_tp) const
    {
      return get()->overload(ret_tp, narg, arg_tp);
    }

    const callable &get_overload(const ndt::type &ret_tp, const std::initializer_list<ndt::type> &arg_tp) const
    {
      return get_overload(ret_tp, arg_tp.size(), arg_tp.begin());
    }

    void set_overload(const ndt::type &ret_tp, intptr_t narg, const ndt::type *arg_tp, const callable &value)
    {
      get()->overload(ret_tp, narg, arg_tp) = value;
    }

    void set_overload(const ndt::type &ret_tp, const std::initializer_list<ndt::type> &arg_tp, const callable &value)
    {
      set_overload(ret_tp, arg_tp.size(), arg_tp.begin(), value);
    }

    array call(size_t args_size, const array *args_values, size_t kwds_size,
               const std::pair<const char *, array> *kwds_values)
    {
      typedef array *DataType;

      std::map<std::string, ndt::type> tp_vars;
      const ndt::callable_type *self_tp = get_type();

      if (!self_tp->is_pos_variadic() && (static_cast<intptr_t>(args_size) < self_tp->get_npos())) {
        std::stringstream ss;
        ss << "callable expected " << self_tp->get_npos() << " positional arguments, but received " << args_size;
        throw std::invalid_argument(ss.str());
      }

      std::vector<ndt::type> args_tp(args_size);
      std::vector<const char *> args_arrmeta(args_size);
      std::vector<DataType> args_data(args_size);

      for (intptr_t i = 0; i < (self_tp->is_pos_variadic() ? static_cast<intptr_t>(args_size) : self_tp->get_npos());
           ++i) {
        detail::check_arg(self_tp, i, args_values[i]->tp, args_values[i]->metadata(), tp_vars);

        args_tp[i] = args_values[i]->tp;
        args_arrmeta[i] = args_values[i]->metadata();
        detail::set_data(args_data[i], args_values[i]);
      }

      array dst;

      intptr_t narg = args_size;

      // ...
      intptr_t nkwd = args_size - self_tp->get_npos();
      if (!self_tp->is_kwd_variadic() && nkwd > self_tp->get_nkwd()) {
        throw std::invalid_argument("too many extra positional arguments");
      }

      std::vector<array> kwds_as_vector(nkwd + self_tp->get_nkwd());
      for (intptr_t i = 0; i < nkwd; ++i) {
        kwds_as_vector[i] = args_values[self_tp->get_npos() + i];
        --narg;
      }

      for (size_t i = 0; i < kwds_size; ++i) {
        intptr_t j = self_tp->get_kwd_index(kwds_values[i].first);
        if (j == -1) {
          if (detail::is_special_kwd(self_tp, dst, kwds_values[i].first, kwds_values[i].second)) {
          }
          else {
            std::stringstream ss;
            ss << "passed an unexpected keyword \"" << kwds_values[i].first << "\" to callable with type " << get()->tp;
            throw std::invalid_argument(ss.str());
          }
        }
        else {
          array &value = kwds_as_vector[j];
          if (!value.is_null()) {
            std::stringstream ss;
            ss << "callable passed keyword \"" << kwds_values[i].first << "\" more than once";
            throw std::invalid_argument(ss.str());
          }
          value = kwds_values[i].second;

          ndt::type expected_tp = self_tp->get_kwd_type(j);
          if (expected_tp.get_id() == option_id) {
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
            actual_tp = ndt::make_type<ndt::option_type>(ndt::make_type<void>());
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
        return get()->call(dst_tp, narg, args_tp.data(), args_arrmeta.data(), args_data.data(), nkwd,
                           kwds_as_vector.data(), tp_vars);
      }

      dst_tp = dst.get_type();
      get()->call(dst_tp, dst->metadata(), &dst, narg, args_tp.data(), args_arrmeta.data(), args_data.data(), nkwd,
                  kwds_as_vector.data(), tp_vars);
      return dst;
    }

    template <typename... ArgTypes>
    array operator()(ArgTypes &&... args)
    {
      array tmp[sizeof...(ArgTypes)] = {std::forward<ArgTypes>(args)...};
      return call(sizeof...(ArgTypes), tmp, 0, nullptr);
    }

    array operator()() { return call(0, nullptr, 0, nullptr); }

    array operator()(const std::initializer_list<array> &args,
                     const std::initializer_list<std::pair<const char *, array>> &kwds)
    {
      return call(args.size(), args.begin(), kwds.size(), kwds.begin());
    }

    template <typename DstType, typename... ArgTypes>
    array operator()(ArgTypes &&... args)
    {
      array tmp[sizeof...(ArgTypes)] = {std::forward<ArgTypes>(args)...};
      std::pair<const char *, array> kwds = {"dst_tp", ndt::make_type<DstType>()};
      return call(sizeof...(ArgTypes), tmp, 1, &kwds);
    }

    template <typename KernelType>
    static typename std::enable_if<ndt::has_traits<KernelType>::value, callable>::type make()
    {
      return callable(ndt::traits<KernelType>::equivalent(), detail::get_targets<KernelType>(),
                      detail::get_ir<KernelType>(), &KernelType::data_init, detail::get_resolve_dst_type<KernelType>(),
                      &KernelType::instantiate);
    }

    template <typename KernelType>
    static typename std::enable_if<!ndt::has_traits<KernelType>::value, callable>::type make(const ndt::type &tp)
    {
      return callable(tp, detail::get_targets<KernelType>(), detail::get_ir<KernelType>(), &KernelType::data_init,
                      detail::get_resolve_dst_type<KernelType>(), &KernelType::instantiate);
    }

    template <typename KernelType, typename StaticDataType>
    static typename std::enable_if<ndt::has_traits<KernelType>::value, callable>::type
    make(StaticDataType &&static_data)
    {
      return callable(ndt::traits<KernelType>::equivalent(), detail::get_targets<KernelType>(),
                      detail::get_ir<KernelType>(), std::forward<StaticDataType>(static_data), &KernelType::data_init,
                      detail::get_resolve_dst_type<KernelType>(), &KernelType::instantiate);
    }

    template <typename KernelType, typename StaticDataType>
    static typename std::enable_if<!ndt::has_traits<KernelType>::value, callable>::type
    make(const ndt::type &tp, StaticDataType &&static_data)
    {
      return callable(tp, detail::get_targets<KernelType>(), detail::get_ir<KernelType>(),
                      std::forward<StaticDataType>(static_data), &KernelType::data_init,
                      detail::get_resolve_dst_type<KernelType>(), &KernelType::instantiate);
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

  template <typename CallableType, typename KernelType, typename... ArgTypes>
  std::enable_if_t<ndt::has_traits<KernelType>::value, callable> make_callable(ArgTypes &&... args)
  {
    return callable(new CallableType(ndt::traits<KernelType>::equivalent(), detail::get_targets<KernelType>(),
                                     detail::get_ir<KernelType>(), &KernelType::data_init,
                                     detail::get_resolve_dst_type<KernelType>(), &KernelType::instantiate,
                                     std::forward<ArgTypes>(args)...),
                    true);
  }

  template <typename CallableType, typename KernelType, typename... ArgTypes>
  std::enable_if_t<!ndt::has_traits<KernelType>::value, callable> make_callable(const ndt::type &tp,
                                                                                ArgTypes &&... args)
  {
    return callable(new CallableType(tp, detail::get_targets<KernelType>(), detail::get_ir<KernelType>(),
                                     &KernelType::data_init, detail::get_resolve_dst_type<KernelType>(),
                                     &KernelType::instantiate, std::forward<ArgTypes>(args)...),
                    true);
  }

  inline std::ostream &operator<<(std::ostream &o, const callable &rhs)
  {
    return o << "<callable <" << rhs.get()->tp << "> at " << reinterpret_cast<const void *>(rhs.get()) << ">";
  }

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
    template <typename DstType, typename... ArgTypes>
    array operator()(ArgTypes &&... args)
    {
      return get().operator()<DstType>(std::forward<ArgTypes>(args)...);
    }

    template <typename... ArgTypes>
    array operator()(ArgTypes &&... args)
    {
      return get()(std::forward<ArgTypes>(args)...);
    }

    array operator()(const std::initializer_list<array> &args,
                     const std::initializer_list<std::pair<const char *, array>> &kwds)
    {
      return get()(args, kwds);
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

  namespace functional {

    /**
     * Makes an callable out of function ``func``, using the provided keyword
     * parameter names. This function takes ``func`` as a template
     * parameter, so can call it efficiently.
     */
    template <kernel_request_t kernreq, typename func_type, func_type func, typename... T>
    callable apply(T &&... names)
    {
      typedef apply_function_kernel<func_type, func, arity_of<func_type>::value - sizeof...(T)> CKT;

      ndt::type self_tp = ndt::make_type<typename funcproto_of<func_type>::type>(std::forward<T>(names)...);

      return callable::make<CKT>(self_tp);
    }

    template <typename func_type, func_type func, typename... T>
    callable apply(T &&... names)
    {
      return apply<kernel_request_host, func_type, func>(std::forward<T>(names)...);
    }

    /**
     * Makes an callable out of the function object ``func``, using the provided
     * keyword parameter names. This version makes a copy of provided ``func``
     * object.
     */
    template <kernel_request_t kernreq, typename func_type, typename... T>
    typename std::enable_if<!is_function_pointer<func_type>::value, callable>::type apply(func_type func, T &&... names)
    {
      typedef apply_callable_kernel<func_type, arity_of<func_type>::value - sizeof...(T)> ck_type;

      ndt::type self_tp = ndt::make_type<typename funcproto_of<func_type>::type>(std::forward<T>(names)...);

      return callable::make<ck_type>(self_tp, func);
    }

    template <typename func_type, typename... T>
    typename std::enable_if<!is_function_pointer<func_type>::value, callable>::type apply(func_type func, T &&... names)
    {
      static_assert(all_char_string_params<T...>::value, "All the names must be strings");
      return apply<kernel_request_host>(func, std::forward<T>(names)...);
    }

    template <kernel_request_t kernreq, typename func_type, typename... T>
    callable apply(func_type *func, T &&... names)
    {
      typedef apply_callable_kernel<func_type *, arity_of<func_type>::value - sizeof...(T)> ck_type;

      ndt::type self_tp = ndt::make_type<typename funcproto_of<func_type>::type>(std::forward<T>(names)...);

      return callable::make<ck_type>(self_tp, func);
    }

    template <typename func_type, typename... T>
    callable apply(func_type *func, T &&... names)
    {
      return apply<kernel_request_host>(func, std::forward<T>(names)...);
    }

    template <kernel_request_t kernreq, typename T, typename R, typename... A, typename... S>
    callable apply(T *obj, R (T::*mem_func)(A...), S &&... names)
    {
      typedef apply_member_function_kernel<T *, R (T::*)(A...), sizeof...(A) - sizeof...(S)> ck_type;

      ndt::type self_tp = ndt::make_type<typename funcproto_of<R (T::*)(A...)>::type>(std::forward<S>(names)...);

      return callable::make<ck_type>(self_tp, typename ck_type::data_type(obj, mem_func));
    }

    template <typename O, typename R, typename... A, typename... T>
    callable apply(O *obj, R (O::*mem_func)(A...), T &&... names)
    {
      return apply<kernel_request_host>(obj, mem_func, std::forward<T>(names)...);
    }

    /**
     * Makes an callable out of the provided function object type, specialized
     * for a memory_type such as cuda_device based on the ``kernreq``.
     */
    template <kernel_request_t kernreq, typename func_type, typename... K, typename... T>
    callable apply(T &&... names)
    {
      typedef construct_then_apply_callable_kernel<func_type, K...> ck_type;

      ndt::type self_tp = ndt::make_type<typename funcproto_of<func_type, K...>::type>(std::forward<T>(names)...);

      return callable::make<ck_type>(self_tp);
    }

    /**
     * Makes an callable out of the provided function object type, which
     * constructs and calls the function object on demand.
     */
    template <typename func_type, typename... K, typename... T>
    callable apply(T &&... names)
    {
      return apply<kernel_request_host, func_type, K...>(std::forward<T>(names)...);
    }

  } // namespace dynd::nd::functional

  template <typename CallableType, typename... T, typename>
  callable::callable(CallableType f, T &&... names)
      : callable(nd::functional::apply(f, std::forward<T>(names)...))
  {
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

} // namespace dynd
