//
// Copyright (C) 2011-15 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#pragma once

#include <memory>

#include <dynd/callables/base_callable.hpp>
#include <dynd/dispatcher.hpp>
#include <dynd/type_registry.hpp>
#include <dynd/callables/apply_function_callable.hpp>
#include <dynd/callables/apply_callable_callable.hpp>
#include <dynd/callables/apply_member_function_callable.hpp>
#include <dynd/callables/construct_then_apply_callable_callable.hpp>
#include <dynd/types/callable_type.hpp>
#include <dynd/types/option_type.hpp>
#include <dynd/types/substitute_typevars.hpp>

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

    template <typename KernelType>
    kernel_targets_t get_targets()
    {
      return kernel_targets_t{reinterpret_cast<void *>(KernelType::single_wrapper), NULL,
                              reinterpret_cast<void *>(NULL)};
    }

    template <typename KernelType>
    const volatile char *get_ir()
    {
      return KernelType::ir;
    }

    template <template <type_id_t...> class KernelType>
    struct make_all;

    template <template <type_id_t...> class KernelType>
    struct new_make_all;

  } // namespace dynd::nd::detail

  typedef array callable_arg_t;
  typedef std::pair<const char *, array> callable_kwd_t;

  /**
   * Holds a single instance of a callable in an nd::array,
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

    void overload(const ndt::type &ret_tp, intptr_t narg, const ndt::type *arg_tp, const callable &value)
    {
      get()->overload(ret_tp, narg, arg_tp, value);
    }

    void overload(const ndt::type &ret_tp, const std::initializer_list<ndt::type> &arg_tp, const callable &value)
    {
      overload(ret_tp, arg_tp.size(), arg_tp.begin(), value);
    }

    const callable &specialize(const ndt::type &ret_tp, intptr_t narg, const ndt::type *arg_tp) const
    {
      return get()->specialize(ret_tp, narg, arg_tp);
    }

    const callable &specialize(const ndt::type &ret_tp, const std::initializer_list<ndt::type> &arg_tp) const
    {
      return specialize(ret_tp, arg_tp.size(), arg_tp.begin());
    }

    array call(size_t args_size, const array *args_values, size_t kwds_size,
               const std::pair<const char *, array> *kwds_values) const;

    template <typename... ArgTypes>
    array operator()(ArgTypes &&... args) const
    {
      array tmp[sizeof...(ArgTypes)] = {std::forward<ArgTypes>(args)...};
      return call(sizeof...(ArgTypes), tmp, 0, nullptr);
    }

    array operator()() const { return call(0, nullptr, 0, nullptr); }

    array operator()(const std::initializer_list<array> &args,
                     const std::initializer_list<std::pair<const char *, array>> &kwds) const
    {
      return call(args.size(), args.begin(), kwds.size(), kwds.begin());
    }

    template <template <type_id_t> class KernelType, typename I0, typename... A>
    static dispatcher<callable> new_make_all(A &&... a)
    {
      std::vector<std::pair<std::vector<type_id_t>, callable>> callables;
      for_each<I0>(detail::new_make_all<KernelType>(), callables, std::forward<A>(a)...);

      return dispatcher<callable>(callables.begin(), callables.end());
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

    template <template <type_id_t, type_id_t, type_id_t...> class KernelType, typename I0, typename I1, typename... I,
              typename... A>
    static dispatcher<callable> new_make_all(A &&... a)
    {
      std::vector<std::pair<std::vector<type_id_t>, callable>> callables;
      for_each<typename outer<I0, I1, I...>::type>(detail::new_make_all<KernelType>(), callables,
                                                   std::forward<A>(a)...);

      return dispatcher<callable>(callables.begin(), callables.end());
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
        callables[TypeID] = make_callable<KernelType<TypeID>>(std::forward<A>(a)...);
      }

      template <typename TypeIDSequence, typename... A>
      void on_each(std::map<std::array<type_id_t, TypeIDSequence::size2()>, callable> &callables, A &&... a) const
      {
        callables[i2a<TypeIDSequence>()] =
            make_callable<typename apply<KernelType, TypeIDSequence>::type>(std::forward<A>(a)...);
      }
    };

    template <template <type_id_t...> class KernelType>
    struct new_make_all {
      template <type_id_t TypeID, typename... A>
      void on_each(std::vector<std::pair<std::vector<type_id_t>, callable>> &callables, A &&... a) const
      {
        callables.push_back({{TypeID}, make_callable<KernelType<TypeID>>(std::forward<A>(a)...)});
      }

      template <typename TypeIDSequence, typename... A>
      void on_each(std::vector<std::pair<std::vector<type_id_t>, callable>> &callables, A &&... a) const
      {
        auto arr = i2a<TypeIDSequence>();
        std::vector<type_id_t> v(arr.size());
        for (size_t i = 0; i < arr.size(); ++i) {
          v[i] = arr[i];
        }

        callables.push_back(
            {{v}, make_callable<typename apply<KernelType, TypeIDSequence>::type>(std::forward<A>(a)...)});
      }
    };

  } // namespace dynd::nd::detail

  template <typename FuncType>
  struct declfunc {
    template <typename DstType, typename... ArgTypes>
    array operator()(ArgTypes &&... args)
    {
      return FuncType::get().template operator()<DstType>(std::forward<ArgTypes>(args)...);
    }

    template <typename... ArgTypes>
    array operator()(ArgTypes &&... args)
    {
      return FuncType::get()(std::forward<ArgTypes>(args)...);
    }

    array operator()(const std::initializer_list<array> &args,
                     const std::initializer_list<std::pair<const char *, array>> &kwds)
    {
      return FuncType::get()(args, kwds);
    }
  };

// A macro for defining the get method.
#define DYND_DEFAULT_DECLFUNC_GET(NAME)                                                                                \
  DYND_API dynd::nd::callable &NAME::get()                                                                             \
  {                                                                                                                    \
    static dynd::nd::callable self = NAME::make();                                                                     \
    return self;                                                                                                       \
  }

  template <typename FuncType>
  std::ostream &operator<<(std::ostream &o, const declfunc<FuncType> &DYND_UNUSED(rhs))
  {
    return o << FuncType::get();
  }

  namespace functional {

    /**
     * Makes a callable out of function ``func``, using the provided keyword
     * parameter names. This function takes ``func`` as a template
     * parameter, so can call it efficiently.
     */
    template <kernel_request_t kernreq, typename func_type, func_type func, typename... T>
    callable apply(T &&... names)
    {
      typedef apply_function_callable<func_type, func, arity_of<func_type>::value - sizeof...(T)> CKT;
      ndt::type self_tp = ndt::make_type<typename funcproto_of<func_type>::type>(std::forward<T>(names)...);
      return make_callable<CKT>(self_tp);
    }

    template <typename func_type, func_type func, typename... T>
    callable apply(T &&... names)
    {
      return apply<kernel_request_host, func_type, func>(std::forward<T>(names)...);
    }

    /**
     * Makes a callable out of the function object ``func``, using the provided
     * keyword parameter names. This version makes a copy of provided ``func``
     * object.
     */
    template <kernel_request_t kernreq, typename func_type, typename... T>
    typename std::enable_if<!is_function_pointer<func_type>::value, callable>::type apply(func_type func, T &&... names)
    {
      typedef apply_callable_callable<func_type, arity_of<func_type>::value - sizeof...(T)> ck_type;

      ndt::type self_tp = ndt::make_type<typename funcproto_of<func_type>::type>(std::forward<T>(names)...);

      return make_callable<ck_type>(self_tp, func);
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
      typedef apply_callable_callable<func_type *, arity_of<func_type>::value - sizeof...(T)> ck_type;

      ndt::type self_tp = ndt::make_type<typename funcproto_of<func_type>::type>(std::forward<T>(names)...);

      return make_callable<ck_type>(self_tp, func);
    }

    template <typename func_type, typename... T>
    callable apply(func_type *func, T &&... names)
    {
      return apply<kernel_request_host>(func, std::forward<T>(names)...);
    }

    template <kernel_request_t kernreq, typename T, typename R, typename... A, typename... S>
    callable apply(T *obj, R (T::*mem_func)(A...), S &&... names)
    {
      typedef apply_member_function_callable<T *, R (T::*)(A...), sizeof...(A) - sizeof...(S)> ck_type;

      ndt::type self_tp = ndt::make_type<typename funcproto_of<R (T::*)(A...)>::type>(std::forward<S>(names)...);

      return make_callable<ck_type>(self_tp, obj, mem_func);
    }

    template <typename O, typename R, typename... A, typename... T>
    callable apply(O *obj, R (O::*mem_func)(A...), T &&... names)
    {
      return apply<kernel_request_host>(obj, mem_func, std::forward<T>(names)...);
    }

    /**
     * Makes a callable out of the provided function object type, specialized
     * for a memory_type such as cuda_device based on the ``kernreq``.
     */
    template <kernel_request_t kernreq, typename func_type, typename... K, typename... T>
    callable apply(T &&... names)
    {
      typedef construct_then_apply_callable_callable<func_type, K...> ck_type;

      ndt::type self_tp = ndt::make_type<typename funcproto_of<func_type, K...>::type>(std::forward<T>(names)...);

      return make_callable<ck_type>(self_tp);
    }

    /**
     * Makes a callable out of the provided function object type, which
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
 * Creates a callable which does the assignment from
 * data of src_tp to dst_tp.
 *
 * \param dst_tp  The type of the destination.
 * \param src_tp  The type of the source.
 * \param errmode  The error mode to use for the assignment.
 */
DYND_API nd::callable make_callable_from_assignment(const ndt::type &dst_tp, const ndt::type &src_tp,
                                                    assign_error_mode errmode);

} // namespace dynd
