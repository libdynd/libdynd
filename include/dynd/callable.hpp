//
// Copyright (C) 2011-16 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#pragma once

#include <map>
#include <memory>

#include <dynd/callables/apply_callable_callable.hpp>
#include <dynd/dispatcher.hpp>
#include <dynd/type_registry.hpp>

namespace dynd {
namespace nd {
  namespace detail {

    /**
     * Presently, there are some specially treated keyword arguments in
     * arrfuncs. The "dst_tp" keyword argument always tells the desired
     * output type, and the "dst" keyword argument always provides an
     * output array.
     */
    inline bool is_special_kwd(array &dst, const std::string &name, const nd::array &value) {
      if (name == "dst_tp") {
        dst = nd::empty(value.as<ndt::type>());
        return true;
      } else if (name == "dst") {
        dst = value;
        return true;
      }

      return false;
    }

    DYND_API void check_narg(const base_callable *self, size_t narg);

    DYND_API void check_arg(const base_callable *self, intptr_t i, const ndt::type &actual_tp,
                            const char *actual_arrmeta, std::map<std::string, ndt::type> &tp_vars);

    template <template <typename...> class KernelType>
    struct make_all;

    template <template <typename...> class KernelType, template <typename...> class Condition>
    struct make_all_if;

  } // namespace dynd::nd::detail

  /**
   * Holds a single instance of a callable in an nd::array,
   * providing some more direct convenient interface.
   */
  class DYND_API callable : public intrusive_ptr<base_callable> {
  public:
    using intrusive_ptr<base_callable>::intrusive_ptr;

    callable() = default;

    template <typename CallableType, typename... T, typename = std::enable_if_t<all_char_string_params<T...>::value>>
    callable(CallableType f, T &&... names)
        : callable(new functional::apply_callable_callable<CallableType, arity_of<CallableType>::value - sizeof...(T)>(
                       f, std::forward<T>(names)...),
                   true) {}

    bool is_null() const { return get() == NULL; }

    callable_property get_flags() const { return right_associative; }

    ndt::type resolve(const ndt::type &dst_tp, size_t nsrc, const ndt::type *src_tp, size_t nkwd, const array *kwds) {
      std::map<std::string, ndt::type> tp_vars;

      call_graph cg;
      return m_ptr->resolve(nullptr, nullptr, cg, dst_tp, nsrc, src_tp, nkwd, kwds, tp_vars);
    }

    ndt::type resolve(const ndt::type &dst_tp, std::initializer_list<ndt::type> src_tp,
                      std::initializer_list<array> kwds) {
      return resolve(dst_tp, src_tp.size(), src_tp.begin(), kwds.size(), kwds.begin());
    }

    ndt::type resolve(std::initializer_list<ndt::type> src_tp, std::initializer_list<array> kwds) {
      return resolve(m_ptr->get_ret_type(), src_tp.size(), src_tp.begin(), kwds.size(), kwds.begin());
    }

    template <template <typename> class CallableType, typename I0>
    void overload() {
      std::vector<callable> callables;
      std::array<int, 1> arr;
      for_each<I0>(detail::make_all<CallableType>(), callables, arr);

      for (const callable &f : callables) {
        get()->overload(f);
      }
    }

    /*
        void overload(const ndt::type &ret_tp, intptr_t narg, const ndt::type *arg_tp, const callable &value) {
          get()->overload(ret_tp, narg, arg_tp, value);
        }

        void overload(const ndt::type &ret_tp, const std::initializer_list<ndt::type> &arg_tp, const callable &value) {
          overload(ret_tp, arg_tp.size(), arg_tp.begin(), value);
        }
    */

    const callable &specialize(const ndt::type &ret_tp, intptr_t narg, const ndt::type *arg_tp) const {
      return get()->specialize(ret_tp, narg, arg_tp);
    }

    const callable &specialize(const ndt::type &ret_tp, const std::initializer_list<ndt::type> &arg_tp) const {
      return specialize(ret_tp, arg_tp.size(), arg_tp.begin());
    }

    array call(size_t narg, const array *args, size_t nkwd, const std::pair<const char *, array> *unordered_kwds) const;

    template <typename... ArgTypes>
    array operator()(ArgTypes &&... args) const {
      array tmp[sizeof...(ArgTypes)] = {std::forward<ArgTypes>(args)...};
      return call(sizeof...(ArgTypes), tmp, 0, nullptr);
    }

    array operator()() const { return call(0, nullptr, 0, nullptr); }

    array operator()(std::initializer_list<std::pair<const char *, array>> kwds) const {
      return call(0, nullptr, kwds.size(), kwds.begin());
    }

    array operator()(const std::initializer_list<array> &args,
                     const std::initializer_list<std::pair<const char *, array>> &kwds) const {
      return call(args.size(), args.begin(), kwds.size(), kwds.begin());
    }

    template <template <typename> class KernelType, typename I0, typename... A>
    static dispatcher<1, callable> make_all(dispatch_t dispatch, A &&... a) {
      std::vector<callable> callables;
      std::array<int, 1> arr;
      for_each<I0>(detail::make_all<KernelType>(), callables, arr, std::forward<A>(a)...);

      return dispatcher<1, callable>(dispatch, callables.begin(), callables.end());
    }

    template <template <typename...> class KernelType, typename I0, typename I1, typename... I, typename... A>
    static dispatcher<2 + sizeof...(I), callable> make_all(dispatch_t dispatch, A &&... a) {
      std::vector<callable> callables;
      std::array<int, 2 + sizeof...(I)> arr;
      for_each<typename outer<I0, I1, I...>::type>(detail::make_all<KernelType>(), callables, arr,
                                                   std::forward<A>(a)...);

      return dispatcher<2 + sizeof...(I), callable>(dispatch, callables.begin(), callables.end());
    }

    template <template <typename> class KernelType, template <typename> class Condition, typename Type0Sequence,
              typename... A>
    static dispatcher<1, callable> make_all_if(dispatch_t dispatch, A &&... a) {
      std::vector<callable> callables;
      std::array<int, 1> arr;
      for_each<Type0Sequence>(detail::make_all_if<KernelType, Condition>(), callables, arr, std::forward<A>(a)...);

      return dispatcher<1, callable>(dispatch, callables.begin(), callables.end());
    }

    template <template <typename, typename, typename...> class KernelType,
              template <typename, typename, typename...> class Condition, typename I0, typename I1, typename... I,
              typename... A>
    static dispatcher<2 + sizeof...(I), callable> make_all_if(dispatch_t dispatch, A &&... a) {
      std::vector<callable> callables;
      std::array<int, 2 + sizeof...(I)> arr;
      for_each<typename outer<I0, I1, I...>::type>(detail::make_all_if<KernelType, Condition>(), callables, arr,
                                                   std::forward<A>(a)...);

      return dispatcher<2 + sizeof...(I), callable>(dispatch, callables.begin(), callables.end());
    }
  };

  template <typename CallableType, typename... ArgTypes>
  std::enable_if_t<std::is_base_of<base_callable, CallableType>::value, callable> make_callable(ArgTypes &&... args) {
    return callable(new CallableType(std::forward<ArgTypes>(args)...), true);
  }

  template <typename KernelType, typename... ArgTypes>
  std::enable_if_t<std::is_base_of<base_kernel<KernelType>, KernelType>::value, callable>
  make_callable(const ndt::type &tp) {
    return make_callable<default_instantiable_callable<KernelType>>(tp);
  }

  template <template <size_t NArg> class CallableType, typename... ArgTypes>
  callable make_callable(size_t narg, ArgTypes &&... args) {
    switch (narg) {
    case 0:
      return make_callable<CallableType<0>>(std::forward<ArgTypes>(args)...);
    case 1:
      return make_callable<CallableType<1>>(std::forward<ArgTypes>(args)...);
    case 2:
      return make_callable<CallableType<2>>(std::forward<ArgTypes>(args)...);
    case 3:
      return make_callable<CallableType<3>>(std::forward<ArgTypes>(args)...);
    case 4:
      return make_callable<CallableType<4>>(std::forward<ArgTypes>(args)...);
    case 5:
      return make_callable<CallableType<5>>(std::forward<ArgTypes>(args)...);
    case 6:
      return make_callable<CallableType<6>>(std::forward<ArgTypes>(args)...);
    case 7:
      return make_callable<CallableType<7>>(std::forward<ArgTypes>(args)...);
    default:
      throw std::runtime_error("callable with nsrc > 7 not implemented yet");
    }
  }

  inline std::ostream &operator<<(std::ostream &o, const callable &rhs) {
    return o << "<callable <" << rhs->get_type() << "> at " << reinterpret_cast<const void *>(rhs.get()) << ">";
  }

  namespace detail {

    template <template <typename...> class CallableType>
    struct make_all {
      template <typename Type, typename... ArgTypes>
      void on_each(std::vector<callable> &callables, std::array<int, 1>, ArgTypes &&... args) const {
        callables.push_back(make_callable<CallableType<Type>>(std::forward<ArgTypes>(args)...));
      }

      template <typename TypeSequence, typename... ArgTypes>
      void on_each(std::vector<callable> &callables, std::array<int, TypeSequence::size()>, ArgTypes &&... args) const {
        typedef instantiate_t<CallableType, TypeSequence> callable_type;
        callable f = make_callable<callable_type>(std::forward<ArgTypes>(args)...);
        callables.push_back(f);
      }
    };

    // insert_callable_if is an internal helper template for make_all_if that reorganizes
    // its template parameters so that when a set of types does not pass the conditional
    // test for whether or not it should be added to a callable, the kernel template used
    // is never instantiated.
    template <bool Enable, template <typename...> class KernelType>
    struct insert_callable_if;

    template <template <typename...> class KernelType>
    struct insert_callable_if<false, KernelType> {
      template <typename Arg0Type, typename... A>
      static void insert(std::vector<callable> &DYND_UNUSED(callables), std::array<int, 1>, A &&... DYND_UNUSED(a)) {}

      template <typename TypeSequence, typename... A>
      static void insert(std::vector<callable> &DYND_UNUSED(callables), std::array<int, TypeSequence::size()>,
                         A &&... DYND_UNUSED(a)) {}
    };

    template <template <typename...> class KernelType>
    struct insert_callable_if<true, KernelType> {
      template <typename Arg0Type, typename... A>
      static void insert(std::vector<callable> &callables, std::array<int, 1>, A &&... a) {
        callable f = make_callable<KernelType<Arg0Type>>(std::forward<A>(a)...);

        callables.push_back(f);
      }

      template <typename TypeSequence, typename... A>
      static void insert(std::vector<callable> &callables, std::array<int, TypeSequence::size()>, A &&... a) {
        callable f = make_callable<instantiate_t<KernelType, TypeSequence>>(std::forward<A>(a)...);

        callables.push_back(f);
      }
    };

    template <template <typename...> class KernelType, template <typename...> class Condition>
    struct make_all_if {
      template <typename Type, typename... A>
      void on_each(std::vector<callable> &callables, std::array<int, 1> arr, A &&... a) const {
        insert_callable_if<Condition<Type>::value, KernelType>::template insert<Type, A...>(callables, arr,
                                                                                            std::forward<A>(a)...);
      }

      template <typename TypeSequence, typename... A>
      void on_each(std::vector<callable> &callables, std::array<int, TypeSequence::size()> arr, A &&... a) const {
        insert_callable_if<instantiate_t<Condition, TypeSequence>::value,
                           KernelType>::template insert<TypeSequence, A...>(callables, arr, std::forward<A>(a)...);
      }
    };

  } // namespace dynd::nd::detail

} // namespace dynd::nd

class registry_entry {
public:
  typedef void (*observer)(const char *, registry_entry *);
  typedef typename std::map<std::string, registry_entry>::iterator iterator;
  typedef typename std::map<std::string, registry_entry>::const_iterator const_iterator;

private:
  bool m_is_namespace;
  nd::callable m_value;
  std::map<std::string, registry_entry> m_namespace;
  std::vector<observer> m_observers;

public:
  registry_entry() = default;

  registry_entry(const nd::callable &entry) : m_is_namespace(false), m_value(entry) {}

  registry_entry(std::initializer_list<std::pair<const std::string, registry_entry>> values)
      : m_is_namespace(true), m_namespace(values) {}

  nd::callable &value() { return m_value; }
  const nd::callable &value() const { return m_value; }

  bool is_namespace() const { return m_is_namespace; }

  void insert(const std::pair<const std::string, registry_entry> &entry) {
    auto subentry = m_namespace.find(entry.first);
    if (subentry == m_namespace.end()) {
      m_namespace.emplace(entry);
    } else {
      for (const auto &pair : entry.second) {
        subentry->second.insert(pair);
      }
    }

    for (observer obs : m_observers) {
      obs(entry.first.c_str(), this);
    }
  }

  iterator find(const std::string &name) { return m_namespace.find(name); }

  void observe(observer obs) { m_observers.emplace_back(obs); }

  registry_entry &operator=(const registry_entry &rhs) {
    m_is_namespace = rhs.m_is_namespace;
    if (m_is_namespace) {
      m_namespace = rhs.m_namespace;
    } else {
      m_value = rhs.m_value;
    }

    return *this;
  }

  registry_entry &operator[](const std::string &path) {
    size_t i = path.find(".");
    std::string name = path.substr(0, i);

    iterator it = find(name);
    if (it == end()) {
      std::stringstream ss;
      ss << "No dynd function ";
      print_escaped_utf8_string(ss, name);
      ss << " has been registered";
      throw std::invalid_argument(ss.str());
    }

    if (i == std::string::npos) {
      return it->second;
    }

    return it->second[path.substr(i + 1)];
  }

  iterator begin() { return m_namespace.begin(); }
  const_iterator begin() const { return m_namespace.begin(); }

  iterator end() { return m_namespace.end(); }
  const_iterator end() const { return m_namespace.end(); }

  const_iterator cbegin() const { return m_namespace.cbegin(); }

  const_iterator cend() const { return m_namespace.cend(); }
};

/**
 * Returns a reference to the map of registered callables.
 */
DYND_API registry_entry &registered();

inline registry_entry &registered(const std::string &path) {
  registry_entry &entry = registered();
  return entry[path];
}

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

namespace nd {

  template <typename... ArgTypes>
  array array::f(const char *name, ArgTypes &&... args) const {
    registry_entry &entry = registered("dynd.nd");

    callable &f = entry[name].value();
    return f(*this, std::forward<ArgTypes>(args)...);
  }

} // namespace dynd::nd
} // namespace dynd
