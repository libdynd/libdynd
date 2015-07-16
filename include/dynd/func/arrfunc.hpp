//
// Copyright (C) 2011-15 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#pragma once

#include <memory>

#include <dynd/config.hpp>
#include <dynd/eval/eval_context.hpp>
#include <dynd/types/base_type.hpp>
#include <dynd/types/arrfunc_type.hpp>
#include <dynd/kernels/ckernel_builder.hpp>
#include <dynd/types/struct_type.hpp>
#include <dynd/types/substitute_typevars.hpp>
#include <dynd/types/type_type.hpp>

namespace dynd {

namespace ndt {
  ndt::type make_option(const ndt::type &value_tp);
} // namespace dynd::ndt

namespace nd {
  namespace detail {

    /**
     * Presently, there are some specially treated keyword arguments in
     * arrfuncs. The "dst_tp" keyword argument always tells the desired
     * output type, and the "dst" keyword argument always provides an
     * output array.
     */
    template <typename T>
    bool is_special_kwd(const ndt::arrfunc_type *DYND_UNUSED(self_tp),
                        const std::string &DYND_UNUSED(name),
                        const T &DYND_UNUSED(value),
                        std::map<nd::string, ndt::type> &DYND_UNUSED(tp_vars))
    {
      return false;
    }

    inline bool is_special_kwd(const ndt::arrfunc_type *DYND_UNUSED(self_tp),
                               array &dst, const std::string &name,
                               const ndt::type &value)
    {
      if (name == "dst_tp") {
        dst = nd::empty(value);
        return true;
      }

      return false;
    }

    inline bool is_special_kwd(const ndt::arrfunc_type *DYND_UNUSED(self_tp),
                               array &dst, const std::string &name,
                               const nd::array &value)
    {
      if (name == "dst_tp") {
        dst = nd::empty(value.as<ndt::type>());
        return true;
      } else if (name == "dst") {
        dst = value;
        return true;
      }

      return false;
    }

    template <typename T>
    void check_name(const ndt::arrfunc_type *af_tp, array &dst,
                    const std::string &name, const T &value, bool &has_dst_tp,
                    ndt::type *kwd_tp, std::vector<intptr_t> &available)
    {
      intptr_t j = af_tp->get_kwd_index(name);
      if (j == -1) {
        if (is_special_kwd(af_tp, dst, name, value)) {
          has_dst_tp = true;
        } else {
          std::stringstream ss;
          ss << "passed an unexpected keyword \"" << name
             << "\" to arrfunc with type " << ndt::type(af_tp, true);
          throw std::invalid_argument(ss.str());
        }
      } else {
        ndt::type &actual_tp = kwd_tp[j];
        if (!actual_tp.is_null()) {
          std::stringstream ss;
          ss << "arrfunc passed keyword \"" << name << "\" more than once";
          throw std::invalid_argument(ss.str());
        }
        actual_tp = ndt::type_of(value);
      }
      available.push_back(j);
    }

    void fill_missing_values(const ndt::type *tp, char *arrmeta,
                             const uintptr_t *arrmeta_offsets, char *data,
                             const uintptr_t *data_offsets,
                             const std::vector<intptr_t> &missing);

    void check_narg(const ndt::arrfunc_type *af_tp, intptr_t npos);

    void check_arg(const ndt::arrfunc_type *af_tp, intptr_t i,
                   const ndt::type &actual_tp, const char *actual_arrmeta,
                   std::map<nd::string, ndt::type> &tp_vars);

    void check_nkwd(const ndt::arrfunc_type *af_tp,
                    const std::vector<intptr_t> &available,
                    const std::vector<intptr_t> &missing);

    void validate_kwd_types(const ndt::arrfunc_type *af_tp,
                            std::vector<ndt::type> &kwd_tp,
                            const std::vector<intptr_t> &available,
                            const std::vector<intptr_t> &missing,
                            std::map<nd::string, ndt::type> &tp_vars);

    inline char *data_of(array &value)
    {
      return const_cast<char *>(value.get_readonly_originptr());
    }

    inline char *data_of(const array &value)
    {
      return const_cast<char *>(value.get_readonly_originptr());
    }

    /** A holder class for the array arguments */
    template <typename... A>
    class args {
      std::tuple<A...> m_values;
      const char *m_arrmeta[sizeof...(A)];

    public:
      args(A &&... a) : m_values(std::forward<A>(a)...)
      {
        validate_types.self = this;

        // Todo: This should be removed, but it seems to trigger an error on
        // travis if it is
        typedef make_index_sequence<sizeof...(A)> I;
        old_index_proxy<I>::template get_arrmeta(m_arrmeta, m_values);
      }

      struct {
        args *self;

        template <size_t I>
        void on_each(const ndt::arrfunc_type *af_tp,
                     std::vector<ndt::type> &src_tp,
                     std::vector<const char *> &src_arrmeta,
                     std::vector<char *> &src_data,
                     std::map<nd::string, ndt::type> &tp_vars) const
        {
          auto &value = std::get<I>(self->m_values);
          const ndt::type &tp = ndt::type_of(value);
          const char *arrmeta = self->m_arrmeta[I];

          check_arg(af_tp, I, tp, arrmeta, tp_vars);

          src_tp[I] = tp;
          src_arrmeta[I] = arrmeta;
          src_data[I] = data_of(value);
        }

        void operator()(const ndt::arrfunc_type *af_tp,
                        std::vector<ndt::type> &src_tp,
                        std::vector<const char *> &src_arrmeta,
                        std::vector<char *> &src_data,
                        std::map<nd::string, ndt::type> &tp_vars) const
        {
          check_narg(af_tp, sizeof...(A));

          typedef make_index_sequence<sizeof...(A)> I;
          index_proxy<I>::for_each(*this, af_tp, src_tp, src_arrmeta, src_data,
                                   tp_vars);
        }
      } validate_types;
    };

    template <>
    class args<> {
    public:
      void validate_types(
          const ndt::arrfunc_type *af_tp,
          std::vector<ndt::type> &DYND_UNUSED(src_tp),
          std::vector<const char *> &DYND_UNUSED(src_arrmeta),
          std::vector<char *> &DYND_UNUSED(src_data),
          std::map<nd::string, ndt::type> &DYND_UNUSED(tp_vars)) const
      {
        check_narg(af_tp, 0);
      }
    };

    /** A way to pass a run-time array of array arguments */
    template <>
    class args<intptr_t, nd::array *> {
      intptr_t m_size;
      array *m_values;

    public:
      args(intptr_t size, nd::array *values) : m_size(size), m_values(values) {}

      void validate_types(const ndt::arrfunc_type *af_tp,
                          std::vector<ndt::type> &src_tp,
                          std::vector<const char *> &src_arrmeta,
                          std::vector<char *> &src_data,
                          std::map<nd::string, ndt::type> &tp_vars) const
      {
        check_narg(af_tp, m_size);

        for (intptr_t i = 0; i < m_size; ++i) {
          array &value = m_values[i];
          const ndt::type &tp = value.get_type();
          const char *arrmeta = value.get_arrmeta();

          check_arg(af_tp, i, tp, arrmeta, tp_vars);

          src_tp[i] = tp;
          src_arrmeta[i] = arrmeta;
          src_data[i] = data_of(value);
        }
      }
    };

    /**
     * A way to pass a run-time array of array arguments, split up into the
     * type/arrmeta/data components
     */
    template <>
    class args<intptr_t, const ndt::type *, const char *const *,
               char *const *> {
      intptr_t m_size;
      const ndt::type *m_types;
      const char *const *m_arrmetas;
      char *const *m_datas;

    public:
      args(intptr_t size, const ndt::type *types, const char *const *arrmetas,
           char *const *datas)
          : m_size(size), m_types(types), m_arrmetas(arrmetas), m_datas(datas)
      {
      }

      void validate_types(const ndt::arrfunc_type *af_tp,
                          std::vector<ndt::type> &src_tp,
                          std::vector<const char *> &src_arrmeta,
                          std::vector<char *> &src_data,
                          std::map<nd::string, ndt::type> &tp_vars) const
      {
        check_narg(af_tp, m_size);

        for (intptr_t i = 0; i < m_size; ++i) {
          const ndt::type &tp = m_types[i];
          const char *arrmeta = m_arrmetas[i];

          check_arg(af_tp, i, tp, arrmeta, tp_vars);

          src_tp[i] = tp;
          src_arrmeta[i] = arrmeta;
          src_data[i] = m_datas[i];
        }
      }
    };

    /**
     * A metafunction to distinguish the general C++ variadic arguments versus
     * the special args<> for bypassing the C++ interface layer.
     */
    template <typename... T>
    struct is_variadic_args {
      enum { value = true };
    };
    template <typename T0, typename T1>
    struct is_variadic_args<T0, T1> {
      enum {
        value = !(std::is_convertible<T0, intptr_t>::value &&
                  std::is_convertible<T1, array *>::value)
      };
    };
    template <typename T0, typename T1, typename T2, typename T3>
    struct is_variadic_args<T0, T1, T2, T3> {
      enum {
        value = !(std::is_convertible<T0, intptr_t>::value &&
                  std::is_convertible<T1, const ndt::type *>::value &&
                  std::is_convertible<T2, const char *const *>::value &&
                  std::is_convertible<T3, char *const *>::value)
      };
    };

    /** A holder class for the keyword arguments */
    template <typename... K>
    class kwds;

    /** The case of no keyword arguments being provided */
    template <>
    class kwds<> {
      void fill_values(const ndt::type *tp, char *arrmeta,
                       const uintptr_t *arrmeta_offsets, char *data,
                       const uintptr_t *data_offsets,
                       const std::vector<intptr_t> &DYND_UNUSED(available),
                       const std::vector<intptr_t> &missing) const
      {
        fill_missing_values(tp, arrmeta, arrmeta_offsets, data, data_offsets,
                            missing);
      }

    public:
      void validate_names(const ndt::arrfunc_type *af_tp,
                          array &DYND_UNUSED(dst),
                          std::vector<ndt::type> &DYND_UNUSED(tp),
                          std::vector<intptr_t> &available,
                          std::vector<intptr_t> &missing) const
      {
        // No keywords provided, so all are missing
        for (intptr_t j : af_tp->get_option_kwd_indices()) {
          missing.push_back(j);
        }

        check_nkwd(af_tp, available, missing);
      }

      /** Converts the keyword args + filled in defaults into an nd::array */
      array as_array(const ndt::type &tp,
                     const std::vector<intptr_t> &available,
                     const std::vector<intptr_t> &missing) const
      {
        array res = empty_shell(tp);
        ndt::struct_type::fill_default_data_offsets(
            res.get_dim_size(),
            tp.extended<ndt::base_struct_type>()->get_field_types_raw(),
            reinterpret_cast<uintptr_t *>(res.get_arrmeta()));

        char *arrmeta = res.get_arrmeta();
        const uintptr_t *arrmeta_offsets =
            res.get_type()
                .extended<ndt::base_struct_type>()
                ->get_arrmeta_offsets_raw();
        char *data = res.get_readwrite_originptr();
        const uintptr_t *data_offsets =
            res.get_type().extended<ndt::base_struct_type>()->get_data_offsets(
                res.get_arrmeta());

        fill_values(tp.extended<ndt::base_struct_type>()->get_field_types_raw(),
                    arrmeta, arrmeta_offsets, data, data_offsets, available,
                    missing);

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
          index_proxy<I>::for_each(*this, names...);
        }
      } set_names;

      struct {
        kwds *self;

        template <size_t I>
        void on_each(const ndt::type *tp, char *arrmeta,
                     const uintptr_t *arrmeta_offsets, char *data,
                     const uintptr_t *data_offsets,
                     const std::vector<intptr_t> &available) const
        {
          intptr_t j = available[I];
          if (j != -1) {
            nd::forward_as_array(tp[j], arrmeta + arrmeta_offsets[j],
                                 data + data_offsets[j],
                                 std::get<I>(self->m_values));
          }
        }

        void operator()(const ndt::type *tp, char *arrmeta,
                        const uintptr_t *arrmeta_offsets, char *data,
                        const uintptr_t *data_offsets,
                        const std::vector<intptr_t> &available) const
        {
          typedef make_index_sequence<sizeof...(K)> I;
          index_proxy<I>::for_each(*this, tp, arrmeta, arrmeta_offsets, data,
                                   data_offsets, available);
        }
      } fill_available_values;

      void fill_values(const ndt::type *tp, char *arrmeta,
                       const uintptr_t *arrmeta_offsets, char *data,
                       const uintptr_t *data_offsets,
                       const std::vector<intptr_t> &available,
                       const std::vector<intptr_t> &missing) const
      {
        fill_available_values(tp, arrmeta, arrmeta_offsets, data, data_offsets,
                              available);
        fill_missing_values(tp, arrmeta, arrmeta_offsets, data, data_offsets,
                            missing);
      }

    public:
      kwds(typename as_<K, const char *>::type... names, K &&... values)
          : m_values(std::forward<K>(values)...)
      {
        set_names.self = this;
        validate_names.self = this;
        fill_available_values.self = this;

        set_names(names...);
      }

      struct {
        kwds *self;

        template <size_t I>
        void on_each(const ndt::arrfunc_type *af_tp, array &dst,
                     bool &has_dst_tp, std::vector<ndt::type> &kwd_tp,
                     std::vector<intptr_t> &available)
        {
          check_name(af_tp, dst, self->m_names[I], std::get<I>(self->m_values),
                     has_dst_tp, kwd_tp.data(), available);
        }

        void operator()(const ndt::arrfunc_type *af_tp, array &dst,
                        std::vector<ndt::type> &tp,
                        std::vector<intptr_t> &available,
                        std::vector<intptr_t> &missing) const
        {
          bool has_dst_tp = false;

          typedef make_index_sequence<sizeof...(K)> I;
          index_proxy<I>::for_each(*this, af_tp, dst, has_dst_tp, tp,
                                   available);

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

      array as_array(const ndt::type &tp,
                     const std::vector<intptr_t> &available,
                     const std::vector<intptr_t> &missing) const
      {
        array res = empty_shell(tp);
        ndt::struct_type::fill_default_data_offsets(
            res.get_dim_size(),
            tp.extended<ndt::base_struct_type>()->get_field_types_raw(),
            reinterpret_cast<uintptr_t *>(res.get_arrmeta()));

        fill_values(
            tp.extended<ndt::base_struct_type>()->get_field_types_raw(),
            res.get_arrmeta(), res.get_type()
                                   .extended<ndt::base_struct_type>()
                                   ->get_arrmeta_offsets_raw(),
            res.get_readwrite_originptr(),
            res.get_type().extended<ndt::base_struct_type>()->get_data_offsets(
                res.get_arrmeta()),
            available, missing);

        return res;
      }
    };

    template <>
    class kwds<intptr_t, const char *const *, array *> {
      intptr_t m_size;
      const char *const *m_names;
      array *m_values;

      void fill_available_values(const ndt::type *tp, char *arrmeta,
                                 const uintptr_t *arrmeta_offsets, char *data,
                                 const uintptr_t *data_offsets,
                                 const std::vector<intptr_t> &available) const
      {
        for (intptr_t i = 0; i < m_size; ++i) {
          intptr_t j = available[i];
          if (j != -1) {
            nd::forward_as_array(tp[j], arrmeta + arrmeta_offsets[j],
                                 data + data_offsets[j], this->m_values[i]);
          }
        }
      }

      void fill_values(const ndt::type *tp, char *arrmeta,
                       const uintptr_t *arrmeta_offsets, char *data,
                       const uintptr_t *data_offsets,
                       const std::vector<intptr_t> &available,
                       const std::vector<intptr_t> &missing) const
      {
        fill_available_values(tp, arrmeta, arrmeta_offsets, data, data_offsets,
                              available);
        fill_missing_values(tp, arrmeta, arrmeta_offsets, data, data_offsets,
                            missing);
      }

    public:
      kwds(intptr_t size, const char *const *names, array *values)
          : m_size(size), m_names(names), m_values(values)
      {
      }

      void validate_names(const ndt::arrfunc_type *af_tp, array &dst,
                          std::vector<ndt::type> &kwd_tp,
                          std::vector<intptr_t> &available,
                          std::vector<intptr_t> &missing) const
      {
        bool has_dst_tp = false;

        for (intptr_t i = 0; i < m_size; ++i) {
          check_name(af_tp, dst, m_names[i], m_values[i], has_dst_tp,
                     kwd_tp.data(), available);
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

      array as_array(const ndt::type &tp,
                     const std::vector<intptr_t> &available,
                     const std::vector<intptr_t> &missing) const
      {
        array res = empty_shell(tp);
        auto field_count =
            tp.extended<ndt::base_struct_type>()->get_field_count();
        auto field_types =
            tp.extended<ndt::base_struct_type>()->get_field_types_raw();
        ndt::struct_type::fill_default_data_offsets(
            field_count, field_types,
            reinterpret_cast<uintptr_t *>(res.get_arrmeta()));

        fill_values(
            field_types, res.get_arrmeta(),
            res.get_type()
                .extended<ndt::base_struct_type>()
                ->get_arrmeta_offsets_raw(),
            res.get_readwrite_originptr(),
            res.get_type().extended<ndt::base_struct_type>()->get_data_offsets(
                res.get_arrmeta()),
            available, missing);

        return res;
      }
    };

    template <typename T>
    struct is_kwds {
      static const bool value = false;
    };

    template <typename... K>
    struct is_kwds<nd::detail::kwds<K...>> {
      static const bool value = true;
    };

    template <typename... K>
    struct is_kwds<const nd::detail::kwds<K...>> {
      static const bool value = true;
    };

    template <typename... K>
    struct is_kwds<const nd::detail::kwds<K...> &> {
      static const bool value = true;
    };

    template <typename... K>
    struct is_kwds<nd::detail::kwds<K...> &> {
      static const bool value = true;
    };

    template <typename... T>
    struct is_variadic_kwds {
      enum { value = true };
    };

    template <typename T0, typename T1, typename T2>
    struct is_variadic_kwds<T0, T1, T2> {
      enum {
        value = !std::is_convertible<T0, intptr_t>::value ||
                !std::is_convertible<T1, const char *const *>::value ||
                !std::is_convertible<T2, nd::array *>::value
      };
    };

    template <typename... T>
    struct as_kwds {
      typedef typename instantiate<
          nd::detail::kwds,
          typename dynd::take<
              type_sequence<T...>,
              make_index_sequence<1, sizeof...(T), 2>>::type>::type type;
    };
  }
} // namespace dynd::nd

/**
 * A function to provide keyword arguments to an arrfunc. The arguments
 * must alternate between the keyword name and the argument value.
 *
 *   arrfunc af = <some arrfunc>;
 *   af(arr1, arr2, kwds("firstkwarg", kwval1, "second", kwval2));
 */
template <typename... T>
typename std::enable_if<nd::detail::is_variadic_kwds<T...>::value,
                        typename nd::detail::as_kwds<T...>::type>::type
kwds(T &&... t)
{
  // Sequence of even integers, for extracting the keyword names
  typedef make_index_sequence<0, sizeof...(T), 2> I;
  // Sequence of odd integers, for extracting the keyword values
  typedef make_index_sequence<1, sizeof...(T), 2> J;

  return index_proxy<typename join<I, J>::type>::template make<
      decltype(kwds(std::forward<T>(t)...))>(std::forward<T>(t)...);
}

/**
 * A special way to provide the keyword argument as an array of
 * names and an array of nd::array values.
 */
template <typename... T>
typename std::enable_if<
    !nd::detail::is_variadic_kwds<T...>::value,
    nd::detail::kwds<intptr_t, const char *const *, nd::array *>>::type
kwds(T &&... t)
{
  return nd::detail::kwds<intptr_t, const char *const *, nd::array *>(
      std::forward<T>(t)...);
}

/**
 * Empty keyword args.
 */
inline nd::detail::kwds<> kwds() { return nd::detail::kwds<>(); }

/**
 * TODO: This `as_array` metafunction should either go somewhere better (this
 *       file is for arrfunc), or be in a detail:: namespace.
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

    DYND_HAS(data_size);
    DYND_HAS(make_type);
    DYND_HAS(data_init);
    DYND_HAS(resolve_dst_type);
    DYND_HAS(instantiate);
    DYND_HAS(static_data_free);

    DYND_GET(data_init, arrfunc_data_init_t, NULL);
    DYND_GET(resolve_dst_type, arrfunc_resolve_dst_type_t, NULL);
    DYND_GET(instantiate, arrfunc_instantiate_t, NULL);
    DYND_GET(static_data_free, arrfunc_static_data_free_t, NULL);

    template <template <type_id_t...> class KernelType>
    struct make_all;

  } // namespace dynd::nd::detail

  template <typename T>
  struct declfunc;

  /**
   * Holds a single instance of an arrfunc in an nd::array,
   * providing some more direct convenient interface.
   */
  class arrfunc {
    nd::array m_value;

  public:
    arrfunc() = default;

    arrfunc(const ndt::type &self_tp, std::size_t data_size,
            arrfunc_data_init_t data_init,
            arrfunc_resolve_dst_type_t resolve_dst_type,
            arrfunc_instantiate_t instantiate)
        : m_value(empty(self_tp))
    {
      new (m_value.get_readwrite_originptr()) arrfunc_type_data(
          data_size, data_init, resolve_dst_type, instantiate);
    }

    template <typename T>
    arrfunc(const ndt::type &self_tp, T &&static_data, std::size_t data_size,
            arrfunc_data_init_t data_init,
            arrfunc_resolve_dst_type_t resolve_dst_type,
            arrfunc_instantiate_t instantiate)
        : m_value(empty(self_tp))
    {
      new (m_value.get_readwrite_originptr())
          arrfunc_type_data(std::forward<T>(static_data), data_size, data_init,
                            resolve_dst_type, instantiate);
    }

    arrfunc(const arrfunc &rhs) : m_value(rhs.m_value) {}

    /**
      * Constructor from an nd::array. Validates that the input
      * has "arrfunc" type.
      */
    arrfunc(const nd::array &rhs);

    arrfunc &operator=(const arrfunc &rhs)
    {
      m_value = rhs.m_value;
      return *this;
    }

    bool is_null() const { return m_value.is_null(); }

    arrfunc_type_data *get()
    {
      return !m_value.is_null()
                 ? const_cast<arrfunc_type_data *>(
                       reinterpret_cast<const arrfunc_type_data *>(
                           m_value.get_readonly_originptr()))
                 : NULL;
    }

    const arrfunc_type_data *get() const
    {
      return !m_value.is_null() ? reinterpret_cast<const arrfunc_type_data *>(
                                      m_value.get_readonly_originptr())
                                : NULL;
    }

    const ndt::arrfunc_type *get_type() const
    {
      return !m_value.is_null()
                 ? m_value.get_type().extended<ndt::arrfunc_type>()
                 : NULL;
    }

    const ndt::type &get_array_type() const { return m_value.get_type(); }

    operator nd::array() const { return m_value; }

    void swap(nd::arrfunc &rhs) { m_value.swap(rhs.m_value); }

    /** Implements the general call operator which returns an array */
    template <typename A, typename K>
    array call(const A &args, const K &kwds)
    {
      arrfunc_type_data *self = const_cast<arrfunc_type_data *>(get());
      const ndt::arrfunc_type *self_tp = get_type();

      array dst;

      // ...
      std::vector<ndt::type> kwd_tp(self_tp->get_nkwd());
      std::vector<intptr_t> available, missing;
      kwds.validate_names(self_tp, dst, kwd_tp, available, missing);

      std::map<nd::string, ndt::type> tp_vars;
      std::vector<ndt::type> arg_tp(self_tp->get_npos());
      std::vector<const char *> arg_arrmeta(self_tp->get_npos());
      std::vector<char *> arg_data(self_tp->get_npos());
      // Validate the array arguments
      args.validate_types(self_tp, arg_tp, arg_arrmeta, arg_data, tp_vars);

      // Validate the destination type, if it was provided
      if (!dst.is_null()) {
        if (!self_tp->get_return_type().match(NULL, dst.get_type(),
                                              dst.get_arrmeta(), tp_vars)) {
          std::stringstream ss;
          ss << "provided \"dst\" type " << dst.get_type()
             << " does not match arrfunc return type "
             << self_tp->get_return_type();
          throw std::invalid_argument(ss.str());
        }
      }

      // Validate the keyword arguments, and does substitutions to make
      // them concrete
      detail::validate_kwd_types(self_tp, kwd_tp, available, missing, tp_vars);

      // ...
      array kwds_as_array =
          kwds.as_array(ndt::struct_type::make(self_tp->get_kwd_names(), kwd_tp),
                        available, missing);

      ndt::type dst_tp;
      if (dst.is_null()) {
        dst_tp = self_tp->get_return_type();
        return (*self)(
            dst_tp, arg_tp.size(), arg_tp.empty() ? NULL : arg_tp.data(),
            arg_arrmeta.empty() ? NULL : arg_arrmeta.data(),
            arg_data.empty() ? NULL : arg_data.data(), kwds_as_array, tp_vars);
      }

      dst_tp = dst.get_type();
      (*self)(dst_tp, dst.get_arrmeta(), dst.get_readwrite_originptr(),
              arg_tp.size(), arg_tp.empty() ? NULL : arg_tp.data(),
              arg_arrmeta.empty() ? NULL : arg_arrmeta.data(),
              arg_data.empty() ? NULL : arg_data.data(), kwds_as_array,
              tp_vars);
      return dst;
    }

    /**
     * operator()()
     */
    nd::array operator()() { return call(detail::args<>(), detail::kwds<>()); }

    /**
    * operator()(kwds<...>(...))
    */
    template <typename... K>
    array operator()(detail::kwds<K...> &&k)
    {
      return call(detail::args<>(), std::forward<detail::kwds<K...>>(k));
    }

    /**
     * operator()(a0, a1, ..., an, kwds<...>(...))
     */
    template <typename... T>
    typename std::enable_if<
        sizeof...(T) != 3 && sizeof...(T) != 5 &&
            detail::is_kwds<typename back<type_sequence<T...>>::type>::value,
        array>::type
    operator()(T &&... a)
    {
      typedef make_index_sequence<sizeof...(T)-1> I;
      typedef typename instantiate<
          detail::args,
          typename to<type_sequence<typename as_array<T>::type...>,
                      sizeof...(T)-1>::type>::type args_type;

      args_type arr =
          index_proxy<I>::template make<args_type>(std::forward<T>(a)...);
      return call(arr, dynd::get<sizeof...(T)-1>(std::forward<T>(a)...));
    }

    template <typename A0, typename A1, typename... K>
    typename std::enable_if<detail::is_variadic_args<A0, A1>::value,
                            array>::type
    operator()(A0 &&a0, A1 &&a1, const detail::kwds<K...> &kwds)
    {
      return call(detail::args<array, array>(array(std::forward<A0>(a0)),
                                             array(std::forward<A1>(a1))),
                  kwds);
    }

    template <typename A0, typename A1, typename... K>
    typename std::enable_if<!detail::is_variadic_args<A0, A1>::value,
                            array>::type
    operator()(A0 &&a0, A1 &&a1, const detail::kwds<K...> &kwds)
    {
      return call(detail::args<intptr_t, array *>(std::forward<A0>(a0),
                                                  std::forward<A1>(a1)),
                  kwds);
    }

    template <typename A0, typename A1, typename A2, typename A3, typename... K>
    typename std::enable_if<detail::is_variadic_args<A0, A1, A2, A3>::value,
                            array>::type
    operator()(A0 &&a0, A1 &&a1, A2 &&a2, A3 &&a3,
               const detail::kwds<K...> &kwds)
    {
      return call(detail::args<array, array, array, array, array>(
                      std::forward<A0>(a0), std::forward<A1>(a1),
                      std::forward<A2>(a2), std::forward<A3>(a3)),
                  kwds);
    }

    template <typename A0, typename A1, typename A2, typename A3, typename... K>
    typename std::enable_if<!detail::is_variadic_args<A0, A1, A2, A3>::value,
                            array>::type
    operator()(A0 &&a0, A1 &&a1, A2 &&a2, A3 &&a3,
               const detail::kwds<K...> &kwds)
    {
      return call(detail::args<intptr_t, const ndt::type *, const char *const *,
                               char *const *>(
                      std::forward<A0>(a0), std::forward<A1>(a1),
                      std::forward<A2>(a2), std::forward<A3>(a3)),
                  kwds);
    }

    /**
     * operator()(a0, a1, ..., an)
     */
    template <typename... A>
    typename std::enable_if<
        !detail::is_kwds<typename back<type_sequence<A...>>::type>::value,
        array>::type
    operator()(A &&... a)
    {
      return (*this)(std::forward<A>(a)..., kwds());
    }

    template <typename KernelType>
    static typename std::enable_if<
        ndt::type::has_equivalent<KernelType>::value &&
            detail::has_data_size<KernelType>::value,
        arrfunc>::type
    make()
    {
      return arrfunc(ndt::type::equivalent<KernelType>::make(),
                     KernelType::data_size, detail::get_data_init<KernelType>(),
                     detail::get_resolve_dst_type<KernelType>(),
                     detail::get_instantiate<KernelType>());
    }

    template <typename KernelType, typename StaticDataType>
    static typename std::enable_if<
        ndt::type::has_equivalent<KernelType>::value &&
            detail::has_data_size<KernelType>::value,
        arrfunc>::type
    make(StaticDataType &&static_data)
    {
      return arrfunc(ndt::type::equivalent<KernelType>::make(),
                     std::forward<StaticDataType>(static_data),
                     KernelType::data_size, detail::get_data_init<KernelType>(),
                     detail::get_resolve_dst_type<KernelType>(),
                     detail::get_instantiate<KernelType>());
    }

    template <typename KernelType>
    static typename std::enable_if<
        ndt::type::has_equivalent<KernelType>::value &&
            !detail::has_data_size<KernelType>::value,
        arrfunc>::type
    make(std::size_t data_size)
    {
      return arrfunc(ndt::type::equivalent<KernelType>::make(), data_size,
                     detail::get_data_init<KernelType>(),
                     detail::get_resolve_dst_type<KernelType>(),
                     detail::get_instantiate<KernelType>());
    }

    template <typename KernelType, typename StaticDataType>
    static typename std::enable_if<
        ndt::type::has_equivalent<KernelType>::value &&
            !detail::has_data_size<KernelType>::value,
        arrfunc>::type
    make(StaticDataType &&static_data, std::size_t data_size)
    {
      return arrfunc(ndt::type::equivalent<KernelType>::make(),
                     std::forward<StaticDataType>(static_data), data_size,
                     detail::get_data_init<KernelType>(),
                     detail::get_resolve_dst_type<KernelType>(),
                     detail::get_instantiate<KernelType>());
    }

    template <typename KernelType>
    static typename std::enable_if<
        !ndt::type::has_equivalent<KernelType>::value &&
            detail::has_data_size<KernelType>::value,
        arrfunc>::type
    make(const ndt::type &self_tp)
    {
      return arrfunc(self_tp, KernelType::data_size,
                     detail::get_data_init<KernelType>(),
                     detail::get_resolve_dst_type<KernelType>(),
                     detail::get_instantiate<KernelType>());
    }

    template <typename KernelType, typename StaticDataType>
    static typename std::enable_if<
        !ndt::type::has_equivalent<KernelType>::value &&
            detail::has_data_size<KernelType>::value,
        arrfunc>::type
    make(const ndt::type &self_tp, StaticDataType &&static_data)
    {
      return arrfunc(self_tp, std::forward<StaticDataType>(static_data),
                     KernelType::data_size, detail::get_data_init<KernelType>(),
                     detail::get_resolve_dst_type<KernelType>(),
                     detail::get_instantiate<KernelType>());
    }

    template <typename KernelType>
    static typename std::enable_if<
        !ndt::type::has_equivalent<KernelType>::value &&
            !detail::has_data_size<KernelType>::value,
        arrfunc>::type
    make(const ndt::type &self_tp, std::size_t data_size)
    {
      return arrfunc(self_tp, data_size, detail::get_data_init<KernelType>(),
                     detail::get_resolve_dst_type<KernelType>(),
                     detail::get_instantiate<KernelType>());
    }

    template <typename KernelType, typename StaticDataType>
    static typename std::enable_if<
        !ndt::type::has_equivalent<KernelType>::value &&
            !detail::has_data_size<KernelType>::value,
        arrfunc>::type
    make(const ndt::type &self_tp, StaticDataType &&static_data,
         std::size_t data_size)
    {
      return arrfunc(self_tp, std::forward<StaticDataType>(static_data),
                     data_size, detail::get_data_init<KernelType>(),
                     detail::get_resolve_dst_type<KernelType>(),
                     detail::get_instantiate<KernelType>());
    }

    template <template <int> class CKT, typename T>
    static arrfunc make(const ndt::type &self_tp, T &&data, size_t data_size)
    {
      switch (self_tp.extended<ndt::arrfunc_type>()->get_npos()) {
      case 0:
        return make<CKT<0>>(self_tp, std::forward<T>(data), data_size);
      case 1:
        return make<CKT<1>>(self_tp, std::forward<T>(data), data_size);
      case 2:
        return make<CKT<2>>(self_tp, std::forward<T>(data), data_size);
      case 3:
        return make<CKT<3>>(self_tp, std::forward<T>(data), data_size);
      case 4:
        return make<CKT<4>>(self_tp, std::forward<T>(data), data_size);
      case 5:
        return make<CKT<5>>(self_tp, std::forward<T>(data), data_size);
      case 6:
        return make<CKT<6>>(self_tp, std::forward<T>(data), data_size);
      case 7:
        return make<CKT<7>>(self_tp, std::forward<T>(data), data_size);
      default:
        throw std::runtime_error("arrfunc with nsrc > 7 not implemented yet");
      }
    }

    template <template <type_id_t> class KernelType, typename I0, typename... A>
    static std::map<type_id_t, arrfunc> make_all(A &&... a)
    {
      std::map<type_id_t, arrfunc> arrfuncs;
      for_each<I0>(detail::make_all<KernelType>(), arrfuncs,
                   std::forward<A>(a)...);

      return arrfuncs;
    }

    template <template <type_id_t, type_id_t, type_id_t...> class KernelType,
              typename I0, typename I1, typename... I, typename... A>
    static std::map<std::array<type_id_t, 2 + sizeof...(I)>, arrfunc>
    make_all(A &&... a)
    {
      std::map<std::array<type_id_t, 2 + sizeof...(I)>, arrfunc> arrfuncs;
      for_each<typename outer<I0, I1, I...>::type>(
          detail::make_all<KernelType>(), arrfuncs, std::forward<A>(a)...);

      return arrfuncs;
    }
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
      void on_each(std::map<type_id_t, arrfunc> &arrfuncs, A &&... a) const
      {
        arrfuncs[TypeID] =
            arrfunc::make<KernelType<TypeID>>(std::forward<A>(a)...);
      }

      template <typename TypeIDSequence, typename... A>
      void on_each(std::map<std::array<type_id_t, TypeIDSequence::size>,
                            arrfunc> &arrfuncs,
                   A &&... a) const
      {
        arrfuncs[TypeIDSequence()] =
            arrfunc::make<typename apply<KernelType, TypeIDSequence>::type>(
                std::forward<A>(a)...);
      }
    };

  } // namespace dynd::nd::detail

  template <typename FuncType>
  struct declfunc {
    operator arrfunc &() { return get(); }

    operator const arrfunc &() const { return get(); }

    template <typename... A>
    array operator()(A &&... a)
    {
      return get()(std::forward<A>(a)...);
    }

    static arrfunc &get()
    {
      static arrfunc self = FuncType::make();
      return self;
    }
  };

  template <typename FuncType>
  std::ostream &operator<<(std::ostream &o, const declfunc<FuncType> &rhs)
  {
    o << static_cast<const arrfunc &>(rhs);

    return o;
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