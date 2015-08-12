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
#include <dynd/types/struct_type.hpp>
#include <dynd/types/substitute_typevars.hpp>
#include <dynd/types/type_type.hpp>

namespace dynd {
namespace nd {
  namespace detail {

    /**
     * Presently, there are some specially treated keyword arguments in
     * arrfuncs. The "dst_tp" keyword argument always tells the desired
     * output type, and the "dst" keyword argument always provides an
     * output array.
     */
    template <typename T>
    bool is_special_kwd(const ndt::callable_type *DYND_UNUSED(self_tp),
                        const std::string &DYND_UNUSED(name),
                        const T &DYND_UNUSED(value),
                        std::map<std::string, ndt::type> &DYND_UNUSED(tp_vars))
    {
      return false;
    }

    inline bool is_special_kwd(const ndt::callable_type *DYND_UNUSED(self_tp),
                               array &dst, const std::string &name,
                               const ndt::type &value)
    {
      if (name == "dst_tp") {
        dst = nd::empty(value);
        return true;
      }

      return false;
    }

    inline bool is_special_kwd(const ndt::callable_type *DYND_UNUSED(self_tp),
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
    void check_name(const ndt::callable_type *af_tp, array &dst,
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
             << "\" to callable with type " << ndt::type(af_tp, true);
          throw std::invalid_argument(ss.str());
        }
      } else {
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

    void fill_missing_values(const ndt::type *tp, char *arrmeta,
                             const uintptr_t *arrmeta_offsets, char *data,
                             const uintptr_t *data_offsets,
                             const std::vector<intptr_t> &missing);

    void check_narg(const ndt::callable_type *af_tp, intptr_t npos);

    void check_arg(const ndt::callable_type *af_tp, intptr_t i,
                   const ndt::type &actual_tp, const char *actual_arrmeta,
                   std::map<std::string, ndt::type> &tp_vars);

    void check_nkwd(const ndt::callable_type *af_tp,
                    const std::vector<intptr_t> &available,
                    const std::vector<intptr_t> &missing);

    void validate_kwd_types(const ndt::callable_type *af_tp,
                            std::vector<ndt::type> &kwd_tp,
                            const std::vector<intptr_t> &available,
                            const std::vector<intptr_t> &missing,
                            std::map<std::string, ndt::type> &tp_vars);

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
      }

      std::size_t size() const
      {
        return sizeof...(A);
      }

      struct {
        args *self;

        template <size_t I>
        void on_each(const ndt::callable_type *af_tp,
                     std::vector<ndt::type> &src_tp,
                     std::vector<const char *> &src_arrmeta,
                     std::vector<char *> &src_data,
                     std::map<std::string, ndt::type> &tp_vars) const
        {
          auto value = std::get<I>(self->m_values);
          const ndt::type &tp = ndt::type::make(value);
          const char *arrmeta = value.get_arrmeta();

          check_arg(af_tp, I, tp, arrmeta, tp_vars);

          src_tp.push_back(tp);
          src_arrmeta.push_back(arrmeta);
          src_data.push_back(data_of(value));
        }

        void operator()(const ndt::callable_type *af_tp,
                        std::vector<ndt::type> &src_tp,
                        std::vector<const char *> &src_arrmeta,
                        std::vector<char *> &src_data,
                        std::map<std::string, ndt::type> &tp_vars) const
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
      std::size_t size() const
      {
        return 0;
      }

      void validate_types(
          const ndt::callable_type *af_tp,
          std::vector<ndt::type> &DYND_UNUSED(src_tp),
          std::vector<const char *> &DYND_UNUSED(src_arrmeta),
          std::vector<char *> &DYND_UNUSED(src_data),
          std::map<std::string, ndt::type> &DYND_UNUSED(tp_vars)) const
      {
        check_narg(af_tp, 0);
      }
    };

    /** A way to pass a run-time array of array arguments */
    template <>
    class args<std::size_t, array *> {
      std::size_t m_size;
      array *m_values;

    public:
      args(std::size_t size, array *values) : m_size(size), m_values(values)
      {
      }

      std::size_t size() const
      {
        return m_size;
      }

      void validate_types(const ndt::callable_type *af_tp,
                          std::vector<ndt::type> &src_tp,
                          std::vector<const char *> &src_arrmeta,
                          std::vector<char *> &src_data,
                          std::map<std::string, ndt::type> &tp_vars) const
      {
        check_narg(af_tp, m_size);

        for (std::size_t i = 0; i < m_size; ++i) {
          array &value = m_values[i];
          const ndt::type &tp = value.get_type();
          const char *arrmeta = value.get_arrmeta();

          check_arg(af_tp, i, tp, arrmeta, tp_vars);

          src_tp.push_back(tp);
          src_arrmeta.push_back(arrmeta);
          src_data.push_back(data_of(value));
        }
      }
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
      void validate_names(const ndt::callable_type *af_tp,
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
        void on_each(const ndt::callable_type *af_tp, array &dst,
                     bool &has_dst_tp, std::vector<ndt::type> &kwd_tp,
                     std::vector<intptr_t> &available)
        {
          check_name(af_tp, dst, self->m_names[I], std::get<I>(self->m_values),
                     has_dst_tp, kwd_tp.data(), available);
        }

        void operator()(const ndt::callable_type *af_tp, array &dst,
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

      void validate_names(const ndt::callable_type *af_tp, array &dst,
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
      enum {
        value = true
      };
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
 * A function to provide keyword arguments to an callable. The arguments
 * must alternate between the keyword name and the argument value.
 *
 *   callable af = <some callable>;
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
inline nd::detail::kwds<> kwds()
{
  return nd::detail::kwds<>();
}

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

    DYND_HAS(data_size);
    DYND_HAS(data_init);
    DYND_HAS(resolve_dst_type);
    DYND_HAS(instantiate);
    DYND_HAS(static_data_free);

    DYND_GET(static_data_free, callable_static_data_free_t, NULL);

    template <typename KernelType>
    typename std::enable_if<std::is_same<decltype(&KernelType::instantiate),
                                         callable_instantiate_t>::value,
                            callable_instantiate_t>::type
    get_instantiate()
    {
      return &KernelType::instantiate;
    }

    template <typename KernelType>
    typename std::enable_if<!std::is_same<decltype(&KernelType::instantiate),
                                          callable_instantiate_t>::value,
                            callable_instantiate_t>::type
    get_instantiate()
    {
      return [](char *static_data, size_t data_size, char *data, void *ckb,
                intptr_t ckb_offset, const ndt::type &dst_tp,
                const char *dst_arrmeta, intptr_t nsrc, const ndt::type *src_tp,
                const char *const *src_arrmeta, kernel_request_t kernreq,
                const eval::eval_context *ectx, const array &kwds,
                const std::map<std::string, ndt::type> &tp_vars) {
        typedef instantiate_traits<decltype(&KernelType::instantiate)> traits;
        intptr_t res_ckb_offset = KernelType::instantiate(
            reinterpret_cast<typename traits::static_data_type *>(static_data),
            data_size, reinterpret_cast<typename traits::data_type *>(data),
            ckb, ckb_offset, dst_tp, dst_arrmeta, nsrc, src_tp, src_arrmeta,
            kernreq, ectx, kwds, tp_vars);
        reinterpret_cast<typename traits::data_type *>(data)->~data_type();
        return res_ckb_offset;
      };
    }

    template <typename KernelType>
    typename std::enable_if<!has_data_init<KernelType>::value,
                            callable_data_init_t>::type
    get_data_init()
    {
      return NULL;
    }

    template <typename KernelType>
    typename std::enable_if<
        has_data_init<KernelType>::value &&std::is_same<
            decltype(&KernelType::data_init), callable_data_init_t>::value,
        callable_data_init_t>::type
    get_data_init()
    {
      return &KernelType::data_init;
    }

    template <typename KernelType>
    typename std::enable_if<has_data_init<KernelType>::value &&
                                !std::is_same<decltype(&KernelType::data_init),
                                              callable_data_init_t>::value,
                            callable_data_init_t>::type
    get_data_init()
    {
      return [](char *static_data, size_t data_size, char *data,
                const ndt::type &dst_tp, intptr_t nsrc, const ndt::type *src_tp,
                const nd::array &kwds,
                const std::map<std::string, ndt::type> &tp_vars) {
        typedef data_init_traits<decltype(&KernelType::data_init)> traits;
        KernelType::data_init(
            reinterpret_cast<typename traits::static_data_type *>(static_data),
            data_size, reinterpret_cast<typename traits::data_type *>(data),
            dst_tp, nsrc, src_tp, kwds, tp_vars);
      };
    }

    template <typename KernelType>
    typename std::enable_if<!has_resolve_dst_type<KernelType>::value,
                            callable_resolve_dst_type_t>::type
    get_resolve_dst_type()
    {
      return NULL;
    }

    template <typename KernelType>
    typename std::enable_if<
        has_resolve_dst_type<KernelType>::value &&
            std::is_same<decltype(&KernelType::resolve_dst_type),
                         callable_resolve_dst_type_t>::value,
        callable_resolve_dst_type_t>::type
    get_resolve_dst_type()
    {
      return &KernelType::resolve_dst_type;
    }

    template <typename KernelType>
    typename std::enable_if<
        has_resolve_dst_type<KernelType>::value &&
            !std::is_same<decltype(&KernelType::resolve_dst_type),
                          callable_resolve_dst_type_t>::value,
        callable_resolve_dst_type_t>::type
    get_resolve_dst_type()
    {
      return [](char *static_data, size_t data_size, char *data,
                ndt::type &dst_tp, intptr_t nsrc, const ndt::type *src_tp,
                const nd::array &kwds,
                const std::map<std::string, ndt::type> &tp_vars) {
        typedef resolve_dst_type_traits<decltype(&KernelType::resolve_dst_type)>
        traits;
        KernelType::resolve_dst_type(
            reinterpret_cast<typename traits::static_data_type *>(static_data),
            data_size, reinterpret_cast<typename traits::data_type *>(data),
            dst_tp, nsrc, src_tp, kwds, tp_vars);
      };
    }

    template <template <type_id_t...> class KernelType>
    struct make_all;

  } // namespace dynd::nd::detail

  template <typename T>
  struct declfunc;

  /**
   * Holds a single instance of an callable in an nd::array,
   * providing some more direct convenient interface.
   */
  class callable {
    nd::array m_value;

  public:
    callable() = default;

    callable(const ndt::type &self_tp, expr_single_t single,
             expr_strided_t strided)
        : m_value(empty(self_tp))
    {
      new (m_value.get_readwrite_originptr())
          callable_type_data(single, strided);
    }

    callable(const ndt::type &self_tp, std::size_t data_size,
             callable_data_init_t data_init,
             callable_resolve_dst_type_t resolve_dst_type,
             callable_instantiate_t instantiate)
        : m_value(empty(self_tp))
    {
      new (m_value.get_readwrite_originptr()) callable_type_data(
          data_size, data_init, resolve_dst_type, instantiate);
    }

    template <typename T>
    callable(const ndt::type &self_tp, T &&static_data, std::size_t data_size,
             callable_data_init_t data_init,
             callable_resolve_dst_type_t resolve_dst_type,
             callable_instantiate_t instantiate)
        : m_value(empty(self_tp))
    {
      new (m_value.get_readwrite_originptr())
          callable_type_data(std::forward<T>(static_data), data_size, data_init,
                             resolve_dst_type, instantiate);
    }

    callable(const callable &rhs) : m_value(rhs.m_value)
    {
    }

    /**
      * Constructor from an nd::array. Validates that the input
      * has "callable" type.
      */
    explicit callable(const nd::array &rhs);

    callable &operator=(const callable &rhs)
    {
      m_value = rhs.m_value;
      return *this;
    }

    bool is_null() const
    {
      return m_value.is_null();
    }

    callable_type_data *get()
    {
      return !m_value.is_null()
                 ? const_cast<callable_type_data *>(
                       reinterpret_cast<const callable_type_data *>(
                           m_value.get_readonly_originptr()))
                 : NULL;
    }

    const callable_type_data *get() const
    {
      return !m_value.is_null() ? reinterpret_cast<const callable_type_data *>(
                                      m_value.get_readonly_originptr())
                                : NULL;
    }

    const ndt::callable_type *get_type() const
    {
      return !m_value.is_null()
                 ? m_value.get_type().extended<ndt::callable_type>()
                 : NULL;
    }

    const ndt::type &get_array_type() const
    {
      return m_value.get_type();
    }

    const ndt::type &get_ret_type() const
    {
      return get_type()->get_return_type();
    }

    const array &get_arg_types() const
    {
      return get_type()->get_pos_types();
    }

    operator nd::array() const
    {
      return m_value;
    }

    void swap(nd::callable &rhs)
    {
      m_value.swap(rhs.m_value);
    }

    /** Implements the general call operator which returns an array */
    template <typename A, typename K>
    array call(const A &args, const K &kwds)
    {
      const ndt::callable_type *self_tp = get_type();

      array dst;

      // ...
      std::vector<ndt::type> kwd_tp(self_tp->get_nkwd());
      std::vector<intptr_t> available, missing;
      kwds.validate_names(self_tp, dst, kwd_tp, available, missing);

      std::map<std::string, ndt::type> tp_vars;
      std::vector<ndt::type> arg_tp;
      std::vector<const char *> arg_arrmeta;
      std::vector<char *> arg_data;

      // Validate the array arguments
      args.validate_types(self_tp, arg_tp, arg_arrmeta, arg_data, tp_vars);

      // Validate the destination type, if it was provided
      if (!dst.is_null()) {
        if (!self_tp->get_return_type().match(NULL, dst.get_type(),
                                              dst.get_arrmeta(), tp_vars)) {
          std::stringstream ss;
          ss << "provided \"dst\" type " << dst.get_type()
             << " does not match callable return type "
             << self_tp->get_return_type();
          throw std::invalid_argument(ss.str());
        }
      }

      // Validate the keyword arguments, and does substitutions to make
      // them concrete
      detail::validate_kwd_types(self_tp, kwd_tp, available, missing, tp_vars);

      // ...
      array kwds_as_array = kwds.as_array(
          ndt::struct_type::make(self_tp->get_kwd_names(), kwd_tp), available,
          missing);

      ndt::type dst_tp;
      if (dst.is_null()) {
        dst_tp = self_tp->get_return_type();
        return (*get())(
            dst_tp, arg_tp.size(), arg_tp.empty() ? NULL : arg_tp.data(),
            arg_arrmeta.empty() ? NULL : arg_arrmeta.data(),
            arg_data.empty() ? NULL : arg_data.data(), kwds_as_array, tp_vars);
      }

      dst_tp = dst.get_type();
      (*get())(dst_tp, dst.get_arrmeta(), dst.get_readwrite_originptr(),
               arg_tp.size(), arg_tp.empty() ? NULL : arg_tp.data(),
               arg_arrmeta.empty() ? NULL : arg_arrmeta.data(),
               arg_data.empty() ? NULL : arg_data.data(), kwds_as_array,
               tp_vars);
      return dst;
    }

    /**
     * operator()()
     */
    nd::array operator()()
    {
      return call(detail::args<>(), detail::kwds<>());
    }

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
        sizeof...(T) != 3 &&
            detail::is_kwds<typename back<type_sequence<T...>>::type>::value,
        array>::type
    operator()(T &&... a)
    {
      typedef make_index_sequence<sizeof...(T) - 1> I;
      typedef typename instantiate<
          detail::args,
          typename to<type_sequence<typename as_array<T>::type...>,
                      sizeof...(T) - 1>::type>::type args_type;

      args_type arr =
          index_proxy<I>::template make<args_type>(std::forward<T>(a)...);
      return call(arr, dynd::get<sizeof...(T) - 1>(std::forward<T>(a)...));
    }

    template <typename A0, typename A1, typename... K>
    typename std::enable_if<!std::is_convertible<A0 &&, std::size_t>::value ||
                                !std::is_convertible<A1 &&, array *>::value,
                            array>::type
    operator()(A0 &&a0, A1 &&a1, const detail::kwds<K...> &kwds)
    {
      return call(detail::args<array, array>(array(std::forward<A0>(a0)),
                                             array(std::forward<A1>(a1))),
                  kwds);
    }

    template <typename A0, typename A1, typename... K>
    typename std::enable_if<std::is_convertible<A0 &&, std::size_t>::value &&
                                std::is_convertible<A1 &&, array *>::value,
                            array>::type
    operator()(A0 &&a0, A1 &&a1, const detail::kwds<K...> &kwds)
    {
      return call(detail::args<std::size_t, array *>(std::forward<A0>(a0),
                                                     std::forward<A1>(a1)),
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
        callable>::type
    make()
    {
      return callable(ndt::type::equivalent<KernelType>::make(),
                      KernelType::data_size,
                      detail::get_data_init<KernelType>(),
                      detail::get_resolve_dst_type<KernelType>(),
                      detail::get_instantiate<KernelType>());
    }

    template <typename KernelType, typename StaticDataType>
    static typename std::enable_if<
        ndt::type::has_equivalent<KernelType>::value &&
            detail::has_data_size<KernelType>::value,
        callable>::type
    make(StaticDataType &&static_data)
    {
      return callable(ndt::type::equivalent<KernelType>::make(),
                      std::forward<StaticDataType>(static_data),
                      KernelType::data_size,
                      detail::get_data_init<KernelType>(),
                      detail::get_resolve_dst_type<KernelType>(),
                      detail::get_instantiate<KernelType>());
    }

    template <typename KernelType>
    static typename std::enable_if<
        ndt::type::has_equivalent<KernelType>::value &&
            !detail::has_data_size<KernelType>::value,
        callable>::type
    make(std::size_t data_size)
    {
      return callable(ndt::type::equivalent<KernelType>::make(), data_size,
                      detail::get_data_init<KernelType>(),
                      detail::get_resolve_dst_type<KernelType>(),
                      detail::get_instantiate<KernelType>());
    }

    template <typename KernelType, typename StaticDataType>
    static typename std::enable_if<
        ndt::type::has_equivalent<KernelType>::value &&
            !detail::has_data_size<KernelType>::value,
        callable>::type
    make(StaticDataType &&static_data, std::size_t data_size)
    {
      return callable(ndt::type::equivalent<KernelType>::make(),
                      std::forward<StaticDataType>(static_data), data_size,
                      detail::get_data_init<KernelType>(),
                      detail::get_resolve_dst_type<KernelType>(),
                      detail::get_instantiate<KernelType>());
    }

    template <typename KernelType>
    static typename std::enable_if<
        !ndt::type::has_equivalent<KernelType>::value &&
            detail::has_data_size<KernelType>::value,
        callable>::type
    make(const ndt::type &self_tp)
    {
      return callable(self_tp, KernelType::data_size,
                      detail::get_data_init<KernelType>(),
                      detail::get_resolve_dst_type<KernelType>(),
                      detail::get_instantiate<KernelType>());
    }

    template <typename KernelType, typename StaticDataType>
    static typename std::enable_if<
        !ndt::type::has_equivalent<KernelType>::value &&
            detail::has_data_size<KernelType>::value,
        callable>::type
    make(const ndt::type &self_tp, StaticDataType &&static_data)
    {
      return callable(self_tp, std::forward<StaticDataType>(static_data),
                      KernelType::data_size,
                      detail::get_data_init<KernelType>(),
                      detail::get_resolve_dst_type<KernelType>(),
                      detail::get_instantiate<KernelType>());
    }

    template <typename KernelType>
    static typename std::enable_if<
        !ndt::type::has_equivalent<KernelType>::value &&
            !detail::has_data_size<KernelType>::value,
        callable>::type
    make(const ndt::type &self_tp, std::size_t data_size)
    {
      return callable(self_tp, data_size, detail::get_data_init<KernelType>(),
                      detail::get_resolve_dst_type<KernelType>(),
                      detail::get_instantiate<KernelType>());
    }

    template <typename KernelType, typename StaticDataType>
    static typename std::enable_if<
        !ndt::type::has_equivalent<KernelType>::value &&
            !detail::has_data_size<KernelType>::value,
        callable>::type
    make(const ndt::type &self_tp, StaticDataType &&static_data,
         std::size_t data_size)
    {
      return callable(self_tp, std::forward<StaticDataType>(static_data),
                      data_size, detail::get_data_init<KernelType>(),
                      detail::get_resolve_dst_type<KernelType>(),
                      detail::get_instantiate<KernelType>());
    }

    template <template <int> class CKT, typename T>
    static callable make(const ndt::type &self_tp, T &&data, size_t data_size)
    {
      switch (self_tp.extended<ndt::callable_type>()->get_npos()) {
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
        throw std::runtime_error("callable with nsrc > 7 not implemented yet");
      }
    }

    template <template <type_id_t> class KernelType, typename I0, typename... A>
    static std::map<type_id_t, callable> make_all(A &&... a)
    {
      std::map<type_id_t, callable> callables;
      for_each<I0>(detail::make_all<KernelType>(), callables,
                   std::forward<A>(a)...);

      return callables;
    }

    template <template <type_id_t, type_id_t, type_id_t...> class KernelType,
              typename I0, typename I1, typename... I, typename... A>
    static std::map<std::array<type_id_t, 2 + sizeof...(I)>, callable>
    make_all(A &&... a)
    {
      std::map<std::array<type_id_t, 2 + sizeof...(I)>, callable> callables;
      for_each<typename outer<I0, I1, I...>::type>(
          detail::make_all<KernelType>(), callables, std::forward<A>(a)...);

      return callables;
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
      void on_each(std::map<type_id_t, callable> &callables, A &&... a) const
      {
        callables[TypeID] =
            callable::make<KernelType<TypeID>>(std::forward<A>(a)...);
      }

      template <typename TypeIDSequence, typename... A>
      void on_each(std::map<std::array<type_id_t, TypeIDSequence::size>,
                            callable> &callables,
                   A &&... a) const
      {
        callables[TypeIDSequence()] =
            callable::make<typename apply<KernelType, TypeIDSequence>::type>(
                std::forward<A>(a)...);
      }
    };

  } // namespace dynd::nd::detail

  template <typename FuncType>
  struct declfunc {
    operator callable &()
    {
      return get();
    }

    operator const callable &() const
    {
      return get();
    }

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
nd::callable make_callable_from_assignment(const ndt::type &dst_tp,
                                           const ndt::type &src_tp,
                                           assign_error_mode errmode);

/**
 * Creates an callable which does the assignment from
 * data of `tp` to its property `propname`
 *
 * \param tp  The type of the source.
 * \param propname  The name of the property.
 */
nd::callable make_callable_from_property(const ndt::type &tp,
                                         const std::string &propname);

} // namespace dynd
