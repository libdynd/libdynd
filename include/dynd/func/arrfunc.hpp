//
// Copyright (C) 2011-15 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#pragma once

#include <dynd/config.hpp>
#include <dynd/eval/eval_context.hpp>
#include <dynd/types/base_type.hpp>
#include <dynd/types/arrfunc_type.hpp>
#include <dynd/kernels/ckernel_builder.hpp>
#include <dynd/types/struct_type.hpp>
#include <dynd/types/substitute_typevars.hpp>
#include <dynd/types/type_type.hpp>

#define DYND_HAS_MEM_FUNC(NAME)                                                \
  template <typename T, typename S>                                            \
  class DYND_PP_PASTE(has_, NAME) {                                            \
    template <typename U, U>                                                   \
    struct test_type;                                                          \
    typedef char true_type[1];                                                 \
    typedef char false_type[2];                                                \
                                                                               \
    template <typename U>                                                      \
    static true_type &test(test_type<S, &U::NAME> *);                          \
    template <typename>                                                        \
    static false_type &test(...);                                              \
                                                                               \
  public:                                                                      \
    static bool const value = sizeof(test<T>(0)) == sizeof(true_type);         \
  }
#define DYND_GET_MEM_FUNC(TYPE, NAME)                                          \
  template <typename T, bool DYND_PP_PASTE(has_, NAME)>                        \
  typename std::enable_if<DYND_PP_PASTE(has_, NAME), TYPE>::type               \
      DYND_PP_PASTE(get_, NAME)()                                              \
  {                                                                            \
    return DYND_PP_META_SCOPE(T, NAME);                                        \
  }                                                                            \
                                                                               \
  template <typename T, bool DYND_PP_PASTE(has_, NAME)>                        \
  typename std::enable_if<!DYND_PP_PASTE(has_, NAME), TYPE>::type              \
      DYND_PP_PASTE(get_, NAME)()                                              \
  {                                                                            \
    return NULL;                                                               \
  }                                                                            \
                                                                               \
  template <typename T>                                                        \
  TYPE DYND_PP_PASTE(get_, NAME)()                                             \
  {                                                                            \
    return DYND_PP_PASTE(                                                      \
        get_, NAME)<T, DYND_PP_PASTE(has_, NAME)<T, TYPE>::value>();           \
  }

namespace dynd {

namespace ndt {
  ndt::type make_option(const ndt::type &value_tp);
}

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
    const arrfunc_type_data *self, const arrfunc_type *self_tp, void *ckb,
    intptr_t ckb_offset, const ndt::type &dst_tp, const char *dst_arrmeta,
    const ndt::type *src_tp, const char *const *src_arrmeta,
    kernel_request_t kernreq, const eval::eval_context *ectx,
    const nd::array &kwds, const std::map<nd::string, ndt::type> &tp_vars);

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
    const arrfunc_type_data *self, const arrfunc_type *self_tp, intptr_t nsrc,
    const ndt::type *src_tp, nd::array &kwds,
    const std::map<nd::string, ndt::type> &tp_vars);

/**
 * A function which deallocates the memory behind data_ptr after
 * freeing any additional resources it might contain.
 */
typedef void (*arrfunc_free_t)(arrfunc_type_data *self);

typedef ndt::type (*arrfunc_make_type_t)();

namespace detail {
  DYND_HAS_MEM_FUNC(make_type);
  DYND_HAS_MEM_FUNC(instantiate);
  DYND_HAS_MEM_FUNC(resolve_option_values);
  DYND_HAS_MEM_FUNC(resolve_dst_type);
  DYND_HAS_MEM_FUNC(free);
  DYND_GET_MEM_FUNC(arrfunc_make_type_t, make_type);
  DYND_GET_MEM_FUNC(arrfunc_instantiate_t, instantiate);
  DYND_GET_MEM_FUNC(arrfunc_resolve_option_values_t, resolve_option_values);
  DYND_GET_MEM_FUNC(arrfunc_resolve_dst_type_t, resolve_dst_type);
  DYND_GET_MEM_FUNC(arrfunc_free_t, free);
} // namespace dynd::detail

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
  char data[4 * 8];

  arrfunc_instantiate_t instantiate;
  arrfunc_resolve_option_values_t resolve_option_values;
  arrfunc_resolve_dst_type_t resolve_dst_type;
  arrfunc_free_t free;

  arrfunc_type_data()
      : instantiate(NULL), resolve_option_values(NULL), resolve_dst_type(NULL),
        free(NULL)
  {
    DYND_STATIC_ASSERT((sizeof(arrfunc_type_data) & 7) == 0,
                       "arrfunc_type_data must have size divisible by 8");
  }

  arrfunc_type_data(arrfunc_instantiate_t instantiate,
                    arrfunc_resolve_option_values_t resolve_option_values,
                    arrfunc_resolve_dst_type_t resolve_dst_type)
      : instantiate(instantiate), resolve_option_values(resolve_option_values),
        resolve_dst_type(resolve_dst_type)
  {
  }

  arrfunc_type_data(arrfunc_instantiate_t instantiate,
                    arrfunc_resolve_option_values_t resolve_option_values,
                    arrfunc_resolve_dst_type_t resolve_dst_type,
                    arrfunc_free_t free)
      : instantiate(instantiate), resolve_option_values(resolve_option_values),
        resolve_dst_type(resolve_dst_type), free(free)
  {
  }

  template <typename T>
  arrfunc_type_data(const T &data, arrfunc_instantiate_t instantiate,
                    arrfunc_resolve_option_values_t resolve_option_values,
                    arrfunc_resolve_dst_type_t resolve_dst_type,
                    arrfunc_free_t free = NULL)
      : instantiate(instantiate), resolve_option_values(resolve_option_values),
        resolve_dst_type(resolve_dst_type),
        free(free == NULL ? &destroy_wrapper<T> : free)
  {
    new (this->data) T(data);
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

namespace nd {
  namespace detail {
    template <typename K>
    typename std::enable_if<!eval::is_eval_context<K>::value, bool>::type is_special_kwd(const arrfunc_type *DYND_UNUSED(self_tp),
                        const std::string &DYND_UNUSED(name),
                        K DYND_UNUSED(value), const eval::eval_context *&DYND_UNUSED(ectx),
                        std::map<nd::string, ndt::type> &DYND_UNUSED(tp_vars))
    {
      return false;
    }

    template <typename K>
    typename std::enable_if<eval::is_eval_context<K>::value, bool>::type
    is_special_kwd(const arrfunc_type *DYND_UNUSED(self_tp),
                   const std::string &name, K value,
                   const eval::eval_context *&ectx,
                   std::map<nd::string, ndt::type> &DYND_UNUSED(tp_vars))
    {
      if (name == "ectx") {
        ectx = value;
        return true;
      }

      return false;
    }

    inline bool is_special_kwd(const arrfunc_type *self_tp,
                               const std::string &name, const ndt::type &value,
                               const eval::eval_context *&DYND_UNUSED(ectx),
                               std::map<nd::string, ndt::type> &tp_vars)
    {
      if (name == "dst_tp") {
        const ndt::type &expected_tp = self_tp->get_return_type();
        if (value.matches(expected_tp, tp_vars)) {
          return true;
        }

        std::stringstream ss;
        ss << "keyword \"dst_tp\" does not match, ";
        ss << "arrfunc expected " << expected_tp << " but passed " << value;
        throw std::invalid_argument(ss.str());
      }

      return false;
    }

    template <typename... A>
    class args {
      std::tuple<A...> m_values;
      const char *m_arrmeta[sizeof...(A)];

    public:
      args(A &&... a) : m_values(std::forward<A>(a)...)
      {
        resolve.self = this;

        typedef make_index_sequence<sizeof...(A)> I;
        nd::index_proxy<I>::template get_arrmeta(m_arrmeta, get_vals());
      }

      size_t size() const { return sizeof...(A); }

      const std::tuple<A...> &get_vals() const { return m_values; }

      struct {
        args *self;

        template <size_t I>
        void operator()(std::vector<ndt::type> &src_tp,
                        std::vector<const char *> &src_arrmeta,
                        std::vector<char *> &src_data) const
        {
          const nd::array &value = std::get<I>(self->m_values);

          src_tp.push_back(ndt::as_type(value));
          src_arrmeta.push_back(self->m_arrmeta[I]);
          src_data.push_back(const_cast<char *>(value.get_readonly_originptr()));
        }

        void operator()(std::vector<ndt::type> &src_tp,
                        std::vector<const char *> &src_arrmeta,
                        std::vector<char *> &src_data) const
        {
          typedef make_index_sequence<sizeof...(A)> I;
          dynd::index_proxy<I>::for_each(*this, src_tp, src_arrmeta, src_data);
        }
      } resolve;
    };

    template <>
    class args<intptr_t, nd::array *> {
      intptr_t m_narg;
      nd::array *m_args;

    public:
      args(intptr_t narg, nd::array *args) : m_narg(narg), m_args(args) {}

      size_t size() const { return m_narg; }

      void resolve(std::vector<ndt::type> &src_tp,
                   std::vector<const char *> &src_arrmeta,
                   std::vector<char *> &src_data) const
      {
        for (intptr_t i = 0; i < m_narg; ++i) {
          src_tp.push_back(m_args[i].get_type());
          src_arrmeta.push_back(m_args[i].get_arrmeta());
          src_data.push_back(
              const_cast<char *>(m_args[i].get_readonly_originptr()));
        }
      }
    };

    template <>
    class args<> {
    public:
      size_t size() const { return 0; }

      void resolve(std::vector<ndt::type> &DYND_UNUSED(src_tp),
                   std::vector<const char *> &DYND_UNUSED(src_arrmeta),
                   std::vector<char *> &DYND_UNUSED(src_data)) const
      {
      }

      std::vector<const char *> get_arrmeta() const
      {
        return std::vector<const char *>();
      }
    };

    template <typename... T>
    class kwds;

    template <>
    class kwds<> {
    public:
      static intptr_t get_size() { return 0; }

      void resolve_available_types(
          const arrfunc_type *DYND_UNUSED(self_tp), const std::vector<intptr_t> &DYND_UNUSED(available),
          ndt::type *DYND_UNUSED(tp),
          std::map<nd::string, ndt::type> &DYND_UNUSED(typevars)) const
      {
      }

      std::vector<intptr_t> is_permutation(
          const arrfunc_type *DYND_UNUSED(self_tp), intptr_t &nkwd, const eval::eval_context *&DYND_UNUSED(ectx),
          std::map<nd::string, ndt::type> &DYND_UNUSED(tp_vars)) const
      {
        nkwd = 0;
        return std::vector<intptr_t>();
      }

      nd::array forward_as_array(const nd::array &names, nd::array &types,
                                 const std::vector<intptr_t> &DYND_UNUSED(available),
                                 const std::vector<intptr_t> &missing) const
      {
        nd::array res = nd::empty_shell(ndt::make_struct(names, types));
        struct_type::fill_default_data_offsets(
            res.get_dim_size(),
            reinterpret_cast<const ndt::type *>(types.get_readonly_originptr()),
            reinterpret_cast<uintptr_t *>(res.get_arrmeta()));

        const ndt::type *tp = reinterpret_cast<const ndt::type *>(types.get_readonly_originptr());
        char *arrmeta = res.get_arrmeta();
        const uintptr_t *arrmeta_offsets = res.get_type()
                                               .extended<base_struct_type>()
                                               ->get_arrmeta_offsets_raw();
        char *data = res.get_readwrite_originptr();
        const uintptr_t *data_offsets = res.get_type().extended<base_struct_type>()->get_data_offsets(
                res.get_arrmeta());

        for (intptr_t j : missing) {
          tp[j].extended()->arrmeta_default_construct(
              arrmeta + arrmeta_offsets[j], true);
          assign_na(tp[j], arrmeta + arrmeta_offsets[j], data + data_offsets[j],
                    &eval::default_eval_context);
        }

        return res;
      }
    };

    template <typename... K>
    class kwds {
      const char *m_names[sizeof...(K)];
      std::tuple<K...> m_values;

      template <size_t I>
      void set_name(const char *name)
      {
        m_names[I] = name;
      }

      struct {
        template <size_t I>
        void operator()(kwds<K...> *self,
                        typename as_<K, const char *>::type... names)
        {
          self->set_name<I>(get<I>(names...));
        }
      } set_names;

      struct {
        template <size_t I>
        void operator()(const kwds<K...> *self,
                        const std::vector<intptr_t> &available,
                        const arrfunc_type *af_tp, ndt::type *kwd_tp_data,
                        std::map<nd::string, ndt::type> &typevars) const
        {
          intptr_t j = available[I];
          if (j != -1) {
            const std::string &name = self->get_name<I>();
            const auto &value = self->get_value<I>();

            ndt::type &actual_tp = kwd_tp_data[j];
            if (!actual_tp.is_null()) {
              std::stringstream ss;
              ss << "arrfunc passed keyword \"" << name << "\" more than once";
              throw std::invalid_argument(ss.str());
            }

            ndt::type expected_tp = af_tp->get_kwd_type(j);
            if (expected_tp.get_type_id() == option_type_id) {
              expected_tp = expected_tp.p("value_type").as<ndt::type>();
            }
            actual_tp = ndt::as_type(value);

            if (!actual_tp.value_type().matches(expected_tp, typevars)) {
              std::stringstream ss;
              ss << "keyword \"" << name << "\" does not match, ";
              ss << "arrfunc expected " << expected_tp << " but passed "
                 << actual_tp;
              throw std::invalid_argument(ss.str());
            }
          }
        }
      } resolve_available_type;

    public:
      kwds(typename as_<K, const char *>::type... names, K &&... values)
          : m_values(std::forward<K>(values)...)
      {
        typedef make_index_sequence<sizeof...(K)> I;
        dynd::index_proxy<I>::for_each(set_names, this, names...);
      }

      static intptr_t get_size() { return sizeof...(K); }

      template <size_t I>
      const char *get_name() const
      {
        return m_names[I];
      }

      template <size_t I>
      const typename std::tuple_element<I, std::tuple<K...>>::type &
      get_value() const
      {
        return std::get<I>(m_values);
      }

      struct {
        template <size_t I>
        void operator()(const kwds<K...> *self, intptr_t &nkwd, const arrfunc_type *af_tp,
                        std::vector<intptr_t> &available, const eval::eval_context *&ectx,
                        std::map<nd::string, ndt::type> &tp_vars)
        {
          const std::string &name = self->get_name<I>();
          const auto &value = self->get_value<I>();

          intptr_t j = af_tp->get_kwd_index(name);
          if (j != -1) {
            ++nkwd;
          } else if (!is_special_kwd(af_tp, name, value, ectx, tp_vars)) {
            std::stringstream ss;
            ss << "passed an unexpected keyword \"" << name
               << "\" to arrfunc with type " << ndt::type(af_tp, true);
            throw std::invalid_argument(ss.str());
          }
          available.push_back(j);
        }
      } is_permutation_ex;

      std::vector<intptr_t>
      is_permutation(const arrfunc_type *af_tp, intptr_t &nkwd, const eval::eval_context *&ectx,
                     std::map<nd::string, ndt::type> &tp_vars) const
      {
        typedef make_index_sequence<sizeof...(K)> I;

        std::vector<intptr_t> permutation;
        nkwd = 0;
        dynd::index_proxy<I>::for_each(is_permutation_ex, this, nkwd, af_tp, permutation, ectx, tp_vars);
        return permutation;
      }

      void resolve_available_types(
          const arrfunc_type *self_tp, const std::vector<intptr_t> &available,
          ndt::type *kwd_tp_data, std::map<nd::string, ndt::type> &typevars) const
      {
        typedef make_index_sequence<sizeof...(K)> I;

        dynd::index_proxy<I>::for_each(resolve_available_type, this, available,
                                       self_tp, kwd_tp_data, typevars);
      }

      struct {
        template <size_t I>
        void operator()(ndt::type *tp,
                        const std::tuple<K...> &values,
                        const std::vector<intptr_t> &available) const
        {
          intptr_t j = available[I];
          if (j != -1) {
            tp[j] = ndt::get_forward_type(std::get<I>(values));
          }
        }
      } get_forward_types_ex;

      struct {
        template <size_t I>
        void operator()(const ndt::type *tp, char *arrmeta,
                        const uintptr_t *arrmeta_offsets, char *data,
                        const uintptr_t *data_offsets,
                        const std::tuple<K...> &values,
                        const intptr_t *available)
        {
          intptr_t j = available[I];
          if (j != -1) {
            nd::forward_as_array(tp[j], arrmeta + arrmeta_offsets[j],
                             data + data_offsets[j], std::get<I>(values));
          }
        }
      } forward_as_array_ex;

      nd::array forward_as_array(const nd::array &names, nd::array &types,
                                 const std::vector<intptr_t> &available,
                                 const std::vector<intptr_t> &missing) const
      {
        typedef make_index_sequence<sizeof...(K)> I;
        dynd::index_proxy<I>::for_each(
            get_forward_types_ex,
            reinterpret_cast<ndt::type *>(types.get_readwrite_originptr()),
            m_values, available);

        nd::array res = nd::empty_shell(ndt::make_struct(names, types));
        struct_type::fill_default_data_offsets(
            res.get_dim_size(),
            reinterpret_cast<const ndt::type *>(types.get_readonly_originptr()),
            reinterpret_cast<uintptr_t *>(res.get_arrmeta()));

        const ndt::type *tp = reinterpret_cast<const ndt::type *>(types.get_readonly_originptr());
        char *arrmeta = res.get_arrmeta();
        const uintptr_t *arrmeta_offsets = res.get_type()
                                               .extended<base_struct_type>()
                                               ->get_arrmeta_offsets_raw();
        char *data = res.get_readwrite_originptr();
        const uintptr_t *data_offsets = res.get_type().extended<base_struct_type>()->get_data_offsets(
                res.get_arrmeta());

        dynd::index_proxy<I>::for_each(
            forward_as_array_ex, tp, arrmeta, arrmeta_offsets, data,
            data_offsets, m_values,
            available.empty() ? NULL : available.data());
        for (intptr_t j : missing) {
          tp[j].extended()->arrmeta_default_construct(
              arrmeta + arrmeta_offsets[j], true);
          assign_na(tp[j], arrmeta + arrmeta_offsets[j], data + data_offsets[j],
                    &eval::default_eval_context);
        }

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
  }
} // namespace nd

template <typename T>
struct builtin_or_array {
  typedef nd::array type;
};

inline nd::detail::kwds<> kwds() { return nd::detail::kwds<>(); }

template <typename... T>
typename instantiate<
    nd::detail::kwds,
    typename take<type_sequence<T...>,
                  make_index_sequence<1, sizeof...(T), 2>>::type>::type
kwds(T &&... args)
{
  // Sequence of even integers, for extracting the keyword names
  typedef make_index_sequence<0, sizeof...(T), 2> I;
  // Sequence of odd integers, for extracting the values
  typedef make_index_sequence<1, sizeof...(T), 2> J;

  return index_proxy<typename join<I, J>::type>::template make<
      decltype(kwds(std::forward<T>(args)...))>(std::forward<T>(args)...);
}

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
  /**
   * Holds a single instance of an arrfunc in an immutable nd::array,
   * providing some more direct convenient interface.
   */
  class arrfunc {
    nd::array m_value;

    // Todo: Delete this constructor. For now, make it private.
    arrfunc(const arrfunc_type_data *self, const ndt::type &self_tp)
        : m_value(empty(self_tp))
    {
      *reinterpret_cast<arrfunc_type_data *>(
          m_value.get_readwrite_originptr()) = *self;
    }

  public:
    arrfunc() {}

    arrfunc(const ndt::type &self_tp, arrfunc_instantiate_t instantiate,
            arrfunc_resolve_option_values_t resolve_option_values,
            arrfunc_resolve_dst_type_t resolve_dst_type)
        : m_value(empty(self_tp))
    {
      new (m_value.get_readwrite_originptr()) arrfunc_type_data(
          instantiate, resolve_option_values, resolve_dst_type);
    }

    template <typename T>
    arrfunc(const ndt::type &self_tp, const T &data,
            arrfunc_instantiate_t instantiate,
            arrfunc_resolve_option_values_t resolve_option_values,
            arrfunc_resolve_dst_type_t resolve_dst_type,
            arrfunc_free_t free = NULL)
        : m_value(empty(self_tp))
    {
      new (m_value.get_readwrite_originptr()) arrfunc_type_data(
          data, instantiate, resolve_option_values, resolve_dst_type, free);
    }

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
    void set_as_option(arrfunc_resolve_option_values_t resolve_option_values,
                       T &&... names)
    {
      // TODO: This function makes some assumptions about types not being option
      //       already, etc. We need this functionality, but probably can find
      //       a better way later.

      intptr_t missing[sizeof...(T)] = {get_type()->get_kwd_index(names)...};

      nd::array new_kwd_types = get_type()->get_kwd_types().eval_copy();
      auto new_kwd_types_ptr = reinterpret_cast<ndt::type *>(
          new_kwd_types.get_readwrite_originptr());
      for (size_t i = 0; i < sizeof...(T); ++i) {
        new_kwd_types_ptr[missing[i]] =
            ndt::make_option(new_kwd_types_ptr[missing[i]]);
      }
      new_kwd_types.flag_as_immutable();

      arrfunc_type_data self(get()->instantiate, resolve_option_values,
                             get()->resolve_dst_type, get()->free);
      std::memcpy(self.data, get()->data, sizeof(get()->data));
      ndt::type self_tp = ndt::make_arrfunc(
          get_type()->get_pos_tuple(),
          ndt::make_struct(get_type()->get_kwd_names(), new_kwd_types),
          get_type()->get_return_type());

      arrfunc(&self, self_tp).swap(*this);
    }

    std::vector<intptr_t>
    resolve_missing_types(ndt::type *kwd_tp,
                          std::map<nd::string, ndt::type> &typevars) const
    {
      std::vector<intptr_t> missing;

      const arrfunc_type *self_tp = m_value.get_type().extended<arrfunc_type>();

      for (intptr_t j : self_tp->get_option_kwd_indices()) {
        ndt::type &actual_tp = kwd_tp[j];
        if (actual_tp.is_null()) {
          actual_tp =
              ndt::substitute(self_tp->get_kwd_type(j), typevars, false);
          if (actual_tp.is_symbolic()) {
            actual_tp = ndt::make_option(ndt::make_type<void>());
          }
          missing.push_back(j);
        }
      }

      return missing;
    }

    template <typename... K>
    ndt::type resolve(intptr_t nsrc, const ndt::type *src_tp,
                 const char *const *src_arrmeta, const detail::kwds<K...> &kwds,
                 array &kwds_as_array, const eval::eval_context *&ectx,
                 std::map<nd::string, ndt::type> &typevars) const
    {
      const arrfunc_type_data *self = get();
      const arrfunc_type *self_tp = m_value.get_type().extended<arrfunc_type>();

      intptr_t nkwd;
      std::vector<intptr_t> available = kwds.is_permutation(self_tp, nkwd, ectx, typevars);

      if (!self_tp->is_pos_variadic() && nsrc != self_tp->get_npos()) {
        std::stringstream ss;
        ss << "arrfunc expected " << self_tp->get_npos()
           << " parameters, but received " << nsrc << ". arrfunc signature is "
           << ndt::type(self_tp, true);
        throw std::invalid_argument(ss.str());
      }
      const ndt::type *param_types = self_tp->get_pos_types_raw();
      for (intptr_t i = 0; i != self_tp->get_npos(); ++i) {
        ndt::type expected_tp = param_types[i];
        if (!src_tp[i].value_type().matches(src_arrmeta[i], expected_tp, NULL,
                                            typevars)) {
          std::stringstream ss;
          ss << "parameter " << (i + 1) << " to arrfunc does not match, ";
          ss << "expected " << expected_tp << ", received " << src_tp[i];
          throw std::invalid_argument(ss.str());
        }
      }

      if (self_tp->get_nkwd() > 0) {
        nd::array kwd_tp = nd::empty(self_tp->get_nkwd(), ndt::make_type());
        kwds.resolve_available_types(
            self_tp, available, reinterpret_cast<ndt::type *>(kwd_tp.get_data()),
            typevars);
        std::vector<intptr_t> missing = resolve_missing_types(
            reinterpret_cast<ndt::type *>(kwd_tp.get_data()), typevars);

        // check that we have the exact number of available parameters
        if (intptr_t(available.size() + missing.size()) < self_tp->get_nkwd()) {
          std::stringstream ss;
          // TODO: Provide the missing keyword parameter names in this error
          //       message
          ss << "arrfunc requires keyword parameters that were not provided. "
                "arrfunc signature " << ndt::type(self_tp, true);
          throw std::invalid_argument(ss.str());
        }

        kwds_as_array = kwds.forward_as_array(self_tp->get_kwd_names(), kwd_tp,
                              available, missing);

        if (self->resolve_option_values != NULL) {
          self->resolve_option_values(self, self_tp, nsrc, src_tp,
                                      kwds_as_array, typevars);
        }
      } else if (nkwd != 0) {
        stringstream ss;
        ss << "arrfunc does not accept keyword arguments, but was provided "
              "keyword arguments. arrfunc signature is "
           << ndt::type(self_tp, true);
        throw std::invalid_argument(ss.str());
      }

      if (self->resolve_dst_type != NULL) {
        ndt::type dst_tp;
        self->resolve_dst_type(self, self_tp, nsrc, src_tp, true, dst_tp,
                               kwds_as_array, typevars);
        return dst_tp;
      }

      return ndt::substitute(self_tp->get_return_type(), typevars, true);
    }

    /** Implements the general call operator */
    template <typename... A, typename... K>
    array call(const detail::args<A...> &args,
               const detail::kwds<K...> &kwds) const
    {
      const eval::eval_context *ectx = NULL;

      const arrfunc_type_data *self = get();
      const arrfunc_type *self_tp = m_value.get_type().extended<arrfunc_type>();

      // Resolve the destination type
      std::vector<ndt::type> src_tp;
      std::vector<const char *> src_arrmeta;
      std::vector<char *> src_data;
      args.resolve(src_tp, src_arrmeta, src_data);

      nd::array kwds_as_array;
      std::map<nd::string, ndt::type> tp_vars;
      ndt::type dst_tp = resolve(args.size(), src_tp.data(), src_arrmeta.data(), kwds,
                                 kwds_as_array, ectx, tp_vars);

      if (ectx == NULL) {
        ectx = &eval::default_eval_context;
      }

      // Construct the destination array
      nd::array res = nd::empty(dst_tp);

      // Generate and evaluate the ckernel
      ckernel_builder<kernel_request_host> ckb;
      self->instantiate(self, self_tp, &ckb, 0, dst_tp, res.get_arrmeta(),
                        src_tp.data(), src_arrmeta.data(), kernel_request_single, ectx,
                        kwds_as_array, tp_vars);
      expr_single_t fn = ckb.get()->get_function<expr_single_t>();
      fn(res.get_readwrite_originptr(), src_data.data(), ckb.get());

      return res;
    }

    /**
     * operator()()
     */
    nd::array operator()() const
    {
      return call(detail::args<>(), detail::kwds<>());
    }

    /**
     * operator()(a0, a1, ..., an, kwds<...>(...))
     */
    template <typename... T>
    typename std::enable_if<
        detail::is_kwds<typename back<type_sequence<T...>>::type>::value,
        array>::type
    operator()(T &&... a) const
    {
      typedef make_index_sequence<sizeof...(T)-1> I;
      typedef typename instantiate<
          detail::args,
          typename to<type_sequence<typename as_array<T>::type...>,
                      sizeof...(T)-1>::type>::type args_type;

      args_type arr =
          dynd::index_proxy<I>::template make<args_type>(std::forward<T>(a)...);
      return call(arr, dynd::get<sizeof...(T)-1>(std::forward<T>(a)...));
    }

    /**
     * operator()(a0, a1, ..., an)
     */
    template <typename... T>
    typename std::enable_if<
        !detail::is_kwds<typename back<type_sequence<T...>>::type>::value,
        nd::array>::type
    operator()(T &&... a) const
    {
      detail::args<typename as_array<T>::type...> arr(std::forward<T>(a)...);
      return call(arr, kwds());
    }

    nd::array operator()(intptr_t narg, nd::array *args) const {
      return call(detail::args<intptr_t, nd::array *>(narg, args), kwds());
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
                      array(), std::map<nd::string, ndt::type>());
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

  namespace functional {

    arrfunc multidispatch(const ndt::type &self_tp,
                          const std::vector<arrfunc> &children);

  } // namespace dynd::nd::functional

  template <typename CKT>
  arrfunc as_arrfunc(const ndt::type &self_tp)
  {
    return arrfunc(self_tp, dynd::detail::get_instantiate<CKT>(),
                   dynd::detail::get_resolve_option_values<CKT>(),
                   dynd::detail::get_resolve_dst_type<CKT>());
  }

  template <typename CKT>
  arrfunc as_arrfunc()
  {
    return as_arrfunc<CKT>(CKT::make_type());
  }

  template <typename CKT0, typename CKT1, typename... CKT>
  arrfunc as_arrfunc(const ndt::type &self_tp)
  {
    return functional::multidispatch(
        self_tp,
        {as_arrfunc<CKT0>(), as_arrfunc<CKT1>(), as_arrfunc<CKT>()...});
  }

  template <typename CKT, typename T>
  arrfunc as_arrfunc(const ndt::type &self_tp, const T &data)
  {
    return arrfunc(self_tp, data, dynd::detail::get_instantiate<CKT>(),
                   dynd::detail::get_resolve_option_values<CKT>(),
                   dynd::detail::get_resolve_dst_type<CKT>());
  }

  template <typename CKT, typename T>
  arrfunc as_arrfunc(const T &data)
  {
    return as_arrfunc<CKT>(CKT::make_type(), data);
  }

  template <typename CKT0, typename CKT1, typename... CKT, typename T>
  arrfunc as_arrfunc(const ndt::type &self_tp, const T &data)
  {
    return functional::multidispatch(self_tp, {as_arrfunc<CKT0>(data),
                                               as_arrfunc<CKT1>(data),
                                               as_arrfunc<CKT>(data)...});
  }

  namespace detail {
    struct as_arrfunc_wrapper {
      template <typename... CKT>
      arrfunc operator()(const ndt::type &self_tp) const
      {
        return as_arrfunc<CKT...>(self_tp);
      }

      template <typename... CKT, typename T>
      arrfunc operator()(const ndt::type &self_tp, const T &data) const
      {
        return as_arrfunc<CKT...>(self_tp, data);
      }
    };
  }

  template <template <typename...> class TCK, typename... A, typename T>
  arrfunc as_arrfunc(const ndt::type &self_tp, const T &data)
  {
    typedef typename for_each<TCK, typename outer<A...>::type,
                              sizeof...(A) >= 2>::type CKT;
    return type_proxy<CKT>::apply(detail::as_arrfunc_wrapper(), self_tp, data);
  }

  template <template <kernel_request_t, typename...> class TCK,
            kernel_request_t kernreq, typename... A, typename T>
  arrfunc as_arrfunc(const ndt::type &self_tp, const T &data)
  {
    typedef typename ex_for_each<TCK, kernreq, typename outer<A...>::type,
                              sizeof...(A) >= 2>::type CKT;
    return type_proxy<CKT>::apply(detail::as_arrfunc_wrapper(), self_tp, data);
  }

  namespace decl {

    template <typename T>
    struct arrfunc {
      const nd::arrfunc &get_self() const
      {
        static nd::arrfunc self = as_arrfunc();
        return self;
      }

      const ndt::type &get_type() const { return get_self().get_array_type(); }

      operator const nd::arrfunc &() const { return get_self(); }

      template <typename... A>
      array operator()(A &&... a) const
      {
        return get_self()(std::forward<A>(a)...);
      }

      static nd::arrfunc as_arrfunc() { return T::as_arrfunc(); }
    };

  } // namespace dynd::nd::decl

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