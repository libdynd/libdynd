//
// Copyright (C) 2011-15 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#pragma once

#include <dynd/gfunc/gcallable.hpp>
#include <dynd/types/struct_type.hpp>
#include <dynd/types/ndarrayarg_type.hpp>
#include <dynd/types/string_type.hpp>
#include <dynd/types/type_type.hpp>

namespace dynd {
namespace gfunc {

  // Metaprogram for determining size
  template <typename T>
  struct dcs_size_of;
  template <>
  struct dcs_size_of<bool1> {
    enum {
      value = 1
    };
  };
  template <>
  struct dcs_size_of<char> {
    enum {
      value = 1
    };
  };
  template <>
  struct dcs_size_of<signed char> {
    enum {
      value = 1
    };
  };
  template <>
  struct dcs_size_of<short> {
    enum {
      value = sizeof(short)
    };
  };
  template <>
  struct dcs_size_of<int> {
    enum {
      value = sizeof(int)
    };
  };
  template <>
  struct dcs_size_of<long> {
    enum {
      value = sizeof(long)
    };
  };
  template <>
  struct dcs_size_of<long long> {
    enum {
      value = sizeof(long long)
    };
  };
  template <>
  struct dcs_size_of<unsigned char> {
    enum {
      value = 1
    };
  };
  template <>
  struct dcs_size_of<unsigned short> {
    enum {
      value = sizeof(unsigned short)
    };
  };
  template <>
  struct dcs_size_of<unsigned int> {
    enum {
      value = sizeof(unsigned int)
    };
  };
  template <>
  struct dcs_size_of<unsigned long> {
    enum {
      value = sizeof(unsigned long)
    };
  };
  template <>
  struct dcs_size_of<unsigned long long> {
    enum {
      value = sizeof(unsigned long long)
    };
  };
  template <>
  struct dcs_size_of<float> {
    enum {
      value = sizeof(long)
    };
  };
  template <>
  struct dcs_size_of<double> {
    enum {
      value = sizeof(double)
    };
  };
  template <>
  struct dcs_size_of<complex<float>> {
    enum {
      value = sizeof(complex<float>)
    };
  };
  template <>
  struct dcs_size_of<complex<double>> {
    enum {
      value = sizeof(complex<double>)
    };
  };
  template <>
  struct dcs_size_of<array_preamble *> {
    enum {
      value = sizeof(array_preamble *)
    };
  };
  template <>
  struct dcs_size_of<const ndt::base_type *> {
    enum {
      value = sizeof(const ndt::base_type *)
    };
  };
  template <>
  struct dcs_size_of<string> {
    enum {
      value = sizeof(string)
    };
  };
  template <typename T, int N>
  struct dcs_size_of<T[N]> {
    enum {
      value = N * sizeof(T)
    };
  };

  /**
   * Metaprogram which returns the field offset of the last field in the
   * template argument list.
   */
  template <typename T0, typename T1 = void, typename T2 = void, typename T3 = void, typename T4 = void>
  struct dcs_offset_of {
    enum {
      partial_offset = dcs_offset_of<T0, T1, T2, T3, void>::value + dcs_size_of<T3>::value
    };
    enum {
      field_align = scalar_align_of<T4>::value
    };

    // The offset to the T4 value
    enum {
      value = partial_offset +
              (((partial_offset & (field_align - 1)) == 0) ? 0 : (field_align - (partial_offset & (field_align - 1))))
    };
  };

  template <typename T0, typename T1, typename T2, typename T3>
  struct dcs_offset_of<T0, T1, T2, T3, void> {
    enum {
      partial_offset = dcs_offset_of<T0, T1, T2, void, void>::value + dcs_size_of<T2>::value
    };
    enum {
      field_align = scalar_align_of<T3>::value
    };

    // The offset to the T3 value
    enum {
      value = partial_offset +
              (((partial_offset & (field_align - 1)) == 0) ? 0 : (field_align - (partial_offset & (field_align - 1))))
    };
  };

  template <typename T0, typename T1, typename T2>
  struct dcs_offset_of<T0, T1, T2, void, void> {
    enum {
      partial_offset = dcs_offset_of<T0, T1, void, void, void>::value + dcs_size_of<T1>::value
    };
    enum {
      field_align = scalar_align_of<T2>::value
    };

    // The offset to the T2 value
    enum {
      value = partial_offset +
              (((partial_offset & (field_align - 1)) == 0) ? 0 : (field_align - (partial_offset & (field_align - 1))))
    };
  };

  template <typename T0, typename T1>
  struct dcs_offset_of<T0, T1, void, void, void> {
    enum {
      partial_offset = dcs_offset_of<T0, void, void, void, void>::value + dcs_size_of<T0>::value
    };
    enum {
      field_align = scalar_align_of<T1>::value
    };

    // The offset to the T1 value
    enum {
      value = partial_offset +
              (((partial_offset & (field_align - 1)) == 0) ? 0 : (field_align - (partial_offset & (field_align - 1))))
    };
  };

  template <typename T0>
  struct dcs_offset_of<T0, void, void, void, void> {
    // The offset to the T0 value
    enum {
      value = 0
    };
  };

  template <typename T>
  struct parameter_type_of;
  template <typename T>
  struct parameter_type_of<T &> : public parameter_type_of<T> {
  };
  template <typename T>
  struct parameter_type_of<const T> : public parameter_type_of<T> {
  };
  template <typename T, int N>
  struct parameter_type_of<T[N]> {
    typedef typename parameter_type_of<T>::type type[N];
  };
  template <>
  struct parameter_type_of<bool> {
    typedef bool1 type;
  };
  template <>
  struct parameter_type_of<signed char> {
    typedef signed char type;
  };
  template <>
  struct parameter_type_of<short> {
    typedef short type;
  };
  template <>
  struct parameter_type_of<int> {
    typedef int type;
  };
  template <>
  struct parameter_type_of<long> {
    typedef long type;
  };
  template <>
  struct parameter_type_of<long long> {
    typedef long long type;
  };
  template <>
  struct parameter_type_of<uint8_t> {
    typedef uint8_t type;
  };
  template <>
  struct parameter_type_of<uint16_t> {
    typedef uint16_t type;
  };
  template <>
  struct parameter_type_of<unsigned int> {
    typedef unsigned int type;
  };
  template <>
  struct parameter_type_of<unsigned long> {
    typedef unsigned long type;
  };
  template <>
  struct parameter_type_of<unsigned long long> {
    typedef unsigned long long type;
  };
  template <>
  struct parameter_type_of<float> {
    typedef float type;
  };
  template <>
  struct parameter_type_of<double> {
    typedef double type;
  };
  template <typename T>
  struct parameter_type_of<complex<T>> {
    typedef complex<T> type;
  };
  template <>
  struct parameter_type_of<nd::array> {
    typedef array_preamble *type;
  };
  template <>
  struct parameter_type_of<ndt::type> {
    typedef const ndt::base_type *type;
  };
  template <>
  struct parameter_type_of<std::string> {
    typedef string type;
  };

  template <typename T>
  struct make_parameter_type {
    inline static ndt::type make()
    {
      return ndt::type::make<typename parameter_type_of<T>::type>();
    }
  };
  template <typename T>
  struct make_parameter_type<T &> : public make_parameter_type<T> {
  };
  template <typename T>
  struct make_parameter_type<const T> : public make_parameter_type<T> {
  };
  template <typename T, int N>
  struct make_parameter_type<T[N]> {
    inline static ndt::type make()
    {
      return ndt::type::make<T[N]>();
    }
  };
  // Use ndarrayarg to pass nd::array as a parameter. This is a borrowed
  // reference
  // to an nd::array owned by the caller.
  template <>
  struct make_parameter_type<nd::array> {
    inline static ndt::type make()
    {
      return ndt::make_ndarrayarg();
    }
  };
  template <>
  struct make_parameter_type<ndt::type> {
    inline static ndt::type make()
    {
      return ndt::make_type();
    }
  };
  template <>
  struct make_parameter_type<std::string> {
    inline static ndt::type make()
    {
      return ndt::string_type::make();
    }
  };

  template <typename T>
  struct box_result {
    inline static typename std::enable_if<is_dynd_scalar<T>::value, array_preamble *>::type box(const T &v)
    {
      return nd::array(v).release();
    }
  };
  template <typename T>
  struct box_result<T &> : public box_result<T> {
  };
  template <typename T>
  struct box_result<const T> : public box_result<T> {
  };
  template <>
  struct box_result<nd::array> {
    inline static array_preamble *box(const nd::array &v)
    {
      // Throwing away v's value is ok here, for the limited use of this
      // function
      return const_cast<nd::array &>(v).release();
    }
  };
  template <>
  struct box_result<ndt::type> {
    inline static array_preamble *box(const ndt::type &v)
    {
      return nd::array(v).release();
    }
  };
  template <>
  struct box_result<std::string> {
    inline static array_preamble *box(const std::string &v)
    {
      return nd::array(v).release();
    }
  };

  template <typename T>
  struct unbox_param {
    inline static typename std::enable_if<is_dynd_scalar<T>::value, const T &>::type unbox(char *v)
    {
      return *reinterpret_cast<T *>(v);
    }
  };
  template <>
  struct unbox_param<bool> {
    inline static bool unbox(char *v)
    {
      return (*v != 0);
    }
  };
  template <typename T>
  struct unbox_param<T &> : public unbox_param<T> {
  };
  template <typename T>
  struct unbox_param<const T> : public unbox_param<T> {
  };
  template <typename T, int N>
  struct unbox_param<T[N]> {
    typedef T (&result_type)[N];
    typedef T (*result_type_ptr)[N];
    inline static result_type unbox(char *v)
    {
      return *reinterpret_cast<result_type_ptr>(v);
    }
  };
  template <>
  struct unbox_param<nd::array> {
    inline static nd::array unbox(char *v)
    {
      return nd::array(*reinterpret_cast<array_preamble **>(v), true);
    }
  };
  template <>
  struct unbox_param<ndt::type> {
    inline static ndt::type unbox(char *v)
    {
      return ndt::type(*reinterpret_cast<ndt::base_type **>(v), true);
    }
  };
  template <>
  struct unbox_param<std::string> {
    inline static std::string unbox(char *v)
    {
      string *p = reinterpret_cast<string *>(v);
      return std::string(p->begin(), p->end());
    }
  };

  namespace detail {
    template <class FN>
    struct callable_maker;

    // Metaprogramming wrapper for 1-parameter function
    template <class R, class P0>
    struct callable_maker<R (*)(P0)> {
      typedef R (*function_pointer_t)(P0);
      typedef typename parameter_type_of<P0>::type T0;
      static array_preamble *wrapper(const array_preamble *params, void *extra)
      {
        char *p = params->data;
        function_pointer_t f = reinterpret_cast<function_pointer_t>(extra);
        return box_result<R>::box(f(unbox_param<P0>::unbox(p + dcs_offset_of<T0>::value)));
      }
      static ndt::type make_parameters_type(const char *name0)
      {
        return ndt::struct_type::make({name0}, {make_parameter_type<P0>::make()});
      }
    };

    // Metaprogramming wrapper for 2-parameter function
    template <class R, class P0, class P1>
    struct callable_maker<R (*)(P0, P1)> {
      typedef R (*function_pointer_t)(P0, P1);
      typedef typename parameter_type_of<P0>::type T0;
      typedef typename parameter_type_of<P1>::type T1;
      static array_preamble *wrapper(const array_preamble *params, void *extra)
      {
        char *p = params->data;
        function_pointer_t f = reinterpret_cast<function_pointer_t>(extra);
        return box_result<R>::box(f(unbox_param<P0>::unbox(p + dcs_offset_of<T0>::value),
                                    unbox_param<P1>::unbox(p + dcs_offset_of<T0, T1>::value)));
      }
      static ndt::type make_parameters_type(const char *name0, const char *name1)
      {
        return ndt::struct_type::make({name0, name1},
                                      {make_parameter_type<P0>::make(), make_parameter_type<P1>::make()});
      }
    };

    // Metaprogramming wrapper for 3-parameter function
    template <class R, class P0, class P1, class P2>
    struct callable_maker<R (*)(P0, P1, P2)> {
      typedef R (*function_pointer_t)(P0, P1, P2);
      typedef typename parameter_type_of<P0>::type T0;
      typedef typename parameter_type_of<P1>::type T1;
      typedef typename parameter_type_of<P2>::type T2;
      static array_preamble *wrapper(const array_preamble *params, void *extra)
      {
        char *p = params->data;
        function_pointer_t f = reinterpret_cast<function_pointer_t>(extra);
        return box_result<R>::box(f(unbox_param<P0>::unbox(p + dcs_offset_of<T0>::value),
                                    unbox_param<P1>::unbox(p + dcs_offset_of<T0, T1>::value),
                                    unbox_param<P2>::unbox(p + dcs_offset_of<T0, T1, T2>::value)));
      }
      static ndt::type make_parameters_type(const char *name0, const char *name1, const char *name2)
      {
        return ndt::struct_type::make(
            {name0, name1, name2},
            {make_parameter_type<P0>::make(), make_parameter_type<P1>::make(), make_parameter_type<P2>::make()});
      }
    };

    // Metaprogramming wrapper for 4-parameter function
    template <class R, class P0, class P1, class P2, class P3>
    struct callable_maker<R (*)(P0, P1, P2, P3)> {
      typedef R (*function_pointer_t)(P0, P1, P2, P3);
      typedef typename parameter_type_of<P0>::type T0;
      typedef typename parameter_type_of<P1>::type T1;
      typedef typename parameter_type_of<P2>::type T2;
      typedef typename parameter_type_of<P3>::type T3;
      static array_preamble *wrapper(const array_preamble *params, void *extra)
      {
        char *p = params->data;
        function_pointer_t f = reinterpret_cast<function_pointer_t>(extra);
        return box_result<R>::box(f(unbox_param<P0>::unbox(p + dcs_offset_of<T0>::value),
                                    unbox_param<P1>::unbox(p + dcs_offset_of<T0, T1>::value),
                                    unbox_param<P2>::unbox(p + dcs_offset_of<T0, T1, T2>::value),
                                    unbox_param<P3>::unbox(p + dcs_offset_of<T0, T1, T2, T3>::value)));
      }
      static ndt::type make_parameters_type(const char *name0, const char *name1, const char *name2, const char *name3)
      {
        ndt::type field_types[4];
        const char *field_names[4] = {name0, name1, name2, name3};
        field_types[0] = make_parameter_type<P0>::make();
        field_types[1] = make_parameter_type<P1>::make();
        field_types[2] = make_parameter_type<P2>::make();
        field_types[3] = make_parameter_type<P3>::make();
        return ndt::struct_type::make(nd::make_strided_string_array(field_names, 4), field_types);
      }
    };

    // Metaprogramming wrapper for 5-parameter function
    template <class R, class P0, class P1, class P2, class P3, class P4>
    struct callable_maker<R (*)(P0, P1, P2, P3, P4)> {
      typedef R (*function_pointer_t)(P0, P1, P2, P3, P4);
      typedef typename parameter_type_of<P0>::type T0;
      typedef typename parameter_type_of<P1>::type T1;
      typedef typename parameter_type_of<P2>::type T2;
      typedef typename parameter_type_of<P3>::type T3;
      typedef typename parameter_type_of<P4>::type T4;
      static array_preamble *wrapper(const array_preamble *params, void *extra)
      {
        char *p = params->data;
        function_pointer_t f = reinterpret_cast<function_pointer_t>(extra);
        return box_result<R>::box(f(unbox_param<P0>::unbox(p + dcs_offset_of<T0>::value),
                                    unbox_param<P1>::unbox(p + dcs_offset_of<T0, T1>::value),
                                    unbox_param<P2>::unbox(p + dcs_offset_of<T0, T1, T2>::value),
                                    unbox_param<P3>::unbox(p + dcs_offset_of<T0, T1, T2, T3>::value),
                                    unbox_param<P4>::unbox(p + dcs_offset_of<T0, T1, T2, T3, T4>::value)));
      }
      static ndt::type make_parameters_type(const char *name0, const char *name1, const char *name2, const char *name3,
                                            const char *name4)
      {
        ndt::type field_types[5];
        const char *field_names[5] = {name0, name1, name2, name3, name4};
        field_types[0] = make_parameter_type<P0>::make();
        field_types[1] = make_parameter_type<P1>::make();
        field_types[2] = make_parameter_type<P2>::make();
        field_types[3] = make_parameter_type<P3>::make();
        field_types[4] = make_parameter_type<P4>::make();
        return ndt::struct_type::make(nd::make_strided_string_array(field_names, 5), field_types);
      }
    };
  } // namespace detail

  // One parameter, no defaults
  template <typename FN>
  inline callable make_callable(FN *f, const char *name0)
  {
    return callable(detail::callable_maker<FN *>::make_parameters_type(name0), &detail::callable_maker<FN *>::wrapper,
                    reinterpret_cast<void *>(f));
  }

  // One parameter, one default
  template <typename FN, typename D0>
  inline callable make_callable_with_default(FN *f, const char *name0, const D0 &default0)
  {
    ndt::type pdt = detail::callable_maker<FN *>::make_parameters_type(name0);
    nd::array defaults = nd::empty(pdt);
    defaults(0).vals() = default0;
    // Make defaults immutable (which is ok, because we have the only reference
    // to it)
    defaults.flag_as_immutable();
    return callable(pdt, &detail::callable_maker<FN *>::wrapper, reinterpret_cast<void *>(f), 0, defaults);
  }

  // Two parameters, no defaults
  template <typename FN>
  inline callable make_callable(FN *f, const char *name0, const char *name1)
  {
    return callable(detail::callable_maker<FN *>::make_parameters_type(name0, name1),
                    &detail::callable_maker<FN *>::wrapper, reinterpret_cast<void *>(f));
  }

  // Two parameters, one default
  template <typename FN, typename D1>
  inline callable make_callable_with_default(FN *f, const char *name0, const char *name1, const D1 &default1)
  {
    ndt::type pdt = detail::callable_maker<FN *>::make_parameters_type(name0, name1);
    nd::array defaults = nd::empty(pdt);
    defaults(1).vals() = default1;
    // Make defaults immutable (which is ok, because we have the only reference
    // to it)
    defaults.flag_as_immutable();
    return callable(pdt, &detail::callable_maker<FN *>::wrapper, reinterpret_cast<void *>(f), 1, defaults);
  }

  // Two parameters, two defaults
  template <typename FN, typename D0, typename D1>
  inline callable make_callable_with_default(FN *f, const char *name0, const char *name1, const D0 &default0,
                                             const D1 &default1)
  {
    ndt::type pdt = detail::callable_maker<FN *>::make_parameters_type(name0, name1);
    nd::array defaults = nd::empty(pdt);
    defaults(0).vals() = default0;
    defaults(1).vals() = default1;
    // Make defaults immutable (which is ok, because we have the only reference
    // to it)
    defaults.flag_as_immutable();
    return callable(pdt, &detail::callable_maker<FN *>::wrapper, reinterpret_cast<void *>(f), 0, defaults);
  }

  // Three parameters, no defaults
  template <typename FN>
  inline callable make_callable(FN *f, const char *name0, const char *name1, const char *name2)
  {
    return callable(detail::callable_maker<FN *>::make_parameters_type(name0, name1, name2),
                    &detail::callable_maker<FN *>::wrapper, reinterpret_cast<void *>(f));
  }

  // Three parameters, one default
  template <typename FN, typename D2>
  inline callable make_callable_with_default(FN *f, const char *name0, const char *name1, const char *name2,
                                             const D2 &default2)
  {
    ndt::type pdt = detail::callable_maker<FN *>::make_parameters_type(name0, name1, name2);
    nd::array defaults = nd::empty(pdt);
    defaults(2).vals() = default2;
    // Make defaults immutable (which is ok, because we have the only reference
    // to it)
    defaults.flag_as_immutable();
    return callable(pdt, &detail::callable_maker<FN *>::wrapper, reinterpret_cast<void *>(f), 2, defaults);
  }

  // Three parameters, two defaults
  template <typename FN, typename D1, typename D2>
  inline callable make_callable_with_default(FN *f, const char *name0, const char *name1, const char *name2,
                                             const D1 &default1, const D2 &default2)
  {
    ndt::type pdt = detail::callable_maker<FN *>::make_parameters_type(name0, name1, name2);
    nd::array defaults = nd::empty(pdt);
    defaults(1).vals() = default1;
    defaults(2).vals() = default2;
    // Make defaults immutable (which is ok, because we have the only reference
    // to it)
    defaults.flag_as_immutable();
    return callable(pdt, &detail::callable_maker<FN *>::wrapper, reinterpret_cast<void *>(f), 1, defaults);
  }

  // Three parameters, three defaults
  template <typename FN, typename D0, typename D1, typename D2>
  inline callable make_callable_with_default(FN *f, const char *name0, const char *name1, const char *name2,
                                             const D0 &default0, const D1 &default1, const D2 &default2)
  {
    ndt::type pdt = detail::callable_maker<FN *>::make_parameters_type(name0, name1, name2);
    nd::array defaults = nd::empty(pdt);
    defaults(0).vals() = default0;
    defaults(1).vals() = default1;
    defaults(2).vals() = default2;
    // Make defaults immutable (which is ok, because we have the only reference
    // to it)
    defaults.flag_as_immutable();
    return callable(pdt, &detail::callable_maker<FN *>::wrapper, reinterpret_cast<void *>(f), 0, defaults);
  }

  // Four parameters, no defaults
  template <typename FN>
  inline callable make_callable(FN *f, const char *name0, const char *name1, const char *name2, const char *name3)
  {
    return callable(detail::callable_maker<FN *>::make_parameters_type(name0, name1, name2, name3),
                    &detail::callable_maker<FN *>::wrapper, reinterpret_cast<void *>(f));
  }

  // Four parameters, one default
  template <typename FN, typename D3>
  inline callable make_callable_with_default(FN *f, const char *name0, const char *name1, const char *name2,
                                             const char *name3, const D3 &default3)
  {
    ndt::type pdt = detail::callable_maker<FN *>::make_parameters_type(name0, name1, name2, name3);
    nd::array defaults = nd::empty(pdt);
    defaults(3).vals() = default3;
    // Make defaults immutable (which is ok, because we have the only reference
    // to it)
    defaults.flag_as_immutable();
    return callable(pdt, &detail::callable_maker<FN *>::wrapper, reinterpret_cast<void *>(f), 3, defaults);
  }

  // Four parameters, two defaults
  template <typename FN, typename D2, typename D3>
  inline callable make_callable_with_default(FN *f, const char *name0, const char *name1, const char *name2,
                                             const char *name3, const D2 &default2, const D3 &default3)
  {
    ndt::type pdt = detail::callable_maker<FN *>::make_parameters_type(name0, name1, name2, name3);
    nd::array defaults = nd::empty(pdt);
    defaults(2).vals() = default2;
    defaults(3).vals() = default3;
    // Make defaults immutable (which is ok, because we have the only reference
    // to it)
    defaults.flag_as_immutable();
    return callable(pdt, &detail::callable_maker<FN *>::wrapper, reinterpret_cast<void *>(f), 2, defaults);
  }

  // Four parameters, three defaults
  template <typename FN, typename D1, typename D2, typename D3>
  inline callable make_callable_with_default(FN *f, const char *name0, const char *name1, const char *name2,
                                             const char *name3, const D1 &default1, const D2 &default2,
                                             const D3 &default3)
  {
    ndt::type pdt = detail::callable_maker<FN *>::make_parameters_type(name0, name1, name2, name3);
    nd::array defaults = nd::empty(pdt);
    defaults(1).vals() = default1;
    defaults(2).vals() = default2;
    defaults(3).vals() = default3;
    // Make defaults immutable (which is ok, because we have the only reference
    // to it)
    defaults.flag_as_immutable();
    return callable(pdt, &detail::callable_maker<FN *>::wrapper, reinterpret_cast<void *>(f), 1, defaults);
  }

  // Four parameters, four defaults
  template <typename FN, typename D0, typename D1, typename D2, typename D3>
  inline callable make_callable_with_default(FN *f, const char *name0, const char *name1, const char *name2,
                                             const char *name3, const D0 &default0, const D1 &default1,
                                             const D2 &default2, const D3 &default3)
  {
    ndt::type pdt = detail::callable_maker<FN *>::make_parameters_type(name0, name1, name2, name3);
    nd::array defaults = nd::empty(pdt);
    defaults(0).vals() = default0;
    defaults(1).vals() = default1;
    defaults(2).vals() = default2;
    defaults(3).vals() = default3;
    // Make defaults immutable (which is ok, because we have the only reference
    // to it)
    defaults.flag_as_immutable();
    return callable(pdt, &detail::callable_maker<FN *>::wrapper, reinterpret_cast<void *>(f), 0, defaults);
  }

  template <typename FN>
  inline callable make_callable(FN *f, const char *name0, const char *name1, const char *name2, const char *name3,
                                const char *name4)
  {
    return callable(detail::callable_maker<FN *>::make_parameters_type(name0, name1, name2, name3, name4),
                    &detail::callable_maker<FN *>::wrapper, reinterpret_cast<void *>(f));
  }
}
} // namespace dynd::gfunc
