//
// Copyright (C) 2011-15 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#pragma once

#include <dynd/gfunc/gcallable.hpp>
#include <dynd/types/struct_type.hpp>
#include <dynd/types/string_type.hpp>
#include <dynd/types/fixed_string_type.hpp>
#include <dynd/types/type_type.hpp>
#include <dynd/typed_data_assign.hpp>

namespace dynd {
namespace gfunc {

  namespace detail {
    template <class T>
    struct callable_argument_setter {
      static typename std::enable_if<is_dynd_scalar<T>::value, void>::type
      set(const ndt::type &paramtype, char *arrmeta, char *data, const T &value)
      {
        if (paramtype.get_type_id() == static_cast<type_id_t>(type_id_of<T>::value)) {
          *reinterpret_cast<T *>(data) = value;
        } else {
          typed_data_assign(paramtype, arrmeta, data, ndt::type::make<T>(), NULL,
                            reinterpret_cast<const char *>(&value));
        }
      }
    };

    template <>
    struct callable_argument_setter<bool> {
      static void set(const ndt::type &paramtype, char *arrmeta, char *data, bool value)
      {
        if (paramtype.get_type_id() == bool_type_id) {
          *data = (value ? 1 : 0);
        } else {
          bool1 tmp(value);
          typed_data_assign(paramtype, arrmeta, data, ndt::type::make<bool1>(), NULL,
                            reinterpret_cast<const char *>(&tmp));
        }
      }
    };

    template <>
    struct callable_argument_setter<nd::array> {
      static void set(const ndt::type &paramtype, char *arrmeta, char *data, const nd::array &value)
      {
        if (paramtype.get_type_id() == ndarrayarg_type_id) {
          *reinterpret_cast<const array_preamble **>(data) = value.get();
        } else {
          typed_data_assign(paramtype, arrmeta, data, value.get_type(), value.get()->metadata(), value.get()->data);
        }
      }
    };

    template <>
    struct callable_argument_setter<ndt::type> {
      static void set(const ndt::type &paramtype, char *DYND_UNUSED(arrmeta), char *data, const ndt::type &value)
      {
        if (paramtype.get_type_id() == type_type_id) {
          *reinterpret_cast<ndt::type *>(data) = value;
        } else {
          std::stringstream ss;
          ss << "cannot pass a dynd type as a parameter to dynd callable parameter of type " << paramtype;
          throw std::runtime_error(ss.str());
        }
      }
    };

    template <int N>
    struct callable_argument_setter<const char[N]> {
      static void set(const ndt::type &paramtype, char *arrmeta, char *data, const char (&value)[N])
      {
        // Setting from a known-sized character string array
        if (paramtype.get_type_id() == string_type_id &&
            paramtype.extended<ndt::string_type>()->get_encoding() == string_encoding_utf_8) {
          reinterpret_cast<string *>(data)->assign(value, N - 1);
        } else {
          typed_data_assign(paramtype, arrmeta, data, ndt::fixed_string_type::make(N, string_encoding_utf_8), NULL,
                            value);
        }
      }
    };

    template <int N>
    struct callable_argument_setter<char[N]> : public callable_argument_setter<const char[N]> {
    };
  } // namespace detail

  inline nd::array callable::call() const
  {
    const ndt::struct_type *fsdt = m_parameters_type.extended<ndt::struct_type>();
    intptr_t parameter_count = fsdt->get_field_count();
    nd::array params = nd::empty(m_parameters_type);
    if (parameter_count != 0) {
      if (m_first_default_parameter <= 0) {
        // Fill the missing parameters with their defaults, if available
        for (intptr_t i = 0; i < parameter_count; ++i) {
          uintptr_t arrmeta_offset = fsdt->get_arrmeta_offset(i);
          uintptr_t data_offset = fsdt->get_data_offsets(params.get()->metadata())[i];
          typed_data_copy(fsdt->get_field_type(i), params.get()->metadata() + arrmeta_offset, params.get()->data + data_offset,
                          m_default_parameters.get()->metadata() + arrmeta_offset,
                          m_default_parameters.get()->data + data_offset);
        }
      } else {
        std::stringstream ss;
        ss << "incorrect number of arguments (received 0) for dynd callable with parameters " << m_parameters_type;
        throw std::runtime_error(ss.str());
      }
    }
    return call_generic(params);
  }

  template <class T>
  inline nd::array callable::call(const T &p0) const
  {
    const ndt::struct_type *fsdt = m_parameters_type.extended<ndt::struct_type>();
    intptr_t parameter_count = fsdt->get_field_count();
    nd::array params = nd::empty(m_parameters_type);
    if (parameter_count != 1) {
      if (parameter_count > 1 && m_first_default_parameter <= 1) {
        // Fill the missing parameters with their defaults, if available
        for (intptr_t i = 1; i < parameter_count; ++i) {
          size_t arrmeta_offset = fsdt->get_arrmeta_offset(i);
          size_t data_offset = fsdt->get_data_offsets(params.get()->metadata())[i];
          typed_data_copy(fsdt->get_field_type(i), params.get()->metadata() + arrmeta_offset, params.get()->data + data_offset,
                          m_default_parameters.get()->metadata() + arrmeta_offset,
                          m_default_parameters.get()->data + data_offset);
        }
      } else {
        std::stringstream ss;
        ss << "incorrect number of arguments (received 1) for dynd callable with parameters " << m_parameters_type;
        throw std::runtime_error(ss.str());
      }
    }
    detail::callable_argument_setter<T>::set(fsdt->get_field_type(0), params.get()->metadata() + fsdt->get_arrmeta_offset(0),
                                             params.get()->data + fsdt->get_data_offsets(params.get()->metadata())[0], p0);
    return call_generic(params);
  }

  template <class T0, class T1>
  inline nd::array callable::call(const T0 &p0, const T1 &p1) const
  {
    const ndt::struct_type *fsdt = m_parameters_type.extended<ndt::struct_type>();
    intptr_t parameter_count = fsdt->get_field_count();
    nd::array params = nd::empty(m_parameters_type);
    if (fsdt->get_field_count() != 2) {
      if (parameter_count > 2 && m_first_default_parameter <= 2) {
        // Fill the missing parameters with their defaults, if available
        for (intptr_t i = 2; i < parameter_count; ++i) {
          size_t arrmeta_offset = fsdt->get_arrmeta_offset(i);
          size_t data_offset = fsdt->get_data_offsets(params.get()->metadata())[i];
          typed_data_copy(fsdt->get_field_type(i), params.get()->metadata() + arrmeta_offset, params.get()->data + data_offset,
                          m_default_parameters.get()->metadata() + arrmeta_offset,
                          m_default_parameters.get()->data + data_offset);
        }
      } else {
        std::stringstream ss;
        ss << "incorrect number of arguments (received 2) for dynd callable with parameters " << m_parameters_type;
        throw std::runtime_error(ss.str());
      }
    }
    detail::callable_argument_setter<T0>::set(fsdt->get_field_type(0), params.get()->metadata() + fsdt->get_arrmeta_offset(0),
                                              params.get()->data + fsdt->get_data_offsets(params.get()->metadata())[0], p0);
    detail::callable_argument_setter<T1>::set(fsdt->get_field_type(1), params.get()->metadata() + fsdt->get_arrmeta_offset(1),
                                              params.get()->data + fsdt->get_data_offsets(params.get()->metadata())[1], p1);
    return call_generic(params);
  }

  template <class T0, class T1, class T2>
  inline nd::array callable::call(const T0 &p0, const T1 &p1, const T2 &p2) const
  {
    const ndt::struct_type *fsdt = m_parameters_type.extended<ndt::struct_type>();
    intptr_t parameter_count = fsdt->get_field_count();
    nd::array params = nd::empty(m_parameters_type);
    if (fsdt->get_field_count() != 3) {
      if (parameter_count > 3 && m_first_default_parameter <= 3) {
        // Fill the missing parameters with their defaults, if available
        for (intptr_t i = 3; i < parameter_count; ++i) {
          size_t arrmeta_offset = fsdt->get_arrmeta_offset(i);
          size_t data_offset = fsdt->get_data_offsets(params.get()->metadata())[i];
          typed_data_copy(fsdt->get_field_type(i), params.get()->metadata() + arrmeta_offset, params.get()->data + data_offset,
                          m_default_parameters.get()->metadata() + arrmeta_offset,
                          m_default_parameters.get()->data + data_offset);
        }
      } else {
        std::stringstream ss;
        ss << "incorrect number of arguments (received 3) for dynd callable with parameters " << m_parameters_type;
        throw std::runtime_error(ss.str());
      }
    }
    detail::callable_argument_setter<T0>::set(fsdt->get_field_type(0), params.get()->metadata() + fsdt->get_arrmeta_offset(0),
                                              params.get()->data + fsdt->get_data_offsets(params.get()->metadata())[0], p0);
    detail::callable_argument_setter<T1>::set(fsdt->get_field_type(1), params.get()->metadata() + fsdt->get_arrmeta_offset(1),
                                              params.get()->data + fsdt->get_data_offsets(params.get()->metadata())[1], p1);
    detail::callable_argument_setter<T2>::set(fsdt->get_field_type(2), params.get()->metadata() + fsdt->get_arrmeta_offset(2),
                                              params.get()->data + fsdt->get_data_offsets(params.get()->metadata())[2], p2);
    return call_generic(params);
  }

  template <class T0, class T1, class T2, class T3>
  inline nd::array callable::call(const T0 &p0, const T1 &p1, const T2 &p2, const T3 &p3) const
  {
    const ndt::struct_type *fsdt = m_parameters_type.extended<ndt::struct_type>();
    intptr_t parameter_count = fsdt->get_field_count();
    nd::array params = nd::empty(m_parameters_type);
    if (fsdt->get_field_count() != 4) {
      if (parameter_count > 4 && m_first_default_parameter <= 4) {
        // Fill the missing parameters with their defaults, if available
        for (intptr_t i = 4; i < parameter_count; ++i) {
          size_t arrmeta_offset = fsdt->get_arrmeta_offset(i);
          size_t data_offset = fsdt->get_data_offsets(params.get()->metadata())[i];
          typed_data_copy(fsdt->get_field_type(i), params.get()->metadata() + arrmeta_offset, params.get()->data + data_offset,
                          m_default_parameters.get()->metadata() + arrmeta_offset,
                          m_default_parameters.get()->data + data_offset);
        }
      } else {
        std::stringstream ss;
        ss << "incorrect number of arguments (received 4) for dynd callable with parameters " << m_parameters_type;
        throw std::runtime_error(ss.str());
      }
    }
    detail::callable_argument_setter<T0>::set(fsdt->get_field_type(0), params.get()->metadata() + fsdt->get_arrmeta_offset(0),
                                              params.get()->data + fsdt->get_data_offsets(params.get()->metadata())[0], p0);
    detail::callable_argument_setter<T1>::set(fsdt->get_field_type(1), params.get()->metadata() + fsdt->get_arrmeta_offset(1),
                                              params.get()->data + fsdt->get_data_offsets(params.get()->metadata())[1], p1);
    detail::callable_argument_setter<T2>::set(fsdt->get_field_type(2), params.get()->metadata() + fsdt->get_arrmeta_offset(2),
                                              params.get()->data + fsdt->get_data_offsets(params.get()->metadata())[2], p2);
    detail::callable_argument_setter<T3>::set(fsdt->get_field_type(3), params.get()->metadata() + fsdt->get_arrmeta_offset(3),
                                              params.get()->data + fsdt->get_data_offsets(params.get()->metadata())[3], p3);
    return call_generic(params);
  }

  template <class T0, class T1, class T2, class T3, class T4>
  inline nd::array callable::call(const T0 &p0, const T1 &p1, const T2 &p2, const T3 &p3, const T4 &p4) const
  {
    const ndt::struct_type *fsdt = m_parameters_type.extended<ndt::struct_type>();
    intptr_t parameter_count = fsdt->get_field_count();
    nd::array params = nd::empty(m_parameters_type);
    if (fsdt->get_field_count() != 5) {
      if (parameter_count > 5 && m_first_default_parameter <= 5) {
        // Fill the missing parameters with their defaults, if available
        for (intptr_t i = 5; i < parameter_count; ++i) {
          size_t arrmeta_offset = fsdt->get_arrmeta_offset(i);
          size_t data_offset = fsdt->get_data_offsets(params.get()->metadata())[i];
          typed_data_copy(fsdt->get_field_type(i), params.get()->metadata() + arrmeta_offset, params.get()->data + data_offset,
                          m_default_parameters.get()->metadata() + arrmeta_offset,
                          m_default_parameters.get()->data + data_offset);
        }
      } else {
        std::stringstream ss;
        ss << "incorrect number of arguments (received 5) for dynd callable with parameters " << m_parameters_type;
        throw std::runtime_error(ss.str());
      }
    }
    detail::callable_argument_setter<T0>::set(fsdt->get_field_type(0), params.get()->metadata() + fsdt->get_arrmeta_offset(0),
                                              params.get()->data + fsdt->get_data_offsets(params.get()->metadata())[0], p0);
    detail::callable_argument_setter<T1>::set(fsdt->get_field_type(1), params.get()->metadata() + fsdt->get_arrmeta_offset(1),
                                              params.get()->data + fsdt->get_data_offsets(params.get()->metadata())[1], p1);
    detail::callable_argument_setter<T2>::set(fsdt->get_field_type(2), params.get()->metadata() + fsdt->get_arrmeta_offset(2),
                                              params.get()->data + fsdt->get_data_offsets(params.get()->metadata())[2], p2);
    detail::callable_argument_setter<T3>::set(fsdt->get_field_type(3), params.get()->metadata() + fsdt->get_arrmeta_offset(3),
                                              params.get()->data + fsdt->get_data_offsets(params.get()->metadata())[3], p3);
    detail::callable_argument_setter<T4>::set(fsdt->get_field_type(4), params.get()->metadata() + fsdt->get_arrmeta_offset(4),
                                              params.get()->data + fsdt->get_data_offsets(params.get()->metadata())[4], p4);
    return call_generic(params);
  }

} // namespace gfunc

//////////////////////////////////////////
// Some functions from nd::array that use callable.call

/** Calls the dynamic function - #include <dynd/gfunc/call_callable.hpp> to use it */
inline nd::array nd::array::f(const char *function_name)
{
  return find_dynamic_function(function_name).call(*this);
}
inline nd::array nd::array::f(const char *function_name) const
{
  return find_dynamic_function(function_name).call(*this);
}

/** Calls the dynamic function - #include <dynd/gfunc/call_callable.hpp> to use it */
template <class T0>
inline nd::array nd::array::f(const char *function_name, const T0 &p0)
{
  return find_dynamic_function(function_name).call(*this, p0);
}

/** Calls the dynamic function - #include <dynd/gfunc/call_callable.hpp> to use it */
template <class T0, class T1>
inline nd::array nd::array::f(const char *function_name, const T0 &p0, const T1 &p1)
{
  return find_dynamic_function(function_name).call(*this, p0, p1);
}

/** Calls the dynamic function - #include <dynd/gfunc/call_callable.hpp> to use it */
template <class T0, class T1, class T2>
inline nd::array nd::array::f(const char *function_name, const T0 &p0, const T1 &p1, const T2 &p2)
{
  return find_dynamic_function(function_name).call(*this, p0, p1, p2);
}

/** Calls the dynamic function - #include <dynd/gfunc/call_callable.hpp> to use it */
template <class T0, class T1, class T2, class T3>
inline nd::array nd::array::f(const char *function_name, const T0 &p0, const T1 &p1, const T2 &p2, const T3 &p3)
{
  return find_dynamic_function(function_name).call(*this, p0, p1, p2, p3);
}

} // namespace dynd
