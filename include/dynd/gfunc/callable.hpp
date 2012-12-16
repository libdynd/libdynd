//
// Copyright (C) 2011-12, Dynamic NDArray Developers
// BSD 2-Clause License, see LICENSE.txt
//

#ifndef _DYND__CALLABLE_HPP_
#define _DYND__CALLABLE_HPP_

#include <dynd/ndobject.hpp>
#include <dynd/dtypes/fixedstruct_dtype.hpp>
#include <dynd/dtypes/string_dtype.hpp>
#include <dynd/dtypes/fixedstring_dtype.hpp>
#include <dynd/dtype_assign.hpp>

namespace dynd { namespace gfunc {

/**
 * Generic prototype for a dynd callable object.
 *
 * \param params  The packed parameters for the function call.
 *                This corresponds to a particular fixedstruct parameters_pack dtype.
 * \param extra  Some static memory to help. TODO: switch to auxdata.
 *
 * \returns  A reference to an ndobject.
 */
typedef ndobject_preamble *(*callable_function_t)(const ndobject_preamble *params, void *extra);

/**
 * Object that provides a dynd-based parameter passing mechanism
 */
class callable {
    /** DType for the parameters, must be a fixedstruct dtype */
    dtype m_parameters_dtype;
    callable_function_t m_function;
    void *m_extra;
    int m_first_default_parameter;
    ndobject m_default_parameters;
public:
    inline callable()
        : m_parameters_dtype(), m_function(), m_extra()
    {}

    inline callable(const dtype& parameters_dtype, callable_function_t function, void *extra = NULL,
                    int first_default_parameter = std::numeric_limits<int>::max(), const ndobject& default_parameters = ndobject())
        : m_parameters_dtype(parameters_dtype), m_function(function), m_extra(extra),
            m_first_default_parameter(first_default_parameter),
            m_default_parameters(default_parameters)

    {
        if (!m_default_parameters.empty()) {
            // Make sure the default parameter values have the correct dtype
            if (m_default_parameters.get_dtype() != m_parameters_dtype) {
                throw std::runtime_error("dynd callable's default arguments have a different type than the parameters");
            }
            // Make sure the default parameter values are immutable
            if ((m_default_parameters.get_access_flags()&immutable_access_flag) == 0) {
                m_default_parameters = m_default_parameters.eval_immutable();
            }
        }
    }

    inline void set(const dtype& parameters_dtype, callable_function_t function, void *extra = NULL,
                    int first_default_parameter = std::numeric_limits<int>::max(), const ndobject& default_parameters = ndobject())
    {
        if (!default_parameters.empty()) {
            // Make sure the default parameter values have the correct dtype
            if (default_parameters.get_dtype() != parameters_dtype) {
                throw std::runtime_error("dynd callable's default arguments have a different type than the parameters");
            }
            // Make sure the default parameter values are immutable
            if ((default_parameters.get_access_flags()&immutable_access_flag) == 0) {
                m_default_parameters = default_parameters.eval_immutable();
            } else {
                m_default_parameters = default_parameters;
            }
        } else {
            m_default_parameters = ndobject();
        }
        m_parameters_dtype = parameters_dtype;
        m_function = function;
        m_extra = extra;
        m_first_default_parameter = first_default_parameter;
    }

    inline const dtype& get_parameters_dtype() const {
        return m_parameters_dtype;
    }
    
    inline void *get_extra() const {
        return m_extra;
    }

    inline callable_function_t get_function() const {
        return m_function;
    }

    inline int get_first_default_parameter() const {
        return m_first_default_parameter;
    }

    inline const ndobject& get_default_parameters() const {
        return m_default_parameters;
    }

    inline ndobject call_generic(const ndobject& n) const {
        return ndobject(m_function(n.get_ndo(), m_extra), false);
    }

    ndobject call() const;

    template<class T>
    ndobject call(const T& p0) const;

    template<class T0, class T1>
    ndobject call(const T0& p0, const T1& p1) const;

    template<class T0, class T1, class T2>
    ndobject call(const T0& p0, const T1& p1, const T2& p2) const;

    template<class T0, class T1, class T2, class T3>
    ndobject call(const T0& p0, const T1& p1, const T2& p2, const T3& p3) const;

    template<class T0, class T1, class T2, class T3, class T4>
    ndobject call(const T0& p0, const T1& p1, const T2& p2, const T3& p3, const T4& p4) const;

    void debug_print(std::ostream& o, const std::string& indent = "") const;
};

namespace detail {
    template<class T>
    struct callable_argument_setter {
        static typename enable_if<is_dtype_scalar<T>::value, void>::type set(const dtype& paramtype, char *metadata, char *data, const T& value) {
            if (paramtype.get_type_id() == static_cast<type_id_t>(type_id_of<T>::value)) {
                *reinterpret_cast<T *>(data) = value;
            } else {
                dtype_assign(paramtype, metadata, data, make_dtype<T>(), NULL, reinterpret_cast<const char *>(&value));
            }
        }
    };

    template<>
    struct callable_argument_setter<bool> {
        static void set(const dtype& paramtype, char *metadata, char *data, bool value) {
            if (paramtype.get_type_id() == bool_type_id) {
               *data = (value ? 1 : 0);
            } else {
                dynd_bool tmp = value;
                dtype_assign(paramtype, metadata, data, make_dtype<dynd_bool>(), NULL, reinterpret_cast<const char *>(&tmp));
            }
        }
    };

    template<>
    struct callable_argument_setter<ndobject> {
        static void set(const dtype& paramtype, char *metadata, char *data, const ndobject& value) {
            if (paramtype.get_type_id() == void_pointer_type_id) {
                // TODO: switch to a better mechanism for passing ndobject references
                *reinterpret_cast<const ndobject_preamble **>(data) = value.get_ndo();
            } else {
                dtype_assign(paramtype, metadata, data, value.get_dtype(), value.get_ndo_meta(), value.get_ndo()->m_data_pointer);
            }
        }
    };

    template<int N>
    struct callable_argument_setter<const char[N]> {
        static void set(const dtype& paramtype, char *metadata, char *data, const char (&value)[N]) {
            // Setting from a known-sized character string array
            if (paramtype.get_type_id() == string_type_id &&
                    static_cast<const string_dtype *>(paramtype.extended())->get_encoding() == string_encoding_utf_8) {
                reinterpret_cast<string_dtype_data*>(data)->begin = const_cast<char *>(value);
                reinterpret_cast<string_dtype_data*>(data)->end = const_cast<char *>(value + N);
            } else {
                dtype_assign(paramtype, metadata, data, make_fixedstring_dtype(string_encoding_utf_8, N),
                        NULL, value);
            }
        }
    };

    template<int N>
    struct callable_argument_setter<char[N]> : public callable_argument_setter<const char[N]> {};
} // namespace detail

inline ndobject callable::call() const
{
    const fixedstruct_dtype *fsdt = static_cast<const fixedstruct_dtype *>(m_parameters_dtype.extended());
    size_t parameter_count = fsdt->get_field_count();
    ndobject params(m_parameters_dtype);
    if (parameter_count != 0) {
        if (m_first_default_parameter <= 0) {
            // Fill the missing parameters with their defaults, if available
            for (size_t i = 0; i < parameter_count; ++i) {
                size_t metadata_offset = fsdt->get_metadata_offsets()[i];
                size_t data_offset = fsdt->get_data_offsets()[i];
                dtype_copy(fsdt->get_field_types()[i],
                                params.get_ndo_meta() + metadata_offset,
                                params.get_ndo()->m_data_pointer + data_offset,
                                m_default_parameters.get_ndo_meta() + metadata_offset,
                                m_default_parameters.get_ndo()->m_data_pointer + data_offset);
            }
        } else {
            std::stringstream ss;
            ss << "incorrect number of arguments (received 1) for dynd callable with parameters " << m_parameters_dtype;
            throw std::runtime_error(ss.str());
        }
    }
    return call_generic(params);
}

template<class T>
inline ndobject callable::call(const T& p0) const
{
    const fixedstruct_dtype *fsdt = static_cast<const fixedstruct_dtype *>(m_parameters_dtype.extended());
    size_t parameter_count = fsdt->get_field_count();
    ndobject params(m_parameters_dtype);
    if (parameter_count != 1) {
        if (parameter_count > 1 && m_first_default_parameter <= 1) {
            // Fill the missing parameters with their defaults, if available
            for (size_t i = 1; i < parameter_count; ++i) {
                size_t metadata_offset = fsdt->get_metadata_offsets()[i];
                size_t data_offset = fsdt->get_data_offsets()[i];
                dtype_copy(fsdt->get_field_types()[i],
                                params.get_ndo_meta() + metadata_offset,
                                params.get_ndo()->m_data_pointer + data_offset,
                                m_default_parameters.get_ndo_meta() + metadata_offset,
                                m_default_parameters.get_ndo()->m_data_pointer + data_offset);
            }
        } else {
            std::stringstream ss;
            ss << "incorrect number of arguments (received 1) for dynd callable with parameters " << m_parameters_dtype;
            throw std::runtime_error(ss.str());
        }
    }
    detail::callable_argument_setter<T>::set(fsdt->get_field_types()[0],
                    params.get_ndo_meta() + fsdt->get_metadata_offsets()[0],
                    params.get_ndo()->m_data_pointer + fsdt->get_data_offsets()[0],
                    p0);
    return call_generic(params);
}

template<class T0, class T1>
inline ndobject callable::call(const T0& p0, const T1& p1) const
{
    const fixedstruct_dtype *fsdt = static_cast<const fixedstruct_dtype *>(m_parameters_dtype.extended());
    size_t parameter_count = fsdt->get_field_count();
    ndobject params(m_parameters_dtype);
    if (fsdt->get_field_types().size() != 2) {
        if (parameter_count > 2 && m_first_default_parameter <= 2) {
            // Fill the missing parameters with their defaults, if available
            for (size_t i = 2; i < parameter_count; ++i) {
                size_t metadata_offset = fsdt->get_metadata_offsets()[i];
                size_t data_offset = fsdt->get_data_offsets()[i];
                dtype_copy(fsdt->get_field_types()[i],
                                params.get_ndo_meta() + metadata_offset,
                                params.get_ndo()->m_data_pointer + data_offset,
                                m_default_parameters.get_ndo_meta() + metadata_offset,
                                m_default_parameters.get_ndo()->m_data_pointer + data_offset);
            }
        } else {
            std::stringstream ss;
            ss << "incorrect number of arguments (received 2) for dynd callable with parameters " << m_parameters_dtype;
            throw std::runtime_error(ss.str());
        }
    }
    detail::callable_argument_setter<T0>::set(fsdt->get_field_types()[0],
                    params.get_ndo_meta() + fsdt->get_metadata_offsets()[0],
                    params.get_ndo()->m_data_pointer + fsdt->get_data_offsets()[0],
                    p0);
    detail::callable_argument_setter<T1>::set(fsdt->get_field_types()[1],
                    params.get_ndo_meta() + fsdt->get_metadata_offsets()[1],
                    params.get_ndo()->m_data_pointer + fsdt->get_data_offsets()[1],
                    p1);
    return call_generic(params);
}

template<class T0, class T1, class T2>
inline ndobject callable::call(const T0& p0, const T1& p1, const T2& p2) const
{
    const fixedstruct_dtype *fsdt = static_cast<const fixedstruct_dtype *>(m_parameters_dtype.extended());
    size_t parameter_count = fsdt->get_field_count();
    ndobject params(m_parameters_dtype);
    if (fsdt->get_field_types().size() != 3) {
        if (parameter_count > 3 && m_first_default_parameter <= 3) {
            // Fill the missing parameters with their defaults, if available
            for (size_t i = 3; i < parameter_count; ++i) {
                size_t metadata_offset = fsdt->get_metadata_offsets()[i];
                size_t data_offset = fsdt->get_data_offsets()[i];
                dtype_copy(fsdt->get_field_types()[i],
                                params.get_ndo_meta() + metadata_offset,
                                params.get_ndo()->m_data_pointer + data_offset,
                                m_default_parameters.get_ndo_meta() + metadata_offset,
                                m_default_parameters.get_ndo()->m_data_pointer + data_offset);
            }
        } else {
            std::stringstream ss;
            ss << "incorrect number of arguments (received 3) for dynd callable with parameters " << m_parameters_dtype;
            throw std::runtime_error(ss.str());
        }
    }
    detail::callable_argument_setter<T0>::set(fsdt->get_field_types()[0],
                    params.get_ndo_meta() + fsdt->get_metadata_offsets()[0],
                    params.get_ndo()->m_data_pointer + fsdt->get_data_offsets()[0],
                    p0);
    detail::callable_argument_setter<T1>::set(fsdt->get_field_types()[1],
                    params.get_ndo_meta() + fsdt->get_metadata_offsets()[1],
                    params.get_ndo()->m_data_pointer + fsdt->get_data_offsets()[1],
                    p1);
    detail::callable_argument_setter<T2>::set(fsdt->get_field_types()[2],
                    params.get_ndo_meta() + fsdt->get_metadata_offsets()[2],
                    params.get_ndo()->m_data_pointer + fsdt->get_data_offsets()[2],
                    p2);
    return call_generic(params);
}

template<class T0, class T1, class T2, class T3>
inline ndobject callable::call(const T0& p0, const T1& p1, const T2& p2, const T3& p3) const
{
    const fixedstruct_dtype *fsdt = static_cast<const fixedstruct_dtype *>(m_parameters_dtype.extended());
    size_t parameter_count = fsdt->get_field_count();
    ndobject params(m_parameters_dtype);
    if (fsdt->get_field_types().size() != 4) {
        if (parameter_count > 4 && m_first_default_parameter <= 4) {
            // Fill the missing parameters with their defaults, if available
            for (size_t i = 4; i < parameter_count; ++i) {
                size_t metadata_offset = fsdt->get_metadata_offsets()[i];
                size_t data_offset = fsdt->get_data_offsets()[i];
                dtype_copy(fsdt->get_field_types()[i],
                                params.get_ndo_meta() + metadata_offset,
                                params.get_ndo()->m_data_pointer + data_offset,
                                m_default_parameters.get_ndo_meta() + metadata_offset,
                                m_default_parameters.get_ndo()->m_data_pointer + data_offset);
            }
        } else {
            std::stringstream ss;
            ss << "incorrect number of arguments (received 4) for dynd callable with parameters " << m_parameters_dtype;
            throw std::runtime_error(ss.str());
        }
    }
    detail::callable_argument_setter<T0>::set(fsdt->get_field_types()[0],
                    params.get_ndo_meta() + fsdt->get_metadata_offsets()[0],
                    params.get_ndo()->m_data_pointer + fsdt->get_data_offsets()[0],
                    p0);
    detail::callable_argument_setter<T1>::set(fsdt->get_field_types()[1],
                    params.get_ndo_meta() + fsdt->get_metadata_offsets()[1],
                    params.get_ndo()->m_data_pointer + fsdt->get_data_offsets()[1],
                    p1);
    detail::callable_argument_setter<T2>::set(fsdt->get_field_types()[2],
                    params.get_ndo_meta() + fsdt->get_metadata_offsets()[2],
                    params.get_ndo()->m_data_pointer + fsdt->get_data_offsets()[2],
                    p2);
    detail::callable_argument_setter<T3>::set(fsdt->get_field_types()[3],
                    params.get_ndo_meta() + fsdt->get_metadata_offsets()[3],
                    params.get_ndo()->m_data_pointer + fsdt->get_data_offsets()[3],
                    p3);
    return call_generic(params);
}

template<class T0, class T1, class T2, class T3, class T4>
inline ndobject callable::call(const T0& p0, const T1& p1, const T2& p2, const T3& p3, const T4& p4) const
{
    const fixedstruct_dtype *fsdt = static_cast<const fixedstruct_dtype *>(m_parameters_dtype.extended());
    size_t parameter_count = fsdt->get_field_count();
    ndobject params(m_parameters_dtype);
    if (fsdt->get_field_types().size() != 5) {
        if (parameter_count > 5 && m_first_default_parameter <= 5) {
            // Fill the missing parameters with their defaults, if available
            for (size_t i = 5; i < parameter_count; ++i) {
                size_t metadata_offset = fsdt->get_metadata_offsets()[i];
                size_t data_offset = fsdt->get_data_offsets()[i];
                dtype_copy(fsdt->get_field_types()[i],
                                params.get_ndo_meta() + metadata_offset,
                                params.get_ndo()->m_data_pointer + data_offset,
                                m_default_parameters.get_ndo_meta() + metadata_offset,
                                m_default_parameters.get_ndo()->m_data_pointer + data_offset);
            }
        } else {
            std::stringstream ss;
            ss << "incorrect number of arguments (received 5) for dynd callable with parameters " << m_parameters_dtype;
            throw std::runtime_error(ss.str());
        }
    }
    detail::callable_argument_setter<T0>::set(fsdt->get_field_types()[0],
                    params.get_ndo_meta() + fsdt->get_metadata_offsets()[0],
                    params.get_ndo()->m_data_pointer + fsdt->get_data_offsets()[0],
                    p0);
    detail::callable_argument_setter<T1>::set(fsdt->get_field_types()[1],
                    params.get_ndo_meta() + fsdt->get_metadata_offsets()[1],
                    params.get_ndo()->m_data_pointer + fsdt->get_data_offsets()[1],
                    p1);
    detail::callable_argument_setter<T2>::set(fsdt->get_field_types()[2],
                    params.get_ndo_meta() + fsdt->get_metadata_offsets()[2],
                    params.get_ndo()->m_data_pointer + fsdt->get_data_offsets()[2],
                    p2);
    detail::callable_argument_setter<T3>::set(fsdt->get_field_types()[3],
                    params.get_ndo_meta() + fsdt->get_metadata_offsets()[3],
                    params.get_ndo()->m_data_pointer + fsdt->get_data_offsets()[3],
                    p3);
    detail::callable_argument_setter<T4>::set(fsdt->get_field_types()[4],
                    params.get_ndo_meta() + fsdt->get_metadata_offsets()[4],
                    params.get_ndo()->m_data_pointer + fsdt->get_data_offsets()[4],
                    p4);
    return call_generic(params);
}


}} // namespace dynd::gfunc

#endif // _DYND__CALLABLE_HPP_
