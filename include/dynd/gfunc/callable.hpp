//
// Copyright (C) 2011-12, Dynamic NDArray Developers
// BSD 2-Clause License, see LICENSE.txt
//

#ifndef _DYND__CALLABLE_HPP_
#define _DYND__CALLABLE_HPP_

#include <dynd/ndobject.hpp>

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

    /** Calls the gfunc - #include <dynd/gfunc/call_callable.hpp> to use it */
    ndobject call() const;

    /** Calls the gfunc - #include <dynd/gfunc/call_callable.hpp> to use it */
    template<class T>
    ndobject call(const T& p0) const;

    /** Calls the gfunc - #include <dynd/gfunc/call_callable.hpp> to use it */
    template<class T0, class T1>
    ndobject call(const T0& p0, const T1& p1) const;

    /** Calls the gfunc - #include <dynd/gfunc/call_callable.hpp> to use it */
    template<class T0, class T1, class T2>
    ndobject call(const T0& p0, const T1& p1, const T2& p2) const;

    /** Calls the gfunc - #include <dynd/gfunc/call_callable.hpp> to use it */
    template<class T0, class T1, class T2, class T3>
    ndobject call(const T0& p0, const T1& p1, const T2& p2, const T3& p3) const;

    /** Calls the gfunc - #include <dynd/gfunc/call_callable.hpp> to use it */
    template<class T0, class T1, class T2, class T3, class T4>
    ndobject call(const T0& p0, const T1& p1, const T2& p2, const T3& p3, const T4& p4) const;

    void debug_print(std::ostream& o, const std::string& indent = "") const;
};


}} // namespace dynd::gfunc

#endif // _DYND__CALLABLE_HPP_
