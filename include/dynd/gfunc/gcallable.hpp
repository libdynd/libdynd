//
// Copyright (C) 2011-15 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#pragma once

#include <dynd/array.hpp>

namespace dynd { namespace gfunc {

/**
 * Generic prototype for a dynd callable object.
 *
 * \param params  The packed parameters for the function call.
 *                This corresponds to a particular cstruct parameters_pack type.
 * \param extra  Some static memory to help. TODO: switch to auxdata.
 *
 * \returns  A reference to an nd::array.
 */
typedef array_preamble *(*callable_function_t)(const array_preamble *params, void *extra);

/**
 * Object that provides a dynd-based parameter passing mechanism
 */
class DYND_API callable {
    /** Type for the parameters, must be a cstruct type */
    ndt::type m_parameters_type;
    callable_function_t m_function;
    void *m_extra;
    int m_first_default_parameter;
    nd::array m_default_parameters;
public:
    inline callable()
        : m_parameters_type(), m_function(), m_extra()
    {}

    inline callable(const ndt::type& parameters_tp, callable_function_t function, void *extra = NULL,
                    int first_default_parameter = std::numeric_limits<int>::max(), const nd::array& default_parameters = nd::array())
        : m_parameters_type(parameters_tp), m_function(function), m_extra(extra),
            m_first_default_parameter(first_default_parameter),
            m_default_parameters(default_parameters)

    {
        if (!m_default_parameters.is_null()) {
            // Make sure the default parameter values have the correct type
            if (m_default_parameters.get_type() != m_parameters_type) {
                throw std::invalid_argument("dynd callable's default arguments have a different type than the parameters");
            }
            // Make sure the default parameter values are immutable
            if ((m_default_parameters.get_access_flags()&nd::immutable_access_flag) == 0) {
                m_default_parameters = m_default_parameters.eval_immutable();
            }
        }
    }

    inline void set(const ndt::type& parameters_tp, callable_function_t function, void *extra = NULL,
                    int first_default_parameter = std::numeric_limits<int>::max(), const nd::array& default_parameters = nd::array())
    {
        if (!default_parameters.is_null()) {
            // Make sure the default parameter values have the correct type
            if (default_parameters.get_type() != parameters_tp) {
                throw std::invalid_argument("dynd callable's default arguments have a different type than the parameters");
            }
            // Make sure the default parameter values are immutable
            if ((default_parameters.get_access_flags()&nd::immutable_access_flag) == 0) {
                m_default_parameters = default_parameters.eval_immutable();
            } else {
                m_default_parameters = default_parameters;
            }
        } else {
            m_default_parameters = nd::array();
        }
        m_parameters_type = parameters_tp;
        m_function = function;
        m_extra = extra;
        m_first_default_parameter = first_default_parameter;
    }

    inline const ndt::type& get_parameters_type() const {
        return m_parameters_type;
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

    inline const nd::array& get_default_parameters() const {
        return m_default_parameters;
    }

    inline nd::array call_generic(const nd::array& n) const {
        return nd::array(m_function(n.get(), m_extra), false);
    }

    /** Calls the gfunc - #include <dynd/gfunc/call_callable.hpp> to use it */
    nd::array call() const;

    /** Calls the gfunc - #include <dynd/gfunc/call_callable.hpp> to use it */
    template<class T>
    nd::array call(const T& p0) const;

    /** Calls the gfunc - #include <dynd/gfunc/call_callable.hpp> to use it */
    template<class T0, class T1>
    nd::array call(const T0& p0, const T1& p1) const;

    /** Calls the gfunc - #include <dynd/gfunc/call_callable.hpp> to use it */
    template<class T0, class T1, class T2>
    nd::array call(const T0& p0, const T1& p1, const T2& p2) const;

    /** Calls the gfunc - #include <dynd/gfunc/call_callable.hpp> to use it */
    template<class T0, class T1, class T2, class T3>
    nd::array call(const T0& p0, const T1& p1, const T2& p2, const T3& p3) const;

    /** Calls the gfunc - #include <dynd/gfunc/call_callable.hpp> to use it */
    template<class T0, class T1, class T2, class T3, class T4>
    nd::array call(const T0& p0, const T1& p1, const T2& p2, const T3& p3, const T4& p4) const;

    void debug_print(std::ostream& o, const std::string& indent = "") const;
};


}} // namespace dynd::gfunc
