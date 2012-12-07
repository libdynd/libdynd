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
public:
    inline callable()
        : m_parameters_dtype(), m_function(), m_extra()
    {}

    inline callable(const dtype& parameters_dtype, callable_function_t function, void *extra = NULL)
        : m_parameters_dtype(parameters_dtype), m_function(function), m_extra(extra)
    {}

    inline void set(const dtype& parameters_dtype, callable_function_t function, void *extra = NULL)
    {
        m_parameters_dtype = parameters_dtype;
        m_function = function;
        m_extra = extra;
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

    inline ndobject call(const ndobject& n) const {
        return ndobject(m_function(n.get_ndo(), m_extra), false);
    }

    void debug_print(std::ostream& o, const std::string& indent = "") const;
};

}} // namespace dynd::gfunc

#endif // _DYND__CALLABLE_HPP_
