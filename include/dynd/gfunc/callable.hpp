//
// Copyright (C) 2011-12, Dynamic NDArray Developers
// BSD 2-Clause License, see LICENSE.txt
//

#ifndef _DYND__CALLABLE_HPP_
#define _DYND__CALLABLE_HPP_

#include <sstream>
#include <deque>
#include <vector>

#include <dynd/ndobject.hpp>
#include <dynd/dtypes/fixedstruct_dtype.hpp>
#include <dynd/dtypes/fixedarray_dtype.hpp>

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
typedef ndobject_preamble *(*callable_function_t)(ndobject_preamble *params, void *extra);

/**
 * Object that provides a dynd-based parameter passing mechanism
 */
class callable {
    /** DType for the parameters, must be a fixedstruct dtype */
    dtype m_parameters_dtype;
    void *m_extra;
    callable_function_t m_function;
public:
    inline callable(const dtype& parameters_dtype, callable_function_t function, void *extra = NULL)
        : m_parameters_dtype(parameters_dtype), m_extra(extra), m_function(function)
    {}

    inline const dtype& get_parameters_dtype() const {
        return m_parameters_dtype;
    }
    
    inline void *get_extra() const {
        return m_extra;
    }

    inline callable_function_t get_function() const {
        return m_function;
    }

    void debug_print(std::ostream& o, const std::string& indent = "") const;
};

template <typename T> struct parameter_type_of;
template <typename T> struct parameter_type_of<T &> : public parameter_type_of<T> {};
template <typename T> struct parameter_type_of<const T> : public parameter_type_of<T> {};
template <typename T, int N> struct parameter_type_of<T[N]> {typedef typename parameter_type_of<T>::type type[N];};
template <> struct parameter_type_of<bool> {typedef dynd_bool type;};
template <> struct parameter_type_of<signed char> {typedef signed char type;};
template <> struct parameter_type_of<short> {typedef short type;};
template <> struct parameter_type_of<int> {typedef int type;};
template <> struct parameter_type_of<long> {typedef long type;};
template <> struct parameter_type_of<long long> {typedef long long type;};
template <> struct parameter_type_of<uint8_t> {typedef uint8_t type;};
template <> struct parameter_type_of<uint16_t> {typedef uint16_t type;};
template <> struct parameter_type_of<unsigned int> {typedef unsigned int type;};
template <> struct parameter_type_of<unsigned long> {typedef unsigned long type;};
template <> struct parameter_type_of<unsigned long long> {typedef unsigned long long type;};
template <> struct parameter_type_of<float> {typedef float type;};
template <> struct parameter_type_of<double> {typedef double type;};
template <typename T> struct parameter_type_of<std::complex<T> > {typedef std::complex<T> type;};
template <> struct parameter_type_of<ndobject> {typedef ndobject_preamble *type;};

template <typename T> struct make_parameter_dtype {inline static dtype make() {
        return make_dtype<parameter_type_of<T>::type>();
    }};
template <typename T> struct make_parameter_dtype<T &> : public make_parameter_dtype<T> {};
template <typename T> struct make_parameter_dtype<const T> : public make_parameter_dtype<T> {};
template <typename T, int N> struct make_parameter_dtype<T[N]> {inline static dtype make() {
        return make_fixedarray_dtype(make_dtype<T>(), N);
    }};

template <typename T> struct box_result {
    inline static typename enable_if<is_dtype_scalar<T>::value, ndobject_preamble *>::type box(const T& v) {
        return ndobject(v).release();
    }
};


namespace detail {
    template <class FN>
    struct callable_maker;

    // Metaprogramming wrapper for 1-parameter function
    template<class R, class P0>
    struct callable_maker<R (*)(P0)> {
        typedef R (*function_pointer_t)(P0);
        struct params_struct {
            typename parameter_type_of<P0>::type p0;
        };
        static ndobject_preamble *wrapper(ndobject_preamble *params, void *extra) {
            params_struct *p = reinterpret_cast<params_struct *>(params->m_data_pointer);
            function_pointer_t f = reinterpret_cast<function_pointer_t>(extra);
            return box_result<R>::box(f(p->p0));
        }
        static dtype make_parameters_dtype(const char *name0) {
            return make_fixedstruct_dtype(make_parameter_dtype<P0>::make(), name0);
        }
    };

    // Metaprogramming wrapper for 2-parameter function
    template<class R, class P0, class P1>
    struct callable_maker<R (*)(P0, P1)> {
        typedef R (*function_pointer_t)(P0, P1);
        struct params_struct {
            typename parameter_type_of<P0>::type p0;
            typename parameter_type_of<P1>::type p1;
        };
        static ndobject_preamble *wrapper(ndobject_preamble *params, void *extra) {
            params_struct *p = reinterpret_cast<params_struct *>(params->m_data_pointer);
            function_pointer_t f = reinterpret_cast<function_pointer_t>(extra);
            return box_result<R>::box(f(p->p0, p->p1));
        }
        static dtype make_parameters_dtype(const char *name0, const char *name1) {
            return make_fixedstruct_dtype(make_parameter_dtype<P0>::make(), name0,
                            make_parameter_dtype<P1>::make(), name1);
        }
    };

    // Metaprogramming wrapper for 3-parameter function
    template<class R, class P0, class P1, class P2>
    struct callable_maker<R (*)(P0, P1, P2)> {
        typedef R (*function_pointer_t)(P0, P1, P2);
        struct params_struct {
            typename parameter_type_of<P0>::type p0;
            typename parameter_type_of<P1>::type p1;
            typename parameter_type_of<P2>::type p2;
        };
        static ndobject_preamble *wrapper(ndobject_preamble *params, void *extra) {
            params_struct *p = reinterpret_cast<params_struct *>(params->m_data_pointer);
            function_pointer_t f = reinterpret_cast<function_pointer_t>(extra);
            return box_result<R>::box(f(p->p0, p->p1, p->p2));
        }
        static dtype make_parameters_dtype(const char *name0, const char *name1, const char *name2) {
            return make_fixedstruct_dtype(make_parameter_dtype<P0>::make(), name0,
                            make_parameter_dtype<P1>::make(), name1,
                            make_parameter_dtype<P2>::make(), name2);
        }
    };
} // namespace detail

template<typename FN>
inline callable make_callable(FN *f, const char *name0) {
    callable result = callable(detail::callable_maker<FN *>::make_parameters_dtype(name0),
                &detail::callable_maker<FN *>::wrapper,
                reinterpret_cast<void *>(f));
    return result;
}

template<typename FN>
inline callable make_callable(FN *f, const char *name0, const char *name1) {
    callable result = callable(detail::callable_maker<FN *>::make_parameters_dtype(name0, name1),
                &detail::callable_maker<FN *>::wrapper,
                reinterpret_cast<void *>(f));
    return result;
}

template<typename FN>
inline callable make_callable(FN *f, const char *name0, const char *name1, const char *name2) {
    callable result = callable(detail::callable_maker<FN *>::make_parameters_dtype(name0, name1, name2),
                &detail::callable_maker<FN *>::wrapper,
                reinterpret_cast<void *>(f));
    return result;
}

}} // namespace dynd::gfunc

#endif // _DYND__CALLABLE_HPP_
