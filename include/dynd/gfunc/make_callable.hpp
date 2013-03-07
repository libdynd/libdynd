//
// Copyright (C) 2011-13, DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#ifndef _DYND__MAKE_CALLABLE_HPP_
#define _DYND__MAKE_CALLABLE_HPP_

#include <dynd/gfunc/callable.hpp>
#include <dynd/dtypes/fixedstruct_dtype.hpp>
#include <dynd/dtypes/fixed_dim_dtype.hpp>
#include <dynd/dtypes/void_pointer_dtype.hpp>
#include <dynd/dtypes/string_dtype.hpp>
#include <dynd/dtypes/dtype_dtype.hpp>

namespace dynd { namespace gfunc {

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
template <> struct parameter_type_of<dtype> {typedef const base_dtype *type;};
template <> struct parameter_type_of<std::string> {typedef string_dtype_data type;};

template <typename T> struct make_parameter_dtype {inline static dtype make() {
        return make_dtype<typename parameter_type_of<T>::type>();
    }};
template <typename T> struct make_parameter_dtype<T &> : public make_parameter_dtype<T> {};
template <typename T> struct make_parameter_dtype<const T> : public make_parameter_dtype<T> {};
template <typename T, int N> struct make_parameter_dtype<T[N]> {inline static dtype make() {
        return make_fixed_dim_dtype(N, make_dtype<T>());
    }};
// Use void* to pass ndobject and dtype as parameters, correctness currently will
// rely on using them in the right context. To pass these properly will require
// dynd to grow the ability to manage object memory.
template <> struct make_parameter_dtype<ndobject> {inline static dtype make() {
        return dtype(new void_pointer_dtype, false);
    }};
template <> struct make_parameter_dtype<dtype> {inline static dtype make() {
        return make_dtype_dtype();
    }};
template <> struct make_parameter_dtype<std::string> {inline static dtype make() {
        return make_string_dtype(string_encoding_utf_8);
    }};

template <typename T> struct box_result {
    inline static typename enable_if<is_dtype_scalar<T>::value, ndobject_preamble *>::type box(const T& v) {
        return ndobject(v).release();
    }
};
template <typename T> struct box_result<T &> : public box_result<T> {};
template <typename T> struct box_result<const T> : public box_result<T> {};
template <> struct box_result<ndobject> {
    inline static ndobject_preamble *box(const ndobject& v) {
        // Throwing away v's value is ok here, for the limited use of this function
        return const_cast<ndobject&>(v).release();
    }
};
template <> struct box_result<dtype> {
    inline static ndobject_preamble *box(const dtype& v) {
        return ndobject(v).release();
    }
};
template <> struct box_result<std::string> {
    inline static ndobject_preamble *box(const std::string& v) {
        return ndobject(v).release();
    }
};

template <typename T> struct unbox_param {
    inline static typename enable_if<is_dtype_scalar<T>::value, const T&>::type unbox(const T& v) {
        return v;
    }
};
template <> struct unbox_param<bool> {
    inline static bool unbox(const dynd_bool& v) {
        return v;
    }
};
template <typename T> struct unbox_param<T &> : public unbox_param<T> {};
template <typename T> struct unbox_param<const T> : public unbox_param<T> {};
template <typename T, int N> struct unbox_param<T[N]> {
    typedef T (&result_type)[N];
    inline static result_type unbox(T (&v)[N]) {
        return v;
    }
};
template <> struct unbox_param<ndobject> {
    inline static ndobject unbox(ndobject_preamble *v) {
        return ndobject(v, true);
    }
};
template <> struct unbox_param<dtype> {
    inline static dtype unbox(const base_dtype *v) {
        return dtype(v, true);
    }
};
template <> struct unbox_param<std::string> {
    inline static std::string unbox(string_dtype_data& v) {
        return std::string(v.begin, v.end);
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
        static ndobject_preamble *wrapper(const ndobject_preamble *params, void *extra) {
            params_struct *p = reinterpret_cast<params_struct *>(params->m_data_pointer);
            function_pointer_t f = reinterpret_cast<function_pointer_t>(extra);
            return box_result<R>::box(f(unbox_param<P0>::unbox(p->p0)));
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
        static ndobject_preamble *wrapper(const ndobject_preamble *params, void *extra) {
            params_struct *p = reinterpret_cast<params_struct *>(params->m_data_pointer);
            function_pointer_t f = reinterpret_cast<function_pointer_t>(extra);
            return box_result<R>::box(f(unbox_param<P0>::unbox(p->p0), unbox_param<P1>::unbox(p->p1)));
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
        static ndobject_preamble *wrapper(const ndobject_preamble *params, void *extra) {
            params_struct *p = reinterpret_cast<params_struct *>(params->m_data_pointer);
            function_pointer_t f = reinterpret_cast<function_pointer_t>(extra);
            return box_result<R>::box(f(unbox_param<P0>::unbox(p->p0), unbox_param<P1>::unbox(p->p1),
                            unbox_param<P2>::unbox(p->p2)));
        }
        static dtype make_parameters_dtype(const char *name0, const char *name1, const char *name2) {
            return make_fixedstruct_dtype(make_parameter_dtype<P0>::make(), name0,
                            make_parameter_dtype<P1>::make(), name1,
                            make_parameter_dtype<P2>::make(), name2);
        }
    };

    // Metaprogramming wrapper for 4-parameter function
    template<class R, class P0, class P1, class P2, class P3>
    struct callable_maker<R (*)(P0, P1, P2, P3)> {
        typedef R (*function_pointer_t)(P0, P1, P2, P3);
        struct params_struct {
            typename parameter_type_of<P0>::type p0;
            typename parameter_type_of<P1>::type p1;
            typename parameter_type_of<P2>::type p2;
            typename parameter_type_of<P3>::type p3;
        };
        static ndobject_preamble *wrapper(const ndobject_preamble *params, void *extra) {
            params_struct *p = reinterpret_cast<params_struct *>(params->m_data_pointer);
            function_pointer_t f = reinterpret_cast<function_pointer_t>(extra);
            return box_result<R>::box(f(unbox_param<P0>::unbox(p->p0), unbox_param<P1>::unbox(p->p1),
                            unbox_param<P2>::unbox(p->p2), unbox_param<P3>::unbox(p->p3)));
        }
        static dtype make_parameters_dtype(const char *name0, const char *name1, const char *name2, const char *name3) {
            dtype field_types[4];
            std::string field_names[4];
            field_types[0] = make_parameter_dtype<P0>::make();
            field_types[1] = make_parameter_dtype<P1>::make();
            field_types[2] = make_parameter_dtype<P2>::make();
            field_types[3] = make_parameter_dtype<P3>::make();
            field_names[0] = name0;
            field_names[1] = name1;
            field_names[2] = name2;
            field_names[3] = name3;
            return make_fixedstruct_dtype(4, field_types, field_names);
        }
    };

    // Metaprogramming wrapper for 5-parameter function
    template<class R, class P0, class P1, class P2, class P3, class P4>
    struct callable_maker<R (*)(P0, P1, P2, P3, P4)> {
        typedef R (*function_pointer_t)(P0, P1, P2, P3, P4);
        struct params_struct {
            typename parameter_type_of<P0>::type p0;
            typename parameter_type_of<P1>::type p1;
            typename parameter_type_of<P2>::type p2;
            typename parameter_type_of<P3>::type p3;
            typename parameter_type_of<P4>::type p4;
        };
        static ndobject_preamble *wrapper(const ndobject_preamble *params, void *extra) {
            params_struct *p = reinterpret_cast<params_struct *>(params->m_data_pointer);
            function_pointer_t f = reinterpret_cast<function_pointer_t>(extra);
            return box_result<R>::box(f(unbox_param<P0>::unbox(p->p0), unbox_param<P1>::unbox(p->p1),
                            unbox_param<P2>::unbox(p->p2), unbox_param<P3>::unbox(p->p3),
                            unbox_param<P4>::unbox(p->p4)));
        }
        static dtype make_parameters_dtype(const char *name0, const char *name1, const char *name2,
                        const char *name3, const char *name4) {
            dtype field_types[5];
            std::string field_names[5];
            field_types[0] = make_parameter_dtype<P0>::make();
            field_types[1] = make_parameter_dtype<P1>::make();
            field_types[2] = make_parameter_dtype<P2>::make();
            field_types[3] = make_parameter_dtype<P3>::make();
            field_types[4] = make_parameter_dtype<P4>::make();
            field_names[0] = name0;
            field_names[1] = name1;
            field_names[2] = name2;
            field_names[3] = name3;
            field_names[4] = name4;
            return make_fixedstruct_dtype(5, field_types, field_names);
        }
    };
} // namespace detail

// One parameter, no defaults
template<typename FN>
inline callable make_callable(FN *f, const char *name0) {
    return callable(detail::callable_maker<FN *>::make_parameters_dtype(name0),
                &detail::callable_maker<FN *>::wrapper,
                reinterpret_cast<void *>(f));
}

// One parameter, one default
template<typename FN, typename D0>
inline callable make_callable_with_default(FN *f, const char *name0, const D0& default0) {
    dtype pdt = detail::callable_maker<FN *>::make_parameters_dtype(name0);
    ndobject defaults = empty(pdt);
    defaults.at(0).vals() = default0;
    // Make defaults immutable (which is ok, because we have the only reference to it)
    defaults.flag_as_immutable();
    return callable(pdt,
                &detail::callable_maker<FN *>::wrapper,
                reinterpret_cast<void *>(f),
                0, defaults);
}

// Two parameters, no defaults
template<typename FN>
inline callable make_callable(FN *f, const char *name0, const char *name1) {
    return callable(detail::callable_maker<FN *>::make_parameters_dtype(name0, name1),
                &detail::callable_maker<FN *>::wrapper,
                reinterpret_cast<void *>(f));
}

// Two parameters, one default
template<typename FN, typename D1>
inline callable make_callable_with_default(FN *f, const char *name0, const char *name1, const D1& default1) {
    dtype pdt = detail::callable_maker<FN *>::make_parameters_dtype(name0, name1);
    ndobject defaults = empty(pdt);
    defaults.at(1).vals() = default1;
    // Make defaults immutable (which is ok, because we have the only reference to it)
    defaults.flag_as_immutable();
    return callable(pdt,
                &detail::callable_maker<FN *>::wrapper,
                reinterpret_cast<void *>(f),
                1, defaults);
}

// Two parameters, two defaults
template<typename FN, typename D0, typename D1>
inline callable make_callable_with_default(FN *f, const char *name0, const char *name1, const D0& default0, const D1& default1) {
    dtype pdt = detail::callable_maker<FN *>::make_parameters_dtype(name0, name1);
    ndobject defaults = empty(pdt);
    defaults.at(0).vals() = default0;
    defaults.at(1).vals() = default1;
    // Make defaults immutable (which is ok, because we have the only reference to it)
    defaults.flag_as_immutable();
    return callable(pdt,
                &detail::callable_maker<FN *>::wrapper,
                reinterpret_cast<void *>(f),
                0, defaults);
}

// Three parameters, no defaults
template<typename FN>
inline callable make_callable(FN *f, const char *name0, const char *name1, const char *name2) {
    return callable(detail::callable_maker<FN *>::make_parameters_dtype(name0, name1, name2),
                &detail::callable_maker<FN *>::wrapper,
                reinterpret_cast<void *>(f));
}

// Three parameters, one default
template<typename FN, typename D2>
inline callable make_callable_with_default(FN *f, const char *name0, const char *name1, const char *name2, const D2& default2) {
    dtype pdt = detail::callable_maker<FN *>::make_parameters_dtype(name0, name1, name2);
    ndobject defaults = empty(pdt);
    defaults.at(2).vals() = default2;
    // Make defaults immutable (which is ok, because we have the only reference to it)
    defaults.flag_as_immutable();
    return callable(pdt,
                &detail::callable_maker<FN *>::wrapper,
                reinterpret_cast<void *>(f),
                2, defaults);
}

// Three parameters, two defaults
template<typename FN, typename D1, typename D2>
inline callable make_callable_with_default(FN *f, const char *name0, const char *name1, const char *name2, const D1& default1, const D2& default2) {
    dtype pdt = detail::callable_maker<FN *>::make_parameters_dtype(name0, name1, name2);
    ndobject defaults = empty(pdt);
    defaults.at(1).vals() = default1;
    defaults.at(2).vals() = default2;
    // Make defaults immutable (which is ok, because we have the only reference to it)
    defaults.flag_as_immutable();
    return callable(pdt,
                &detail::callable_maker<FN *>::wrapper,
                reinterpret_cast<void *>(f),
                1, defaults);
}

// Three parameters, three defaults
template<typename FN, typename D0, typename D1, typename D2>
inline callable make_callable_with_default(FN *f, const char *name0, const char *name1, const char *name2, const D0& default0, const D1& default1, const D2& default2) {
    dtype pdt = detail::callable_maker<FN *>::make_parameters_dtype(name0, name1, name2);
    ndobject defaults = empty(pdt);
    defaults.at(0).vals() = default0;
    defaults.at(1).vals() = default1;
    defaults.at(2).vals() = default2;
    // Make defaults immutable (which is ok, because we have the only reference to it)
    defaults.flag_as_immutable();
    return callable(pdt,
                &detail::callable_maker<FN *>::wrapper,
                reinterpret_cast<void *>(f),
                0, defaults);
}

// Four parameters, no defaults
template<typename FN>
inline callable make_callable(FN *f, const char *name0, const char *name1, const char *name2, const char *name3) {
    return callable(detail::callable_maker<FN *>::make_parameters_dtype(name0, name1, name2, name3),
                &detail::callable_maker<FN *>::wrapper,
                reinterpret_cast<void *>(f));
}

// Four parameters, one default
template<typename FN, typename D3>
inline callable make_callable_with_default(FN *f, const char *name0, const char *name1, const char *name2,
                const char *name3, const D3& default3) {
    dtype pdt = detail::callable_maker<FN *>::make_parameters_dtype(name0, name1, name2, name3);
    ndobject defaults = empty(pdt);
    defaults.at(3).vals() = default3;
    // Make defaults immutable (which is ok, because we have the only reference to it)
    defaults.flag_as_immutable();
    return callable(pdt,
                &detail::callable_maker<FN *>::wrapper,
                reinterpret_cast<void *>(f),
                3, defaults);
}

// Four parameters, two defaults
template<typename FN, typename D2, typename D3>
inline callable make_callable_with_default(FN *f, const char *name0, const char *name1, const char *name2,
                const char *name3, const D2& default2, const D3& default3) {
    dtype pdt = detail::callable_maker<FN *>::make_parameters_dtype(name0, name1, name2, name3);
    ndobject defaults = empty(pdt);
    defaults.at(2).vals() = default2;
    defaults.at(3).vals() = default3;
    // Make defaults immutable (which is ok, because we have the only reference to it)
    defaults.flag_as_immutable();
    return callable(pdt,
                &detail::callable_maker<FN *>::wrapper,
                reinterpret_cast<void *>(f),
                2, defaults);
}

// Four parameters, three defaults
template<typename FN, typename D1, typename D2, typename D3>
inline callable make_callable_with_default(FN *f, const char *name0, const char *name1, const char *name2,
                const char *name3, const D1& default1, const D2& default2, const D3& default3) {
    dtype pdt = detail::callable_maker<FN *>::make_parameters_dtype(name0, name1, name2, name3);
    ndobject defaults = empty(pdt);
    defaults.at(1).vals() = default1;
    defaults.at(2).vals() = default2;
    defaults.at(3).vals() = default3;
    // Make defaults immutable (which is ok, because we have the only reference to it)
    defaults.flag_as_immutable();
    return callable(pdt,
                &detail::callable_maker<FN *>::wrapper,
                reinterpret_cast<void *>(f),
                1, defaults);
}

// Four parameters, four defaults
template<typename FN, typename D0, typename D1, typename D2, typename D3>
inline callable make_callable_with_default(FN *f, const char *name0, const char *name1, const char *name2,
                const char *name3, const D0& default0, const D1& default1, const D2& default2, const D3& default3) {
    dtype pdt = detail::callable_maker<FN *>::make_parameters_dtype(name0, name1, name2, name3);
    ndobject defaults = empty(pdt);
    defaults.at(0).vals() = default0;
    defaults.at(1).vals() = default1;
    defaults.at(2).vals() = default2;
    defaults.at(3).vals() = default3;
    // Make defaults immutable (which is ok, because we have the only reference to it)
    defaults.flag_as_immutable();
    return callable(pdt,
                &detail::callable_maker<FN *>::wrapper,
                reinterpret_cast<void *>(f),
                0, defaults);
}

template<typename FN>
inline callable make_callable(FN *f, const char *name0, const char *name1, const char *name2,
                const char *name3, const char *name4) {
    return callable(detail::callable_maker<FN *>::make_parameters_dtype(name0, name1, name2, name3, name4),
                &detail::callable_maker<FN *>::wrapper,
                reinterpret_cast<void *>(f));
}

}} // namespace dynd::gfunc

#endif // _DYND__MAKE_CALLABLE_HPP_
