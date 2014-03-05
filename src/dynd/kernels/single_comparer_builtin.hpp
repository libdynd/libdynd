//
// Copyright (C) 2011-14 Mark Wiebe, DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

// This file is an internal implementation detail of built-in value comparison

#include <dynd/type.hpp>
#include <complex>

using namespace std;

namespace dynd {

    // Bigger type metaprogram
    template<class S, class T, bool Sbigger>
    struct big_type_helper {
        typedef T type;
    };
    template<class S, class T>
    struct big_type_helper<S, T, true> {
        typedef S type;
    };
#define DYND_FORCE_BIG_TYPE(orig, forced) \
    template<class S> \
    struct big_type_helper<S, orig, true> { \
        typedef forced type; \
    }; \
    template<class S> \
    struct big_type_helper<S, orig, false> { \
        typedef forced type; \
    }; \
    template<class T> \
    struct big_type_helper<orig, T, true> { \
        typedef forced type; \
    }; \
    template<class T> \
    struct big_type_helper<orig, T, false> { \
        typedef forced type; \
    }; \
    template<> \
    struct big_type_helper<orig, orig, true> { \
        typedef orig type; \
    }; \
    template<> \
    struct big_type_helper<orig, orig, false> { \
        typedef orig type; \
    }
DYND_FORCE_BIG_TYPE(dynd_float16, double);
DYND_FORCE_BIG_TYPE(dynd_float128, dynd_float128);
    template<>
    struct big_type_helper<dynd_float16, dynd_float128, false> {
        typedef dynd_float128 type;
    };
    template<>
    struct big_type_helper<dynd_float128, dynd_float16, true> {
        typedef dynd_float128 type;
    };
    template<class S, class T>
    struct big_type {
        typedef typename big_type_helper<S, T, (sizeof(S) > sizeof(T))>::type type;
    };

    // Base classes for comparisons
    template<class src0_type, class src1_type, type_kind_t src0_kind,
                    type_kind_t src1_kind, bool src0_bigger, bool src1_bigger>
    struct op_sort_lt {
        inline static bool f(const src0_type& v0, const src1_type& v1)
        {
            typedef typename big_type<src0_type,src1_type>::type BT;
            return BT(v0) < BT(v1);
        }
    };
    template<class src0_type, class src1_type, type_kind_t src0_kind,
                    type_kind_t src1_kind, bool src0_bigger, bool src1_bigger>
    struct op_lt {
        inline static bool f(const src0_type& v0, const src1_type& v1)
        {
            typedef typename big_type<src0_type,src1_type>::type BT;
            return BT(v0) < BT(v1);
        }
    };
    template<class src0_type, class src1_type, type_kind_t src0_kind,
                    type_kind_t src1_kind, bool src0_bigger, bool src1_bigger>
    struct op_le {
        inline static bool f(const src0_type& v0, const src1_type& v1)
        {
            typedef typename big_type<src0_type,src1_type>::type BT;
            return BT(v0) <= BT(v1);
        }
    };
    template<class src0_type, class src1_type, type_kind_t src0_kind,
                    type_kind_t src1_kind, bool src0_bigger, bool src1_bigger>
    struct op_eq {
        inline static bool f(const src0_type& v0, const src1_type& v1)
        {
            typedef typename big_type<src0_type,src1_type>::type BT;
            return BT(v0) == BT(v1);
        }
    };
    template<class src0_type, class src1_type, type_kind_t src0_kind,
                    type_kind_t src1_kind, bool src0_bigger, bool src1_bigger>
    struct op_ne {
        inline static bool f(const src0_type& v0, const src1_type& v1)
        {
            typedef typename big_type<src0_type,src1_type>::type BT;
            return BT(v0) != BT(v1);
        }
    };
    template<class src0_type, class src1_type, type_kind_t src0_kind,
                    type_kind_t src1_kind, bool src0_bigger, bool src1_bigger>
    struct op_ge {
        inline static bool f(const src0_type& v0, const src1_type& v1)
        {
            typedef typename big_type<src0_type,src1_type>::type BT;
            return BT(v0) >= BT(v1);
        }
    };
    template<class src0_type, class src1_type, type_kind_t src0_kind,
                    type_kind_t src1_kind, bool src0_bigger, bool src1_bigger>
    struct op_gt {
        inline static bool f(const src0_type& v0, const src1_type& v1)
        {
            typedef typename big_type<src0_type,src1_type>::type BT;
            return BT(v0) > BT(v1);
        }
    };

    // int, uint, sizeof(src0_type) <= sizeof(src1_type)
    template<class src0_type, class src1_type, bool src1_bigger>
    struct op_sort_lt<src0_type, src1_type, int_kind, uint_kind, false, src1_bigger> {
        inline static bool f(const src0_type& v0, const src1_type& v1)
        {
            return v0 < src0_type(0) || static_cast<src1_type>(v0) < v1;
        }
    };
    //
    template<class src0_type, class src1_type, bool src1_bigger>
    struct op_lt<src0_type, src1_type, int_kind, uint_kind, false, src1_bigger>
        : public op_sort_lt<src0_type, src1_type, int_kind, uint_kind, false, src1_bigger> {};
    //
    template<class src0_type, class src1_type, bool src1_bigger>
    struct op_le<src0_type, src1_type, int_kind, uint_kind, false, src1_bigger> {
        inline static bool f(const src0_type& v0, const src1_type& v1)
        {
            return v0 <= src0_type(0) || static_cast<src1_type>(v0) <= v1;
        }
    };
    //
    template<class src0_type, class src1_type, bool src1_bigger>
    struct op_eq<src0_type, src1_type, int_kind, uint_kind, false, src1_bigger> {
        inline static bool f(const src0_type& v0, const src1_type& v1)
        {
            return v0 >= src0_type(0) && static_cast<src1_type>(v0) == v1;
        }
    };
    //
    template<class src0_type, class src1_type, bool src1_bigger>
    struct op_ne<src0_type, src1_type, int_kind, uint_kind, false, src1_bigger> {
        inline static bool f(const src0_type& v0, const src1_type& v1)
        {
            return v0 < src0_type(0) || static_cast<src1_type>(v0) != v1;
        }
    };
    //
    template<class src0_type, class src1_type, bool src1_bigger>
    struct op_ge<src0_type, src1_type, int_kind, uint_kind, false, src1_bigger> {
        inline static bool f(const src0_type& v0, const src1_type& v1)
        {
            return v0 >= src0_type(0) && static_cast<src1_type>(v0) >= v1;
        }
    };
    //
    template<class src0_type, class src1_type, bool src1_bigger>
    struct op_gt<src0_type, src1_type, int_kind, uint_kind, false, src1_bigger> {
        inline static bool f(const src0_type& v0, const src1_type& v1)
        {
            return v0 > src0_type(0) && static_cast<src1_type>(v0) > v1;
        }
    };

    // uint, int, sizeof(src0_type) >= sizeof(src1_type)
    template<class src0_type, class src1_type, bool src0_bigger>
    struct op_sort_lt<src0_type, src1_type, uint_kind, int_kind, src0_bigger, false> {
        inline static bool f(const src0_type& v0, const src1_type& v1)
        {
            return v1 > src1_type(0) && v0 < static_cast<src0_type>(v1);
        }
    };
    //
    template<class src0_type, class src1_type, bool src0_bigger>
    struct op_lt<src0_type, src1_type, uint_kind, int_kind, src0_bigger, false>
        : public op_sort_lt<src0_type, src1_type, uint_kind, int_kind, src0_bigger, false> {};
    //
    template<class src0_type, class src1_type, bool src0_bigger>
    struct op_le<src0_type, src1_type, uint_kind, int_kind, src0_bigger, false> {
        inline static bool f(const src0_type& v0, const src1_type& v1)
        {
            return v1 >= src1_type(0) && v0 <= static_cast<src0_type>(v1);
        }
    };
    //
    template<class src0_type, class src1_type, bool src0_bigger>
    struct op_eq<src0_type, src1_type, uint_kind, int_kind, src0_bigger, false> {
        inline static bool f(const src0_type& v0, const src1_type& v1)
        {
            return v1 >= src1_type(0) && v0 == static_cast<src0_type>(v1);
        }
    };
    //
    template<class src0_type, class src1_type, bool src0_bigger>
    struct op_ne<src0_type, src1_type, uint_kind, int_kind, src0_bigger, false> {
        inline static bool f(const src0_type& v0, const src1_type& v1)
        {
            return v1 < src1_type(0) || v0 != static_cast<src0_type>(v1);
        }
    };
    //
    template<class src0_type, class src1_type, bool src0_bigger>
    struct op_ge<src0_type, src1_type, uint_kind, int_kind, src0_bigger, false> {
        inline static bool f(const src0_type& v0, const src1_type& v1)
        {
            return v1 <= src1_type(0) || v0 >= static_cast<src0_type>(v1);
        }
    };
    //
    template<class src0_type, class src1_type, bool src0_bigger>
    struct op_gt<src0_type, src1_type, uint_kind, int_kind, src0_bigger, false> {
        inline static bool f(const src0_type& v0, const src1_type& v1)
        {
            return v1 < src1_type(0) || v0 > static_cast<src0_type>(v1);
        }
    };

    template<class src0_type, class src1_type, comparison_type_t comptype>
    struct op_cant_compare {
        inline static bool f(const src0_type& DYND_UNUSED(src0), const src1_type& DYND_UNUSED(src1))
        {
            throw not_comparable_error(ndt::make_type<src0_type>(), ndt::make_type<src1_type>(), comptype);
        }
    };

#define NOT_ORDERABLE(nord_kind) \
    template<class src0_type, class src1_type, type_kind_t src0_kind, bool src0_bigger, bool src1_bigger> \
    struct op_lt<src0_type, src1_type, src0_kind, nord_kind, src0_bigger, src1_bigger> \
        : public op_cant_compare<src0_type, src1_type, comparison_type_less> {}; \
    template<class src0_type, class src1_type, type_kind_t src0_kind, bool src0_bigger, bool src1_bigger> \
    struct op_le<src0_type, src1_type, src0_kind, nord_kind, src0_bigger, src1_bigger> \
        : public op_cant_compare<src0_type, src1_type, comparison_type_less_equal> {}; \
    template<class src0_type, class src1_type, type_kind_t src0_kind, bool src0_bigger, bool src1_bigger> \
    struct op_ge<src0_type, src1_type, src0_kind, nord_kind, src0_bigger, src1_bigger> \
        : public op_cant_compare<src0_type, src1_type, comparison_type_greater_equal> {}; \
    template<class src0_type, class src1_type, type_kind_t src0_kind, bool src0_bigger, bool src1_bigger> \
    struct op_gt<src0_type, src1_type, src0_kind, nord_kind, src0_bigger, src1_bigger> \
        : public op_cant_compare<src0_type, src1_type, comparison_type_greater> {}; \
    \
    template<class src0_type, class src1_type, type_kind_t src1_kind, bool src0_bigger, bool src1_bigger> \
    struct op_lt<src0_type, src1_type, nord_kind, src1_kind, src0_bigger, src1_bigger> \
        : public op_cant_compare<src0_type, src1_type, comparison_type_less> {}; \
    template<class src0_type, class src1_type, type_kind_t src1_kind, bool src0_bigger, bool src1_bigger> \
    struct op_le<src0_type, src1_type, nord_kind, src1_kind, src0_bigger, src1_bigger> \
        : public op_cant_compare<src0_type, src1_type, comparison_type_less_equal> {}; \
    template<class src0_type, class src1_type, type_kind_t src1_kind, bool src0_bigger, bool src1_bigger> \
    struct op_ge<src0_type, src1_type, nord_kind, src1_kind, src0_bigger, src1_bigger> \
        : public op_cant_compare<src0_type, src1_type, comparison_type_greater_equal> {}; \
    template<class src0_type, class src1_type, type_kind_t src1_kind, bool src0_bigger, bool src1_bigger> \
    struct op_gt<src0_type, src1_type, nord_kind, src1_kind, src0_bigger, src1_bigger> \
        : public op_cant_compare<src0_type, src1_type, comparison_type_greater> {}; \
    \
    template<class src0_type, class src1_type, bool src0_bigger, bool src1_bigger> \
    struct op_lt<src0_type, src1_type, nord_kind, nord_kind, src0_bigger, src1_bigger> \
        : public op_cant_compare<src0_type, src1_type, comparison_type_less> {}; \
    template<class src0_type, class src1_type, bool src0_bigger, bool src1_bigger> \
    struct op_le<src0_type, src1_type, nord_kind, nord_kind, src0_bigger, src1_bigger> \
        : public op_cant_compare<src0_type, src1_type, comparison_type_less_equal> {}; \
    template<class src0_type, class src1_type, bool src0_bigger, bool src1_bigger> \
    struct op_ge<src0_type, src1_type, nord_kind, nord_kind, src0_bigger, src1_bigger> \
        : public op_cant_compare<src0_type, src1_type, comparison_type_greater_equal> {}; \
    template<class src0_type, class src1_type, bool src0_bigger, bool src1_bigger> \
    struct op_gt<src0_type, src1_type, nord_kind, nord_kind, src0_bigger, src1_bigger> \
        : public op_cant_compare<src0_type, src1_type, comparison_type_greater> {};

#define NOT_ORDERABLE_PAIR(nord0_kind, nord1_kind) \
    template<class src0_type, class src1_type, bool src0_bigger, bool src1_bigger> \
    struct op_lt<src0_type, src1_type, nord0_kind, nord1_kind, src0_bigger, src1_bigger> \
        : public op_cant_compare<src0_type, src1_type, comparison_type_less> {}; \
    template<class src0_type, class src1_type, bool src0_bigger, bool src1_bigger> \
    struct op_le<src0_type, src1_type, nord0_kind, nord1_kind, src0_bigger, src1_bigger> \
        : public op_cant_compare<src0_type, src1_type, comparison_type_less_equal> {}; \
    template<class src0_type, class src1_type, bool src0_bigger, bool src1_bigger> \
    struct op_ge<src0_type, src1_type, nord0_kind, nord1_kind, src0_bigger, src1_bigger> \
        : public op_cant_compare<src0_type, src1_type, comparison_type_greater_equal> {}; \
    template<class src0_type, class src1_type, bool src0_bigger, bool src1_bigger> \
    struct op_gt<src0_type, src1_type, nord0_kind, nord1_kind, src0_bigger, src1_bigger> \
        : public op_cant_compare<src0_type, src1_type, comparison_type_greater> {};

    // real, real sorting comparison
    template<class src0_type, class src1_type,
                    bool src0_bigger, bool src1_bigger>
    struct op_sort_lt<src0_type, src1_type, real_kind, real_kind, src0_bigger, src1_bigger> {
        inline static bool f(const src0_type& v0, const src1_type& v1)
        {
            typedef typename big_type<src0_type,src1_type>::type BT;
            // This puts NaNs at the end
            return BT(v0) < BT(v1) || (v1 != v1 && v0 == v0);
        }
    };

    // int/uint, real comparison
    template<class src0_type, class src1_type,
                    bool src0_bigger, bool src1_bigger>
    struct op_eq<src0_type, src1_type, int_kind, real_kind, src0_bigger, src1_bigger> {
        inline static bool f(const src0_type& v0, const src1_type& v1)
        {
            // Slower, but more rigorous test
            return v0 == static_cast<src0_type>(v1) &&
                static_cast<src1_type>(v0) == v1;
        }
    };
    template<class src0_type,
                    bool src0_bigger, bool src1_bigger>
    struct op_eq<src0_type, dynd_float16, int_kind, real_kind, src0_bigger, src1_bigger> {
        inline static bool f(const src0_type& v0, const dynd_float16& v1)
        {
            // Slower, but more rigorous test
            return v0 == static_cast<src0_type>(double(v1)) &&
                double(v0) == v1;
        }
    };
    template<class src0_type, class src1_type,
                    bool src0_bigger, bool src1_bigger>
    struct op_ne<src0_type, src1_type, int_kind, real_kind, src0_bigger, src1_bigger> {
        inline static bool f(const src0_type& v0, const src1_type& v1)
        {
            // Slower, but more rigorous test
            return v0 != static_cast<src0_type>(v1) ||
                static_cast<src1_type>(v0) != v1;
        }
    };
    template<class src0_type,
                    bool src0_bigger, bool src1_bigger>
    struct op_ne<src0_type, dynd_float16, int_kind, real_kind, src0_bigger, src1_bigger> {
        inline static bool f(const src0_type& v0, const dynd_float16& v1)
        {
            // Slower, but more rigorous test
            return v0 != static_cast<src0_type>(double(v1)) ||
                double(v0) != v1;
        }
    };
    template<class src0_type, class src1_type,
                    bool src0_bigger, bool src1_bigger>
    struct op_eq<src0_type, src1_type, uint_kind, real_kind, src0_bigger, src1_bigger> {
        inline static bool f(const src0_type& v0, const src1_type& v1)
        {
            // Slower, but more rigorous test
            return v0 == static_cast<src0_type>(v1) &&
                static_cast<src1_type>(v0) == v1;
        }
    };
    template<class src0_type, class src1_type,
                    bool src0_bigger, bool src1_bigger>
    struct op_ne<src0_type, src1_type, uint_kind, real_kind, src0_bigger, src1_bigger> {
        inline static bool f(const src0_type& v0, const src1_type& v1)
        {
            // Slower, but more rigorous test
            return v0 != static_cast<src0_type>(v1) ||
                static_cast<src1_type>(v0) != v1;
        }
    };

    // real, int/uint comparison
    template<class src0_type, class src1_type,
                    bool src0_bigger, bool src1_bigger>
    struct op_eq<src0_type, src1_type, real_kind, int_kind, src0_bigger, src1_bigger> {
        inline static bool f(const src0_type& v0, const src1_type& v1)
        {
            // Slower, but more rigorous test
            return v0 == static_cast<src0_type>(v1) &&
                static_cast<src1_type>(v0) == v1;
        }
    };
    template<class src0_type, class src1_type,
                    bool src0_bigger, bool src1_bigger>
    struct op_ne<src0_type, src1_type, real_kind, int_kind, src0_bigger, src1_bigger> {
        inline static bool f(const src0_type& v0, const src1_type& v1)
        {
            // Slower, but more rigorous test
            return v0 != static_cast<src0_type>(v1) ||
                static_cast<src1_type>(v0) != v1;
        }
    };
    template<class src0_type, class src1_type,
                    bool src0_bigger, bool src1_bigger>
    struct op_eq<src0_type, src1_type, real_kind, uint_kind, src0_bigger, src1_bigger> {
        inline static bool f(const src0_type& v0, const src1_type& v1)
        {
            // Slower, but more rigorous test
            return v0 == static_cast<src0_type>(v1) &&
                static_cast<src1_type>(v0) == v1;
        }
    };
    template<class src0_type, class src1_type,
                    bool src0_bigger, bool src1_bigger>
    struct op_ne<src0_type, src1_type, real_kind, uint_kind, src0_bigger, src1_bigger> {
        inline static bool f(const src0_type& v0, const src1_type& v1)
        {
            // Slower, but more rigorous test
            return v0 != static_cast<src0_type>(v1) ||
                static_cast<src1_type>(v0) != v1;
        }
    };

    // Complex isn't comparable (except sorting_less, equal, not_equal)
    NOT_ORDERABLE(complex_kind);

    // Complex sorting comparison (lexicographic)
    template<class src0_real_type, class src1_real_type,
                    bool src0_bigger, bool src1_bigger>
    struct op_sort_lt<dynd_complex<src0_real_type>, dynd_complex<src1_real_type>, complex_kind, complex_kind, src0_bigger, src1_bigger> {
        inline static bool f(const dynd_complex<src0_real_type>& v0, const dynd_complex<src1_real_type>& v1)
        {
            // Sorts in the order like NumPy, [R + Rj, R + nanj, nan + Rj, nan + nanj]
            if (v0.real() < v1.real()) {
                return !DYND_ISNAN(v0.imag()) || DYND_ISNAN(v1.imag());
            } else if (v0.real() > v1.real()) {
                return DYND_ISNAN(v1.imag()) && !DYND_ISNAN(v0.imag());
            } else if (v0.real() == v1.real() || (DYND_ISNAN(v0.real()) && DYND_ISNAN(v1.real()))) {
                return v0.imag() < v1.imag() || (DYND_ISNAN(v1.imag()) && !DYND_ISNAN(v0.imag()));
            } else {
                return DYND_ISNAN(v1.real());
            }
        }
    };
    template<class src0_type, class src1_real_type, type_kind_t src0_kind,
                    bool src0_bigger, bool src1_bigger>
    struct op_sort_lt<src0_type, dynd_complex<src1_real_type>, src0_kind, complex_kind, src0_bigger, src1_bigger> {
        inline static bool f(const src0_type& v0, const dynd_complex<src1_real_type>& v1)
        {
            typedef typename big_type<src0_type,src1_real_type>::type BT;
            return BT(v0) < BT(v1.real()) ||
                (BT(v0) == BT(v1.real()) && 0 < v1.imag());
        }
    };
    template<class src0_real_type, class src1_type, type_kind_t src1_kind,
                    bool src0_bigger, bool src1_bigger>
    struct op_sort_lt<dynd_complex<src0_real_type>, src1_type, complex_kind, src1_kind, src0_bigger, src1_bigger> {
        inline static bool f(const dynd_complex<src0_real_type>& v0, const src1_type& v1)
        {
            typedef typename big_type<src0_real_type,src1_type>::type BT;
            return BT(v0.real()) < BT(v1) ||
                (BT(v0.real()) == BT(v1) && v0.imag() < 0);
        }
    };
    template<class src0_type, class src1_real_type,
                    bool src0_bigger, bool src1_bigger>
    struct op_sort_lt<src0_type, dynd_complex<src1_real_type>, bool_kind, complex_kind, src0_bigger, src1_bigger> {
        inline static bool f(const src0_type& v0, const dynd_complex<src1_real_type>& v1)
        {
            return static_cast<src1_real_type>(v0) < v1.real() ||
                (static_cast<src1_real_type>(v0) == v1.real() && 0 < v1.imag());
        }
    };
    template<class src0_real_type, class src1_type,
                    bool src0_bigger, bool src1_bigger>
    struct op_sort_lt<dynd_complex<src0_real_type>, src1_type, complex_kind, bool_kind, src0_bigger, src1_bigger> {
        inline static bool f(const dynd_complex<src0_real_type>& v0, const src1_type& v1)
        {
            return v0.real() < static_cast<src0_real_type>(v1) ||
                (v0.real() == static_cast<src0_real_type>(v1) && v0.imag() < 0);
        }
    };

    // complex, complex equality
    template<class src0_real_type, class src1_real_type,
                    bool src0_bigger, bool src1_bigger>
    struct op_eq<dynd_complex<src0_real_type>, dynd_complex<src1_real_type>, complex_kind, complex_kind, src0_bigger, src1_bigger> {
        inline static bool f(const dynd_complex<src0_real_type>& v0, const dynd_complex<src1_real_type>& v1)
        {
            return v0.real() == v1.real() && v0.imag() == v1.imag();
        }
    };

    // complex, complex inequality
    template<class src0_real_type, class src1_real_type,
                    bool src0_bigger, bool src1_bigger>
    struct op_ne<dynd_complex<src0_real_type>, dynd_complex<src1_real_type>, complex_kind, complex_kind, src0_bigger, src1_bigger> {
        inline static bool f(const dynd_complex<src0_real_type>& v0, const dynd_complex<src1_real_type>& v1)
        {
            return v0.real() != v1.real() || v0.imag() != v1.imag();
        }
    };

    // int/uint/real, complex equality
    template<class src0_type, class src1_real_type,
                    bool src0_bigger, bool src1_bigger>
    struct op_eq<src0_type, dynd_complex<src1_real_type>, int_kind, complex_kind, src0_bigger, src1_bigger> {
        inline static bool f(const src0_type& v0, const dynd_complex<src1_real_type>& v1)
        {
            return v1.imag() == 0 && v0 == static_cast<src0_type>(v1.real()) &&
                            static_cast<src1_real_type>(v0) == v1.real();
        }
    };
    template<class src0_type, class src1_real_type,
                    bool src0_bigger, bool src1_bigger>
    struct op_eq<src0_type, dynd_complex<src1_real_type>, uint_kind, complex_kind, src0_bigger, src1_bigger> {
        inline static bool f(const src0_type& v0, const dynd_complex<src1_real_type>& v1)
        {
            return v1.imag() == 0 && v0 == static_cast<src0_type>(v1.real()) &&
                            static_cast<src1_real_type>(v0) == v1.real();
        }
    };
    template<class src0_type, class src1_real_type,
                    bool src0_bigger, bool src1_bigger>
    struct op_eq<src0_type, dynd_complex<src1_real_type>, real_kind, complex_kind, src0_bigger, src1_bigger> {
        inline static bool f(const src0_type& v0, const dynd_complex<src1_real_type>& v1)
        {
            typedef typename big_type<src0_type,src1_real_type>::type BT;
            return v1.imag() == 0 && BT(v0) == BT(v1.real());
        }
    };

    // complex, int/uint/real equality
    template<class src0_real_type, class src1_type,
                    bool src0_bigger, bool src1_bigger>
    struct op_eq<dynd_complex<src0_real_type>, src1_type, complex_kind, int_kind, src0_bigger, src1_bigger> {
        inline static bool f(const dynd_complex<src0_real_type>& v0, const src1_type& v1)
        {
            return v0.imag() == 0 && v0.real() == static_cast<src0_real_type>(v1) &&
                            static_cast<src1_type>(v0.real()) == v1;
        }
    };
    template<class src0_real_type, class src1_type,
                    bool src0_bigger, bool src1_bigger>
    struct op_eq<dynd_complex<src0_real_type>, src1_type, complex_kind, uint_kind, src0_bigger, src1_bigger> {
        inline static bool f(const dynd_complex<src0_real_type>& v0, const src1_type& v1)
        {
            return v0.imag() == 0 && v0.real() == static_cast<src0_real_type>(v1) &&
                            static_cast<src1_type>(v0.real()) == v1;
        }
    };
    template<class src0_real_type, class src1_type,
                    bool src0_bigger, bool src1_bigger>
    struct op_eq<dynd_complex<src0_real_type>, src1_type, complex_kind, real_kind, src0_bigger, src1_bigger> {
        inline static bool f(const dynd_complex<src0_real_type>& v0, const src1_type& v1)
        {
            typedef typename big_type<src0_real_type,src1_type>::type BT;
            return v0.imag() == 0 && BT(v0.real()) == BT(v1);
        }
    };

    // int/uint/real, complex inequality
    template<class src0_type, class src1_real_type,
                    bool src0_bigger, bool src1_bigger>
    struct op_ne<src0_type, dynd_complex<src1_real_type>, uint_kind, complex_kind, src0_bigger, src1_bigger> {
        inline static bool f(const src0_type& v0, const dynd_complex<src1_real_type>& v1)
        {
            return v1.imag() != 0 || v0 != static_cast<src0_type>(v1.real()) ||
                            static_cast<src1_real_type>(v0) != v1.real();
        }
    };
    template<class src0_type, class src1_real_type,
                    bool src0_bigger, bool src1_bigger>
    struct op_ne<src0_type, dynd_complex<src1_real_type>, int_kind, complex_kind, src0_bigger, src1_bigger> {
        inline static bool f(const src0_type& v0, const dynd_complex<src1_real_type>& v1)
        {
            return v1.imag() != 0 || v0 == static_cast<src0_type>(v1.real()) ||
                            static_cast<src1_real_type>(v0) != v1.real();
        }
    };
    template<class src0_type, class src1_real_type,
                    bool src0_bigger, bool src1_bigger>
    struct op_ne<src0_type, dynd_complex<src1_real_type>, real_kind, complex_kind, src0_bigger, src1_bigger> {
        inline static bool f(const src0_type& v0, const dynd_complex<src1_real_type>& v1)
        {
            typedef typename big_type<src0_type,src1_real_type>::type BT;
            return v1.imag() != 0 || BT(v0) != BT(v1.real());
        }
    };

    // complex, int/uint/real inequality
    template<class src0_real_type, class src1_type,
                    bool src0_bigger, bool src1_bigger>
    struct op_ne<dynd_complex<src0_real_type>, src1_type, complex_kind, int_kind, src0_bigger, src1_bigger> {
        inline static bool f(const dynd_complex<src0_real_type>& v0, const src1_type& v1)
        {
            return v0.imag() != 0 || v0.real() != static_cast<src0_real_type>(v1) ||
                            static_cast<src1_type>(v0.real()) != v1;
        }
    };
    template<class src0_real_type, class src1_type,
                    bool src0_bigger, bool src1_bigger>
    struct op_ne<dynd_complex<src0_real_type>, src1_type, complex_kind, uint_kind, src0_bigger, src1_bigger> {
        inline static bool f(const dynd_complex<src0_real_type>& v0, const src1_type& v1)
        {
            return v0.imag() != 0 || v0.real() != static_cast<src0_real_type>(v1) ||
                            static_cast<src1_type>(v0.real()) != v1;
        }
    };
    template<class src0_real_type, class src1_type,
                    bool src0_bigger, bool src1_bigger>
    struct op_ne<dynd_complex<src0_real_type>, src1_type, complex_kind, real_kind, src0_bigger, src1_bigger> {
        inline static bool f(const dynd_complex<src0_real_type>& v0, const src1_type& v1)
        {
            typedef typename big_type<src0_real_type,src1_type>::type BT;
            return v0.imag() != 0 || BT(v0.real()) != BT(v1);
        }
    };

    // Bool isn't comparable (except sorting_less, equal, not_equal)
    NOT_ORDERABLE(bool_kind);
    NOT_ORDERABLE_PAIR(bool_kind, complex_kind);
    NOT_ORDERABLE_PAIR(complex_kind, bool_kind);

    // Bool sorting comparison
    template<class src0_type, class src1_type, type_kind_t src1_kind,
                    bool src0_bigger, bool src1_bigger>
    struct op_sort_lt<src0_type, src1_type, bool_kind, src1_kind, src0_bigger, src1_bigger> {
        inline static bool f(const src0_type& v0, const src1_type& v1)
        {
            return static_cast<src1_type>(v0) < v1;
        }
    };
    template<class src0_type, class src1_type, type_kind_t src0_kind,
                    bool src0_bigger, bool src1_bigger>
    struct op_sort_lt<src0_type, src1_type, src0_kind, bool_kind, src0_bigger, src1_bigger> {
        inline static bool f(const src0_type& v0, const src1_type& v1)
        {
            return v0 < static_cast<src0_type>(v1);
        }
    };
    template<class src0_type, class src1_type,
                    bool src0_bigger, bool src1_bigger>
    struct op_sort_lt<src0_type, src1_type, bool_kind, bool_kind, src0_bigger, src1_bigger> {
        inline static bool f(const src0_type& v0, const src1_type& v1)
        {
            return static_cast<int>(v0) < static_cast<int>(v1);
        }
    };

    // Bool equality comparison
    template<class src0_type, class src1_type, type_kind_t src1_kind,
                    bool src0_bigger, bool src1_bigger>
    struct op_eq<src0_type, src1_type, bool_kind, src1_kind, src0_bigger, src1_bigger> {
        inline static bool f(const src0_type& v0, const src1_type& v1)
        {
            return static_cast<src1_type>(v0) == v1;
        }
    };
    template<class src0_type, class src1_type, type_kind_t src0_kind,
                    bool src0_bigger, bool src1_bigger>
    struct op_eq<src0_type, src1_type, src0_kind, bool_kind, src0_bigger, src1_bigger> {
        inline static bool f(const src0_type& v0, const src1_type& v1)
        {
            return v0 == static_cast<src0_type>(v1);
        }
    };
    template<class src0_type, class src1_type,
                    bool src0_bigger, bool src1_bigger>
    struct op_eq<src0_type, src1_type, bool_kind, bool_kind, src0_bigger, src1_bigger> {
        inline static bool f(const src0_type& v0, const src1_type& v1)
        {
            return v0 == v1;
        }
    };

    // Bool inequality comparison
    template<class src0_type, class src1_type, type_kind_t src1_kind,
                    bool src0_bigger, bool src1_bigger>
    struct op_ne<src0_type, src1_type, bool_kind, src1_kind, src0_bigger, src1_bigger> {
        inline static bool f(const src0_type& v0, const src1_type& v1)
        {
            return static_cast<src1_type>(v0) != v1;
        }
    };
    template<class src0_type, class src1_type, type_kind_t src0_kind,
                    bool src0_bigger, bool src1_bigger>
    struct op_ne<src0_type, src1_type, src0_kind, bool_kind, src0_bigger, src1_bigger> {
        inline static bool f(const src0_type& v0, const src1_type& v1)
        {
            return v0 != static_cast<src0_type>(v1);
        }
    };
    template<class src0_type, class src1_type,
                    bool src0_bigger, bool src1_bigger>
    struct op_ne<src0_type, src1_type, bool_kind, bool_kind, src0_bigger, src1_bigger> {
        inline static bool f(const src0_type& v0, const src1_type& v1)
        {
            return v0 != v1;
        }
    };

    // Comparison operations
    template<class src0_type, class src1_type>
    struct single_comparison_builtin {
        inline static int sorting_less(const char *src0, const char *src1,
                        ckernel_prefix *DYND_UNUSED(extra))
        {
            src0_type v0 = *reinterpret_cast<const src0_type *>(src0);
            src1_type v1 = *reinterpret_cast<const src1_type *>(src1);
            return op_sort_lt<src0_type, src1_type,
                            dynd_kind_of<src0_type>::value,
                            dynd_kind_of<src1_type>::value,
                            (sizeof(src0_type) > sizeof(src1_type)),
                            (sizeof(src0_type) < sizeof(src1_type))>::f(v0, v1);
        }
        inline static int less(const char *src0, const char *src1,
                        ckernel_prefix *DYND_UNUSED(extra))
        {
            src0_type v0 = *reinterpret_cast<const src0_type *>(src0);
            src1_type v1 = *reinterpret_cast<const src1_type *>(src1);
            return op_lt<src0_type, src1_type,
                            dynd_kind_of<src0_type>::value,
                            dynd_kind_of<src1_type>::value,
                            (sizeof(src0_type) > sizeof(src1_type)),
                            (sizeof(src0_type) < sizeof(src1_type))>::f(v0, v1);
        }
        inline static int less_equal(const char *src0, const char *src1,
                        ckernel_prefix *DYND_UNUSED(extra))
        {
            src0_type v0 = *reinterpret_cast<const src0_type *>(src0);
            src1_type v1 = *reinterpret_cast<const src1_type *>(src1);
            return op_le<src0_type, src1_type,
                            dynd_kind_of<src0_type>::value,
                            dynd_kind_of<src1_type>::value,
                            (sizeof(src0_type) > sizeof(src1_type)),
                            (sizeof(src0_type) < sizeof(src1_type))>::f(v0, v1);
        }
        inline static int equal(const char *src0, const char *src1,
                        ckernel_prefix *DYND_UNUSED(extra))
        {
            src0_type v0 = *reinterpret_cast<const src0_type *>(src0);
            src1_type v1 = *reinterpret_cast<const src1_type *>(src1);
            return op_eq<src0_type, src1_type,
                            dynd_kind_of<src0_type>::value,
                            dynd_kind_of<src1_type>::value,
                            (sizeof(src0_type) > sizeof(src1_type)),
                            (sizeof(src0_type) < sizeof(src1_type))>::f(v0, v1);
        }
        inline static int not_equal(const char *src0, const char *src1,
                        ckernel_prefix *DYND_UNUSED(extra))
        {
            src0_type v0 = *reinterpret_cast<const src0_type *>(src0);
            src1_type v1 = *reinterpret_cast<const src1_type *>(src1);
            return op_ne<src0_type, src1_type,
                            dynd_kind_of<src0_type>::value,
                            dynd_kind_of<src1_type>::value,
                            (sizeof(src0_type) > sizeof(src1_type)),
                            (sizeof(src0_type) < sizeof(src1_type))>::f(v0, v1);
        }
        inline static int greater_equal(const char *src0, const char *src1,
                        ckernel_prefix *DYND_UNUSED(extra))
        {
            src0_type v0 = *reinterpret_cast<const src0_type *>(src0);
            src1_type v1 = *reinterpret_cast<const src1_type *>(src1);
            return op_ge<src0_type, src1_type,
                            dynd_kind_of<src0_type>::value,
                            dynd_kind_of<src1_type>::value,
                            (sizeof(src0_type) > sizeof(src1_type)),
                            (sizeof(src0_type) < sizeof(src1_type))>::f(v0, v1);
        }
        inline static int greater(const char *src0, const char *src1,
                        ckernel_prefix *DYND_UNUSED(extra))
        {
            src0_type v0 = *reinterpret_cast<const src0_type *>(src0);
            src1_type v1 = *reinterpret_cast<const src1_type *>(src1);
            return op_gt<src0_type, src1_type,
                            dynd_kind_of<src0_type>::value,
                            dynd_kind_of<src1_type>::value,
                            (sizeof(src0_type) > sizeof(src1_type)),
                            (sizeof(src0_type) < sizeof(src1_type))>::f(v0, v1);
        }
    };

} // namespace dynd
