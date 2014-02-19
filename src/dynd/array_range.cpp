//
// Copyright (C) 2011-14 Mark Wiebe, DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <stdexcept>
#include <sstream>

#include <dynd/array.hpp>
#include <dynd/array_range.hpp>
#include <dynd/type_promotion.hpp>

using namespace std;
using namespace dynd;

namespace {
    template<class T>
    struct range_specialization {
        static void range(const void *beginval, const void *stepval, nd::array& result) {
            T begin = *reinterpret_cast<const T *>(beginval);
            T step = *reinterpret_cast<const T *>(stepval);
            intptr_t count = result.get_shape()[0], stride = result.get_strides()[0];
            //cout << "range with count " << count << endl;
            char *dst = result.get_readwrite_originptr();
            for (intptr_t i = 0; i < count; ++i, dst += stride) {
                *reinterpret_cast<T *>(dst) = static_cast<T>(begin + i * step);
            }
        }
    };

    template<class T, type_kind_t kind>
    struct range_counter {
        static intptr_t count(const void *beginval, const void *endval, const void *stepval) {
            T begin = *reinterpret_cast<const T *>(beginval);
            T end = *reinterpret_cast<const T *>(endval);
            T step = *reinterpret_cast<const T *>(stepval);
            //cout << "range params " << begin << "  " << end << "  " << step << "\n";
            if (step > 0) {
                if (end <= begin) {
                    return 0;
                }
                return ((intptr_t)end - (intptr_t)begin + (intptr_t)step - 1) / (intptr_t)step;
            } else if (step < 0) {
                if (end >= begin) {
                    return 0;
                }
                step = -step;
                return ((intptr_t)begin - (intptr_t)end + (intptr_t)step - 1) / (intptr_t)step;
            } else {
                throw std::runtime_error("nd::range cannot have a zero-sized step");
            }
        }
    };

    template<class T>
    struct range_counter<T, uint_kind> {
        static intptr_t count(const void *beginval, const void *endval, const void *stepval) {
            T begin = *reinterpret_cast<const T *>(beginval);
            T end = *reinterpret_cast<const T *>(endval);
            T step = *reinterpret_cast<const T *>(stepval);
            //cout << "range params " << begin << "  " << end << "  " << step << "\n";
            if (step > 0) {
                if (end <= begin) {
                    return 0;
                }
                return ((intptr_t)end - (intptr_t)begin + (intptr_t)step - 1) / (intptr_t)step;
            } else {
                throw std::runtime_error("nd::range cannot have a zero-sized step");
            }
        }
    };

    template<class T>
    struct range_counter<T, real_kind> {
        static intptr_t count(const void *beginval, const void *endval, const void *stepval) {
            T begin = *reinterpret_cast<const T *>(beginval);
            T end = *reinterpret_cast<const T *>(endval);
            T step = *reinterpret_cast<const T *>(stepval);
            //cout << "nd::range params " << begin << "  " << end << "  " << step << "\n";
            if (step > 0) {
                if (end <= begin) {
                    return 0;
                }
                // Add half a step to make the count robust. This requires
                // that the range given is approximately a multiple of step
                return (intptr_t)floor((end - begin + 0.5 * step) / step);
            } else if (step < 0) {
                if (end >= begin) {
                    return 0;
                }
                // Add half a step to make the count robust. This requires
                // that the range given is approximately a multiple of step
                return (intptr_t)floor((end - begin + 0.5 * step) / step);
            } else {
                throw std::runtime_error("nd::range cannot have a zero-sized step");
            }
        }
    };
} // anonymous namespace

nd::array dynd::nd::range(const ndt::type& scalar_tp, const void *beginval, const void *endval, const void *stepval)
{
#define ONE_ARANGE_SPECIALIZATION(type) \
    case type_id_of<type>::value: { \
        intptr_t dim_size = range_counter<type, dynd_kind_of<type>::value>::count(beginval, endval, stepval); \
        nd::array result = \
                nd::make_strided_array(dim_size, scalar_tp); \
        range_specialization<type>::range(beginval, stepval, result); \
        return DYND_MOVE(result); \
    }

    switch (scalar_tp.get_type_id()) {
        ONE_ARANGE_SPECIALIZATION(int8_t);
        ONE_ARANGE_SPECIALIZATION(int16_t);
        ONE_ARANGE_SPECIALIZATION(int32_t);
        ONE_ARANGE_SPECIALIZATION(int64_t);
        ONE_ARANGE_SPECIALIZATION(uint8_t);
        ONE_ARANGE_SPECIALIZATION(uint16_t);
        ONE_ARANGE_SPECIALIZATION(uint32_t);
        ONE_ARANGE_SPECIALIZATION(uint64_t);
        ONE_ARANGE_SPECIALIZATION(float);
        ONE_ARANGE_SPECIALIZATION(double);
        default:
            break;
    }

#undef ONE_ARANGE_SPECIALIZATION

    stringstream ss;
    ss << "dynd nd::range doesn't support type " << scalar_tp;
    throw runtime_error(ss.str());
}

static void linspace_specialization(float start, float stop, intptr_t count, nd::array& result)
{
    intptr_t stride = result.get_strides()[0];
    char *dst = result.get_readwrite_originptr();
    for (intptr_t i = 0; i < count; ++i, dst += stride) {
        double val = ((count - i - 1) * double(start) + i * double(stop)) / double(count - 1);
        *reinterpret_cast<float *>(dst) = static_cast<float>(val);
    }
}

static void linspace_specialization(double start, double stop, intptr_t count, nd::array& result)
{
    intptr_t stride = result.get_strides()[0];
    char *dst = result.get_readwrite_originptr();
    for (intptr_t i = 0; i < count; ++i, dst += stride) {
        double val = ((count - i - 1) * start + i * stop) / double(count - 1);
        *reinterpret_cast<double *>(dst) = val;
    }
}

static void linspace_specialization(dynd_complex<float> start, dynd_complex<float> stop, intptr_t count, nd::array& result)
{
    intptr_t stride = result.get_strides()[0];
    char *dst = result.get_readwrite_originptr();
    for (intptr_t i = 0; i < count; ++i, dst += stride) {
        dynd_complex<double> val = (double(count - i - 1) * dynd_complex<double>(start) +
                        double(i) * dynd_complex<double>(stop)) / double(count - 1);
        *reinterpret_cast<dynd_complex<float> *>(dst) = dynd_complex<float>(val);
    }
}

static void linspace_specialization(dynd_complex<double> start, dynd_complex<double> stop, intptr_t count, nd::array& result)
{
    intptr_t stride = result.get_strides()[0];
    char *dst = result.get_readwrite_originptr();
    for (intptr_t i = 0; i < count; ++i, dst += stride) {
        dynd_complex<double> val = (double(count - i - 1) * dynd_complex<double>(start) +
                        double(i) * dynd_complex<double>(stop)) / double(count - 1);
        *reinterpret_cast<dynd_complex<double> *>(dst) = val;
    }
}

nd::array dynd::nd::linspace(const nd::array& start, const nd::array& stop, intptr_t count, const ndt::type& dt)
{
    nd::array start_cleaned = start.ucast(dt).eval();
    nd::array stop_cleaned = stop.ucast(dt).eval();

    if (start_cleaned.is_scalar() && stop_cleaned.is_scalar()) {
        return linspace(dt, start_cleaned.get_readonly_originptr(), stop_cleaned.get_readonly_originptr(), count);
    } else {
        throw runtime_error("dynd::linspace presently only supports scalar parameters");
    }
}

nd::array dynd::nd::linspace(const nd::array& start, const nd::array& stop, intptr_t count)
{
    ndt::type dt = promote_types_arithmetic(start.get_dtype(), stop.get_dtype());
    // Make sure it's at least floating point
    if (dt.get_kind() == bool_kind || dt.get_kind() == int_kind || dt.get_kind() == uint_kind) {
        dt = ndt::make_type<double>();
    }
    return linspace(start, stop, count, dt);
}

nd::array dynd::nd::linspace(const ndt::type& dt, const void *startval, const void *stopval, intptr_t count)
{
    if (count < 2) {
        throw runtime_error("linspace needs a count of at least 2");
    }

    switch (dt.get_type_id()) {
        case float32_type_id: {
            nd::array result = nd::make_strided_array(count, dt);
            linspace_specialization(*reinterpret_cast<const float *>(startval),
                            *reinterpret_cast<const float *>(stopval), count, result);
            return DYND_MOVE(result);
        }
        case float64_type_id: {
            nd::array result = nd::make_strided_array(count, dt);
            linspace_specialization(*reinterpret_cast<const double *>(startval),
                            *reinterpret_cast<const double *>(stopval), count, result);
            return DYND_MOVE(result);
        }
        case complex_float32_type_id: {
            nd::array result = nd::make_strided_array(count, dt);
            linspace_specialization(*reinterpret_cast<const dynd_complex<float> *>(startval),
                            *reinterpret_cast<const dynd_complex<float> *>(stopval), count, result);
            return DYND_MOVE(result);
        }
        case complex_float64_type_id: {
            nd::array result = nd::make_strided_array(count, dt);
            linspace_specialization(*reinterpret_cast<const dynd_complex<double> *>(startval),
                            *reinterpret_cast<const dynd_complex<double> *>(stopval), count, result);
            return DYND_MOVE(result);
        }
        default:
            break;
    }

    stringstream ss;
    ss << "dynd linspace doesn't support type " << dt;
    throw runtime_error(ss.str());
}
