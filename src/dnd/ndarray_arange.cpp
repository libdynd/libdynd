//
// Copyright (C) 2011 Mark Wiebe (mwwiebe@gmail.com)
// All rights reserved.
//
// This is unreleased proprietary software.
//

#include <stdexcept>
#include <sstream>

#include <dnd/ndarray.hpp>
#include <dnd/ndarray_arange.hpp>

using namespace std;
using namespace dnd;

namespace {
    template<class T>
    struct arange_specialization {
        static void arange(const void *beginval, const void *stepval, ndarray& result) {
            T begin = *reinterpret_cast<const T *>(beginval);
            T step = *reinterpret_cast<const T *>(stepval);
            intptr_t count = result.get_shape(0), stride = result.get_strides(0);
            //cout << "arange with count " << count << endl;
            char *dst = result.get_originptr();
            for (intptr_t i = 0; i < count; ++i, dst += stride) {
                *reinterpret_cast<T *>(dst) = begin + i * step;
            }
        }
    };

    template<class T, dtype_kind kind>
    struct arange_counter {
        static intptr_t count(const void *beginval, const void *endval, const void *stepval) {
            T begin = *reinterpret_cast<const T *>(beginval);
            T end = *reinterpret_cast<const T *>(endval);
            T step = *reinterpret_cast<const T *>(stepval);
            //cout << "arange params " << begin << "  " << end << "  " << step << "\n";
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
                throw std::runtime_error("arange cannot have a zero-sized step");
            }
        }
    };

    template<class T>
    struct arange_counter<T, uint_kind> {
        static intptr_t count(const void *beginval, const void *endval, const void *stepval) {
            T begin = *reinterpret_cast<const T *>(beginval);
            T end = *reinterpret_cast<const T *>(endval);
            T step = *reinterpret_cast<const T *>(stepval);
            //cout << "arange params " << begin << "  " << end << "  " << step << "\n";
            if (step > 0) {
                if (end <= begin) {
                    return 0;
                }
                return ((intptr_t)end - (intptr_t)begin + (intptr_t)step - 1) / (intptr_t)step;
            } else {
                throw std::runtime_error("arange cannot have a zero-sized step");
            }
        }
    };

    template<class T>
    struct arange_counter<T, float_kind> {
        static intptr_t count(const void *beginval, const void *endval, const void *stepval) {
            T begin = *reinterpret_cast<const T *>(beginval);
            T end = *reinterpret_cast<const T *>(endval);
            T step = *reinterpret_cast<const T *>(stepval);
            //cout << "arange params " << begin << "  " << end << "  " << step << "\n";
            if (step > 0) {
                if (end <= begin) {
                    return 0;
                }
                intptr_t count = (intptr_t)floor((end - begin) / step);
                while (begin + count * step < end) {
                    ++count;
                }
                return count;
            } else if (step < 0) {
                if (end >= begin) {
                    return 0;
                }
                intptr_t count = (intptr_t)floor((end - begin) / step);
                while (begin + count * step > end) {
                    ++count;
                }
                return count;
            } else {
                throw std::runtime_error("arange cannot have a zero-sized step");
            }
        }
    };
} // anonymous namespace

ndarray dnd::arange(const dtype& dt, const void *beginval, const void *endval, const void *stepval)
{
    if (dt.extended() == NULL && !dt.is_byteswapped()) {

#define ONE_ARANGE_SPECIALIZATION(type) \
        case type_id_of<type>::value: { \
            ndarray result(arange_counter<type, dtype_kind_of<type>::value>::count(beginval, endval, stepval), dt); \
            arange_specialization<type>::arange(beginval, stepval, result); \
            return std::move(result); \
        }

        switch (dt.type_id()) {
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
        ss << "arange doesn't support built-in dtype " << dt;
        throw runtime_error(ss.str());
    } else {
        stringstream ss;
        ss << "arange doesn't support extended dtype " << dt;
        throw runtime_error(ss.str());
    }
}

static void linspace_specialization(float start, float stop, intptr_t count, ndarray& result)
{
    intptr_t stride = result.get_strides(0);
    char *dst = result.get_originptr();
    for (intptr_t i = 0; i < count; ++i, dst += stride) {
        float alpha = float(double(i) / double(count - 1));
        *reinterpret_cast<float *>(dst) = (1 - alpha) * start + alpha * stop;
    }
}

static void linspace_specialization(double start, double stop, intptr_t count, ndarray& result)
{
    intptr_t stride = result.get_strides(0);
    char *dst = result.get_originptr();
    for (intptr_t i = 0; i < count; ++i, dst += stride) {
        double alpha = double(i) / double(count - 1);
        *reinterpret_cast<double *>(dst) = (1 - alpha) * start + alpha * stop;
    }
}

ndarray dnd::linspace(const dtype& dt, const void *startval, const void *stopval, intptr_t count)
{
    if (count < 2) {
        throw runtime_error("linspace needs a count of at least 2");
    }

    if (dt.extended() == NULL && !dt.is_byteswapped()) {
        switch (dt.type_id()) {
            case float32_type_id: {
                ndarray result(count, dt);
                linspace_specialization(*reinterpret_cast<const float *>(startval),
                                *reinterpret_cast<const float *>(stopval), count, result);
                return std::move(result);
            }
            case float64_type_id: {
                ndarray result(count, dt);
                linspace_specialization(*reinterpret_cast<const double *>(startval),
                                *reinterpret_cast<const double *>(stopval), count, result);
                return std::move(result);
            }
            default:
                break;
        }

        stringstream ss;
        ss << "linspace doesn't support built-in dtype " << dt;
        throw runtime_error(ss.str());
    } else {
        stringstream ss;
        ss << "linspace doesn't support extended dtype " << dt;
        throw runtime_error(ss.str());
    }
}
