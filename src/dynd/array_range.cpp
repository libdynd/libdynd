//
// Copyright (C) 2011-16 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <sstream>
#include <stdexcept>

#include <dynd/array.hpp>
#include <dynd/array_range.hpp>
#include <dynd/type_promotion.hpp>

using namespace std;
using namespace dynd;

namespace {
template <class T>
struct range_specialization {
  static void old_range(const void *beginval, const void *stepval, nd::array &result) {
    T begin = *reinterpret_cast<const T *>(beginval);
    T step = *reinterpret_cast<const T *>(stepval);
    intptr_t count = result.get_shape()[0], stride = result.get_strides()[0];
    // cout << "range with count " << count << endl;
    char *dst = result.data();
    for (intptr_t i = 0; i < count; ++i, dst += stride) {
      *reinterpret_cast<T *>(dst) = static_cast<T>(begin + i * step);
    }
  }
};

template <>
struct range_specialization<int128> {
  static void old_range(const void *beginval, const void *stepval, nd::array &result) {
    int128 begin = *reinterpret_cast<const int128 *>(beginval);
    int128 step = *reinterpret_cast<const int128 *>(stepval);
    intptr_t count = result.get_shape()[0], stride = result.get_strides()[0];
    // cout << "range with count " << count << endl;
    char *dst = result.data();
    for (intptr_t i = 0; i < count; ++i, dst += stride) {
      *reinterpret_cast<int128 *>(dst) = begin;
      begin = begin + step;
    }
  }
};

template <class T, type_id_t BaseID>
struct range_counter {
  static intptr_t count(const void *beginval, const void *endval, const void *stepval) {
    T begin = *reinterpret_cast<const T *>(beginval);
    T end = *reinterpret_cast<const T *>(endval);
    T step = *reinterpret_cast<const T *>(stepval);
    // cout << "range params " << begin << "  " << end << "  " << step << "\n";
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

template <class T>
struct range_counter<T, uint_kind_id> {
  static intptr_t count(const void *beginval, const void *endval, const void *stepval) {
    T begin = *reinterpret_cast<const T *>(beginval);
    T end = *reinterpret_cast<const T *>(endval);
    T step = *reinterpret_cast<const T *>(stepval);
    // cout << "range params " << begin << "  " << end << "  " << step << "\n";
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

template <class T>
struct range_counter<T, float_kind_id> {
  static intptr_t count(const void *beginval, const void *endval, const void *stepval) {
    T begin = *reinterpret_cast<const T *>(beginval);
    T end = *reinterpret_cast<const T *>(endval);
    T step = *reinterpret_cast<const T *>(stepval);
    // cout << "nd::range params " << begin << "  " << end << "  " << step <<
    // "\n";
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

nd::array dynd::nd::old_range(const ndt::type &scalar_tp, const void *beginval, const void *endval,
                              const void *stepval) {
#define ONE_ARANGE_SPECIALIZATION(type)                                                                                \
  case ndt::id_of<type>::value: {                                                                                      \
    intptr_t dim_size =                                                                                                \
        range_counter<type, base_id_of<ndt::id_of<type>::value>::value>::count(beginval, endval, stepval);             \
    nd::array result = nd::empty(dim_size, scalar_tp);                                                                 \
    range_specialization<type>::old_range(beginval, stepval, result);                                                  \
    return result;                                                                                                     \
  }

  switch (scalar_tp.get_id()) {
    ONE_ARANGE_SPECIALIZATION(int8_t);
    ONE_ARANGE_SPECIALIZATION(int16_t);
    ONE_ARANGE_SPECIALIZATION(int32_t);
    ONE_ARANGE_SPECIALIZATION(int64_t);
    ONE_ARANGE_SPECIALIZATION(int128);
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
  throw type_error(ss.str());
}

static void old_linspace_specialization(float start, float stop, intptr_t count, nd::array &result) {
  intptr_t stride = result.get_strides()[0];
  char *dst = result.data();
  for (intptr_t i = 0; i < count; ++i, dst += stride) {
    double val = ((count - i - 1) * double(start) + i * double(stop)) / double(count - 1);
    *reinterpret_cast<float *>(dst) = static_cast<float>(val);
  }
}

static void old_linspace_specialization(double start, double stop, intptr_t count, nd::array &result) {
  intptr_t stride = result.get_strides()[0];
  char *dst = result.data();
  for (intptr_t i = 0; i < count; ++i, dst += stride) {
    double val = ((count - i - 1) * start + i * stop) / double(count - 1);
    *reinterpret_cast<double *>(dst) = val;
  }
}

static void old_linspace_specialization(dynd::complex<float> start, dynd::complex<float> stop, intptr_t count,
                                        nd::array &result) {
  intptr_t stride = result.get_strides()[0];
  char *dst = result.data();
  for (intptr_t i = 0; i < count; ++i, dst += stride) {
    dynd::complex<double> val =
        (double(count - i - 1) * dynd::complex<double>(start) + double(i) * dynd::complex<double>(stop)) /
        double(count - 1);
    *reinterpret_cast<dynd::complex<float> *>(dst) = dynd::complex<float>(val);
  }
}

static void old_linspace_specialization(dynd::complex<double> start, dynd::complex<double> stop, intptr_t count,
                                        nd::array &result) {
  intptr_t stride = result.get_strides()[0];
  char *dst = result.data();
  for (intptr_t i = 0; i < count; ++i, dst += stride) {
    dynd::complex<double> val =
        (double(count - i - 1) * dynd::complex<double>(start) + double(i) * dynd::complex<double>(stop)) /
        double(count - 1);
    *reinterpret_cast<dynd::complex<double> *>(dst) = val;
  }
}

nd::array dynd::nd::old_linspace(const nd::array &start, const nd::array &stop, intptr_t count, const ndt::type &dt) {
  nd::array start_cleaned = nd::empty(dt).assign(start);
  nd::array stop_cleaned = nd::empty(dt).assign(stop);

  if (start_cleaned.is_scalar() && stop_cleaned.is_scalar()) {
    return old_linspace(dt, start_cleaned.cdata(), stop_cleaned.cdata(), count);
  } else {
    throw runtime_error("dynd::linspace presently only supports scalar parameters");
  }
}

nd::array dynd::nd::old_linspace(const nd::array &start, const nd::array &stop, intptr_t count) {
  ndt::type dt = promote_types_arithmetic(start.get_dtype(), stop.get_dtype());
  // Make sure it's at least floating point
  if (dt.get_base_id() == bool_kind_id || dt.get_base_id() == int_kind_id || dt.get_base_id() == uint_kind_id) {
    dt = ndt::make_type<double>();
  }
  return old_linspace(start, stop, count, dt);
}

nd::array dynd::nd::old_linspace(const ndt::type &dt, const void *startval, const void *stopval, intptr_t count) {
  if (count < 2) {
    throw runtime_error("linspace needs a count of at least 2");
  }

  switch (dt.get_id()) {
  case float32_id: {
    nd::array result = nd::empty(count, dt);
    old_linspace_specialization(*reinterpret_cast<const float *>(startval), *reinterpret_cast<const float *>(stopval),
                                count, result);
    return result;
  }
  case float64_id: {
    nd::array result = nd::empty(count, dt);
    old_linspace_specialization(*reinterpret_cast<const double *>(startval), *reinterpret_cast<const double *>(stopval),
                                count, result);
    return result;
  }
  case complex_float32_id: {
    nd::array result = nd::empty(count, dt);
    old_linspace_specialization(*reinterpret_cast<const dynd::complex<float> *>(startval),
                                *reinterpret_cast<const dynd::complex<float> *>(stopval), count, result);
    return result;
  }
  case complex_float64_id: {
    nd::array result = nd::empty(count, dt);
    old_linspace_specialization(*reinterpret_cast<const dynd::complex<double> *>(startval),
                                *reinterpret_cast<const dynd::complex<double> *>(stopval), count, result);
    return result;
  }
  default:
    break;
  }

  stringstream ss;
  ss << "dynd linspace doesn't support type " << dt;
  throw runtime_error(ss.str());
}
