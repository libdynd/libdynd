//
// Copyright (C) 2011-12, Dynamic NDArray Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <cmath>

#include <dnd/config.hpp>

#ifdef _WIN32
# define EXPORT extern "C" __declspec(dllexport)
#else
# define EXPORT extern "C"
#endif

////////////////////////////
// "add" kernels

EXPORT int32_t add_int32(int32_t x, int32_t y) {
    return x + y;
}

EXPORT int64_t add_int64(int64_t x, int64_t y) {
    return x + y;
}

EXPORT uint32_t add_uint32(uint32_t x, uint32_t y) {
    return x + y;
}

EXPORT uint64_t add_uint64(uint64_t x, uint64_t y) {
    return x + y;
}

EXPORT float add_float32(float x, float y) {
    return x + y;
}

EXPORT double add_float64(double x, double y) {
    return x + y;
}

////////////////////////////
// "subtract" kernels

EXPORT int32_t subtract_int32(int32_t x, int32_t y) {
    return x - y;
}

EXPORT int64_t subtract_int64(int64_t x, int64_t y) {
    return x - y;
}

EXPORT uint32_t subtract_uint32(uint32_t x, uint32_t y) {
    return x - y;
}

EXPORT uint64_t subtract_uint64(uint64_t x, uint64_t y) {
    return x - y;
}

EXPORT float subtract_float32(float x, float y) {
    return x - y;
}

EXPORT double subtract_float64(double x, double y) {
    return x - y;
}

////////////////////////////
// "multiply" kernels

EXPORT int32_t multiply_int32(int32_t x, int32_t y) {
    return x * y;
}

EXPORT int64_t multiply_int64(int64_t x, int64_t y) {
    return x * y;
}

EXPORT uint32_t multiply_uint32(uint32_t x, uint32_t y) {
    return x * y;
}

EXPORT uint64_t multiply_uint64(uint64_t x, uint64_t y) {
    return x * y;
}

EXPORT float multiply_float32(float x, float y) {
    return x * y;
}

EXPORT double multiply_float64(double x, double y) {
    return x * y;
}

////////////////////////////
// "divide" kernels

EXPORT int32_t divide_int32(int32_t x, int32_t y) {
    return x / y;
}

EXPORT int64_t divide_int64(int64_t x, int64_t y) {
    return x / y;
}

EXPORT uint32_t divide_uint32(uint32_t x, uint32_t y) {
    return x / y;
}

EXPORT uint64_t divide_uint64(uint64_t x, uint64_t y) {
    return x / y;
}

EXPORT float divide_float32(float x, float y) {
    return x / y;
}

EXPORT double divide_float64(double x, double y) {
    return x / y;
}

////////////////////////////
// "maximum" kernels

EXPORT int32_t maximum2_int32(int32_t x, int32_t y) {
    return (x > y) ? x : y;
}

EXPORT int64_t maximum2_int64(int64_t x, int64_t y) {
    return (x > y) ? x : y;
}

EXPORT uint32_t maximum2_uint32(uint32_t x, uint32_t y) {
    return (x > y) ? x : y;
}

EXPORT uint64_t maximum2_uint64(uint64_t x, uint64_t y) {
    return (x > y) ? x : y;
}

EXPORT float maximum2_float32(float x, float y) {
    return (x > y) ? x : y;
}

EXPORT double maximum2_float64(double x, double y) {
    return (x > y) ? x : y;
}

EXPORT int32_t maximum3_int32(int32_t x, int32_t y, int32_t z) {
    return (x > y) ? ((x > z) ? x : z) : ((y > z) ? y : z);
}

EXPORT int64_t maximum3_int64(int64_t x, int64_t y, int64_t z) {
    return (x > y) ? ((x > z) ? x : z) : ((y > z) ? y : z);
}

EXPORT uint32_t maximum3_uint32(uint32_t x, uint32_t y, uint32_t z) {
    return (x > y) ? ((x > z) ? x : z) : ((y > z) ? y : z);
}

EXPORT uint64_t maximum3_uint64(uint64_t x, uint64_t y, uint64_t z) {
    return (x > y) ? ((x > z) ? x : z) : ((y > z) ? y : z);
}

EXPORT float maximum3_float32(float x, float y, float z) {
    return (x > y) ? ((x > z) ? x : z) : ((y > z) ? y : z);
}

EXPORT double maximum3_float64(double x, double y, double z) {
    return (x > y) ? ((x > z) ? x : z) : ((y > z) ? y : z);
}

////////////////////////////
// "minimum" kernels

EXPORT int32_t minimum2_int32(int32_t x, int32_t y) {
    return (x < y) ? x : y;
}

EXPORT int64_t minimum2_int64(int64_t x, int64_t y) {
    return (x < y) ? x : y;
}

EXPORT uint32_t minimum2_uint32(uint32_t x, uint32_t y) {
    return (x < y) ? x : y;
}

EXPORT uint64_t minimum2_uint64(uint64_t x, uint64_t y) {
    return (x < y) ? x : y;
}

EXPORT float minimum2_float32(float x, float y) {
    return (x < y) ? x : y;
}

EXPORT double minimum2_float64(double x, double y) {
    return (x < y) ? x : y;
}

EXPORT int32_t minimum3_int32(int32_t x, int32_t y, int32_t z) {
    return (x < y) ? ((x < z) ? x : z) : ((y < z) ? y : z);
}

EXPORT int64_t minimum3_int64(int64_t x, int64_t y, int64_t z) {
    return (x < y) ? ((x < z) ? x : z) : ((y < z) ? y : z);
}

EXPORT uint32_t minimum3_uint32(uint32_t x, uint32_t y, uint32_t z) {
    return (x < y) ? ((x < z) ? x : z) : ((y < z) ? y : z);
}

EXPORT uint64_t minimum3_uint64(uint64_t x, uint64_t y, uint64_t z) {
    return (x < y) ? ((x < z) ? x : z) : ((y < z) ? y : z);
}

EXPORT float minimum3_float32(float x, float y, float z) {
    return (x < y) ? ((x < z) ? x : z) : ((y < z) ? y : z);
}

EXPORT double minimum3_float64(double x, double y, double z) {
    return (x < y) ? ((x < z) ? x : z) : ((y < z) ? y : z);
}

////////////////////////////
// "square" kernels

EXPORT int32_t square_int32(int32_t x) {
    return x * x;
}

EXPORT int64_t square_int64(int64_t x) {
    return x * x;
}

EXPORT uint32_t square_uint32(uint32_t x) {
    return x * x;
}

EXPORT uint64_t square_uint64(uint64_t x) {
    return x * x;
}

EXPORT float square_float32(float x) {
    return x * x;
}

EXPORT double square_float64(double x) {
    return x * x;
}

////////////////////////////
// "abs" kernels

EXPORT int32_t abs_int32(int32_t x) {
    return (x >= 0) ? x : -x;
}

EXPORT int64_t abs_int64(int64_t x) {
    return (x >= 0) ? x : -x;
}

EXPORT uint32_t abs_uint32(uint32_t x) {
    return x;
}

EXPORT uint64_t abs_uint64(uint64_t x) {
    return x;
}

EXPORT float abs_float32(float x) {
    return (x >= 0) ? x : -x;
}

EXPORT double abs_float64(double x) {
    return (x >= 0) ? x : -x;
}

////////////////////////////
// "floor" kernels

EXPORT int32_t floor_int32(int32_t x) {
    return x;
}

EXPORT int64_t floor_int64(int64_t x) {
    return x;
}

EXPORT uint32_t floor_uint32(uint32_t x) {
    return x;
}

EXPORT uint64_t floor_uint64(uint64_t x) {
    return x;
}

EXPORT float floor_float32(float x) {
    return floorf(x);
}

EXPORT double floor_float64(double x) {
    return floor(x);
}

////////////////////////////
// "ceil" kernels

EXPORT int32_t ceil_int32(int32_t x) {
    return x;
}

EXPORT int64_t ceil_int64(int64_t x) {
    return x;
}

EXPORT uint32_t ceil_uint32(uint32_t x) {
    return x;
}

EXPORT uint64_t ceil_uint64(uint64_t x) {
    return x;
}

EXPORT float ceil_float32(float x) {
    return ceilf(x);
}

EXPORT double ceil_float64(double x) {
    return ceil(x);
}

