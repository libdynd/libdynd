//
// Copyright (C) 2011-16 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include "inc_gtest.hpp"

#include <utility>

#include <dynd/array.hpp>
#include <dynd/type.hpp>
#include <dynd/types/cuda_host_type.hpp>
#include <dynd/types/cuda_device_type.hpp>
#include <dynd/types/type_id.hpp>
#include <dynd/types/callable_type.hpp>

using namespace std;
using namespace dynd;

typedef void None;

namespace testing {

// This is needed to pretty print ::testing::Types
template <typename T0, typename T1>
struct Types2 : ::testing::Types<T0, T1> {
};

}

/**
 * This class describes a memory space for testing purposes.
 *
 * The idea is that we implement a template specialization for each memory type
 * to test. The specialization then handles details like the transfer of data back
 * and forth.
 *
 * Since default memory has no memory type, we use the C++ type 'None' from Google Test
 * to represent it.
 */
template <typename T>
class Memory;

/**
 * This class is a pair of memory spaces, which is needed by some tests.
 */
template <typename T>
class MemoryPair;

template <typename T0, typename T1>
class MemoryPair< ::testing::Types2<T0, T1> > : public ::testing::Test {
public:
    typedef Memory<T0> First;
    typedef Memory<T1> Second;
};

template <>
class Memory<None> : public ::testing::Test {
public:
    typedef None Type;

    static inline bool IsTypeID(type_id_t DYND_UNUSED(type_id)) {
        return false;
    }

    static inline ndt::type MakeType(const ndt::type& target_tp) {
        return target_tp;
    }

    template <typename T>
    static inline T Dereference(const T* ptr) {
        return *ptr;
    }

    static inline nd::array To(const nd::array& a) {
#ifdef DYND_CUDA
        return a.to_host();
#else
        return a;
#endif // DYND_CUDA
    }
};

typedef ::testing::Types<None> DefaultMemory;
typedef ::testing::Types< ::testing::Types2<None, None> > DefaultMemoryPairs;

#ifdef DYND_CUDA

template <>
class Memory<cuda_host_type> : public ::testing::Test {
    static const type_id_t TypeID = cuda_host_id;
public:
    typedef cuda_host_type Type;

    static inline bool IsTypeID(type_id_t type_id) {
        return TypeID == type_id;
    }

    static inline ndt::type MakeType(const ndt::type& target_tp, unsigned int cuda_host_flags = cudaHostAllocDefault) {
        return make_cuda_host(target_tp, cuda_host_flags);
    }

    template <typename T>
    static inline T Dereference(const T* ptr) {
        return *ptr;
    }

    static inline nd::array To(const nd::array& a, unsigned int cuda_host_flags = cudaHostAllocDefault) {
        return a.to_cuda_host(cuda_host_flags);
    }
};

template <>
class Memory<cuda_device_type> : public ::testing::Test {
    static const type_id_t TypeID = cuda_device_id;
public:
    typedef cuda_device_type Type;

    static inline bool IsTypeID(type_id_t type_id) {
        return TypeID == type_id;
    }

    static inline ndt::type MakeType(const ndt::type& target_tp) {
        return make_cuda_device(target_tp);
    }

    template <typename T>
    static inline T Dereference(const T* ptr) {
        T tmp;
        cuda_throw_if_not_success(cudaMemcpy(&tmp, ptr, sizeof(T), cudaMemcpyDeviceToHost));
        return tmp;
    }

    static inline nd::array To(const nd::array& a) {
        return a.to_cuda_device();
    }
};

typedef ::testing::Types<cuda_host_type, cuda_device_type> CUDAMemory;
typedef ::testing::Types< ::testing::Types2<None, cuda_host_type>, ::testing::Types2<None, cuda_device_type>,
    ::testing::Types2<cuda_host_type, None>, ::testing::Types2<cuda_device_type, None>,
    ::testing::Types2<cuda_host_type, cuda_host_type>, ::testing::Types2<cuda_host_type, cuda_device_type>,
    ::testing::Types2<cuda_device_type, cuda_host_type>, ::testing::Types2<cuda_device_type, cuda_device_type> > CUDAMemoryPairs;

#endif // DYND_CUDA
