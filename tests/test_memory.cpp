#include "inc_gtest.hpp"

#include <dynd/type.hpp>
#include <dynd/types/cuda_host_type.hpp>
#include <dynd/types/cuda_device_type.hpp>
#include <dynd/types/type_id.hpp>

using namespace std;
using namespace dynd;

template <typename T>
class Memory;

template <>
class Memory<void> : public ::testing::Test {
public:
    static const bool IsDefaultMemoryType = true;

    static inline ndt::type MakeMemoryType(const ndt::type& target_tp) {
        return target_tp;
    }
};

typedef ::testing::Types<void> DefaultMemory;

#ifdef DYND_CUDA

template <>
class Memory<cuda_host_type> : public ::testing::Test {
public:
    typedef cuda_host_type MemoryType;

    static const bool IsDefaultMemoryType = false;
    static const type_id_t MemoryTypeID = cuda_host_type_id;

    static inline ndt::type MakeMemoryType(const ndt::type& target_tp, unsigned int cuda_host_flags = cudaHostAllocDefault) {
        return make_cuda_host(target_tp, cuda_host_flags);
    }
};

template <>
class Memory<cuda_device_type> : public ::testing::Test {
public:
    typedef cuda_device_type MemoryType;

    static const bool IsDefaultMemoryType = false;
    static const type_id_t MemoryTypeID = cuda_device_type_id;

    static inline ndt::type MakeMemoryType(const ndt::type& target_tp) {
        return make_cuda_device(target_tp);
    }
};

typedef ::testing::Types<cuda_host_type, cuda_device_type> CUDAMemory;

#endif // DYND_CUDA
