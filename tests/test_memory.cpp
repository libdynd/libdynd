#include "inc_gtest.hpp"

#include <dynd/array.hpp>
#include <dynd/type.hpp>
#include <dynd/types/cuda_host_type.hpp>
#include <dynd/types/cuda_device_type.hpp>
#include <dynd/types/type_id.hpp>

using namespace std;
using namespace dynd;

typedef ::testing::internal::None None;

template<typename T>
class Memory;

template<typename T>
class MemoryPair : public ::testing::Test {
public:
    typedef Memory<typename T::Head> First;
    typedef Memory<typename T::Tail::Head> Second;
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

    static inline nd::array To(nd::array a) {
#ifdef DYND_CUDA
        return a.to_host();
#else
        return a;
#endif // DYND_CUDA
    }
};

typedef ::testing::Types<None> DefaultMemory;
typedef ::testing::Types< ::testing::internal::Types2<None, None> > DefaultMemoryPairs;

#ifdef DYND_CUDA

template <>
class Memory<cuda_host_type> : public ::testing::Test {
    static const type_id_t TypeID = cuda_host_type_id;
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

    static inline nd::array To(nd::array a, unsigned int cuda_host_flags = cudaHostAllocDefault) {
        return a.to_cuda_host(cuda_host_flags);
    }
};

template <>
class Memory<cuda_device_type> : public ::testing::Test {
    static const type_id_t TypeID = cuda_device_type_id;
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
        throw_if_not_cuda_success(cudaMemcpy(&tmp, ptr, sizeof(T), cudaMemcpyDeviceToHost));
        return tmp;
    }

    static inline nd::array To(nd::array a) {
        return a.to_cuda_device();
    }
};

typedef ::testing::Types<cuda_host_type, cuda_device_type> CUDAMemory;
typedef ::testing::Types< ::testing::internal::Types2<None, cuda_host_type>,
    ::testing::internal::Types2<None, cuda_device_type>,
    ::testing::internal::Types2<cuda_host_type, None>,
    ::testing::internal::Types2<cuda_device_type, None>,
    ::testing::internal::Types2<cuda_host_type, cuda_host_type>,
    ::testing::internal::Types2<cuda_host_type, cuda_device_type>,
    ::testing::internal::Types2<cuda_device_type, cuda_host_type>,
    ::testing::internal::Types2<cuda_device_type, cuda_device_type> > CUDAMemoryPairs;

#endif // DYND_CUDA
