#include <chrono>

#include <dynd/func/elwise.hpp>
#include <dynd/func/random.hpp>
#include <dynd/func/multidispatch.hpp>
#include <dynd/kernels/uniform_kernel.hpp>

using namespace std;
using namespace dynd;

template <typename GeneratorType>
struct uniform_kernel_alias {
  template <type_id_t DstTypeID>
  using type = nd::random::uniform_kernel<DstTypeID, GeneratorType>;
};

nd::callable nd::random::uniform::make()
{
  typedef type_id_sequence<int32_type_id, int64_type_id, uint32_type_id,
                           uint64_type_id, float32_type_id, float64_type_id,
                           complex_float32_type_id,
                           complex_float64_type_id> numeric_type_ids;

  std::random_device random_device;

  auto children =
      callable::make_all<uniform_kernel_alias<std::default_random_engine>::type,
                         numeric_type_ids>(0);

  return functional::elwise(functional::multidispatch<1>(
      ndt::type("(a: ?R, b: ?R) -> R"), std::move(children), callable(), {-1}));
}

struct nd::random::uniform nd::random::uniform;

/*

#ifdef DYND_CUDA

template <kernel_request_t kernreq>
typename std::enable_if<kernreq == kernel_request_cuda_device,
                        nd::callable>::type
nd::random::uniform::make()
{
  unsigned int blocks_per_grid = 512;
  unsigned int threads_per_block = 512;

  curandState_t *s;
  cudaMalloc(&s, blocks_per_grid * threads_per_block * sizeof(curandState_t));
  cuda_device_curand_init << <blocks_per_grid, threads_per_block>>> (s);

  return nd::as_callable<uniform_ck, kernel_request_cuda_device, curandState_t,
                        type_sequence<double, dynd::complex<double>>>(
      ndt::type("(a: ?R, b: ?R) -> cuda_device[R]"), s);
}

#endif

#ifdef DYND_CUDA
  return nd::functional::elwise(nd::functional::multidispatch(
      ndt::type("(a: ?R, b: ?R) -> M[R]"),
      {make<kernel_request_host>(),
       make<kernel_request_cuda_device>()}));
#else

#ifdef __CUDACC__

__global__ void cuda_device_curand_init(curandState_t *s)
{
  curand_init(clock64(), blockIdx.x * blockDim.x + threadIdx.x, 0, s);
}

#endif

*/