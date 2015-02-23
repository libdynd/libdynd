#include <chrono>

#include <dynd/func/elwise.hpp>
#include <dynd/func/random.hpp>

using namespace std;
using namespace dynd;

#ifdef __CUDACC__

__global__ void cuda_device_curand_init(curandState_t *s)
{
  curand_init(clock64(), blockIdx.x * blockDim.x + threadIdx.x, 0, s);
}

#endif

template <kernel_request_t kernreq>
typename std::enable_if<kernreq == kernel_request_host, nd::arrfunc>::type
nd::random::uniform::make()
{
  std::random_device random_device;
  return nd::as_arrfunc<uniform_ck, kernel_request_host,
                        std::default_random_engine, numeric_types>(
      ndt::type("(a: ?R, b: ?R) -> R"),
      std::shared_ptr<std::default_random_engine>(
          new std::default_random_engine(random_device())));
}

#ifdef DYND_CUDA

template <kernel_request_t kernreq>
typename std::enable_if<kernreq == kernel_request_cuda_device,
                        nd::arrfunc>::type
nd::random::uniform::make()
{
  unsigned int blocks_per_grid = 512;
  unsigned int threads_per_block = 512;

  curandState_t *s;
  cudaMalloc(&s, blocks_per_grid * threads_per_block * sizeof(curandState_t));
  cuda_device_curand_init << <blocks_per_grid, threads_per_block>>> (s);

  return nd::as_arrfunc<uniform_ck, kernel_request_cuda_device, curandState_t,
                        type_sequence<double, dynd::complex<double>>>(
      ndt::type("(a: ?R, b: ?R) -> cuda_device[R]"), s);
}

#endif

nd::arrfunc nd::random::uniform::make()
{
#ifdef DYND_CUDA
  return nd::functional::elwise(nd::functional::multidispatch(
      ndt::type("(a: ?R, b: ?R) -> M[R]"),
      {make<kernel_request_host>(),
       make<kernel_request_cuda_device>()}));
#else
  return nd::functional::elwise(make<kernel_request_host>());
#endif
}

struct nd::random::uniform nd::random::uniform;