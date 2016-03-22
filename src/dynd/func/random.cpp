#include <chrono>

#include <dynd/func/random.hpp>
#include <dynd/functional.hpp>
#include <dynd/kernels/uniform_kernel.hpp>

using namespace std;
using namespace dynd;

template <typename GeneratorType>
struct uniform_kernel_alias {
  template <type_id_t DstTypeID>
  using type = nd::random::uniform_kernel<DstTypeID, GeneratorType>;
};

DYND_API nd::callable nd::random::uniform::make()
{
  typedef type_id_sequence<int32_id, int64_id, uint32_id, uint64_id, float32_id, float64_id, complex_float32_id,
                           complex_float64_id> numeric_ids;

  std::random_device random_device;

  auto dispatcher = callable::new_make_all<uniform_kernel_alias<std::default_random_engine>::type, numeric_ids>();
  return functional::elwise(functional::dispatch(
      ndt::type("(a: ?R, b: ?R) -> R"), [dispatcher](const ndt::type &dst_tp, intptr_t DYND_UNUSED(nsrc),
                                                     const ndt::type *DYND_UNUSED(src_tp)) mutable -> callable & {
        return const_cast<callable &>(dispatcher(dst_tp.get_id()));
      }));
}

DYND_DEFAULT_DECLFUNC_GET(nd::random::uniform)

DYND_API struct nd::random::uniform nd::random::uniform;

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
