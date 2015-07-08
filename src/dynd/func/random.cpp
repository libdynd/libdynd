#include <chrono>

#include <dynd/func/elwise.hpp>
#include <dynd/func/random.hpp>
#include <dynd/func/multidispatch.hpp>

using namespace std;
using namespace dynd;

template <template <typename, type_id_t> class KernelType, typename G>
struct bind2 {
  template <type_id_t DstTypeID>
  using type = KernelType<G, DstTypeID>;
};

template <kernel_request_t kernreq>
typename std::enable_if<kernreq == kernel_request_host, nd::arrfunc>::type
nd::random::uniform::make()
{
  typedef type_id_sequence<int32_type_id, int64_type_id, uint32_type_id,
                           uint64_type_id, float32_type_id, float64_type_id,
                           complex_float32_type_id,
                           complex_float64_type_id> numeric_type_ids;

  std::random_device random_device;

  auto children = arrfunc::make_all<
      bind2<uniform_ck, std::default_random_engine>::type, numeric_type_ids>();

  return functional::multidispatch<1>(ndt::type("(a: ?R, b: ?R) -> R"),
                                      std::move(children), arrfunc(), {-1});

  /*
    arrfunc self =
        as_arrfunc<bind2<uniform_ck, std::default_random_engine>::type,
    numeric_types>(
            ndt::type("(a: ?R, b: ?R) -> R"),
            std::shared_ptr<std::default_random_engine>(
                new std::default_random_engine(random_device())));
  */
}

nd::arrfunc nd::random::uniform::make()
{
  return nd::functional::elwise(make<kernel_request_host>());
}

struct nd::random::uniform nd::random::uniform;

/*

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