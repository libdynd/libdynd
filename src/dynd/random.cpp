#include <dynd/random.hpp>


using namespace std;
using namespace dynd;

// Currently using c stdlib rand(), which is known to
// be a terrible thing to do, this function needs to
// be replaced before it's reasonable for serious
// applications
inline void bad_quality_random(float *out)
{
  *out = ::rand() / ((float)RAND_MAX);
}

inline void bad_quality_random(double *out)
{
  *out = ::rand() / ((double)RAND_MAX);
}

inline void bad_quality_random(dynd_complex<float> *out)
{
  *out = dynd_complex<float>(::rand() / ((float)RAND_MAX),
                             ::rand() / ((float)RAND_MAX));
}

inline void bad_quality_random(dynd_complex<double> *out)
{
  *out = dynd_complex<double>(::rand() / ((double)RAND_MAX),
                              ::rand() / ((double)RAND_MAX));
}

template<class T>
static void fill_random(const nd::array &a, intptr_t count)
{
  T *v = reinterpret_cast<T *>(a.get_readwrite_originptr());
  for (intptr_t i = 0; i < count; ++i) {
    bad_quality_random(&v[i]);
  }
}

nd::array nd::rand(const ndt::type &tp) {
  intptr_t strided_ndim = tp.get_strided_ndim();
  ndt::type dtp = tp.get_type_at_dimension(NULL, strided_ndim);

  nd::array res = nd::empty(tp);

  // Get the total number of elements to fill in randomly
  intptr_t total_size = 1;
  const size_stride_t *arrmeta =
      reinterpret_cast<const size_stride_t *>(res.get_arrmeta());
  for (intptr_t i = 0; i < strided_ndim; ++i) {
    total_size *= arrmeta[i].dim_size;
  }

  switch (dtp.get_type_id()) {
  case float32_type_id:
    fill_random<float>(res, total_size);
    break;
  case float64_type_id:
    fill_random<double>(res, total_size);
    break;
  case complex_float32_type_id:
    fill_random<dynd_complex<float> >(res, total_size);
    break;
  case complex_float64_type_id:
    fill_random<dynd_complex<double> >(res, total_size);
    break;
  default: {
    stringstream ss;
    ss << "dynd rand: unsupported dtype " << dtp;
    throw std::runtime_error(ss.str());
  }
  }

  return res;
}
