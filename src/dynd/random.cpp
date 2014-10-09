#include <dynd/random.hpp>

using namespace std;
using namespace dynd;

nd::array nd::dtyped_rand(intptr_t ndim, const intptr_t *shape, const ndt::type &tp) {
    type_id_t dtp_id = tp.get_dtype().get_type_id();

    nd::array res = nd::dtyped_empty(ndim, shape, tp);

    try {
        array_iter<1, 0> iter(res);
        do {
            switch (dtp_id) {
                case float32_type_id:
                    *reinterpret_cast<float *>(iter.data())
                        = ::rand() / ((float) RAND_MAX);
                    break;
                case float64_type_id:
                    *reinterpret_cast<double *>(iter.data())
                        = ::rand() / ((double) RAND_MAX);
                    break;
                case complex_float32_type_id:
                    *reinterpret_cast<dynd_complex<float> *>(iter.data())
                        = dynd_complex<float>(::rand() / ((float) RAND_MAX), ::rand() / ((float) RAND_MAX));
                    break;
                case complex_float64_type_id:
                    *reinterpret_cast<dynd_complex<double> *>(iter.data())
                        = dynd_complex<double>(::rand() / ((double) RAND_MAX), ::rand() / ((double) RAND_MAX));
                    break;
                default:
                    throw std::runtime_error("rand: unsupported dtype");
            }
        } while (iter.next());
    } catch (...) {
      res.vals() =
          nd::dtyped_rand(0, NULL, ndt::make_type(ndim, shape, tp.get_dtype()));
    }

    return res.ucast(tp.get_dtype()).eval();
}
