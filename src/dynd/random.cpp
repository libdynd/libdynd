#include <dynd/random.hpp>

using namespace std;
using namespace dynd;

nd::array nd::typed_rand(intptr_t ndim, const intptr_t *shape, const ndt::type &tp) {
    type_id_t dtp_id = tp.get_dtype().get_type_id();

    nd::array res = nd::typed_empty(ndim, shape, tp);

    try {
        array_iter<1, 0> iter(res);
        do {
            switch (dtp_id) {
                case float32_type_id:
                    *reinterpret_cast<float *>(iter.data())
                        = rand() / ((float) RAND_MAX);
                    break;
                case float64_type_id:
                    *reinterpret_cast<double *>(iter.data())
                        = rand() / ((double) RAND_MAX);
                    break;
                case complex_float32_type_id:
                    *reinterpret_cast<dynd_complex<float> *>(iter.data())
                        = dynd_complex<float>(rand() / ((float) RAND_MAX), rand() / ((float) RAND_MAX));
                    break;
                case complex_float64_type_id:
                    *reinterpret_cast<dynd_complex<double> *>(iter.data())
                        = dynd_complex<double>(rand() / ((double) RAND_MAX), rand() / ((double) RAND_MAX));
                    break;
                default:
                    throw std::runtime_error("rand: unsupported dtype");
            }
        } while (iter.next());
    } catch (...) {
        ndt::type strided_tp = tp.get_dtype();
        for (intptr_t i = 0; i < ndim; ++i) {
            strided_tp = ndt::make_strided_dim(strided_tp);
        }

        res.vals() = nd::typed_rand(ndim, shape, strided_tp);
    }

    return res.ucast(tp.get_dtype()).eval();
}
