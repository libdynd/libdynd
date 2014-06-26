#include <dynd/random.hpp>

using namespace std;
using namespace dynd;

nd::array nd::typed_rand(intptr_t ndim, const intptr_t *shape, const ndt::type &tp) {
    nd::array res = nd::typed_empty(ndim, shape, tp.with_replaced_dtype(ndt::make_type<double>()));

    try {
        array_iter<1, 0> iter(res);
        do {
            *reinterpret_cast<double *>(iter.data()) = rand() / ((double) RAND_MAX);
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
