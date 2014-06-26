//
// Copyright (C) 2011-14 Irwin Zaid, DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <cstdlib>

#include <dynd/array.hpp>
#include <dynd/array_iter.hpp>

namespace dynd { namespace nd {
/*
    nd::array rand(int n, const ndt::type& dtp) {
        srand(time(NULL));

        nd::array x = nd::empty(n, ndt::make_strided_dim(dtp));
        for (int i = 0; i < n; i++) {
            x(i).vals() = dynd_complex<double>(std::rand() / ((double) RAND_MAX), std::rand() / ((double) RAND_MAX));
        }

        return x;
    }

    nd::array rand(intptr_t dim0, intptr_t dim1, const ndt::type& dtp) {
        srand(time(NULL));

        nd::array x = nd::zeros(dim0, dim1, ndt::make_strided_dim(ndt::make_strided_dim(dtp)));
        for (int i = 0; i < dim0; i++) {
            for (int j = 0; j < dim1; j++) {
                if (dtp.get_type_id() == complex_float64_type_id) {
                    x(i, j).vals() = dynd_complex<double>(std::rand() / ((double) RAND_MAX), std::rand() / ((double) RAND_MAX));
                } else {
                    x(i, j).vals() = std::rand() / ((double) RAND_MAX);
                }
            }
        }

        return x;
    }
*/

    nd::array typed_rand(intptr_t ndim, const intptr_t *shape, const ndt::type &tp);

}} // namespace dynd::nd
