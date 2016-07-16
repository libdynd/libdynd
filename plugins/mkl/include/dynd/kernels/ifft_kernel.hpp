//
// Copyright (C) 2011-16 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#pragma once

#include <mkl_dfti.h>

#include <dynd/kernels/base_strided_kernel.hpp>

namespace dynd {
namespace nd {
  namespace mkl {

    template <typename ComplexType>
    struct ifft_kernel : base_strided_kernel<ifft_kernel<ComplexType>, 1> {
      typedef ComplexType complex_type;
      typedef typename ComplexType::value_type real_type;

      DFTI_DESCRIPTOR_HANDLE descriptor;

      ifft_kernel(size_t ndim, const char *src0_metadata, real_type scale) {

        if (ndim == 1) {
          DftiCreateDescriptor(&descriptor, DFTI_DOUBLE, DFTI_COMPLEX, 1,
                                        reinterpret_cast<const size_stride_t *>(src0_metadata)->dim_size);
        } else {
/*
          MKL_LONG src0_size[3];
          for (size_t i = 0; i < ndim; ++i) {
            src0_size[i] = reinterpret_cast<const size_stride_t *>(src0_metadata)->dim_size;
            src0_metadata += sizeof(size_stride_t);
          }
*/
        }

        DftiSetValue(descriptor, DFTI_BACKWARD_SCALE, scale);
        DftiSetValue(descriptor, DFTI_PLACEMENT, DFTI_NOT_INPLACE);

        DftiCommitDescriptor(descriptor);
      }

      ~ifft_kernel() { DftiFreeDescriptor(&descriptor); }

      void single(char *dst, char *const *src) { DftiComputeBackward(descriptor, src[0], dst); }
    };

  } // namespace dynd::nd::mkl
} // namespace dynd::nd
} // namespace dynd
