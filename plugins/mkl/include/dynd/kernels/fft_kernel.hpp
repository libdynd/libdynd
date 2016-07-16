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
    struct fft_kernel : base_strided_kernel<fft_kernel<ComplexType>, 1> {
      typedef ComplexType complex_type;

      DFTI_DESCRIPTOR_HANDLE descriptor;

      fft_kernel(size_t ndim, const size_stride_t *src0_size_stride) {
        switch (ndim) {
        case 1: {
          MKL_LONG src0_size = src0_size_stride[0].dim_size;
          DftiCreateDescriptor(&descriptor, DFTI_DOUBLE, DFTI_COMPLEX, 1, src0_size);
          break;
        }
        case 2: {
          MKL_LONG src0_size[2] = {src0_size_stride[0].dim_size, src0_size_stride[1].dim_size};
          DftiCreateDescriptor(&descriptor, DFTI_DOUBLE, DFTI_COMPLEX, 2, src0_size);
          break;
        }
        default:
          break;
        }

        DftiSetValue(descriptor, DFTI_PLACEMENT, DFTI_NOT_INPLACE);

        DftiCommitDescriptor(descriptor);
      }

      ~fft_kernel() { DftiFreeDescriptor(&descriptor); }

      void single(char *dst, char *const *src) { DftiComputeForward(descriptor, src[0], dst); }
    };

  } // namespace dynd::nd::mkl
} // namespace dynd::nd
} // namespace dynd

#undef DftiCreateDescriptor
