//
// Copyright (C) 2011-16 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#pragma once

#include <mkl_vsl.h>

#include <dynd/kernels/base_strided_kernel.hpp>
#include <dynd/mkl.hpp>

namespace dynd {
namespace nd {
  namespace mkl {
    namespace detail {

      template <typename DataType>
      struct conv_traits;

      template <>
      struct conv_traits<float> {
        static int vslConvNewTask(VSLConvTaskPtr *task, const MKL_INT mode, const MKL_INT dims, const MKL_INT xshape[],
                                  const MKL_INT yshape[], const MKL_INT zshape[]) {
          return vslsConvNewTask(task, mode, dims, xshape, yshape, zshape);
        }

        static int vslConvExec(VSLConvTaskPtr task, const float x[], const MKL_INT xstride[], const float y[],
                               const MKL_INT ystride[], float z[], const MKL_INT zstride[]) {
          return vslsConvExec(task, x, xstride, y, ystride, z, zstride);
        }
      };

      template <>
      struct conv_traits<double> {
        static int vslConvNewTask(VSLConvTaskPtr *task, const MKL_INT mode, const MKL_INT dims, const MKL_INT xshape[],
                                  const MKL_INT yshape[], const MKL_INT zshape[]) {
          return vsldConvNewTask(task, mode, dims, xshape, yshape, zshape);
        }

        static int vslConvExec(VSLConvTaskPtr task, const double x[], const MKL_INT xstride[], const double y[],
                               const MKL_INT ystride[], double z[], const MKL_INT zstride[]) {
          return vsldConvExec(task, x, xstride, y, ystride, z, zstride);
        }
      };

      template <>
      struct conv_traits<complex<float>> {
        static int vslConvNewTask(VSLConvTaskPtr *task, const MKL_INT mode, const MKL_INT dims, const MKL_INT xshape[],
                                  const MKL_INT yshape[], const MKL_INT zshape[]) {
          return vslcConvNewTask(task, mode, dims, xshape, yshape, zshape);
        }

        static int vslConvExec(VSLConvTaskPtr task, const MKL_Complex8 x[], const MKL_INT xstride[],
                               const MKL_Complex8 y[], const MKL_INT ystride[], MKL_Complex8 z[],
                               const MKL_INT zstride[]) {
          return vslcConvExec(task, x, xstride, y, ystride, z, zstride);
        }
      };

      template <>
      struct conv_traits<complex<double>> {
        static int vslConvNewTask(VSLConvTaskPtr *task, const MKL_INT mode, const MKL_INT dims, const MKL_INT xshape[],
                                  const MKL_INT yshape[], const MKL_INT zshape[]) {
          return vslzConvNewTask(task, mode, dims, xshape, yshape, zshape);
        }

        static int vslConvExec(VSLConvTaskPtr task, const MKL_Complex16 x[], const MKL_INT xstride[],
                               const MKL_Complex16 y[], const MKL_INT ystride[], MKL_Complex16 z[],
                               const MKL_INT zstride[]) {
          return vslzConvExec(task, x, xstride, y, ystride, z, zstride);
        }
      };

    } // namespace dynd::nd::mkl::detail

    template <size_t NDim, typename DataType>
    struct conv_kernel : base_strided_kernel<conv_kernel<NDim, DataType>, 2>, detail::conv_traits<DataType> {
      using detail::conv_traits<DataType>::vslConvNewTask;
      using detail::conv_traits<DataType>::vslConvExec;

      VSLConvTaskPtr task;
      MKL_INT ret_stride[NDim];
      MKL_INT arg0_stride[NDim];
      MKL_INT arg1_stride[NDim];

      conv_kernel(MKL_INT mode, const size_stride_t *ret_size_stride, const size_stride_t *arg0_size_stride,
                  const size_stride_t *arg1_size_stride) {
        MKL_INT ret_size[NDim];
        MKL_INT arg0_size[NDim];
        MKL_INT arg1_size[NDim];

        for (size_t i = 0; i < NDim; ++i) {
          ret_size[i] = ret_size_stride[i].dim_size;
          ret_stride[i] = ret_size_stride[i].stride / sizeof(DataType);

          arg0_size[i] = arg0_size_stride[i].dim_size;
          arg0_stride[i] = arg0_size_stride[i].stride / sizeof(DataType);

          arg1_size[i] = arg1_size_stride[i].dim_size;
          arg1_stride[i] = arg1_size_stride[i].stride / sizeof(DataType);
        }

        vslConvNewTask(&task, mode, NDim, arg0_size, arg1_size, ret_size);
      }

      ~conv_kernel() { vslConvDeleteTask(&task); }

      void single(char *ret, char *const *arg) {
        vslConvExec(task, reinterpret_cast<mkl_type_t<DataType> *>(arg[0]), arg0_stride,
                    reinterpret_cast<mkl_type_t<DataType> *>(arg[1]), arg1_stride,
                    reinterpret_cast<mkl_type_t<DataType> *>(ret), ret_stride);
      }
    };

  } // namespace dynd::nd::mkl
} // namespace dynd::nd
} // namespace dynd
