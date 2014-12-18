//
// Copyright (C) 2011-14 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <dynd/array_range.hpp>
#include <dynd/kernels/fft.hpp>

using namespace std;
using namespace dynd;

/*
template <int sign>
intptr_t fftw_ck<complex_float64_type_id, complex_float64_type_id, FFTW_FORWARD>::instantiate(const arrfunc_type_data *DYND_UNUSED(self),
    dynd::ckernel_builder *ckb, intptr_t ckb_offset,
    const ndt::type &dst_tp, const char *dst_arrmeta,
    const ndt::type *src_tp, const char *const *src_arrmeta,
    kernel_request_t kernreq, const eval::eval_context *DYND_UNUSED(ectx),
    const nd::array &DYND_UNUSED(args), const nd::array &DYND_UNUSED(kwds))
{
    nd::array axes = nd::range(src_tp[0].get_ndim());

    const size_stride_t *src_size_stride = reinterpret_cast<const size_stride_t *>(src_arrmeta[0]);
    const size_stride_t *dst_size_stride = reinterpret_cast<const size_stride_t *>(dst_arrmeta);

    int rank = axes.get_ndim();
    shortvector<fftw_iodim> dims(rank);
    for (intptr_t i = 0; i < rank; ++i)
    {
        intptr_t j = axes(i).as<intptr_t>();
        dims[i].n = src_size_stride[j].dim_size;
        dims[i].is = src_size_stride[j].stride / sizeof(fftw_complex);
        dims[i].os = dst_size_stride[j].stride / sizeof(fftw_complex);
    }

    int howmany_rank = src_tp[0].get_ndim() - rank;
    shortvector<fftw_iodim> howmany_dims(howmany_rank);
    for (intptr_t i = 0, j = 0, k = 0; i < howmany_rank; ++i, ++j)
    {
        for (; k < rank && j == axes(k).as<intptr_t>(); ++j, ++k)
        {
        }
        howmany_dims[i].n = src_size_stride[j].dim_size;
        howmany_dims[i].is = src_size_stride[j].stride / sizeof(fftw_complex);
        howmany_dims[i].os = dst_size_stride[j].stride / sizeof(fftw_complex);
    }

    nd::array src = nd::empty(src_tp[0]);
    nd::array dst = nd::empty(dst_tp);

    fftw_ck::create(ckb, kernreq, ckb_offset, fftw_plan_guru_dft(rank, dims.get(), howmany_rank, howmany_dims.get(),
        reinterpret_cast<fftw_complex *>(src.get_readwrite_originptr()), reinterpret_cast<fftw_complex *>(dst.get_readwrite_originptr()),
        FFTW_FORWARD, FFTW_ESTIMATE));
    return ckb_offset;
}
*/
