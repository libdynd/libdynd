//
// Copyright (C) 2011-14 Irwin Zaid, DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <dynd/fft.hpp>

#include </opt/local/include/fftw3.h>

using namespace dynd;

namespace dynd { namespace fftw {

// make_plan
// get_plan

fftw_plan plan(size_t ndim, double *src, std::vector<intptr_t> src_shape, std::vector<intptr_t> src_strides,
                    fftw_complex *dst, std::vector<intptr_t> dst_strides, unsigned int flags) {
    fftw_iodim iodims[ndim];
    for (size_t i = 0; i < ndim; ++i) {
        iodims[i].n = src_shape[i];
        iodims[i].is = src_strides[i] / sizeof(double);
        iodims[i].os = dst_strides[i] / sizeof(fftw_complex);
    }

    return fftw_plan_guru_dft_r2c(ndim, iodims, 0, NULL, src, dst, flags);
}

fftwf_plan plan(size_t ndim, fftwf_complex *src, std::vector<intptr_t> src_shape, std::vector<intptr_t> src_strides,
                    fftwf_complex *dst, std::vector<intptr_t> dst_strides, int sign, unsigned int flags) {
    fftw_iodim iodims[ndim];
    for (size_t i = 0; i < ndim; ++i) {
        iodims[i].n = src_shape[i];
        iodims[i].is = src_strides[i] / sizeof(fftwf_complex);
        iodims[i].os = dst_strides[i] / sizeof(fftwf_complex);
    }

    return fftwf_plan_guru_dft(ndim, iodims, 0, NULL, src, dst, sign, flags);
}

fftw_plan plan(size_t ndim, fftw_complex *src, std::vector<intptr_t> src_shape, std::vector<intptr_t> src_strides,
                    fftw_complex *dst, std::vector<intptr_t> dst_strides, int sign, unsigned int flags) {
    fftw_iodim iodims[ndim];
    for (size_t i = 0; i < ndim; ++i) {
        iodims[i].n = src_shape[i];
        iodims[i].is = src_strides[i] / sizeof(fftw_complex);
        iodims[i].os = dst_strides[i] / sizeof(fftw_complex);
    }

    return fftw_plan_guru_dft(ndim, iodims, 0, NULL, src, dst, sign, flags);
}

void destroy_plan(fftw_plan plan) {
    fftw_destroy_plan(plan);
}

void destroy_plan(fftwf_plan plan) {
    fftwf_destroy_plan(plan);
}

}} // namespace dynd::fftw


nd::array dynd::fftw::fft1(const nd::array &x) {
    nd::array y = nd::empty_like(x);

    type_id_t dtp_id = x.get_dtype().get_type_id();
    if (dtp_id == float32_type_id) {

    } else if (dtp_id == float64_type_id) {

    } else if (dtp_id == complex_float32_type_id) {
        fftw_iodim dims[1];
        dims[0].n = x.get_dim_size();
        dims[0].is = x.get_strides()[0] / sizeof(fftwf_complex);
        dims[0].os = y.get_strides()[0] / sizeof(fftwf_complex);

        fftwf_plan plan = fftwf_plan_guru_dft(1, dims, 0, NULL,
            reinterpret_cast<fftwf_complex *>(x.get_readwrite_originptr()),
            reinterpret_cast<fftwf_complex *>(y.get_readwrite_originptr()),
            FFTW_FORWARD, FFTW_ESTIMATE);
        fftwf_execute(plan);
        fftwf_destroy_plan(plan);
    } else if (dtp_id == complex_float64_type_id) {
        fftw_iodim dims[1];
        dims[0].n = x.get_dim_size();
        dims[0].is = x.get_strides()[0] / sizeof(fftw_complex);
        dims[0].os = y.get_strides()[0] / sizeof(fftw_complex);

        fftw_plan plan = fftw_plan_guru_dft(1, dims, 0, NULL,
            reinterpret_cast<fftw_complex *>(x.get_readwrite_originptr()),
            reinterpret_cast<fftw_complex *>(y.get_readwrite_originptr()),
            FFTW_FORWARD, FFTW_ESTIMATE);
        fftw_execute(plan);
        fftw_destroy_plan(plan);
    } else {
        // error
    }

    return y;
}

nd::array dynd::fftw::ifft1(const nd::array &x) {
    nd::array y = nd::empty_like(x);

    type_id_t dtp_id = x.get_dtype().get_type_id();
    if (dtp_id == float32_type_id) {

    } else if (dtp_id == float64_type_id) {

    } else if (dtp_id == complex_float32_type_id) {
        fftwf_plan plan = fftwf_plan_dft_1d(x.get_dim_size(),
            reinterpret_cast<fftwf_complex *>(x.get_readwrite_originptr()),
            reinterpret_cast<fftwf_complex *>(y.get_readwrite_originptr()),
            FFTW_BACKWARD, FFTW_ESTIMATE);
        fftwf_execute(plan);
        fftwf_destroy_plan(plan);
    } else if (dtp_id == complex_float64_type_id) {
        fftw_plan plan = fftw_plan_dft_1d(x.get_dim_size(),
            reinterpret_cast<fftw_complex *>(x.get_readwrite_originptr()),
            reinterpret_cast<fftw_complex *>(y.get_readwrite_originptr()),
            FFTW_BACKWARD, FFTW_ESTIMATE);
        fftw_execute(plan);
        fftw_destroy_plan(plan);
    } else {
        // error
    }

    return y;
}

nd::array dynd::fftw::fft(const nd::array &x) {
    type_id_t dtp_id = x.get_dtype().get_type_id();
    if (dtp_id == float32_type_id) {
        throw std::runtime_error("");
    } else if (dtp_id == float64_type_id) {
        throw std::runtime_error("");
    } else if (dtp_id == complex_float32_type_id) {
        nd::array y = nd::empty_like(x);
        fftwf_plan plan = fftw::plan(x.get_ndim(),
            reinterpret_cast<fftwf_complex *>(x.get_readwrite_originptr()), x.get_shape(), x.get_strides(),
            reinterpret_cast<fftwf_complex *>(y.get_readwrite_originptr()), y.get_strides(),
            FFTW_FORWARD, FFTW_ESTIMATE);
        fftwf_execute(plan);
        fftw::destroy_plan(plan);
        return y;
    } else if (dtp_id == complex_float64_type_id) {
        nd::array y = nd::empty_like(x);
        fftw_plan plan = fftw::plan(x.get_ndim(),
            reinterpret_cast<fftw_complex *>(x.get_readwrite_originptr()), x.get_shape(), x.get_strides(),
            reinterpret_cast<fftw_complex *>(y.get_readwrite_originptr()), y.get_strides(),
            FFTW_FORWARD, FFTW_ESTIMATE);
        fftw_execute(plan);
        fftw::destroy_plan(plan);
        return y;
    }

    throw std::runtime_error("");
}

nd::array dynd::fftw::ifft(const nd::array &x) {
    type_id_t dtp_id = x.get_dtype().get_type_id();
    if (dtp_id == float32_type_id) {
        throw std::runtime_error("");
    } else if (dtp_id == float64_type_id) {
        throw std::runtime_error("");
    } else if (dtp_id == complex_float32_type_id) {
        nd::array y = nd::empty_like(x);
        fftwf_plan plan = fftw::plan(x.get_ndim(),
            reinterpret_cast<fftwf_complex *>(x.get_readwrite_originptr()), x.get_shape(), x.get_strides(),
            reinterpret_cast<fftwf_complex *>(y.get_readwrite_originptr()), y.get_strides(),
            FFTW_BACKWARD, FFTW_ESTIMATE);
        fftwf_execute(plan);
        fftw::destroy_plan(plan);
        return y;
    } else if (dtp_id == complex_float64_type_id) {
        nd::array y = nd::empty_like(x);
        fftw_plan plan = fftw::plan(x.get_ndim(),
            reinterpret_cast<fftw_complex *>(x.get_readwrite_originptr()), x.get_shape(), x.get_strides(),
            reinterpret_cast<fftw_complex *>(y.get_readwrite_originptr()), y.get_strides(),
            FFTW_BACKWARD, FFTW_ESTIMATE);
        fftw_execute(plan);
        fftw::destroy_plan(plan);
        return y;
    }

    throw std::runtime_error("");
}
