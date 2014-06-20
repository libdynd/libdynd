//
// Copyright (C) 2011-14 Irwin Zaid, DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <map>
#include <tr1/tuple>

#include <dynd/fft.hpp>

using namespace std;
using namespace dynd;

#ifdef DYND_FFTW

fftwf_plan dynd::fftw::fftplan(size_t ndim, fftwf_complex *src, vector<intptr_t> src_shape, vector<intptr_t> src_strides,
                              fftwf_complex *dst, vector<intptr_t> dst_strides, int sign, unsigned int flags, bool cache) {
    typedef tr1::tuple<vector<intptr_t>, vector<intptr_t>, vector<intptr_t>, int, unsigned int> key_type;
    static map<key_type, fftwf_plan> plans;

    key_type key(src_shape, src_strides, dst_strides, sign, flags);
    if (plans.find(key) != plans.end()) {
        return plans[key];
    }

    fftw_iodim iodims[ndim];
    for (size_t i = 0; i < ndim; ++i) {
        iodims[i].n = src_shape[i];
        iodims[i].is = src_strides[i] / sizeof(fftwf_complex);
        iodims[i].os = dst_strides[i] / sizeof(fftwf_complex);
    }

    fftwf_plan plan = fftwf_plan_guru_dft(ndim, iodims, 0, NULL, src, dst, sign, flags);
    if (cache) {
        plans[key] = plan;
    }

    return plan;
}

fftw_plan dynd::fftw::fftplan(size_t ndim, fftw_complex *src, vector<intptr_t> src_shape, vector<intptr_t> src_strides,
                              fftw_complex *dst, vector<intptr_t> dst_strides, int sign, unsigned int flags, bool cache) {
    typedef tr1::tuple<vector<intptr_t>, vector<intptr_t>, vector<intptr_t>, int, unsigned int> key_type;
    static map<key_type, fftw_plan> plans;

    key_type key(src_shape, src_strides, dst_strides, sign, flags);
    if (plans.find(key) != plans.end()) {
        return plans[key];
    }

    fftw_iodim iodims[ndim];
    for (size_t i = 0; i < ndim; ++i) {
        iodims[i].n = src_shape[i];
        iodims[i].is = src_strides[i] / sizeof(fftw_complex);
        iodims[i].os = dst_strides[i] / sizeof(fftw_complex);
    }

    fftw_plan plan = fftw_plan_guru_dft(ndim, iodims, 0, NULL, src, dst, sign, flags);
    if (cache) {
        plans[key] = plan;
    }

    return plan;
}

nd::array dynd::fftw::fft1(const nd::array &x, unsigned int flags, bool cache) {
    return dynd::fftw::fft(x, flags, cache);
}

nd::array dynd::fftw::ifft1(const nd::array &x, unsigned int flags, bool cache) {
    return dynd::fftw::ifft(x, flags, cache);
}

// real -> hermitian
// hermitian -> real

nd::array dynd::fftw::fft(const nd::array &x, unsigned int flags, bool cache) {
    nd::array y = nd::empty_like(x);

    type_id_t dtp_id = x.get_dtype().get_type_id();
    if (dtp_id == complex_float32_type_id) {
        fftwf_complex *src = reinterpret_cast<fftwf_complex *>(x.get_readwrite_originptr());
        fftwf_complex *dst = reinterpret_cast<fftwf_complex *>(y.get_readwrite_originptr());
        fftwf_plan plan = fftw::fftplan(x.get_ndim(), src, x.get_shape(), x.get_strides(),
            dst, y.get_strides(), FFTW_FORWARD, flags);
        fftwf_execute_dft(plan, src, dst);
        if (!cache) {
            fftwf_destroy_plan(plan);
        }
    } else if (dtp_id == complex_float64_type_id) {
        fftw_complex *src = reinterpret_cast<fftw_complex *>(x.get_readwrite_originptr());
        fftw_complex *dst = reinterpret_cast<fftw_complex *>(y.get_readwrite_originptr());
        fftw_plan plan = fftw::fftplan(x.get_ndim(), src, x.get_shape(), x.get_strides(),
            dst, y.get_strides(), FFTW_FORWARD, flags, cache);
        fftw_execute_dft(plan, src, dst);
        if (!cache) {
            fftw_destroy_plan(plan);
        }
    } else {
        throw std::runtime_error("");
    }

    return y;
}

nd::array dynd::fftw::ifft(const nd::array &x, unsigned int flags, bool cache) {
    nd::array y = nd::empty_like(x);

    type_id_t dtp_id = x.get_dtype().get_type_id();
    if (dtp_id == complex_float32_type_id) {
        fftwf_complex *src = reinterpret_cast<fftwf_complex *>(x.get_readwrite_originptr());
        fftwf_complex *dst = reinterpret_cast<fftwf_complex *>(y.get_readwrite_originptr());
        fftwf_plan plan = fftw::fftplan(x.get_ndim(), src, x.get_shape(), x.get_strides(),
            dst, y.get_strides(), FFTW_BACKWARD, flags, cache);
        fftwf_execute_dft(plan, src, dst);
        if (!cache) {
            fftwf_destroy_plan(plan);
        }
    } else if (dtp_id == complex_float64_type_id) {
        fftw_complex *src = reinterpret_cast<fftw_complex *>(x.get_readwrite_originptr());
        fftw_complex *dst = reinterpret_cast<fftw_complex *>(y.get_readwrite_originptr());
        fftw_plan plan = fftw::fftplan(x.get_ndim(), src, x.get_shape(), x.get_strides(),
            dst, y.get_strides(), FFTW_BACKWARD, flags, cache);
        fftw_execute_dft(plan, src, dst);
        if (!cache) {
            fftw_destroy_plan(plan);
        }
    } else {
        throw std::runtime_error("");
    }

    return y;
}

#endif // DYND_FFTW
