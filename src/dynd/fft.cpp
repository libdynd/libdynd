//
// Copyright (C) 2011-14 Irwin Zaid, DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <map>
#include <tr1/tuple>

#include <dynd/fft.hpp>
#include <dynd/types/strided_dim_type.hpp>

using namespace std;
using namespace dynd;

#ifdef DYND_FFTW

namespace dynd { namespace fftw {
typedef tr1::tuple<vector<intptr_t>, type_id_t, vector<intptr_t>, int,
    type_id_t, vector<intptr_t>, int, int, unsigned int> key_type;
static map<key_type, void *> plans;
static int cleanup_atexit = atexit(dynd::fftw::fftcleanup);
}} // namespace dynd::fftw

#define FFTPLAN_C2C_1(SRC_ID, DST_ID)
    fftwf_plan dynd::fftw::fftplan(size_t ndim, vector<intptr_t> shape, fftwf_complex *src, vector<intptr_t> src_strides, \
                                   fftwf_complex *dst, vector<intptr_t> dst_strides, int sign, unsigned int flags, bool cache) { \
    key_type key(shape, complex_float32_type_id, src_strides, fftwf_alignment_of(reinterpret_cast<float *>(src)), \
        complex_float32_type_id, dst_strides, fftwf_alignment_of(reinterpret_cast<float *>(dst)), sign, flags); \
    if (plans.find(key) != plans.end()) { \
        return reinterpret_cast<fftwf_plan>(plans[key]); \
    } \
\
    fftwf_iodim iodims[ndim]; \
    for (size_t i = 0; i < ndim; ++i) { \
        iodims[i].n = shape[i]; \
        iodims[i].is = src_strides[i] / sizeof(fftwf_complex); \
        iodims[i].os = dst_strides[i] / sizeof(fftwf_complex); \
    } \
\
    fftwf_plan plan = fftwf_plan_guru_dft(ndim, iodims, 0, NULL, src, dst, sign, flags); \
    if (cache) { \
        plans[key] = plan; \
    } \
\
    return plan; \
}

FFTPLAN_C2C_1(complex_float32_type_id, complex_float32_type_id)

fftw_plan dynd::fftw::fftplan(size_t ndim, vector<intptr_t> shape, fftw_complex *src, vector<intptr_t> src_strides,
                              fftw_complex *dst, vector<intptr_t> dst_strides, int sign, unsigned int flags, bool cache) {
    key_type key(shape, complex_float64_type_id, src_strides, fftw_alignment_of(reinterpret_cast<double *>(src)),
        complex_float64_type_id, dst_strides, fftw_alignment_of(reinterpret_cast<double *>(dst)), sign, flags);
    if (plans.find(key) != plans.end()) {
        return reinterpret_cast<fftw_plan>(plans[key]);
    }

    fftw_iodim iodims[ndim];
    for (size_t i = 0; i < ndim; ++i) {
        iodims[i].n = shape[i];
        iodims[i].is = src_strides[i] / sizeof(fftw_complex);
        iodims[i].os = dst_strides[i] / sizeof(fftw_complex);
    }

    fftw_plan plan = fftw_plan_guru_dft(ndim, iodims, 0, NULL, src, dst, sign, flags);
    if (cache) {
        plans[key] = plan;
    }

    return plan;
}

fftw_plan dynd::fftw::fftplan(size_t ndim, vector<intptr_t> shape, double *src, vector<intptr_t> src_strides,
                              fftw_complex *dst, vector<intptr_t> dst_strides, unsigned int flags, bool cache) {
    key_type key(shape, float64_type_id, src_strides, fftw_alignment_of(reinterpret_cast<double *>(src)),
        complex_float64_type_id, dst_strides, fftw_alignment_of(reinterpret_cast<double *>(dst)), FFTW_FORWARD, flags);
    if (plans.find(key) != plans.end()) {
        return reinterpret_cast<fftw_plan>(plans[key]);
    }

    fftw_iodim iodims[ndim];
    for (size_t i = 0; i < ndim; ++i) {
        iodims[i].n = shape[i];
        iodims[i].is = src_strides[i] / sizeof(double);
        iodims[i].os = dst_strides[i] / sizeof(fftw_complex);
    }

    fftw_plan plan = fftw_plan_guru_dft_r2c(ndim, iodims, 0, NULL, src, dst, flags);
    if (cache) {
        plans[key] = plan;
    }

    return plan;
}

fftw_plan dynd::fftw::fftplan(size_t ndim, vector<intptr_t> shape, fftw_complex *src, vector<intptr_t> src_strides,
                              double *dst, vector<intptr_t> dst_strides, unsigned int flags, bool cache) {
    key_type key(shape, complex_float64_type_id, src_strides, fftw_alignment_of(reinterpret_cast<double *>(src)),
        float64_type_id, dst_strides, fftw_alignment_of(reinterpret_cast<double *>(dst)), FFTW_BACKWARD, flags);
    if (plans.find(key) != plans.end()) {
        return reinterpret_cast<fftw_plan>(plans[key]);
    }

    fftw_iodim iodims[ndim];
    for (size_t i = 0; i < ndim; ++i) {
        iodims[i].n = shape[i];
        iodims[i].is = src_strides[i] / sizeof(fftw_complex);
        iodims[i].os = dst_strides[i] / sizeof(double);
    }

    fftw_plan plan = fftw_plan_guru_dft_c2r(ndim, iodims, 0, NULL, src, dst, flags);
    if (cache) {
        plans[key] = plan;
    }

    return plan;
}

void dynd::fftw::fftcleanup() {
    for (map<key_type, void *>::iterator iter = plans.begin(); iter != plans.end(); ++iter) {
        type_id_t tp_id = tr1::get<1>(iter->first);
        if (tp_id == float32_type_id || tp_id == complex_float32_type_id) {
            fftwf_destroy_plan(reinterpret_cast<fftwf_plan>(iter->second));
        } else if (tp_id == float64_type_id || tp_id == complex_float64_type_id) {
            fftw_destroy_plan(reinterpret_cast<fftw_plan>(iter->second));
        } else {
            throw std::runtime_error("");
        }
    }

    fftwf_cleanup();
    fftw_cleanup();
}

nd::array dynd::fftw::fft1(const nd::array &x, unsigned int flags, bool cache) {
    return dynd::fftw::fft(x, x.get_shape(), flags, cache);
}

nd::array dynd::fftw::ifft1(const nd::array &x, unsigned int flags, bool cache) {
    return dynd::fftw::ifft(x, x.get_shape(), flags, cache);
}

nd::array dynd::fftw::fft(const nd::array &x, const std::vector<intptr_t> &shape, bool redundant, unsigned int flags, bool cache) {
    type_id_t dtp_id = x.get_dtype().get_type_id();

    if (dtp_id == complex_float32_type_id) {
        nd::array y = nd::empty_like(x);

        fftwf_complex *src = reinterpret_cast<fftwf_complex *>(x.get_readwrite_originptr());
        fftwf_complex *dst = reinterpret_cast<fftwf_complex *>(y.get_readwrite_originptr());
        fftwf_plan plan = fftw::fftplan(x.get_ndim(), shape, src, x.get_strides(),
            dst, y.get_strides(), FFTW_FORWARD, flags);
        fftwf_execute_dft(plan, src, dst);
        if (!cache) {
            fftwf_destroy_plan(plan);
        }

        return y;
    }

    if (dtp_id == float64_type_id) {
        if (redundant) {
            throw std::runtime_error("cannot");
        }

        vector<intptr_t> dst_shape = shape;
        dst_shape[x.get_ndim() - 1] = dst_shape[x.get_ndim() - 1] / 2 + 1;

        nd::array y = nd::empty(dst_shape[0], dst_shape[1],
            ndt::make_strided_dim(ndt::make_strided_dim(ndt::make_type<dynd_complex<double> >())));

        double *src = reinterpret_cast<double *>(x.get_readwrite_originptr());
        fftw_complex *dst = reinterpret_cast<fftw_complex *>(y.get_readwrite_originptr());
        fftw_plan plan = fftw::fftplan(x.get_ndim(), shape, src, x.get_strides(),
            dst, y.get_strides(), flags, cache);
        fftw_execute_dft_r2c(plan, src, dst);
        if (!cache) {
            fftw_destroy_plan(plan);
        }

        return y;
    }

    if (dtp_id == complex_float64_type_id) {
        nd::array y = nd::empty_like(x);

        fftw_complex *src = reinterpret_cast<fftw_complex *>(x.get_readwrite_originptr());
        fftw_complex *dst = reinterpret_cast<fftw_complex *>(y.get_readwrite_originptr());
        fftw_plan plan = fftw::fftplan(x.get_ndim(), shape, src, x.get_strides(),
            dst, y.get_strides(), FFTW_FORWARD, flags, cache);
        fftw_execute_dft(plan, src, dst);
        if (!cache) {
            fftw_destroy_plan(plan);
        }

        return y;
    }

    throw std::runtime_error("");
}

nd::array dynd::fftw::ifft(const nd::array &x, const vector<intptr_t> &shape, bool redundant, unsigned int flags, bool cache) {
    type_id_t dtp_id = x.get_dtype().get_type_id();

    if (dtp_id == complex_float32_type_id) {
        nd::array y = nd::empty_like(x);

        fftwf_complex *src = reinterpret_cast<fftwf_complex *>(x.get_readwrite_originptr());
        fftwf_complex *dst = reinterpret_cast<fftwf_complex *>(y.get_readwrite_originptr());
        fftwf_plan plan = fftw::fftplan(x.get_ndim(), x.get_shape(), src, x.get_strides(),
            dst, y.get_strides(), FFTW_BACKWARD, flags, cache);
        fftwf_execute_dft(plan, src, dst);
        if (!cache) {
            fftwf_destroy_plan(plan);
        }

        return y;
    }

    if (dtp_id == complex_float64_type_id) {
        if (redundant) {
            nd::array y = nd::empty_like(x);

            fftw_complex *src = reinterpret_cast<fftw_complex *>(x.get_readwrite_originptr());
            fftw_complex *dst = reinterpret_cast<fftw_complex *>(y.get_readwrite_originptr());
            fftw_plan plan = fftw::fftplan(x.get_ndim(), shape, src, x.get_strides(),
                dst, y.get_strides(), FFTW_BACKWARD, flags, cache);
            fftw_execute_dft(plan, src, dst);
            if (!cache) {
                fftw_destroy_plan(plan);
            }

            return y;
        }

        nd::array y = nd::empty(shape[0], shape[1],
            ndt::make_strided_dim(ndt::make_strided_dim(ndt::make_type<double>())));

        fftw_complex *src = reinterpret_cast<fftw_complex *>(x.get_readwrite_originptr());
        double *dst = reinterpret_cast<double *>(y.get_readwrite_originptr());
        fftw_plan plan = fftw::fftplan(x.get_ndim(), shape, src, x.get_strides(),
            dst, y.get_strides(), flags, cache);
        fftw_execute_dft_c2r(plan, src, dst);
        if (!cache) {
            fftw_destroy_plan(plan);
        }

        return y;
    }

    throw std::runtime_error("");
}

#endif // DYND_FFTW

nd::array dynd::fftshift(const nd::array &x) {
    return x;
}
