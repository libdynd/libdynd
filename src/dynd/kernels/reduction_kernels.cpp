//
// Copyright (C) 2011-14 Mark Wiebe, DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <dynd/kernels/reduction_kernels.hpp>
#include <dynd/array.hpp>
#include <dynd/types/arrfunc_type.hpp>
#include <dynd/types/strided_dim_type.hpp>
#include <dynd/func/lift_reduction_arrfunc.hpp>

using namespace std;
using namespace dynd;

namespace {
    template<class T, class Accum>
    struct sum_reduction {
        static void single(char *dst, const char *src,
                        ckernel_prefix *DYND_UNUSED(ckp))
        {
            *reinterpret_cast<T *>(dst) = *reinterpret_cast<T *>(dst) + *reinterpret_cast<const T *>(src);
        }

        static void strided(char *dst, intptr_t dst_stride,
                        const char *src, intptr_t src_stride,
                        size_t count, ckernel_prefix *DYND_UNUSED(ckp))
        {
            if (dst_stride == 0) {
                Accum s = 0;
                for (size_t i = 0; i < count; ++i) {
                    s = s + *reinterpret_cast<const T *>(src);
                    src += src_stride;
                }
                *reinterpret_cast<T *>(dst) = static_cast<T>(*reinterpret_cast<const T *>(dst) + s);
            } else {
                for (size_t i = 0; i < count; ++i) {
                    *reinterpret_cast<T *>(dst) = *reinterpret_cast<T *>(dst) + *reinterpret_cast<const T *>(src);
                    dst += dst_stride;
                    src += src_stride;
                }
            }
        }
    };
} // anonymous namespace


intptr_t kernels::make_builtin_sum_reduction_ckernel(
                dynd::ckernel_builder *out_ckb, intptr_t ckb_offset,
                type_id_t tid,
                kernel_request_t kernreq)
{
    ckernel_prefix *ckp = out_ckb->get_at<ckernel_prefix>(ckb_offset);
    if (kernreq == kernel_request_single) {
        switch (tid) {
            case int32_type_id:
                ckp->set_function<unary_single_operation_t>(&sum_reduction<int32_t, int32_t>::single);
                break;
            case int64_type_id:
                ckp->set_function<unary_single_operation_t>(&sum_reduction<int64_t, int64_t>::single);
                break;
            case float32_type_id:
                ckp->set_function<unary_single_operation_t>(&sum_reduction<float, double>::single);
                break;
            case float64_type_id:
                ckp->set_function<unary_single_operation_t>(&sum_reduction<double, double>::single);
                break;
            case complex_float32_type_id:
                ckp->set_function<unary_single_operation_t>(&sum_reduction<dynd_complex<float>, dynd_complex<double> >::single);
                break;
            case complex_float64_type_id:
                ckp->set_function<unary_single_operation_t>(&sum_reduction<dynd_complex<double>, dynd_complex<double> >::single);
                break;
            default: {
                stringstream ss;
                ss << "make_builtin_sum_reduction_ckernel: data type ";
                ss << ndt::type(tid) << " is not supported";
                throw type_error(ss.str());
            }
        }
    } else if (kernreq == kernel_request_strided) {
        switch (tid) {
            case int32_type_id:
                ckp->set_function<unary_strided_operation_t>(&sum_reduction<int32_t, int32_t>::strided);
                break;
            case int64_type_id:
                ckp->set_function<unary_strided_operation_t>(&sum_reduction<int64_t, int64_t>::strided);
                break;
            case float32_type_id:
                // For float32, use float64 as the accumulator in the strided loop for a touch more accuracy
                ckp->set_function<unary_strided_operation_t>(&sum_reduction<float, double>::strided);
                break;
            case float64_type_id:
                ckp->set_function<unary_strided_operation_t>(&sum_reduction<double, double>::strided);
                break;
            case complex_float32_type_id:
                // For float32, use float64 as the accumulator in the strided loop for a touch more accuracy
                ckp->set_function<unary_strided_operation_t>(&sum_reduction<dynd_complex<float>, dynd_complex<double> >::strided);
                break;
            case complex_float64_type_id:
                ckp->set_function<unary_strided_operation_t>(&sum_reduction<dynd_complex<double>, dynd_complex<double> >::strided);
                break;
            default: {
                stringstream ss;
                ss << "make_builtin_sum_reduction_ckernel: data type ";
                ss << ndt::type(tid) << " is not supported";
                throw type_error(ss.str());
            }
        }
    } else {
        throw runtime_error("unsupported kernel request in make_builtin_sum_reduction_ckernel");
    }

    return ckb_offset + sizeof(ckernel_prefix);
}

static intptr_t instantiate_builtin_sum_reduction_arrfunc(
    const arrfunc_type_data *DYND_UNUSED(self_data_ptr), dynd::ckernel_builder *ckb,
    intptr_t ckb_offset, const ndt::type &dst_tp,
    const char *DYND_UNUSED(dst_arrmeta), const ndt::type *src_tp,
    const char *const *DYND_UNUSED(src_arrmeta), uint32_t kernreq,
    const eval::eval_context *DYND_UNUSED(ectx))
{
    if (dst_tp != src_tp[0]) {
        stringstream ss;
        ss << "dynd sum reduction: the source type, " << src_tp[0]
           << ", does not match the destination type, " << dst_tp;
        throw type_error(ss.str());
    }
    return kernels::make_builtin_sum_reduction_ckernel(
        ckb, ckb_offset, dst_tp.get_type_id(), (kernel_request_t)kernreq);
}

void kernels::make_builtin_sum_reduction_arrfunc(
                arrfunc_type_data *out_af,
                type_id_t tid)
{
    if (tid < 0 || tid >= builtin_type_id_count) {
        stringstream ss;
        ss << "make_builtin_sum_reduction_ckernel: data type ";
        ss << ndt::type(tid) << " is not supported";
        throw type_error(ss.str());
    }
    out_af->ckernel_funcproto = unary_operation_funcproto;
    out_af->func_proto = ndt::make_funcproto(ndt::type(tid), ndt::type(tid));
    out_af->data_ptr = reinterpret_cast<void *>(tid);
    out_af->instantiate = &instantiate_builtin_sum_reduction_arrfunc;
    out_af->free_func = NULL;
}

nd::arrfunc kernels::make_builtin_sum1d_arrfunc(type_id_t tid)
{
    nd::arrfunc sum_ew = kernels::make_builtin_sum_reduction_arrfunc(tid);
    nd::array sum_1d = nd::empty(ndt::make_arrfunc());
    bool reduction_dimflags[1] = {true};
    lift_reduction_arrfunc(
        reinterpret_cast<arrfunc_type_data *>(sum_1d.get_readwrite_originptr()),
        sum_ew, ndt::make_strided_dim(ndt::type(tid)), nd::array(), false, 1,
        reduction_dimflags, true, true, false, 0);
    sum_1d.flag_as_immutable();
    return sum_1d;
}

namespace {
    struct double_mean1d_ck : public kernels::assignment_ck<double_mean1d_ck> {
        intptr_t m_minp;
        intptr_t m_src_dim_size, m_src_stride;

        inline void single(char *dst, const char *src)
        {
            intptr_t minp = m_minp, countp = 0;
            intptr_t src_dim_size = m_src_dim_size, src_stride = m_src_stride;
            double result = 0;
            for (intptr_t i = 0; i < src_dim_size; ++i) {
                double v = *reinterpret_cast<const double *>(src);
                if (!DYND_ISNAN(v)) {
                    result += v;
                    ++countp;
                }
                src += src_stride;
            }
            if (countp >= minp) {
                *reinterpret_cast<double *>(dst) = result / countp;
            } else {
                *reinterpret_cast<double *>(dst) = numeric_limits<double>::quiet_NaN();
            }
        }
    };

    struct mean1d_arrfunc_data {
        intptr_t minp;

        static void free(void *data_ptr) {
            delete reinterpret_cast<mean1d_arrfunc_data *>(data_ptr);
        }

        static intptr_t
        instantiate(const arrfunc_type_data *af_self, dynd::ckernel_builder *ckb,
                    intptr_t ckb_offset, const ndt::type &DYND_UNUSED(dst_tp),
                    const char *DYND_UNUSED(dst_arrmeta), const ndt::type *src_tp,
                    const char *const *src_arrmeta, uint32_t kernreq,
                    const eval::eval_context *DYND_UNUSED(ectx))
        {
            typedef double_mean1d_ck self_type;
            mean1d_arrfunc_data *data =
                reinterpret_cast<mean1d_arrfunc_data *>(af_self->data_ptr);
            self_type *self = self_type::create_leaf(ckb, ckb_offset,
                                                     (kernel_request_t)kernreq);
            intptr_t src_dim_size, src_stride;
            ndt::type src_el_tp;
            const char *src_el_arrmeta;
            if (!src_tp[0].get_as_strided_dim(src_arrmeta[0], src_dim_size,
                                              src_stride, src_el_tp,
                                              src_el_arrmeta)) {
                stringstream ss;
                ss << "mean1d: could not process type " << src_tp[0];
                ss << " as a strided dimension";
                throw type_error(ss.str());
            }
            self->m_minp = data->minp;
            if (self->m_minp <= 0) {
                if (self->m_minp <= -src_dim_size) {
                    throw invalid_argument("minp parameter is too large of a negative number");
                }
                self->m_minp += src_dim_size;
            }
            self->m_src_dim_size = src_dim_size;
            self->m_src_stride = src_stride;
            return ckb_offset + sizeof(self_type);
        }
    };
} // anonymous namespace

nd::arrfunc kernels::make_builtin_mean1d_arrfunc(type_id_t tid, intptr_t minp)
{
    if (tid != float64_type_id) {
        stringstream ss;
        ss << "make_builtin_mean1d_arrfunc: data type ";
        ss << ndt::type(tid) << " is not supported";
        throw type_error(ss.str());
    }
    nd::array mean1d = nd::empty(ndt::make_arrfunc());
    arrfunc_type_data *out_af =
        reinterpret_cast<arrfunc_type_data *>(mean1d.get_readwrite_originptr());
    out_af->ckernel_funcproto = unary_operation_funcproto;
    mean1d_arrfunc_data *data = new mean1d_arrfunc_data;
    data->minp = minp;
    out_af->func_proto =
        ndt::make_funcproto(ndt::make_strided_dim(ndt::make_type<double>()),
                            ndt::make_type<double>());
    out_af->data_ptr = data;
    out_af->instantiate = &mean1d_arrfunc_data::instantiate;
    out_af->free_func = &mean1d_arrfunc_data::free;
    mean1d.flag_as_immutable();
    return mean1d;
}
