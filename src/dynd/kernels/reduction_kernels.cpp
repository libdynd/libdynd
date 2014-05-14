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

static ndt::type builtin_type_pairs[builtin_type_id_count][2] = {
    {ndt::type((type_id_t)0), ndt::type((type_id_t)0)},
    {ndt::type((type_id_t)1), ndt::type((type_id_t)1)},
    {ndt::type((type_id_t)2), ndt::type((type_id_t)2)},
    {ndt::type((type_id_t)3), ndt::type((type_id_t)3)},
    {ndt::type((type_id_t)4), ndt::type((type_id_t)4)},
    {ndt::type((type_id_t)5), ndt::type((type_id_t)5)},
    {ndt::type((type_id_t)6), ndt::type((type_id_t)6)},
    {ndt::type((type_id_t)7), ndt::type((type_id_t)7)},
    {ndt::type((type_id_t)8), ndt::type((type_id_t)8)},
    {ndt::type((type_id_t)9), ndt::type((type_id_t)9)},
    {ndt::type((type_id_t)10), ndt::type((type_id_t)10)},
    {ndt::type((type_id_t)11), ndt::type((type_id_t)11)},
    {ndt::type((type_id_t)12), ndt::type((type_id_t)12)},
    {ndt::type((type_id_t)13), ndt::type((type_id_t)13)},
    {ndt::type((type_id_t)14), ndt::type((type_id_t)14)},
    {ndt::type((type_id_t)15), ndt::type((type_id_t)15)},
    {ndt::type((type_id_t)16), ndt::type((type_id_t)16)},
    {ndt::type((type_id_t)17), ndt::type((type_id_t)17)},
    {ndt::type((type_id_t)18), ndt::type((type_id_t)18)},
};

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
                kernel_request_t kerntype)
{
    ckernel_prefix *ckp = out_ckb->get_at<ckernel_prefix>(ckb_offset);
    if (kerntype == kernel_request_single) {
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
    } else if (kerntype == kernel_request_strided) {
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

static intptr_t instantiate_builtin_sum_reduction_ckernel_deferred(
    void *self_data_ptr, dynd::ckernel_builder *out_ckb, intptr_t ckb_offset,
    const char *const *DYND_UNUSED(dynd_metadata), uint32_t kerntype,
    const eval::eval_context *DYND_UNUSED(ectx))
{
    type_id_t tid = static_cast<type_id_t>(reinterpret_cast<uintptr_t>(self_data_ptr));
    return kernels::make_builtin_sum_reduction_ckernel(out_ckb, ckb_offset, tid, (kernel_request_t)kerntype);
}

void kernels::make_builtin_sum_reduction_ckernel_deferred(
                arrfunc *out_ckd,
                type_id_t tid)
{
    if (tid < 0 || tid >= builtin_type_id_count) {
        stringstream ss;
        ss << "make_builtin_sum_reduction_ckernel: data type ";
        ss << ndt::type(tid) << " is not supported";
        throw type_error(ss.str());
    }
    out_ckd->ckernel_funcproto = unary_operation_funcproto;
    out_ckd->data_types_size = 2;
    out_ckd->data_dynd_types = builtin_type_pairs[tid];
    out_ckd->data_ptr = reinterpret_cast<void *>(tid);
    out_ckd->instantiate_func = &instantiate_builtin_sum_reduction_ckernel_deferred;
    out_ckd->free_func = NULL;
}

nd::array kernels::make_builtin_sum1d_ckernel_deferred(type_id_t tid)
{
    nd::array sum_ew = nd::empty(ndt::make_arrfunc());
    kernels::make_builtin_sum_reduction_ckernel_deferred(
        reinterpret_cast<arrfunc *>(sum_ew.get_readwrite_originptr()),
        tid);
    nd::array sum_1d = nd::empty(ndt::make_arrfunc());
    bool reduction_dimflags[1] = {true};
    lift_reduction_arrfunc(
        reinterpret_cast<arrfunc *>(sum_1d.get_readwrite_originptr()),
        sum_ew, ndt::make_strided_dim(ndt::type(tid)), nd::array(), false, 1,
        reduction_dimflags, true, true, false, 0);
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

    struct mean1d_ckernel_deferred_data {
        ndt::type data_types[2];
        intptr_t minp;

        static void free(void *data_ptr) {
            delete reinterpret_cast<mean1d_ckernel_deferred_data *>(data_ptr);
        }

        static intptr_t instantiate(
            void *self_data_ptr, dynd::ckernel_builder *ckb,
            intptr_t ckb_offset, const char *const *dynd_metadata,
            uint32_t kernreq, const eval::eval_context *DYND_UNUSED(ectx))
        {
            typedef double_mean1d_ck self_type;
            mean1d_ckernel_deferred_data *data =
                reinterpret_cast<mean1d_ckernel_deferred_data *>(self_data_ptr);
            self_type *self = self_type::create_leaf(ckb, ckb_offset,
                                                     (kernel_request_t)kernreq);
            const strided_dim_type_metadata *src_md =
                reinterpret_cast<const strided_dim_type_metadata *>(
                    dynd_metadata[1]);
            self->m_minp = data->minp;
            if (self->m_minp <= 0) {
                if (self->m_minp <= -src_md->size) {
                    throw invalid_argument("minp parameter is too large of a negative number");
                }
                self->m_minp += src_md->size;
            }
            self->m_src_dim_size = src_md->size;
            self->m_src_stride = src_md->stride;
            return ckb_offset + sizeof(self_type);
        }
    };
} // anonymous namespace

nd::array kernels::make_builtin_mean1d_ckernel_deferred(type_id_t tid, intptr_t minp)
{
    if (tid != float64_type_id) {
        stringstream ss;
        ss << "make_builtin_mean1d_ckernel_deferred: data type ";
        ss << ndt::type(tid) << " is not supported";
        throw type_error(ss.str());
    }
    nd::array mean1d = nd::empty(ndt::make_arrfunc());
    arrfunc *out_ckd =
        reinterpret_cast<arrfunc *>(mean1d.get_readwrite_originptr());
    out_ckd->ckernel_funcproto = unary_operation_funcproto;
    out_ckd->data_types_size = 2;
    mean1d_ckernel_deferred_data *data = new mean1d_ckernel_deferred_data;
    data->data_types[0] = ndt::make_type<double>();
    data->data_types[1] = ndt::make_strided_dim(ndt::make_type<double>());
    data->minp = minp;
    out_ckd->data_dynd_types = data->data_types;
    out_ckd->data_ptr = data;
    out_ckd->instantiate_func = &mean1d_ckernel_deferred_data::instantiate;
    out_ckd->free_func = &mean1d_ckernel_deferred_data::free;
    return mean1d;
}
