//
// Copyright (C) 2011-14 Mark Wiebe, DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <dynd/kernels/reduction_kernels.hpp>

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
            Accum s = 0;
            for (size_t i = 0; i < count; ++i) {
                s = s + *reinterpret_cast<const T *>(src);
                src += src_stride;
            }
            *reinterpret_cast<T *>(dst) = static_cast<T>(*reinterpret_cast<const T *>(dst) + s);
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

static intptr_t instantiate_builtin_sum_reduction_ckernel_deferred(void *self_data_ptr,
                dynd::ckernel_builder *out_ckb, intptr_t ckb_offset,
                const char *const* DYND_UNUSED(dynd_metadata), uint32_t kerntype)
{
    type_id_t tid = static_cast<type_id_t>(reinterpret_cast<uintptr_t>(self_data_ptr));
    return kernels::make_builtin_sum_reduction_ckernel(out_ckb, ckb_offset, tid, (kernel_request_t)kerntype);
}

void kernels::make_builtin_sum_reduction_ckernel_deferred(
                ckernel_deferred *out_ckd,
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
