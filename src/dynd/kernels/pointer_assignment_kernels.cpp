//
// Copyright (C) 2011-14 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <dynd/kernels/pointer_assignment_kernels.hpp>

using namespace std;

namespace {
    template <typename T>
    struct value_to_pointer_ck : dynd::kernels::expr_ck<value_to_pointer_ck<T>, kernel_request_host, 1> {
        void single(char *dst, const char *const *src) {
            *reinterpret_cast<T **>(dst) = const_cast<T *>(*reinterpret_cast<const T *const *>(src));
        }
    };
} // anonymous namespace

size_t dynd::make_builtin_value_to_pointer_assignment_kernel(
                ckernel_builder *ckb, intptr_t ckb_offset,
                type_id_t tp_id, kernel_request_t kernreq)
{
    switch (tp_id) {
    case bool_type_id:
        value_to_pointer_ck<dynd_bool>::create(ckb, kernreq, ckb_offset);
        break;
    case int8_type_id:
        value_to_pointer_ck<int8_t>::create(ckb, kernreq, ckb_offset);
        break;
    case int16_type_id:
        value_to_pointer_ck<int16_t>::create(ckb, kernreq, ckb_offset);
        break;
    case int32_type_id:
        value_to_pointer_ck<int32_t>::create(ckb, kernreq, ckb_offset);
        break;
    case int64_type_id:
        value_to_pointer_ck<int64_t>::create(ckb, kernreq, ckb_offset);
        break;
    case int128_type_id:
        value_to_pointer_ck<dynd_int128>::create(ckb, kernreq, ckb_offset);
        break;
    case uint8_type_id:
        value_to_pointer_ck<uint8_t>::create(ckb, kernreq, ckb_offset);
        break;
    case uint16_type_id:
        value_to_pointer_ck<uint16_t>::create(ckb, kernreq, ckb_offset);
        break;
    case uint32_type_id:
        value_to_pointer_ck<uint32_t>::create(ckb, kernreq, ckb_offset);
        break;
    case uint64_type_id:
        value_to_pointer_ck<uint64_t>::create(ckb, kernreq, ckb_offset);
        break;
    case uint128_type_id:
        value_to_pointer_ck<dynd_uint128>::create(ckb, kernreq, ckb_offset);
        break;
    case float16_type_id:
        value_to_pointer_ck<dynd_float16>::create(ckb, kernreq, ckb_offset);
        break;
    case float32_type_id:
        value_to_pointer_ck<float>::create(ckb, kernreq, ckb_offset);
        break;
    case float64_type_id:
        value_to_pointer_ck<double>::create(ckb, kernreq, ckb_offset);
        break;
    case float128_type_id:
        value_to_pointer_ck<dynd_float128>::create(ckb, kernreq, ckb_offset);
        break;
    case complex_float32_type_id:
        value_to_pointer_ck<dynd_complex<float> >::create(ckb, kernreq, ckb_offset);
        break;
    case complex_float64_type_id:
        value_to_pointer_ck<dynd_complex<double> >::create(ckb, kernreq, ckb_offset);
        break;
    default: {
        stringstream ss;
        ss << "make_builtin_value_to_pointer_assignment_kernel: unrecognized type_id " << tp_id;
        throw runtime_error(ss.str());
        break;
    }
    }
    return ckb_offset;
}

size_t dynd::make_value_to_pointer_assignment_kernel(
                ckernel_builder *ckb, intptr_t ckb_offset,
                const ndt::type &tp, kernel_request_t kernreq)
{
    if (tp.is_builtin()) {
        return make_builtin_value_to_pointer_assignment_kernel(ckb, ckb_offset,
            tp.get_type_id(), kernreq);
    }

    stringstream ss;
    ss << "make_value_to_pointer_assignment_kernel: unrecognized type " << tp;
    throw runtime_error(ss.str());
}
