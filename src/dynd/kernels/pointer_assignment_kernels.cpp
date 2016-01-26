//
// Copyright (C) 2011-15 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <dynd/kernels/pointer_assignment_kernels.hpp>
#include <dynd/callable.hpp>

using namespace std;
using namespace dynd;

namespace {
template <typename T>
struct value_to_pointer_ck : nd::base_kernel<value_to_pointer_ck<T>, 1> {
  void single(char *dst, char *const *src) { *reinterpret_cast<T **>(dst) = *reinterpret_cast<T **>(src[0]); }
};
} // anonymous namespace

void dynd::make_builtin_value_to_pointer_assignment_kernel(nd::kernel_builder *ckb, type_id_t tp_id,
                                                           kernel_request_t kernreq)
{
  switch (tp_id) {
  case bool_type_id:
    ckb->emplace_back<value_to_pointer_ck<bool1>>(kernreq);
    break;
  case int8_type_id:
    ckb->emplace_back<value_to_pointer_ck<int8>>(kernreq);
    break;
  case int16_type_id:
    ckb->emplace_back<value_to_pointer_ck<int16>>(kernreq);
    break;
  case int32_type_id:
    ckb->emplace_back<value_to_pointer_ck<int32>>(kernreq);
    break;
  case int64_type_id:
    ckb->emplace_back<value_to_pointer_ck<int64>>(kernreq);
    break;
  case int128_type_id:
    ckb->emplace_back<value_to_pointer_ck<int128>>(kernreq);
    break;
  case uint8_type_id:
    ckb->emplace_back<value_to_pointer_ck<uint8>>(kernreq);
    break;
  case uint16_type_id:
    ckb->emplace_back<value_to_pointer_ck<uint16>>(kernreq);
    break;
  case uint32_type_id:
    ckb->emplace_back<value_to_pointer_ck<uint32>>(kernreq);
    break;
  case uint64_type_id:
    ckb->emplace_back<value_to_pointer_ck<uint64>>(kernreq);
    break;
  case uint128_type_id:
    ckb->emplace_back<value_to_pointer_ck<uint128>>(kernreq);
    break;
  case float16_type_id:
    ckb->emplace_back<value_to_pointer_ck<float16>>(kernreq);
    break;
  case float32_type_id:
    ckb->emplace_back<value_to_pointer_ck<float32>>(kernreq);
    break;
  case float64_type_id:
    ckb->emplace_back<value_to_pointer_ck<float64>>(kernreq);
    break;
  case float128_type_id:
    ckb->emplace_back<value_to_pointer_ck<float128>>(kernreq);
    break;
  case complex_float32_type_id:
    ckb->emplace_back<value_to_pointer_ck<complex<float>>>(kernreq);
    break;
  case complex_float64_type_id:
    ckb->emplace_back<value_to_pointer_ck<complex<double>>>(kernreq);
    break;
  default: {
    stringstream ss;
    ss << "make_builtin_value_to_pointer_assignment_kernel: unrecognized "
          "type_id "
       << tp_id;
    throw runtime_error(ss.str());
    break;
  }
  }
}

void dynd::make_value_to_pointer_assignment_kernel(nd::kernel_builder *ckb, const ndt::type &tp,
                                                   kernel_request_t kernreq)
{
  if (tp.is_builtin()) {
    make_builtin_value_to_pointer_assignment_kernel(ckb, tp.get_type_id(), kernreq);
    return;
  }

  stringstream ss;
  ss << "make_value_to_pointer_assignment_kernel: unrecognized type " << tp;
  throw runtime_error(ss.str());
}
