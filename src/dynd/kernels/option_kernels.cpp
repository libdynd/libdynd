//
// Copyright (C) 2011-15 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <dynd/type.hpp>
#include <dynd/kernels/option_kernels.hpp>
#include <dynd/kernels/assignment_kernels.hpp>
#include <dynd/types/option_type.hpp>
#include <dynd/types/typevar_type.hpp>

using namespace std;
using namespace dynd;

void nd::is_avail_ck<bool1>::single(char *dst, char *const *src)
{
  // Available if the value is 0 or 1
  *dst = **reinterpret_cast<unsigned char *const *>(src) <= 1;
}

void nd::is_avail_ck<bool1>::strided(char *dst, intptr_t dst_stride,
                                     char *const *src,
                                     const intptr_t *src_stride, size_t count)
{
  // Available if the value is 0 or 1
  char *src0 = src[0];
  intptr_t src0_stride = src_stride[0];
  for (size_t i = 0; i != count; ++i) {
    *dst = *reinterpret_cast<unsigned char *>(src0) <= 1;
    dst += dst_stride;
    src0 += src0_stride;
  }
}

void nd::is_avail_ck<float>::single(char *dst, char *const *src)
{
  *dst = dynd::isnan(**reinterpret_cast<float *const *>(src)) == 0;
}

void nd::is_avail_ck<float>::strided(char *dst, intptr_t dst_stride,
                                     char *const *src,
                                     const intptr_t *src_stride, size_t count)
{
  char *src0 = src[0];
  intptr_t src0_stride = src_stride[0];
  for (size_t i = 0; i != count; ++i) {
    *dst = dynd::isnan(*reinterpret_cast<float *>(src0)) == 0;
    dst += dst_stride;
    src0 += src0_stride;
  }
}

void nd::is_avail_ck<double>::single(char *dst, char *const *src)
{
  *dst = dynd::isnan(**reinterpret_cast<double *const *>(src)) == 0;
}

void nd::is_avail_ck<double>::strided(char *dst, intptr_t dst_stride,
                                      char *const *src,
                                      const intptr_t *src_stride, size_t count)
{
  char *src0 = src[0];
  intptr_t src0_stride = src_stride[0];
  for (size_t i = 0; i != count; ++i) {
    *dst = dynd::isnan(*reinterpret_cast<double *>(src0)) == 0;
    dst += dst_stride;
    src0 += src0_stride;
  }
}

void nd::is_avail_ck<dynd::complex<float>>::single(char *dst, char *const *src)
{
  *dst =
      (*reinterpret_cast<uint32_t *const *>(src))[0] !=
          DYND_FLOAT32_NA_AS_UINT &&
      (*reinterpret_cast<uint32_t *const *>(src))[1] != DYND_FLOAT32_NA_AS_UINT;
}

void nd::is_avail_ck<dynd::complex<float>>::strided(char *dst,
                                                    intptr_t dst_stride,
                                                    char *const *src,
                                                    const intptr_t *src_stride,
                                                    size_t count)
{
  char *src0 = src[0];
  intptr_t src0_stride = src_stride[0];
  for (size_t i = 0; i != count; ++i) {
    *dst = reinterpret_cast<uint32_t *>(src0)[0] != DYND_FLOAT32_NA_AS_UINT &&
           reinterpret_cast<uint32_t *>(src0)[1] != DYND_FLOAT32_NA_AS_UINT;
    dst += dst_stride;
    src0 += src0_stride;
  }
}

void nd::is_avail_ck<dynd::complex<double>>::single(char *dst, char *const *src)
{
  *dst =
      (*reinterpret_cast<uint64_t *const *>(src))[0] !=
          DYND_FLOAT64_NA_AS_UINT &&
      (*reinterpret_cast<uint64_t *const *>(src))[1] != DYND_FLOAT64_NA_AS_UINT;
}

void nd::is_avail_ck<dynd::complex<double>>::strided(char *dst,
                                                     intptr_t dst_stride,
                                                     char *const *src,
                                                     const intptr_t *src_stride,
                                                     size_t count)
{
  // Available if the value is 0 or 1
  char *src0 = src[0];
  intptr_t src0_stride = src_stride[0];
  for (size_t i = 0; i != count; ++i) {
    *dst = reinterpret_cast<uint64_t *>(src0)[0] != DYND_FLOAT64_NA_AS_UINT &&
           reinterpret_cast<uint64_t *>(src0)[1] != DYND_FLOAT64_NA_AS_UINT;
    dst += dst_stride;
    src0 += src0_stride;
  }
}

void nd::is_avail_ck<void>::single(char *dst, char *const *DYND_UNUSED(src))
{
  *dst = 0;
}

void nd::is_avail_ck<void>::strided(char *dst, intptr_t dst_stride,
                                    char *const *DYND_UNUSED(src),
                                    const intptr_t *DYND_UNUSED(src_stride),
                                    size_t count)
{
  // Available if the value is 0 or 1
  for (size_t i = 0; i != count; ++i) {
    *dst = 0;
    dst += dst_stride;
  }
}

void nd::assign_na_ck<bool1>::single(char *dst, char *const *DYND_UNUSED(src))
{
  *dst = 2;
}

void nd::assign_na_ck<bool1>::strided(char *dst, intptr_t dst_stride,
                                      char *const *DYND_UNUSED(src),
                                      const intptr_t *DYND_UNUSED(src_stride),
                                      size_t count)
{
  if (dst_stride == 1) {
    memset(dst, 2, count);
  } else {
    for (size_t i = 0; i != count; ++i, dst += dst_stride) {
      *dst = 2;
    }
  }
}

void nd::assign_na_ck<float>::single(char *dst, char *const *DYND_UNUSED(src))
{
  *reinterpret_cast<uint32_t *>(dst) = DYND_FLOAT32_NA_AS_UINT;
}

void nd::assign_na_ck<float>::strided(char *dst, intptr_t dst_stride,
                                      char *const *DYND_UNUSED(src),
                                      const intptr_t *DYND_UNUSED(src_stride),
                                      size_t count)
{
  for (size_t i = 0; i != count; ++i, dst += dst_stride) {
    *reinterpret_cast<uint32_t *>(dst) = DYND_FLOAT32_NA_AS_UINT;
  }
}

void nd::assign_na_ck<double>::single(char *dst, char *const *DYND_UNUSED(src))
{
  *reinterpret_cast<uint64_t *>(dst) = DYND_FLOAT64_NA_AS_UINT;
}

void nd::assign_na_ck<double>::strided(char *dst, intptr_t dst_stride,
                                       char *const *DYND_UNUSED(src),
                                       const intptr_t *DYND_UNUSED(src_stride),
                                       size_t count)
{
  for (size_t i = 0; i != count; ++i, dst += dst_stride) {
    *reinterpret_cast<uint64_t *>(dst) = DYND_FLOAT64_NA_AS_UINT;
  }
}

void
nd::assign_na_ck<dynd::complex<float>>::single(char *dst,
                                               char *const *DYND_UNUSED(src))
{
  reinterpret_cast<uint32_t *>(dst)[0] = DYND_FLOAT32_NA_AS_UINT;
  reinterpret_cast<uint32_t *>(dst)[1] = DYND_FLOAT32_NA_AS_UINT;
}

void nd::assign_na_ck<dynd::complex<float>>::strided(
    char *dst, intptr_t dst_stride, char *const *DYND_UNUSED(src),
    const intptr_t *DYND_UNUSED(src_stride), size_t count)
{
  for (size_t i = 0; i != count; ++i, dst += dst_stride) {
    reinterpret_cast<uint32_t *>(dst)[0] = DYND_FLOAT32_NA_AS_UINT;
    reinterpret_cast<uint32_t *>(dst)[1] = DYND_FLOAT32_NA_AS_UINT;
  }
}

void
nd::assign_na_ck<dynd::complex<double>>::single(char *dst,
                                                char *const *DYND_UNUSED(src))
{
  reinterpret_cast<uint64_t *>(dst)[0] = DYND_FLOAT64_NA_AS_UINT;
  reinterpret_cast<uint64_t *>(dst)[1] = DYND_FLOAT64_NA_AS_UINT;
}

void nd::assign_na_ck<dynd::complex<double>>::strided(
    char *dst, intptr_t dst_stride, char *const *DYND_UNUSED(src),
    const intptr_t *DYND_UNUSED(src_stride), size_t count)
{
  for (size_t i = 0; i != count; ++i, dst += dst_stride) {
    reinterpret_cast<uint64_t *>(dst)[0] = DYND_FLOAT64_NA_AS_UINT;
    reinterpret_cast<uint64_t *>(dst)[1] = DYND_FLOAT64_NA_AS_UINT;
  }
}

void nd::assign_na_ck<void>::single(char *DYND_UNUSED(dst),
                                    char *const *DYND_UNUSED(src))
{
}

void nd::assign_na_ck<void>::strided(char *DYND_UNUSED(dst),
                                     intptr_t DYND_UNUSED(dst_stride),
                                     char *const *DYND_UNUSED(src),
                                     const intptr_t *DYND_UNUSED(src_stride),
                                     size_t DYND_UNUSED(count))
{
}

//////////////////////////////////////
// option[T] for signed integer T
// NA is the smallest negative value

//////////////////////////////////////
// option[float]
// NA is 0x7f8007a2
// Special rule adopted from R: Any NaN is NA

//////////////////////////////////////
// option[double]
// NA is 0x7ff00000000007a2ULL
// Special rule adopted from R: Any NaN is NA

//////////////////////////////////////
// option[complex[float]]
// NA is two float NAs

//////////////////////////////////////
// option[complex[double]]
// NA is two double NAs

//////////////////////////////////////
// option[pointer[T]]

// c{is_avail : (T) -> bool, assign_na : () -> T}
// naf.p("assign_na").vals() =
// nd::as_arrfunc<nd::assign_na_ck<T>>(naf.p("assign_na").get_type(), 0);

template <typename T>
struct nafunc {
  typedef typename std::remove_pointer<T>::type nafunc_type;

  static nd::array get()
  {
    nd::array naf = nd::empty(ndt::option_type::make_nafunc_type());
    arrfunc_type_data *is_avail =
        reinterpret_cast<arrfunc_type_data *>(naf.get_ndo()->m_data_pointer);
    arrfunc_type_data *assign_na = is_avail + 1;

    new (is_avail) arrfunc_type_data(0, nd::is_avail_ck<T>::instantiate, NULL,
                                     nd::is_avail_ck<T>::resolve_dst_type);
    new (assign_na)
        arrfunc_type_data(0, nd::assign_na_ck<T>::instantiate, NULL, NULL);
    return naf;
  }
};

intptr_t kernels::fixed_dim_is_avail_ck::instantiate(
    const arrfunc_type_data *DYND_UNUSED(self),
    const ndt::arrfunc_type *DYND_UNUSED(af_tp), char *DYND_UNUSED(static_data),
    size_t DYND_UNUSED(data_size), char *DYND_UNUSED(data), void *ckb,
    intptr_t ckb_offset, const ndt::type &DYND_UNUSED(dst_tp),
    const char *DYND_UNUSED(dst_arrmeta), intptr_t DYND_UNUSED(nsrc),
    const ndt::type *src_tp, const char *const *DYND_UNUSED(src_arrmeta),
    kernel_request_t kernreq, const eval::eval_context *DYND_UNUSED(ectx),
    const nd::array &DYND_UNUSED(kwds),
    const std::map<nd::string, ndt::type> &DYND_UNUSED(tp_vars))
{
  switch (src_tp->get_dtype().get_type_id()) {
  case bool_type_id:
    nd::is_avail_ck<bool1>::make(ckb, kernreq, ckb_offset);
    return ckb_offset;
  case int8_type_id:
    nd::is_avail_ck<int8_t>::make(ckb, kernreq, ckb_offset);
    return ckb_offset;
  case int16_type_id:
    nd::is_avail_ck<int16_t>::make(ckb, kernreq, ckb_offset);
    return ckb_offset;
  case int32_type_id:
    nd::is_avail_ck<int32_t>::make(ckb, kernreq, ckb_offset);
    return ckb_offset;
  case int64_type_id:
    nd::is_avail_ck<int64_t>::make(ckb, kernreq, ckb_offset);
    return ckb_offset;
  case int128_type_id:
    nd::is_avail_ck<int128>::make(ckb, kernreq, ckb_offset);
    return ckb_offset;
  case float32_type_id:
    nd::is_avail_ck<float>::make(ckb, kernreq, ckb_offset);
    return ckb_offset;
  case float64_type_id:
    nd::is_avail_ck<double>::make(ckb, kernreq, ckb_offset);
    return ckb_offset;
  case complex_float32_type_id:
    nd::is_avail_ck<dynd::complex<float>>::make(ckb, kernreq, ckb_offset);
    return ckb_offset;
  case complex_float64_type_id:
    nd::is_avail_ck<dynd::complex<double>>::make(ckb, kernreq, ckb_offset);
    return ckb_offset;
  default:
    throw type_error("fixed_dim_is_avail: expected built-in type");
  }
}

intptr_t kernels::fixed_dim_assign_na_ck::instantiate(
    const arrfunc_type_data *DYND_UNUSED(self),
    const ndt::arrfunc_type *DYND_UNUSED(af_tp), char *DYND_UNUSED(static_data),
    size_t DYND_UNUSED(data_size), char *DYND_UNUSED(data), void *ckb,
    intptr_t ckb_offset, const ndt::type &dst_tp,
    const char *DYND_UNUSED(dst_arrmeta), intptr_t DYND_UNUSED(nsrc),
    const ndt::type *DYND_UNUSED(src_tp),
    const char *const *DYND_UNUSED(src_arrmeta), kernel_request_t kernreq,
    const eval::eval_context *DYND_UNUSED(ectx),
    const nd::array &DYND_UNUSED(kwds),
    const std::map<nd::string, ndt::type> &DYND_UNUSED(tp_vars))
{
  switch (dst_tp.get_dtype().get_type_id()) {
  case bool_type_id:
    nd::assign_na_ck<bool1>::make(ckb, kernreq, ckb_offset);
    return ckb_offset;
  case int8_type_id:
    nd::assign_na_ck<int8_t>::make(ckb, kernreq, ckb_offset);
    return ckb_offset;
  case int16_type_id:
    nd::assign_na_ck<int16_t>::make(ckb, kernreq, ckb_offset);
    return ckb_offset;
  case int32_type_id:
    nd::assign_na_ck<int32_t>::make(ckb, kernreq, ckb_offset);
    return ckb_offset;
  case int64_type_id:
    nd::assign_na_ck<int64_t>::make(ckb, kernreq, ckb_offset);
    return ckb_offset;
  case int128_type_id:
    nd::assign_na_ck<int128>::make(ckb, kernreq, ckb_offset);
    return ckb_offset;
  case float32_type_id:
    nd::assign_na_ck<float>::make(ckb, kernreq, ckb_offset);
    return ckb_offset;
  case float64_type_id:
    nd::assign_na_ck<double>::make(ckb, kernreq, ckb_offset);
    return ckb_offset;
  case complex_float32_type_id:
    nd::assign_na_ck<dynd::complex<float>>::make(ckb, kernreq, ckb_offset);
    return ckb_offset;
  case complex_float64_type_id:
    nd::assign_na_ck<dynd::complex<double>>::make(ckb, kernreq, ckb_offset);
    return ckb_offset;
  default:
    throw type_error("fixed_dim_assign_na: expected built-in type");
  }
}

const nd::array &kernels::get_option_builtin_nafunc(type_id_t tid)
{
  static nd::array bna = nafunc<bool1>::get();
  static nd::array i8na = nafunc<int8_t>::get();
  static nd::array i16na = nafunc<int16_t>::get();
  static nd::array i32na = nafunc<int32_t>::get();
  static nd::array i64na = nafunc<int64_t>::get();
  static nd::array i128na = nafunc<int128>::get();
  static nd::array f32na = nafunc<float>::get();
  static nd::array f64na = nafunc<double>::get();
  static nd::array cf32na = nafunc<dynd::complex<float>>::get();
  static nd::array cf64na = nafunc<dynd::complex<double>>::get();
  static nd::array vna = nafunc<void>::get();
  static nd::array nullarr;
  switch (tid) {
  case bool_type_id:
    return bna;
  case int8_type_id:
    return i8na;
  case int16_type_id:
    return i16na;
  case int32_type_id:
    return i32na;
  case int64_type_id:
    return i64na;
  case int128_type_id:
    return i128na;
  case float32_type_id:
    return f32na;
  case float64_type_id:
    return f64na;
  case complex_float32_type_id:
    return cf32na;
  case complex_float64_type_id:
    return cf64na;
  case void_type_id:
    return vna;
  default:
    return nullarr;
  }
}

const nd::array &kernels::get_option_builtin_pointer_nafunc(type_id_t tid)
{
  static nd::array bna = nafunc<bool1 *>::get();
  static nd::array i8na = nafunc<int8_t *>::get();
  static nd::array i16na = nafunc<int16_t *>::get();
  static nd::array i32na = nafunc<int32_t *>::get();
  static nd::array i64na = nafunc<int64_t *>::get();
  static nd::array i128na = nafunc<int128 *>::get();
  static nd::array f32na = nafunc<float *>::get();
  static nd::array f64na = nafunc<double *>::get();
  static nd::array cf32na = nafunc<dynd::complex<float> *>::get();
  static nd::array cf64na = nafunc<dynd::complex<double> *>::get();
  static nd::array nullarr;
  switch (tid) {
  case bool_type_id:
    return bna;
  case int8_type_id:
    return i8na;
  case int16_type_id:
    return i16na;
  case int32_type_id:
    return i32na;
  case int64_type_id:
    return i64na;
  case int128_type_id:
    return i128na;
  case float32_type_id:
    return f32na;
  case float64_type_id:
    return f64na;
  case complex_float32_type_id:
    return cf32na;
  case complex_float64_type_id:
    return cf64na;
  default:
    return nullarr;
  }
}
