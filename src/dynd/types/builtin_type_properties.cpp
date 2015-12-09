//
// Copyright (C) 2011-15 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <dynd/types/builtin_type_properties.hpp>
#include <dynd/kernels/assignment_kernels.hpp>
#include <dynd/kernels/real_kernel.hpp>
#include <dynd/kernels/imag_kernel.hpp>
#include <dynd/types/property_type.hpp>
#include <dynd/func/callable.hpp>
#include <dynd/func/elwise.hpp>

using namespace std;
using namespace dynd;

namespace dynd {
namespace nd {

  struct complex_conj_kernel : nd::base_kernel<complex_conj_kernel> {
    static const size_t data_size = 0;

    array self;

    complex_conj_kernel(const array &self) : self(self) {}

    void single(array *dst, array *const *DYND_UNUSED(src)) { *dst = helper(self); }

    static void resolve_dst_type(char *DYND_UNUSED(static_data), char *DYND_UNUSED(data), ndt::type &dst_tp,
                                 intptr_t DYND_UNUSED(nsrc), const ndt::type *DYND_UNUSED(src_tp),
                                 intptr_t DYND_UNUSED(nkwd), const array *kwds,
                                 const std::map<std::string, ndt::type> &DYND_UNUSED(tp_vars))
    {
      dst_tp = helper(kwds[0]).get_type();
    }

    static intptr_t instantiate(char *DYND_UNUSED(static_data), char *DYND_UNUSED(data), void *ckb, intptr_t ckb_offset,
                                const ndt::type &DYND_UNUSED(dst_tp), const char *DYND_UNUSED(dst_arrmeta),
                                intptr_t DYND_UNUSED(nsrc), const ndt::type *DYND_UNUSED(src_tp),
                                const char *const *DYND_UNUSED(src_arrmeta), kernel_request_t kernreq,
                                const eval::eval_context *DYND_UNUSED(ectx), intptr_t DYND_UNUSED(nkwd),
                                const array *kwds, const std::map<std::string, ndt::type> &DYND_UNUSED(tp_vars))
    {
      complex_conj_kernel::make(ckb, kernreq, ckb_offset, kwds[0]);
      return ckb_offset;
    }

    static nd::array helper(const nd::array &n)
    {
      return n.replace_dtype(ndt::property_type::make(n.get_dtype(), "conj"));
    }
  };

} // namespace dynd::nd

namespace ndt {

  template <>
  struct type::equivalent<nd::complex_conj_kernel> {
    static type make() { return type("(self: Any) -> Any"); }
  };

} // namespace dynd::ndt
} // namespace dynd

const std::map<std::string, nd::callable> &complex32_array_properties()
{
  static const std::map<std::string, nd::callable> complex_array_properties{
      {"real", nd::functional::elwise(nd::callable::make<nd::real_kernel<float32_type_id>>())},
      {"imag", nd::functional::elwise(nd::callable::make<nd::imag_kernel<float32_type_id>>())},
      {"conj", nd::callable::make<nd::complex_conj_kernel>()}};

  return complex_array_properties;
}

const std::map<std::string, nd::callable> &complex64_array_properties()
{
  static const std::map<std::string, nd::callable> complex_array_properties{
      {"real", nd::functional::elwise(nd::callable::make<nd::real_kernel<float64_type_id>>())},
      {"imag", nd::functional::elwise(nd::callable::make<nd::imag_kernel<float64_type_id>>())},
      {"conj", nd::callable::make<nd::complex_conj_kernel>()}};

  return complex_array_properties;
}

void dynd::get_builtin_type_dynamic_array_properties(type_id_t builtin_type_id,
                                                     std::map<std::string, nd::callable> &properties)
{
  switch (builtin_type_id) {
  case complex_float32_type_id:
    properties = complex32_array_properties();
    break;
  case complex_float64_type_id:
    properties = complex64_array_properties();
    break;
  default:
    break;
  }
}

size_t dynd::get_builtin_type_elwise_property_index(type_id_t builtin_type_id, const std::string &property_name)
{
  switch (builtin_type_id) {
  case complex_float32_type_id:
  case complex_float64_type_id:
    if (property_name == "real") {
      return 0;
    }
    else if (property_name == "imag") {
      return 1;
    }
    else if (property_name == "conj") {
      return 2;
    }
  default:
    break;
  }

  std::stringstream ss;
  ss << "the dynd type " << ndt::type(builtin_type_id);
  ss << " doesn't have a property \"" << property_name << "\"";
  throw std::runtime_error(ss.str());
}

ndt::type dynd::get_builtin_type_elwise_property_type(type_id_t builtin_type_id, size_t elwise_property_index,
                                                      bool &out_readable, bool &out_writable)
{
  switch (builtin_type_id) {
  case complex_float32_type_id:
    switch (elwise_property_index) {
    case 0: // real
    case 1: // imag
      out_readable = true;
      out_writable = false;
      return ndt::type(float32_type_id);
    case 2: // conj
      out_readable = true;
      out_writable = true;
      return ndt::type(complex_float32_type_id);
    default:
      break;
    }
    break;
  case complex_float64_type_id:
    switch (elwise_property_index) {
    case 0: // real
    case 1: // imag
      out_readable = true;
      out_writable = false;
      return ndt::type(float64_type_id);
    case 2: // conj
      out_readable = true;
      out_writable = true;
      return ndt::type(complex_float64_type_id);
    default:
      break;
    }
    break;
  default:
    break;
  }
  out_readable = false;
  out_writable = false;
  return ndt::type();
}

struct get_property_kernel_complex_float32_imag_kernel
    : nd::base_kernel<get_property_kernel_complex_float32_imag_kernel, 1> {
  void single(char *dst, char *const *src)
  {
    *reinterpret_cast<uint32_t *>(dst) = (*reinterpret_cast<uint32_t *const *>(src))[1];
  }
};

struct get_property_kernel_complex_float64_imag_kernel
    : nd::base_kernel<get_property_kernel_complex_float64_imag_kernel, 1> {
  void single(char *dst, char *const *src)
  {
    *reinterpret_cast<uint64_t *>(dst) = (*reinterpret_cast<uint64_t *const *>(src))[1];
  }
};

struct get_or_set_property_kernel_complex_float32_conj_kernel
    : nd::base_kernel<get_or_set_property_kernel_complex_float32_conj_kernel, 1> {
  void single(char *dst, char *const *src)
  {
    dynd::complex<float> value = **reinterpret_cast<dynd::complex<float> *const *>(src);
    *reinterpret_cast<dynd::complex<float> *>(dst) = dynd::complex<float>(value.real(), -value.imag());
  }
};

struct get_or_set_property_kernel_complex_float64_conj_kernel
    : nd::base_kernel<get_or_set_property_kernel_complex_float64_conj_kernel, 1> {
  void single(char *dst, char *const *src)
  {
    dynd::complex<double> value = **reinterpret_cast<dynd::complex<double> *const *>(src);
    *reinterpret_cast<dynd::complex<double> *>(dst) = dynd::complex<double>(value.real(), -value.imag());
  }
};

size_t dynd::make_builtin_type_elwise_property_getter_kernel(void *ckb, intptr_t ckb_offset, type_id_t builtin_type_id,
                                                             const char *dst_arrmeta, const char *src_arrmeta,
                                                             size_t src_elwise_property_index, kernel_request_t kernreq,
                                                             const eval::eval_context *ectx)
{
  switch (builtin_type_id) {
  case complex_float32_type_id:
    switch (src_elwise_property_index) {
    case 0:
      return nd::real_kernel<float32_type_id>::instantiate(NULL, NULL, ckb, ckb_offset, ndt::type(), dst_arrmeta, 1,
                                                           NULL, &src_arrmeta, kernreq, ectx, 0, NULL,
                                                           std::map<std::string, ndt::type>());
    case 1:
      return nd::imag_kernel<float32_type_id>::instantiate(NULL, NULL, ckb, ckb_offset, ndt::type(), dst_arrmeta, 1,
                                                           NULL, &src_arrmeta, kernreq, ectx, 0, NULL,
                                                           std::map<std::string, ndt::type>());
    case 2:
      return get_or_set_property_kernel_complex_float32_conj_kernel::instantiate(
          NULL, NULL, ckb, ckb_offset, ndt::type(), dst_arrmeta, 1, NULL, &src_arrmeta, kernreq, ectx, 0, NULL,
          std::map<std::string, ndt::type>());
    default:
      break;
    }
    break;
  case complex_float64_type_id:
    switch (src_elwise_property_index) {
    case 0:
      return nd::real_kernel<float64_type_id>::instantiate(NULL, NULL, ckb, ckb_offset, ndt::type(), dst_arrmeta, 1,
                                                           NULL, &src_arrmeta, kernreq, ectx, 0, NULL,
                                                           std::map<std::string, ndt::type>());
    case 1:
      return nd::real_kernel<float64_type_id>::instantiate(NULL, NULL, ckb, ckb_offset, ndt::type(), dst_arrmeta, 1,
                                                           NULL, &src_arrmeta, kernreq, ectx, 0, NULL,
                                                           std::map<std::string, ndt::type>());
    case 2:
      return get_or_set_property_kernel_complex_float64_conj_kernel::instantiate(
          NULL, NULL, ckb, ckb_offset, ndt::type(), dst_arrmeta, 1, NULL, &src_arrmeta, kernreq, ectx, 0, NULL,
          std::map<std::string, ndt::type>());
    default:
      break;
    }
    break;
  default:
    break;
  }
  stringstream ss;
  ss << "dynd type " << ndt::type(builtin_type_id) << " given an invalid property index " << src_elwise_property_index;
  throw runtime_error(ss.str());
}

size_t dynd::make_builtin_type_elwise_property_setter_kernel(void *ckb, intptr_t ckb_offset, type_id_t builtin_type_id,
                                                             const char *dst_arrmeta, size_t dst_elwise_property_index,
                                                             const char *src_arrmeta, kernel_request_t kernreq,
                                                             const eval::eval_context *ectx)
{
  switch (builtin_type_id) {
  case complex_float32_type_id:
    switch (dst_elwise_property_index) {
    case 2:
      return get_or_set_property_kernel_complex_float32_conj_kernel::instantiate(
          NULL, NULL, ckb, ckb_offset, ndt::type(), dst_arrmeta, 1, NULL, &src_arrmeta, kernreq, ectx, 0, NULL,
          std::map<std::string, ndt::type>());
    default:
      break;
    }
    break;
  case complex_float64_type_id:
    switch (dst_elwise_property_index) {
    case 2:
      return get_or_set_property_kernel_complex_float64_conj_kernel::instantiate(
          NULL, NULL, ckb, ckb_offset, ndt::type(), dst_arrmeta, 1, NULL, &src_arrmeta, kernreq, ectx, 0, NULL,
          std::map<std::string, ndt::type>());
    default:
      break;
    }
    break;
  default:
    break;
  }
  stringstream ss;
  ss << "dynd type " << ndt::type(builtin_type_id) << " given an invalid property index " << dst_elwise_property_index;
  throw runtime_error(ss.str());
}
