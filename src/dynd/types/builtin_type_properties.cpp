//
// Copyright (C) 2011-15 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <dynd/types/builtin_type_properties.hpp>
#include <dynd/kernels/assignment_kernels.hpp>
#include <dynd/kernels/real_kernel.hpp>
#include <dynd/kernels/imag_kernel.hpp>
#include <dynd/kernels/conj_kernel.hpp>
#include <dynd/types/property_type.hpp>
#include <dynd/func/callable.hpp>
#include <dynd/func/elwise.hpp>

using namespace std;
using namespace dynd;

const std::map<std::string, nd::callable> &complex32_array_properties()
{
  static const std::map<std::string, nd::callable> complex_array_properties{
      {"real", nd::functional::elwise(nd::callable::make<nd::real_kernel<float32_type_id>>())},
      {"imag", nd::functional::elwise(nd::callable::make<nd::imag_kernel<float32_type_id>>())},
      {"conj", nd::functional::elwise(nd::callable::make<nd::conj_kernel<float32_type_id>>())}};

  return complex_array_properties;
}

const std::map<std::string, nd::callable> &complex64_array_properties()
{
  static const std::map<std::string, nd::callable> complex_array_properties{
      {"real", nd::functional::elwise(nd::callable::make<nd::real_kernel<float64_type_id>>())},
      {"imag", nd::functional::elwise(nd::callable::make<nd::imag_kernel<float64_type_id>>())},
      {"conj", nd::functional::elwise(nd::callable::make<nd::conj_kernel<float64_type_id>>())}};

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
      return nd::conj_kernel<float32_type_id>::instantiate(NULL, NULL, ckb, ckb_offset, ndt::type(), dst_arrmeta, 1,
                                                           NULL, &src_arrmeta, kernreq, ectx, 0, NULL,
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
      return nd::imag_kernel<float64_type_id>::instantiate(NULL, NULL, ckb, ckb_offset, ndt::type(), dst_arrmeta, 1,
                                                           NULL, &src_arrmeta, kernreq, ectx, 0, NULL,
                                                           std::map<std::string, ndt::type>());
    case 2:
      return nd::conj_kernel<float64_type_id>::instantiate(NULL, NULL, ckb, ckb_offset, ndt::type(), dst_arrmeta, 1,
                                                           NULL, &src_arrmeta, kernreq, ectx, 0, NULL,
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
      return nd::conj_kernel<float32_type_id>::instantiate(NULL, NULL, ckb, ckb_offset, ndt::type(), dst_arrmeta, 1,
                                                           NULL, &src_arrmeta, kernreq, ectx, 0, NULL,
                                                           std::map<std::string, ndt::type>());
    default:
      break;
    }
    break;
  case complex_float64_type_id:
    switch (dst_elwise_property_index) {
    case 2:
      return nd::conj_kernel<float64_type_id>::instantiate(NULL, NULL, ckb, ckb_offset, ndt::type(), dst_arrmeta, 1,
                                                           NULL, &src_arrmeta, kernreq, ectx, 0, NULL,
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
