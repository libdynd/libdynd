//
// Copyright (C) 2011-15 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <dynd/type.hpp>
#include <dynd/shape_tools.hpp>
#include <dynd/func/apply.hpp>

#include <dynd/types/expr_type.hpp>
#include <dynd/kernels/string_algorithm_kernels.hpp>
#include <dynd/kernels/expr_kernel_generator.hpp>
#include <dynd/kernels/elwise_expr_kernels.hpp>

using namespace std;
using namespace dynd;

ndt::base_string_type::~base_string_type()
{
}

std::string ndt::base_string_type::get_utf8_string(const char *arrmeta, const char *data,
                                                   assign_error_mode errmode) const
{
  const char *begin, *end;
  get_string_range(&begin, &end, arrmeta, data);
  return string_range_as_utf8_string(get_encoding(), begin, end, errmode);
}

size_t ndt::base_string_type::get_iterdata_size(intptr_t DYND_UNUSED(ndim)) const
{
  return 0;
}

static void get_extended_string_encoding(const ndt::type &dt)
{
  const ndt::base_string_type *d = dt.extended<ndt::base_string_type>();
  stringstream ss;
  ss << d->get_encoding();
  //  return ss.str();
}

static size_t base_string_type_properties_size()
{
  return 1;
}

static const pair<std::string, nd::callable> *base_string_type_properties()
{
  static pair<std::string, nd::callable> base_string_type_properties[1] = {
      pair<std::string, nd::callable>("encoding", nd::functional::apply(&get_extended_string_encoding))};

  return base_string_type_properties;
}

void ndt::base_string_type::get_dynamic_type_properties(const std::pair<std::string, nd::callable> **out_properties,
                                                        size_t *out_count) const
{
  *out_properties = base_string_type_properties();
  *out_count = base_string_type_properties_size();
}

namespace {
// TODO: The representation of deferred operations needs work,
//       this way is too verbose and boilerplatey
class string_find_kernel_generator : public expr_kernel_generator {
  ndt::type m_rdt, m_op1dt, m_op2dt;
  expr_operation_pair m_op_pair;
  const char *m_name;

  typedef kernels::string_find_kernel extra_type;

public:
  string_find_kernel_generator(const ndt::type &rdt, const ndt::type &op1dt, const ndt::type &op2dt,
                               const expr_operation_pair &op_pair, const char *name)
      : expr_kernel_generator(true), m_rdt(rdt), m_op1dt(op1dt), m_op2dt(op2dt), m_op_pair(op_pair), m_name(name)
  {
  }

  virtual ~string_find_kernel_generator()
  {
  }

  size_t make_expr_kernel(void *ckb, intptr_t ckb_offset, const ndt::type &dst_tp, const char *dst_arrmeta,
                          size_t src_count, const ndt::type *src_tp, const char *const *src_arrmeta,
                          kernel_request_t kernreq, const eval::eval_context *ectx) const
  {
    if (src_count != 2) {
      stringstream ss;
      ss << "The " << m_name << " kernel requires 2 src operands, ";
      ss << "received " << src_count;
      throw runtime_error(ss.str());
    }
    if (dst_tp != m_rdt || src_tp[0] != m_op1dt || src_tp[1] != m_op2dt) {
      // If the types don't match the ones for this generator,
      // call the elementwise dimension handler to handle one dimension
      // or handle input/output buffering, giving 'this' as the next
      // kernel generator to call
      return make_elwise_dimension_expr_kernel(ckb, ckb_offset, dst_tp, dst_arrmeta, src_count, src_tp, src_arrmeta,
                                               kernreq, ectx, this);
    }
    // This is a leaf kernel, so no additional allocation is needed
    extra_type *e = reinterpret_cast<ckernel_builder<kernel_request_host> *>(ckb)->alloc_ck<extra_type>(ckb_offset);
    switch (kernreq) {
    case kernel_request_single:
      e->base().function = reinterpret_cast<void *>(m_op_pair.single);
      break;
    case kernel_request_strided:
      e->base().function = reinterpret_cast<void *>(m_op_pair.strided);
      break;
    default: {
      stringstream ss;
      ss << "generic_kernel_generator: unrecognized request " << (int)kernreq;
      throw runtime_error(ss.str());
    }
    }
    e->init(src_tp, src_arrmeta);
    return ckb_offset;
  }

  void print_type(std::ostream &o) const
  {
    o << m_name << "(op0, op1)";
  }
};
} // anonymous namespace

/*
static nd::array array_function_find(const nd::array& self, const nd::array&
sub)
{
    nd::array ops[2] = {self, sub};

    // Get the broadcasted shape
    size_t ndim = max(ops[0].get_ndim(), ops[1].get_ndim());
    dimvector result_shape(ndim), tmp_shape(ndim);
    for (size_t j = 0; j != ndim; ++j) {
        result_shape[j] = 1;
    }
    for (size_t i = 0; i != 2; ++i) {
        size_t ndim_i = ops[i].get_ndim();
        if (ndim_i > 0) {
            ops[i].get_shape(tmp_shape.get());
            incremental_broadcast(ndim, result_shape.get(), ndim_i,
tmp_shape.get());
        }
    }

    // Assemble the destination value type
    ndt::type rdt = ndt::make_type<intptr_t>();
    ndt::type result_vdt = ndt::make_type(ndim, result_shape.get(), rdt);

    // Create the result
    nd::array result = combine_into_tuple(2, ops);
    expr_operation_pair expr_ops;
    expr_ops.single = &kernels::string_find_kernel::single;
    expr_ops.strided = &kernels::string_find_kernel::strided;
    // Because the expr type's operand is the result's type,
    // we can swap it in as the type
    ndt::type edt = ndt::make_expr(result_vdt,
                    result.get_type(),
                    new string_find_kernel_generator(rdt,
ops[0].get_dtype().value_type(),
                                    ops[1].get_dtype().value_type(), expr_ops,
"string.find"));
    edt.swap(result.get_ndo()->m_type);
    return result;
}

static size_t base_string_array_functions_size() { return 1; }

static const pair<string, gfunc::callable> *base_string_array_functions()
{
  static pair<string, gfunc::callable> base_string_array_functions[1] = {
      pair<string, gfunc::callable>(
          "find", gfunc::make_callable(&array_function_find, "self", "sub"))};
    std::cout << "done" << std::endl;

  return base_string_array_functions;
}
*/

void ndt::base_string_type::get_dynamic_array_functions(
    const std::pair<std::string, gfunc::callable> **DYND_UNUSED(out_functions), size_t *DYND_UNUSED(out_count)) const
{
  /*
      *out_functions = base_string_array_functions();
      *out_count = base_string_array_functions_size();
  */
}
