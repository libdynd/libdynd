//
// Copyright (C) 2011-15 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <dynd/callable.hpp>
#include <dynd/type.hpp>
#include <dynd/shape_tools.hpp>

using namespace std;
using namespace dynd;

ndt::base_string_type::~base_string_type() {}

std::string ndt::base_string_type::get_utf8_string(const char *arrmeta, const char *data,
                                                   assign_error_mode errmode) const
{
  const char *begin, *end;
  get_string_range(&begin, &end, arrmeta, data);
  return string_range_as_utf8_string(get_encoding(), begin, end, errmode);
}

size_t ndt::base_string_type::get_iterdata_size(intptr_t DYND_UNUSED(ndim)) const { return 0; }

static void get_extended_string_encoding(const ndt::type &dt)
{
  const ndt::base_string_type *d = dt.extended<ndt::base_string_type>();
  stringstream ss;
  ss << d->get_encoding();
  //  return ss.str();
}

static const std::map<std::string, nd::callable> &base_string_type_properties()
{
  static const std::map<std::string, nd::callable> base_string_type_properties{
      {"encoding", nd::functional::apply(&get_extended_string_encoding)}};

  return base_string_type_properties;
}

std::map<std::string, nd::callable> ndt::base_string_type::get_dynamic_type_properties() const
{
  return base_string_type_properties();
}

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

std::map<std::string, nd::callable> ndt::base_string_type::get_dynamic_array_functions() const
{
  return std::map<std::string, nd::callable>();
  /*
      *out_functions = base_string_array_functions();
      *out_count = base_string_array_functions_size();
  */
}
