//
// Copyright (C) 2011-15 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <dynd/func/rolling.hpp>
#include <dynd/kernels/rolling.hpp>
#include <dynd/types/typevar_dim_type.hpp>

using namespace std;
using namespace dynd;

nd::arrfunc nd::functional::rolling(const nd::arrfunc &window_op,
                                    intptr_t window_size)
{
  // Validate the input arrfunc
  if (window_op.is_null()) {
    throw invalid_argument("make_rolling_arrfunc() 'window_op' cannot be null");
  }
  const arrfunc_type *window_af_tp = window_op.get_type();
  if (window_af_tp->get_npos() != 1) {
    stringstream ss;
    ss << "To make a rolling window arrfunc, an operation with one "
          "argument is required, got " << window_af_tp;
    throw invalid_argument(ss.str());
  }
  const ndt::type &window_src_tp = window_af_tp->get_pos_type(0);
  if (window_src_tp.get_ndim() < 1) {
    stringstream ss;
    ss << "To make a rolling window arrfunc, an operation with which "
          "accepts a dimension is required, got " << window_af_tp;
    throw invalid_argument(ss.str());
  }

  nd::string rolldimname("RollDim");
  ndt::type roll_src_tp = ndt::make_typevar_dim(
      rolldimname, window_src_tp.get_type_at_dimension(NULL, 1));
  ndt::type roll_dst_tp =
      ndt::make_typevar_dim(rolldimname, window_af_tp->get_return_type());

  // Create the data for the arrfunc
  std::shared_ptr<rolling_arrfunc_data> data(new rolling_arrfunc_data);
  data->window_size = window_size;
  data->window_op = window_op;

  return as_arrfunc<rolling_ck>(
      ndt::make_arrfunc(ndt::make_tuple(roll_src_tp), roll_dst_tp), data, 0);
}
