//
// Copyright (C) 2011-15 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <iostream>

#include <dynd/shape_tools.hpp>
#include <dynd/exceptions.hpp>
#include <dynd/types/fixed_dim_type.hpp>

using namespace std;
using namespace dynd;

bool dynd::shape_can_broadcast(intptr_t dst_ndim, const intptr_t *dst_shape, intptr_t src_ndim,
                               const intptr_t *src_shape)
{
  if (dst_ndim >= src_ndim) {
    dst_shape += (dst_ndim - src_ndim);
    for (intptr_t i = 0; i < src_ndim; ++i) {
      if (src_shape[i] != 1 && src_shape[i] != dst_shape[i]) {
        return false;
      }
    }

    return true;
  }
  else {
    return false;
  }
}

void dynd::broadcast_to_shape(intptr_t dst_ndim, const intptr_t *dst_shape, intptr_t src_ndim,
                              const intptr_t *src_shape, const intptr_t *src_strides, intptr_t *out_strides)
{
  // cout << "broadcast_to_shape(" << dst_ndim << ", (";
  // for (int i = 0; i < dst_ndim; ++i) cout << dst_shape[i] << " ";
  // cout << "), " << src_ndim << ", (";
  // for (int i = 0; i < src_ndim; ++i) cout << src_shape[i] << " ";
  // cout << "), (";
  // for (int i = 0; i < src_ndim; ++i) cout << src_strides[i] << " ";
  // cout << ")\n";

  if (src_ndim > dst_ndim) {
    throw broadcast_error(dst_ndim, dst_shape, src_ndim, src_shape);
  }

  intptr_t dimdelta = dst_ndim - src_ndim;
  for (intptr_t i = 0; i < dimdelta; ++i) {
    out_strides[i] = 0;
  }
  for (intptr_t i = dimdelta; i < dst_ndim; ++i) {
    intptr_t src_i = i - dimdelta;
    if (src_shape[src_i] == 1) {
      out_strides[i] = 0;
    }
    else if (src_shape[src_i] == dst_shape[i]) {
      out_strides[i] = src_strides[src_i];
    }
    else {
      throw broadcast_error(dst_ndim, dst_shape, src_ndim, src_shape);
    }
  }

  // cout << "output strides: ";
  // for (int i = 0; i < dst_ndim; ++i) cout << out_strides[i] << " ";
  // cout << "\n";
}

void dynd::incremental_broadcast(intptr_t out_undim, intptr_t *out_shape, intptr_t undim, const intptr_t *shape)
{
  if (out_undim < undim) {
    throw broadcast_error(out_undim, out_shape, undim, shape);
  }

  out_shape += (out_undim - undim);
  for (intptr_t i = 0; i < undim; ++i) {
    intptr_t shape_i = shape[i];
    if (shape_i != 1) {
      if (shape_i == -1) {
        if (out_shape[i] == 1) {
          out_shape[i] = -1;
        }
      }
      else if (out_shape[i] == 1 || out_shape[i] == -1) {
        out_shape[i] = shape_i;
      }
      else if (shape_i != out_shape[i]) {
        throw broadcast_error(out_undim, out_shape - (out_undim - undim), undim, shape);
      }
    }
  }
}

static inline intptr_t intptr_abs(intptr_t x) { return x >= 0 ? x : -x; }

namespace {

class abs_intptr_compare {
  const intptr_t *m_strides;

public:
  abs_intptr_compare(const intptr_t *strides) : m_strides(strides) {}

  bool operator()(int i, int j) { return intptr_abs(m_strides[i]) < intptr_abs(m_strides[j]); }
};

} // anonymous namespace

void dynd::strides_to_axis_perm(intptr_t ndim, const intptr_t *strides, int *out_axis_perm)
{
  switch (ndim) {
  case 0: {
    break;
  }
  case 1: {
    out_axis_perm[0] = 0;
    break;
  }
  case 2: {
    if (intptr_abs(strides[0]) >= intptr_abs(strides[1])) {
      out_axis_perm[0] = 1;
      out_axis_perm[1] = 0;
    }
    else {
      out_axis_perm[0] = 0;
      out_axis_perm[1] = 1;
    }
    break;
  }
  case 3: {
    intptr_t abs_strides[3] = {intptr_abs(strides[0]), intptr_abs(strides[1]), intptr_abs(strides[2])};
    if (abs_strides[0] >= abs_strides[1]) {
      if (abs_strides[1] >= abs_strides[2]) {
        out_axis_perm[0] = 2;
        out_axis_perm[1] = 1;
        out_axis_perm[2] = 0;
      }
      else { // abs_strides[1] < abs_strides[2]
        if (abs_strides[0] >= abs_strides[2]) {
          out_axis_perm[0] = 1;
          out_axis_perm[1] = 2;
          out_axis_perm[2] = 0;
        }
        else { // abs_strides[0] < abs_strides[2]
          out_axis_perm[0] = 1;
          out_axis_perm[1] = 0;
          out_axis_perm[2] = 2;
        }
      }
    }
    else { // abs_strides[0] < abs_strides[1]
      if (abs_strides[1] >= abs_strides[2]) {
        if (abs_strides[0] >= abs_strides[2]) {
          out_axis_perm[0] = 2;
          out_axis_perm[1] = 0;
          out_axis_perm[2] = 1;
        }
        else { // abs_strides[0] < abs_strides[2]
          out_axis_perm[0] = 0;
          out_axis_perm[1] = 2;
          out_axis_perm[2] = 1;
        }
      }
      else { // strides[1] < strides[2]
        out_axis_perm[0] = 0;
        out_axis_perm[1] = 1;
        out_axis_perm[2] = 2;
      }
    }
    break;
  }
  default: {
    // Initialize to a reversal perm (i.e. so C-order is a no-op)
    for (intptr_t i = 0; i < ndim; ++i) {
      out_axis_perm[i] = int(ndim - i - 1);
    }
    // Sort based on the absolute value of the strides
    std::sort(out_axis_perm, out_axis_perm + ndim, abs_intptr_compare(strides));
    break;
  }
  }
}

void dynd::axis_perm_to_strides(intptr_t ndim, const int *axis_perm, const intptr_t *shape, intptr_t element_size,
                                intptr_t *out_strides)
{
  for (intptr_t i = 0; i < ndim; ++i) {
    int i_perm = axis_perm[i];
    intptr_t dim_size = shape[i_perm];
    out_strides[i_perm] = dim_size > 1 ? element_size : 0;
    out_strides[i_perm] = dim_size;
    element_size *= dim_size;
  }
}

/**
 * Compares the strides of the operands for axes 'i' and 'j', and returns whether
 * the comparison is ambiguous and, when it's not ambiguous, whether 'i' should occur
 * before 'j'.
 */
static inline void compare_strides(int i, int j, int noperands, const intptr_t **operstrides, bool *out_ambiguous,
                                   bool *out_lessthan)
{
  *out_ambiguous = true;

  for (int ioperand = 0; ioperand < noperands; ++ioperand) {
    intptr_t stride_i = operstrides[ioperand][i];
    intptr_t stride_j = operstrides[ioperand][j];
    if (stride_i != 0 && stride_j != 0) {
      if (intptr_abs(stride_i) <= intptr_abs(stride_j)) {
        // Set 'lessthan' even if it's already not ambiguous, since
        // less than beats greater than when there's a conflict
        *out_lessthan = true;
        *out_ambiguous = false;
        return;
      }
      else if (*out_ambiguous) {
        // Only set greater than when the comparison is still ambiguous
        *out_lessthan = false;
        *out_ambiguous = false;
        // Can't return yet, because a 'lessthan' might override this choice
      }
    }
  }
}

void dynd::multistrides_to_axis_perm(intptr_t ndim, int noperands, const intptr_t **operstrides, int *out_axis_perm)
{
  switch (ndim) {
  case 0: {
    break;
  }
  case 1: {
    out_axis_perm[0] = 0;
    break;
  }
  case 2: {
    bool ambiguous = true, lessthan = false;

    // TODO: The comparison function is quite complicated, maybe there's a way to
    //       simplify all this while retaining the generality?
    compare_strides(0, 1, noperands, operstrides, &ambiguous, &lessthan);

    if (ambiguous || !lessthan) {
      out_axis_perm[0] = 1;
      out_axis_perm[1] = 0;
    }
    else {
      out_axis_perm[0] = 0;
      out_axis_perm[1] = 1;
    }
    break;
  }
  default: {
    // Initialize to a reversal perm (i.e. so C-order is a no-op)
    for (intptr_t i = 0; i < ndim; ++i) {
      out_axis_perm[i] = int(ndim - i - 1);
    }
    // Here we do a custom stable insertion sort, which avoids a swap when a comparison
    // is ambiguous
    for (intptr_t i0 = 1; i0 < ndim; ++i0) {
      // 'ipos' is the position where axis_perm[i0] will get inserted
      ptrdiff_t ipos = i0;
      int perm_i0 = out_axis_perm[i0];

      for (ptrdiff_t i1 = (ptrdiff_t)i0 - 1; i1 >= 0; --i1) {
        bool ambiguous = true, lessthan = false;
        int perm_i1 = out_axis_perm[i1];

        compare_strides(perm_i1, perm_i0, noperands, operstrides, &ambiguous, &lessthan);

        // If the comparison was unambiguous, either shift 'ipos' to 'i1', or
        // stop looking for an insertion point
        if (!ambiguous) {
          if (!lessthan) {
            ipos = int(i1);
          }
          else {
            break;
          }
        }
      }

      // Insert axis_perm[i0] into axis_perm[ipos]
      if (ipos != (ptrdiff_t)i0) {
        for (ptrdiff_t i = (ptrdiff_t)i0; i > ipos; --i) {
          out_axis_perm[i] = out_axis_perm[i - 1];
        }
        out_axis_perm[ipos] = perm_i0;
      }
    }
    break;
  }
  }
}

void dynd::print_shape(std::ostream &o, intptr_t ndim, const intptr_t *shape)
{
  o << "(";
  for (intptr_t i = 0; i < ndim; ++i) {
    intptr_t size = shape[i];
    if (size >= 0) {
      o << size;
    }
    else {
      o << "var";
    }
    if (i != ndim - 1) {
      o << ", ";
    }
  }
  o << ")";
}

void dynd::apply_single_linear_index(const irange &irnge, intptr_t dimension_size, intptr_t error_i,
                                     const ndt::type *error_tp, bool &out_remove_dimension, intptr_t &out_start_index,
                                     intptr_t &out_index_stride, intptr_t &out_dimension_size)
{
  intptr_t step = irnge.step();
  if (step == 0) {
    // A single index
    out_remove_dimension = true;
    intptr_t idx = irnge.start();
    if (idx >= 0) {
      if (idx < dimension_size) {
        out_start_index = idx;
        out_index_stride = 1;
        out_dimension_size = 1;
      }
      else {
        if (error_tp) {
          intptr_t ndim = error_tp->extended()->get_ndim();
          dimvector shape(ndim);
          error_tp->extended()->get_shape(ndim, 0, shape.get(), NULL, NULL);
          throw index_out_of_bounds(idx, error_i, ndim, shape.get());
        }
        else {
          throw index_out_of_bounds(idx, dimension_size);
        }
      }
    }
    else if (idx >= -dimension_size) {
      out_start_index = idx + dimension_size;
      out_index_stride = 1;
      out_dimension_size = 1;
    }
    else {
      if (error_tp) {
        intptr_t ndim = error_tp->get_ndim();
        dimvector shape(ndim);
        error_tp->extended()->get_shape(ndim, 0, shape.get(), NULL, NULL);
        throw index_out_of_bounds(idx, error_i, ndim, shape.get());
      }
      else {
        throw index_out_of_bounds(idx, dimension_size);
      }
    }
  }
  else if (step > 0) {
    // A range with a positive step
    intptr_t start = irnge.start();
    if (start >= 0) {
      if (start < dimension_size) {
        // Starts with a positive index
      }
      else {
        if (error_tp) {
          intptr_t ndim = error_tp->get_ndim();
          dimvector shape(ndim);
          // check that we get here
          error_tp->extended()->get_shape(ndim, 0, shape.get(), NULL, NULL);
          throw irange_out_of_bounds(irnge, error_i, ndim, shape.get());
        }
        else {
          throw irange_out_of_bounds(irnge, dimension_size);
        }
      }
    }
    else if (start >= -dimension_size) {
      // Starts with Python style negative index
      start += dimension_size;
    }
    else {
      // Signal for "from the beginning" whenever the index
      // is more negative
      start = 0;
    }

    intptr_t end = irnge.finish();
    if (end >= 0) {
      if (end <= dimension_size) {
        // Ends with a positive index, or the end of the array
      }
      else {
        // Any end value greater or equal to the dimension size
        // signals to slice to the end
        end = dimension_size;
      }
    }
    else if (end >= -dimension_size) {
      // Ends with a Python style negative index
      end += dimension_size;
    }
    else {
      if (error_tp) {
        intptr_t ndim = error_tp->get_ndim();
        dimvector shape(ndim);
        error_tp->extended()->get_shape(ndim, 0, shape.get(), NULL, NULL);
        throw irange_out_of_bounds(irnge, error_i, ndim, shape.get());
      }
      else {
        throw irange_out_of_bounds(irnge, dimension_size);
      }
    }

    intptr_t size = end - start;
    out_remove_dimension = false;
    if (size > 0) {
      if (step == 1) {
        // Simple range
        out_start_index = start;
        out_index_stride = 1;
        out_dimension_size = size;
      }
      else {
        // Range with a stride
        out_start_index = start;
        out_index_stride = step;
        out_dimension_size = (size + step - 1) / step;
      }
    }
    else {
      // Empty slice
      out_start_index = 0;
      out_index_stride = 1;
      out_dimension_size = 0;
    }
  }
  else {
    // A range with a negative step
    intptr_t start = irnge.start();
    if (start >= 0) {
      if (start < dimension_size) {
        // Starts with a positive index
      }
      else {
        if (error_tp) {
          intptr_t ndim = error_tp->get_ndim();
          dimvector shape(ndim);
          error_tp->extended()->get_shape(ndim, 0, shape.get(), NULL, NULL);
          throw irange_out_of_bounds(irnge, error_i, ndim, shape.get());
        }
        else {
          throw irange_out_of_bounds(irnge, dimension_size);
        }
      }
    }
    else if (start >= -dimension_size) {
      // Starts with Python style negative index
      start += dimension_size;
    }
    else if (start == std::numeric_limits<intptr_t>::min()) {
      // Signal for "from the beginning" (which means the last element)
      start = dimension_size - 1;
    }
    else {
      if (error_tp) {
        intptr_t ndim = error_tp->get_ndim();
        dimvector shape(ndim);
        error_tp->extended()->get_shape(ndim, 0, shape.get(), NULL, NULL);
        throw irange_out_of_bounds(irnge, error_i, ndim, shape.get());
      }
      else {
        throw irange_out_of_bounds(irnge, dimension_size);
      }
    }

    intptr_t end = irnge.finish();
    if (end >= 0) {
      if (end < dimension_size) {
        // Ends with a positive index, or the end of the array
      }
      else if (end == std::numeric_limits<intptr_t>::max()) {
        // Signal for "until the end" (which means towards index 0 of the data)
        end = -1;
      }
      else {
        if (error_tp) {
          intptr_t ndim = error_tp->get_ndim();
          dimvector shape(ndim);
          error_tp->extended()->get_shape(ndim, 0, shape.get(), NULL, NULL);
          throw irange_out_of_bounds(irnge, error_i, ndim, shape.get());
        }
        else {
          throw irange_out_of_bounds(irnge, dimension_size);
        }
      }
    }
    else if (end >= -dimension_size) {
      // Ends with a Python style negative index
      end += dimension_size;
    }
    else {
      // If the value is too negative, -1 means to go all the
      // way to the beginning (with the negative step)
      end = -1;
    }

    intptr_t size = start - end;
    out_remove_dimension = false;
    if (size > 0) {
      if (step == -1) {
        // Simple range
        out_start_index = start;
        out_index_stride = -1;
        out_dimension_size = size;
      }
      else {
        // Range with a stride
        out_start_index = start;
        out_index_stride = step;
        out_dimension_size = (size + (-step) - 1) / (-step);
      }
    }
    else {
      // Empty slice
      out_start_index = 0;
      out_index_stride = 1;
      out_dimension_size = 0;
    }
  }
}

axis_order_classification_t dynd::classify_strided_axis_order(intptr_t current_stride, const ndt::type &element_tp,
                                                              const char *element_arrmeta)
{
  switch (element_tp.get_id()) {
  case fixed_dim_id: {
    const ndt::fixed_dim_type *edt = element_tp.extended<ndt::fixed_dim_type>();
    const fixed_dim_type_arrmeta *emd = reinterpret_cast<const fixed_dim_type_arrmeta *>(element_arrmeta);
    intptr_t estride = intptr_abs(emd->stride);
    if (estride != 0) {
      axis_order_classification_t aoc;
      // Get the classification from the next dimension onward
      if (edt->get_ndim() > 1) {
        aoc = classify_strided_axis_order(current_stride, edt->get_element_type(),
                                          element_arrmeta + sizeof(fixed_dim_type_arrmeta));
      }
      else {
        aoc = axis_order_none;
      }
      if (current_stride > estride) {
        // C order
        return (aoc == axis_order_none || aoc == axis_order_c) ? axis_order_c : axis_order_neither;
      }
      else {
        // F order
        return (aoc == axis_order_none || aoc == axis_order_f) ? axis_order_f : axis_order_neither;
      }
    }
    else if (element_tp.get_ndim() > 1) {
      // Skip the zero-stride dimensions (DyND requires that the stride
      // be zero when the dimension size is one)
      return classify_strided_axis_order(current_stride, edt->get_element_type(),
                                         element_arrmeta + sizeof(fixed_dim_type_arrmeta));
    }
    else {
      // There was only one dimension with a nonzero stride
      return axis_order_none;
    }
  }
  case pointer_id:
  case var_dim_id: {
    // A pointer or a var type is treated like C-order
    axis_order_classification_t aoc = element_tp.extended()->classify_axis_order(element_arrmeta);
    return (aoc == axis_order_none || aoc == axis_order_c) ? axis_order_c : axis_order_neither;
  }
  default: {
    stringstream ss;
    ss << "classify_strided_axis_order not implemented for dynd type ";
    ss << element_tp;
    throw runtime_error(ss.str());
  }
  }
}
