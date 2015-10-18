//
// Copyright (C) 2011-15 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <dynd/view.hpp>
#include <dynd/types/fixed_dim_kind_type.hpp>
#include <dynd/types/fixed_dim_type.hpp>
#include <dynd/types/var_dim_type.hpp>
#include <dynd/types/pointer_type.hpp>
#include <dynd/types/bytes_type.hpp>
#include <dynd/shape_tools.hpp>
#include <dynd/types/substitute_shape.hpp>

using namespace std;
using namespace dynd;

/**
 * Scans through the types, and tries to view data
 * for 'tp'/'arrmeta' as 'view_tp'. For this to be
 * possible, one must be able to construct
 * arrmeta for 'tp' corresponding to the same data.
 *
 * \param tp  The type of the data.
 * \param arrmeta  The array arrmeta of the data.
 * \param view_tp  The type the data should be viewed as.
 * \param view_arrmeta The array arrmeta of the view, which should be populated.
 * \param embedded_reference  The containing memory block in case the data was embedded.
 *
 * \returns If it worked, returns true, otherwise false.
 */
static bool try_view(const ndt::type &tp, const char *arrmeta, const ndt::type &view_tp, char *view_arrmeta,
                     dynd::memory_block_data *embedded_reference)
{
  switch (tp.get_type_id()) {
  case fixed_dim_type_id: {
    // All the strided dim types share the same arrmeta, so can be
    // treated uniformly here
    const ndt::base_dim_type *sdt = tp.extended<ndt::base_dim_type>();
    const fixed_dim_type_arrmeta *md = reinterpret_cast<const fixed_dim_type_arrmeta *>(arrmeta);
    switch (view_tp.get_type_id()) {
    case fixed_dim_type_id: { // strided as fixed
      const ndt::fixed_dim_type *view_fdt = view_tp.extended<ndt::fixed_dim_type>();
      // The size must match exactly in this case
      if (md->dim_size != view_fdt->get_fixed_dim_size()) {
        return false;
      }
      fixed_dim_type_arrmeta *view_md = reinterpret_cast<fixed_dim_type_arrmeta *>(view_arrmeta);
      if (try_view(sdt->get_element_type(), arrmeta + sizeof(fixed_dim_type_arrmeta), view_fdt->get_element_type(),
                   view_arrmeta + sizeof(fixed_dim_type_arrmeta), embedded_reference)) {
        *view_md = *md;
        return true;
      } else {
        return false;
      }
    }
    default: // other cases cannot be handled
      return false;
    }
  }
  default:
    if (tp == view_tp) {
      // require equal types otherwise
      if (tp.get_arrmeta_size() > 0) {
        tp.extended()->arrmeta_copy_construct(view_arrmeta, arrmeta, embedded_reference);
      }
      return true;
    } else if (tp.is_pod() && view_tp.is_pod() && tp.get_data_size() == view_tp.get_data_size() &&
               tp.get_data_alignment() >= view_tp.get_data_alignment()) {
      // POD types with matching properties
      if (view_tp.get_arrmeta_size() > 0) {
        view_tp.extended()->arrmeta_default_construct(view_arrmeta, true);
      }
      return true;
    } else {
      return false;
    }
  }
}

static void refine_bytes_view(memory_block_ptr &data_ref, char *&data_ptr, ndt::type &data_tp, const char *&data_meta,
                              intptr_t &data_dim_size, intptr_t &data_stride)
{
  // Handle sequence of strided dims
  intptr_t dim_size, stride;
  ndt::type el_tp;
  const char *el_meta;
  if (data_tp.get_as_strided(data_meta, &dim_size, &stride, &el_tp, &el_meta)) {
    dimvector shape(data_tp.get_ndim());
    dimvector strides(data_tp.get_ndim());
    intptr_t ndim = 1;
    shape[0] = dim_size;
    strides[0] = stride;
    bool csorted = true;
    // Get all the strided dimensions we can in a row
    while (el_tp.get_as_strided(el_meta, &dim_size, &stride, &el_tp, &el_meta)) {
      shape[ndim] = dim_size;
      strides[ndim] = stride;
      if (stride > strides[ndim - 1]) {
        csorted = false;
      }
      ++ndim;
    }
    if (!csorted) {
      // If the strides weren't sorted in C order, sort them
      shortvector<int> axis_perm(ndim);
      strides_to_axis_perm(ndim, strides.get(), axis_perm.get());
      dimvector shape_sorted(ndim);
      dimvector strides_sorted(ndim);
      for (intptr_t i = 0; i < ndim; ++i) {
        int i_perm = axis_perm[i];
        shape_sorted[ndim - i - 1] = shape[i_perm];
        strides_sorted[ndim - i - 1] = strides[i_perm];
      }
      shape.swap(shape_sorted);
      strides.swap(strides_sorted);
    }
    // Try to collapse the shape/strides into a single strided array
    intptr_t i = 0;
    while (data_dim_size == -1 && i < ndim) {
      // If there's not already a dim_size/stride, start one
      if (shape[i] != 1) {
        data_dim_size = shape[i];
        data_stride = strides[i];
      }
      ++i;
    }
    for (; i < ndim; ++i) {
      if (shape[i] != 1) {
        if (shape[i] * strides[i] != data_stride) {
          // Indicate we couldn't view this as bytes
          data_tp = ndt::type();
          data_dim_size = -1;
          return;
        }
        data_dim_size *= shape[i];
        data_stride = strides[i];
      }
    }
    data_tp = el_tp;
    data_meta = el_meta;
    return;
  }

  switch (data_tp.get_type_id()) {
  case var_dim_type_id: {
    // We can only allow leading var_dim
    if (data_dim_size != -1) {
      data_tp = ndt::type();
      data_dim_size = -1;
      return;
    }
    const var_dim_type_arrmeta *meta = reinterpret_cast<const var_dim_type_arrmeta *>(data_meta);
    if (meta->blockref != NULL) {
      data_ref = meta->blockref;
    }
    var_dim_type_data *d = reinterpret_cast<var_dim_type_data *>(data_ptr);
    data_ptr = d->begin + meta->offset;
    if (d->size != 1) {
      data_dim_size = d->size;
      data_stride = meta->stride;
    }
    data_tp = data_tp.extended<ndt::var_dim_type>()->get_element_type();
    data_meta += sizeof(var_dim_type_arrmeta);
    return;
  }
  case pointer_type_id: {
    // We can only strip away leading pointers
    if (data_dim_size != -1) {
      data_tp = ndt::type();
      data_dim_size = -1;
      return;
    }
    const pointer_type_arrmeta *meta = reinterpret_cast<const pointer_type_arrmeta *>(data_meta);
    if (meta->blockref != NULL) {
      data_ref = meta->blockref;
    }
    data_ptr = *reinterpret_cast<char **>(data_ptr) + meta->offset;
    data_tp = data_tp.extended<ndt::pointer_type>()->get_target_type();
    data_meta += sizeof(pointer_type_arrmeta);
    return;
  }
  case string_type_id: {
    // We can only view leading strings
    if (data_dim_size != -1) {
      data_tp = ndt::type();
      data_dim_size = -1;
      return;
    }
    // Look at the actual string data, not the pointer to it
    const string_type_arrmeta *meta = reinterpret_cast<const string_type_arrmeta *>(data_meta);
    if (meta->blockref != NULL) {
      data_ref = meta->blockref;
    }
    dynd::string *str_ptr = reinterpret_cast<dynd::string *>(data_ptr);
    data_ptr = str_ptr->begin();
    data_tp = ndt::type();
    data_dim_size = str_ptr->end() - str_ptr->begin();
    data_stride = 1;
    return;
  }
  default:
    break;
  }

  // If the data type has a fixed size, check if it fits the strides
  size_t data_tp_size = data_tp.get_data_size();
  if (data_tp_size > 0) {
    if (data_dim_size == -1) {
      // Indicate success (one item)
      data_tp = ndt::type();
      data_dim_size = data_tp_size;
      data_stride = 1;
      return;
    } else if ((intptr_t)data_tp_size == data_stride) {
      data_tp = ndt::type();
      data_dim_size *= data_tp_size;
      data_stride = 1;
      return;
    }
  }

  // Indicate we couldn't view this as bytes
  data_tp = ndt::type();
  data_dim_size = -1;
}

static nd::array view_as_bytes(const nd::array &arr, const ndt::type &tp)
{
  if (arr.get_type().get_flags() & type_flag_destructor) {
    // Can't view arrays of object type
    return nd::array();
  }

  // Get the essential components of the array to analyze
  memory_block_ptr data_ref = arr.get_data_memblock();
  char *data_ptr = arr.get_ndo()->data.ptr;
  ndt::type data_tp = arr.get_type();
  const char *data_meta = arr.get_arrmeta();
  intptr_t data_dim_size = -1, data_stride = 0;
  // Repeatedly refine the data
  while (data_tp.get_type_id() != uninitialized_type_id) {
    refine_bytes_view(data_ref, data_ptr, data_tp, data_meta, data_dim_size, data_stride);
  }
  // Check that it worked, and that the resulting data pointer is aligned
  if (data_dim_size < 0 ||
      !offset_is_aligned(reinterpret_cast<size_t>(data_ptr), tp.extended<ndt::bytes_type>()->get_target_alignment())) {
    // This signals we could not view the data as a
    // contiguous chunk of bytes
    return nd::array();
  }

  char *result_data_ptr = NULL;
  nd::array result(make_array_memory_block(tp.extended()->get_arrmeta_size(), tp.get_data_size(),
                                           tp.get_data_alignment(), &result_data_ptr));
  // Set the bytes extents
  reinterpret_cast<bytes *>(result_data_ptr)->assign(data_ptr, data_dim_size);
  // Set the array arrmeta
  array_preamble *ndo = result.get_ndo();
  ndo->m_type = ndt::type(tp).release();
  ndo->data.ptr = result_data_ptr;
  ndo->data.ref = NULL;
  ndo->m_flags = arr.get_flags();
  // Set the bytes arrmeta
  bytes_type_arrmeta *ndo_meta = reinterpret_cast<bytes_type_arrmeta *>(result.get_arrmeta());
  ndo_meta->blockref = data_ref.release();
  return result;
}

static nd::array view_from_bytes(const nd::array &arr, const ndt::type &tp)
{
  if (tp.get_flags() & (type_flag_blockref | type_flag_destructor | type_flag_not_host_readable)) {
    // Bytes cannot be viewed as blockref types, types which require
    // destruction, or types not on host memory.
    return nd::array();
  }

  const bytes_type_arrmeta *bytes_meta = reinterpret_cast<const bytes_type_arrmeta *>(arr.get_arrmeta());
  bytes_type_data *bytes_d = reinterpret_cast<bytes_type_data *>(arr.get_ndo()->data.ptr);
  memory_block_ptr data_ref;
  if (bytes_meta->blockref != NULL) {
    data_ref = bytes_meta->blockref;
  } else {
    data_ref = arr.get_data_memblock();
  }
  char *data_ptr = bytes_d->begin();
  intptr_t data_size = bytes_d->end() - data_ptr;

  size_t tp_data_size = tp.get_data_size();
  if (tp_data_size > 0) {
    // If the data type has a single chunk of POD memory, it's ok
    if ((intptr_t)tp_data_size == data_size &&
        offset_is_aligned(reinterpret_cast<size_t>(data_ptr), tp.get_data_alignment())) {
      // Allocate a result array to attempt the view in it
      nd::array result(make_array_memory_block(tp.get_arrmeta_size()));
      // Initialize the fields
      result.get_ndo()->data.ptr = data_ptr;
      result.get_ndo()->data.ref = data_ref.release();
      result.get_ndo()->m_type = ndt::type(tp).release();
      result.get_ndo()->m_flags = arr.get_ndo()->m_flags;
      if (tp.get_arrmeta_size() > 0) {
        tp.extended()->arrmeta_default_construct(result.get_arrmeta(), true);
      }
      return result;
    }
  } else if (tp.get_type_id() == fixed_dim_type_id) {
    ndt::type arr_tp = tp;
    ndt::type el_tp = arr_tp.extended<ndt::base_dim_type>()->get_element_type();
    size_t el_data_size = el_tp.get_data_size();
    // If the element type has a single chunk of POD memory, and
    // it divides into the memory size, it's ok
    if (data_size % (intptr_t)el_data_size == 0 &&
        offset_is_aligned(reinterpret_cast<size_t>(data_ptr), arr_tp.get_data_alignment())) {
      intptr_t dim_size = data_size / el_data_size;
      if (arr_tp.get_kind() != kind_kind) {
        if (arr_tp.extended<ndt::fixed_dim_type>()->get_fixed_dim_size() != dim_size) {
          return nd::array();
        }
      } else {
        // Transform the symbolic fixed type into a concrete one
        arr_tp = ndt::make_fixed_dim(dim_size, el_tp);
      }
      // Allocate a result array to attempt the view in it
      nd::array result(make_array_memory_block(arr_tp.get_arrmeta_size()));
      // Initialize the fields
      result.get_ndo()->data.ptr = data_ptr;
      result.get_ndo()->data.ref = data_ref.release();
      result.get_ndo()->m_type = ndt::type(arr_tp).release();
      result.get_ndo()->m_flags = arr.get_ndo()->m_flags;
      if (el_tp.get_arrmeta_size() > 0) {
        el_tp.extended()->arrmeta_default_construct(result.get_arrmeta() + sizeof(fixed_dim_type_arrmeta), true);
      }
      fixed_dim_type_arrmeta *fixed_meta = reinterpret_cast<fixed_dim_type_arrmeta *>(result.get_arrmeta());
      fixed_meta->dim_size = dim_size;
      fixed_meta->stride = el_data_size;
      return result;
    }
  }

  // No view could be produced
  return nd::array();
}

static nd::array view_concrete(const nd::array &arr, const ndt::type &tp)
{
  // Allocate a result array to attempt the view in it
  nd::array result(make_array_memory_block(tp.get_arrmeta_size()));
  // Copy the fields
  result.get_ndo()->data.ptr = arr.get_ndo()->data.ptr;
  if (arr.get_ndo()->data.ref == NULL) {
    // Embedded data, need reference to the array
    result.get_ndo()->data.ref = arr.get_memblock().release();
  } else {
    // Use the same data reference, avoid producing a chain
    result.get_ndo()->data.ref = arr.get_data_memblock().release();
  }
  result.get_ndo()->m_type = ndt::type(tp).release();
  result.get_ndo()->m_flags = arr.get_ndo()->m_flags;
  // First handle a special case of viewing outermost "var" as "fixed[#]"
  if (arr.get_type().get_type_id() == var_dim_type_id && tp.get_type_id() == fixed_dim_type_id) {
    const var_dim_type_arrmeta *in_am = reinterpret_cast<const var_dim_type_arrmeta *>(arr.get_arrmeta());
    const var_dim_type_data *in_dat = reinterpret_cast<const var_dim_type_data *>(arr.get_readonly_originptr());
    fixed_dim_type_arrmeta *out_am = reinterpret_cast<fixed_dim_type_arrmeta *>(result.get_arrmeta());
    out_am->dim_size = tp.extended<ndt::fixed_dim_type>()->get_fixed_dim_size();
    out_am->stride = in_am->stride;
    if ((intptr_t)in_dat->size == out_am->dim_size) {
      // Use the more specific data reference from the var arrmeta if possible
      if (in_am->blockref != NULL) {
        memory_block_decref(result.get_ndo()->data.ref);
        memory_block_incref(in_am->blockref);
        result.get_ndo()->data.ref = in_am->blockref;
      }
      result.get_ndo()->data.ptr = in_dat->begin + in_am->offset;
      // Try to copy the rest of the arrmeta as a view
      if (try_view(arr.get_type().extended<ndt::base_dim_type>()->get_element_type(),
                   arr.get_arrmeta() + sizeof(var_dim_type_arrmeta),
                   tp.extended<ndt::base_dim_type>()->get_element_type(),
                   result.get_arrmeta() + sizeof(fixed_dim_type_arrmeta), arr.get_memblock().get())) {
        return result;
      }
    }
  }
  // Otherwise try to copy the arrmeta as a view
  else if (try_view(arr.get_type(), arr.get_arrmeta(), tp, result.get_arrmeta(), arr.get_memblock().get())) {
    // If it succeeded, return it
    return result;
  }

  stringstream ss;
  ss << "Unable to view nd::array of type " << arr.get_type();
  ss << " as type " << tp;
  throw type_error(ss.str());
}

nd::array nd::view(const nd::array &arr, const ndt::type &tp)
{
  if (arr.get_type() == tp) {
    // If the types match exactly, simply return 'arr'
    return arr;
  } else if (tp.get_type_id() == bytes_type_id) {
    // If it's a request to view the data as raw bytes
    nd::array result = view_as_bytes(arr, tp);
    if (!result.is_null()) {
      return result;
    }
  } else if (arr.get_type().get_type_id() == bytes_type_id) {
    // If it's a request to view raw bytes as something else
    nd::array result = view_from_bytes(arr, tp);
    if (!result.is_null()) {
      return result;
    }
  } else if (arr.get_ndim() == tp.get_ndim()) {
    // If the type is symbolic, e.g. has a "Fixed" symbolic dimension,
    // first substitute in the shape from the array
    if (tp.is_symbolic()) {
      dimvector shape(arr.get_ndim());
      arr.get_shape(shape.get());
      return view_concrete(arr, substitute_shape(tp, arr.get_ndim(), shape.get()));
    } else {
      return view_concrete(arr, tp);
    }
  }

  stringstream ss;
  ss << "Unable to view nd::array of type " << arr.get_type();
  ss << " as type " << tp;
  throw type_error(ss.str());
}
