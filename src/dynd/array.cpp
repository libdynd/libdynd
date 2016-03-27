//
// Copyright (C) 2011-15 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <dynd/arithmetic.hpp>
#include <dynd/array.hpp>
#include <dynd/comparison.hpp>




#include <dynd/array_iter.hpp>
#include <dynd/types/datashape_formatter.hpp>
#include <dynd/math.hpp>
#include <dynd/assignment.hpp>
#include <dynd/option.hpp>
#include <dynd/struct.hpp>
#include <dynd/types/var_dim_type.hpp>
#include <dynd/types/fixed_dim_type.hpp>
#include <dynd/types/tuple_type.hpp>
#include <dynd/types/string_type.hpp>
#include <dynd/types/bytes_type.hpp>
#include <dynd/types/fixed_bytes_type.hpp>
#include <dynd/types/type_type.hpp>
#include <dynd/types/datashape_formatter.hpp>
#include <dynd/types/base_memory_type.hpp>
#include <dynd/types/cuda_host_type.hpp>
#include <dynd/types/cuda_device_type.hpp>
#include <dynd/types/option_type.hpp>
#include <dynd/kernels/assignment_kernels.hpp>
#include <dynd/kernels/field_access_kernel.hpp>
#include <dynd/exceptions.hpp>
#include <dynd/types/categorical_type.hpp>
#include <dynd/memblock/memmap_memory_block.hpp>
#include <dynd/view.hpp>

using namespace std;
using namespace dynd;

nd::array nd::make_strided_array_from_data(const ndt::type &uniform_tp, intptr_t ndim, const intptr_t *shape,
                                           const intptr_t *strides, int64_t access_flags, char *data_ptr,
                                           const intrusive_ptr<memory_block_data> &data_reference,
                                           char **out_uniform_arrmeta)
{
  if (out_uniform_arrmeta == NULL && !uniform_tp.is_builtin() && uniform_tp.extended()->get_arrmeta_size() > 0) {
    stringstream ss;
    ss << "Cannot make a strided array with type " << uniform_tp << " from a preexisting data pointer";
    throw runtime_error(ss.str());
  }

  ndt::type array_type = ndt::make_fixed_dim(ndim, shape, uniform_tp);

  // Allocate the array arrmeta and data in one memory block
  intrusive_ptr<memory_block_data> result = make_array_memory_block(array_type.get_arrmeta_size());

  // Fill in the preamble arrmeta
  array_preamble *ndo = reinterpret_cast<array_preamble *>(result.get());
  ndo->tp = array_type;
  ndo->data = data_ptr;
  ndo->owner = data_reference;
  ndo->flags = access_flags;

  // Fill in the array arrmeta with the shape and strides
  fixed_dim_type_arrmeta *meta = reinterpret_cast<fixed_dim_type_arrmeta *>(ndo + 1);
  for (intptr_t i = 0; i < ndim; ++i) {
    intptr_t dim_size = shape[i];
    meta[i].stride = dim_size > 1 ? strides[i] : 0;
    meta[i].dim_size = dim_size;
  }

  // Return a pointer to the arrmeta for uniform_tp.
  if (out_uniform_arrmeta != NULL) {
    *out_uniform_arrmeta = reinterpret_cast<char *>(meta + ndim);
  }

  return nd::array(ndo, true);
}

/**
 * Clones the arrmeta and swaps in a new type. The type must
 * have identical arrmeta, but this function doesn't check that.
 */
static nd::array make_array_clone_with_new_type(const nd::array &n, const ndt::type &new_dt)
{
  nd::array result(reinterpret_cast<array_preamble *>(
                       shallow_copy_array_memory_block(intrusive_ptr<memory_block_data>(n.get(), true)).get()),
                   true);
  array_preamble *preamble = result.get();
  // Swap in the type
  preamble->tp = new_dt;
  return result;
}

namespace {
static void as_storage_type(const ndt::type &dt, intptr_t DYND_UNUSED(arrmeta_offset), void *DYND_UNUSED(self),
                            ndt::type &out_transformed_tp, bool &out_was_transformed)
{
  // If the type is a simple POD, switch it to a bytes type. Otherwise, keep it
  // the same so that the arrmeta layout is identical.
  if (dt.is_scalar() && dt.get_id() != pointer_id) {
    const ndt::type &storage_dt = dt.storage_type();
    if (storage_dt.is_builtin()) {
      out_transformed_tp = ndt::make_fixed_bytes(storage_dt.get_data_size(), storage_dt.get_data_alignment());
      out_was_transformed = true;
    }
    else if (storage_dt.is_pod() && storage_dt.extended()->get_arrmeta_size() == 0) {
      out_transformed_tp = ndt::make_fixed_bytes(storage_dt.get_data_size(), storage_dt.get_data_alignment());
      out_was_transformed = true;
    }
    else if (storage_dt.get_id() == string_id) {
      out_transformed_tp =
          ndt::bytes_type::make(static_cast<const ndt::string_type *>(storage_dt.extended())->get_target_alignment());
      out_was_transformed = true;
    }
    else {
      if (dt.get_base_id() == expr_kind_id) {
        out_transformed_tp = storage_dt;
        out_was_transformed = true;
      }
      else {
        // No transformation
        out_transformed_tp = dt;
      }
    }
  }
  else {
    dt.extended()->transform_child_types(&as_storage_type, 0, NULL, out_transformed_tp, out_was_transformed);
  }
}
} // anonymous namespace

nd::array nd::array::storage() const
{
  ndt::type storage_dt = get_type();
  bool was_transformed = false;
  as_storage_type(get_type(), 0, NULL, storage_dt, was_transformed);
  if (was_transformed) {
    return make_array_clone_with_new_type(*this, storage_dt);
  }
  else {
    return *this;
  }
}

nd::array nd::array::at_array(intptr_t nindices, const irange *indices, bool collapse_leading) const
{
  if (!get_type().is_indexable()) {
    if (nindices != 0) {
      throw too_many_indices(get_type(), nindices, 0);
    }
    return *this;
  }
  else {
    ndt::type this_dt = get()->tp;
    ndt::type dt = get()->tp->apply_linear_index(nindices, indices, 0, this_dt, collapse_leading);
    array result;
    if (!dt.is_builtin()) {
      intrusive_ptr<memory_block_data> memblock = make_array_memory_block(dt.extended()->get_arrmeta_size());
      result = array(reinterpret_cast<array_preamble *>(memblock.get()), true);
      result.get()->tp = dt;
    }
    else {
      result = array(reinterpret_cast<array_preamble *>(make_array_memory_block(0).get()), true);
      result.get()->tp = dt;
    }
    result.get()->data = get()->data;
    if (get()->owner) {
      result.get()->owner = get()->owner;
    }
    else {
      // If the data reference is NULL, the data is embedded in the array itself
      result.get()->owner = get();
    }
    intptr_t offset = get()->tp->apply_linear_index(nindices, indices, get()->metadata(), dt, result.get()->metadata(),
                                                    intrusive_ptr<memory_block_data>(get(), true), 0, this_dt,
                                                    collapse_leading, &result.get()->data, result.get()->owner);
    result.get()->data += offset;
    result.get()->flags = get()->flags;
    return result;
  }
}

nd::array nd::array::assign(const array &rhs, assign_error_mode error_mode) const
{
  return nd::assign({rhs}, {{"error_mode", error_mode}, {"dst", *this}});
}

nd::array nd::array::assign_na() const { return nd::assign_na({}, {{"dst", *this}}); }

void nd::array::flag_as_immutable()
{
  // If it's already immutable, everything's ok
  if ((get_flags() & immutable_access_flag) != 0) {
    return;
  }

  // Check that nobody else is peeking into our data
  bool ok = true;
  if (intrusive_ptr<array_preamble>::get()->m_use_count != 1) {
    // More than one reference to the array itself
    ok = false;
  }
  else if (get()->owner && (get()->owner->m_use_count != 1 ||
                            !(get()->owner->m_type == fixed_size_pod_memory_block_type ||
                              get()->owner->m_type == pod_memory_block_type))) {
    // More than one reference to the array's data, or the reference is to
    // something
    // other than a memblock owning its data, such as an external memblock.
    ok = false;
  }
  else if (!get()->tp.is_builtin() && !get()->tp->is_unique_data_owner(get()->metadata())) {
    ok = false;
  }

  if (ok) {
    // Finalize any allocated data in the arrmeta
    if (!get()->tp.is_builtin()) {
      get()->tp->arrmeta_finalize_buffers(get()->metadata());
    }
    // Clear the write flag, and set the immutable flag
    get()->flags = (get()->flags & ~(uint64_t)write_access_flag) | immutable_access_flag;
  }
  else {
    stringstream ss;
    ss << "Unable to flag array of type " << get_type() << " as immutable, because ";
    ss << "it does not uniquely own all of its data";
    throw runtime_error(ss.str());
  }
}

nd::array nd::array::p(const char *name) const { return nd::make_field_access_kernel(get_dtype(), name)(eval()); }

nd::array nd::array::p(const std::string &name) const { return p(name.c_str()); }

// Unpack a type property.
template <typename T>
static inline nd::array unpack(bool is_vector, const char *data)
{
  if (is_vector) {
    return *reinterpret_cast<const std::vector<T> *>(data);
  }
  else {
    return *reinterpret_cast<const T *>(data);
  }
}

nd::array nd::array::from_type_property(const std::pair<ndt::type, const char *> &pair)
{
  type_id_t id = pair.first.get_id();
  bool is_vector = false;

  if (id == fixed_dim_id) {
    id = pair.first.get_dtype().get_id();
    is_vector = true;
  }

  switch (id) {
  case bool_id:
    return unpack<bool1>(is_vector, pair.second);
  case int8_id:
    return unpack<int8>(is_vector, pair.second);
  case int16_id:
    return unpack<int16>(is_vector, pair.second);
  case int32_id:
    return unpack<int32>(is_vector, pair.second);
  case int64_id:
    return unpack<int64>(is_vector, pair.second);
  case int128_id:
    return unpack<int128>(is_vector, pair.second);
  case uint8_id:
    return unpack<uint8>(is_vector, pair.second);
  case uint16_id:
    return unpack<uint16>(is_vector, pair.second);
  case uint32_id:
    return unpack<uint32>(is_vector, pair.second);
  case uint64_id:
    return unpack<uint64>(is_vector, pair.second);
  case uint128_id:
    return unpack<uint128>(is_vector, pair.second);
  case float16_id:
    return unpack<float16>(is_vector, pair.second);
  case float32_id:
    return unpack<float32>(is_vector, pair.second);
  case float64_id:
    return unpack<float64>(is_vector, pair.second);
  case float128_id:
    return unpack<float128>(is_vector, pair.second);
  case complex_float32_id:
    return unpack<complex64>(is_vector, pair.second);
  case complex_float64_id:
    return unpack<complex128>(is_vector, pair.second);
  case type_id:
    return unpack<ndt::type>(is_vector, pair.second);
  case string_id:
    return unpack<std::string>(is_vector, pair.second);
  default:
    throw runtime_error("invalid type property");
  }
}

nd::array nd::array::eval() const
{
  const ndt::type &current_tp = get_type();
  if (!current_tp.is_expression()) {
    return *this;
  }
  else {
    // Create a canonical type for the result
    const ndt::type &dt = current_tp.get_canonical_type();
    array result(empty(dt));
    if (dt.get_id() == fixed_dim_id) {
      // Reorder strides of output strided dimensions in a KEEPORDER fashion
      dt.extended<ndt::fixed_dim_type>()->reorder_default_constructed_strides(result.get()->metadata(), get_type(),
                                                                              get()->metadata());
    }
    result.assign(*this);
    return result;
  }
}

nd::array nd::array::eval_immutable() const
{
  const ndt::type &current_tp = get_type();
  if ((get_access_flags() & immutable_access_flag) && !current_tp.is_expression()) {
    return *this;
  }
  else {
    // Create a canonical type for the result
    const ndt::type &dt = current_tp.get_canonical_type();
    array result(nd::empty(dt));
    if (dt.get_id() == fixed_dim_id) {
      // Reorder strides of output strided dimensions in a KEEPORDER fashion
      dt.extended<ndt::fixed_dim_type>()->reorder_default_constructed_strides(result.get()->metadata(), get_type(),
                                                                              get()->metadata());
    }
    result.assign(*this);
    result.get()->flags = immutable_access_flag | read_access_flag;
    return result;
  }
}

nd::array nd::array::eval_copy(uint32_t access_flags) const
{
  const ndt::type &current_tp = get_type();
  const ndt::type &dt = current_tp.get_canonical_type();
  array result(nd::empty(dt));
  if (dt.get_id() == fixed_dim_id) {
    // Reorder strides of output strided dimensions in a KEEPORDER fashion
    dt.extended<ndt::fixed_dim_type>()->reorder_default_constructed_strides(result.get()->metadata(), get_type(),
                                                                            get()->metadata());
  }
  result.assign(*this);
  // If the access_flags are 0, use the defaults
  access_flags = access_flags ? access_flags : (int32_t)nd::default_access_flags;
  // If the access_flags are just readonly, add immutable
  // because we just created a unique instance
  access_flags =
      (access_flags != nd::read_access_flag) ? access_flags : (nd::read_access_flag | nd::immutable_access_flag);
  result.get()->flags = access_flags;
  return result;
}

nd::array nd::array::to_host() const
{
  array result = empty(get_type().without_memory_type());
  result.assign(*this);

  return result;
}

#ifdef DYND_CUDA

nd::array nd::array::to_cuda_host(unsigned int cuda_host_flags) const
{
  array result = empty(make_cuda_host(get_type().without_memory_type(), cuda_host_flags));
  result.assign(*this);

  return result;
}

nd::array nd::array::to_cuda_device() const
{
  array result = empty(make_cuda_device(get_type().without_memory_type()));
  result.assign(*this);

  return result;
}

#endif // DYND_CUDA

bool nd::array::is_na() const
{
  ndt::type tp = get_type();
  if (tp.get_id() == option_id) {
    return !nd::old_is_avail(tp, get()->metadata(), cdata());
  }

  return false;
}

bool nd::array::equals_exact(const array &rhs) const
{
  if (get() == rhs.get()) {
    return true;
  }
  else if (get_type() != rhs.get_type()) {
    return false;
  }
  else if (get_ndim() == 0) {
    return (*this == rhs).as<bool>();
  }
  else if (get_type().get_id() == var_dim_id) {
    // If there's a leading var dimension, convert it to strided and compare
    // (Note: this is an inefficient hack)
    ndt::type tp = ndt::make_fixed_dim(get_shape()[0], get_type().extended<ndt::base_dim_type>()->get_element_type());
    return nd::old_view(*this, tp).equals_exact(nd::old_view(rhs, tp));
  }
  else {
    // First compare the shape, to avoid triggering an exception in common cases
    size_t ndim = get_ndim();
    if (ndim == 1 && get_dim_size() == 0 && rhs.get_dim_size() == 0) {
      return true;
    }
    dimvector shape0(ndim), shape1(ndim);
    get_shape(shape0.get());
    rhs.get_shape(shape1.get());
    if (memcmp(shape0.get(), shape1.get(), ndim * sizeof(intptr_t)) != 0) {
      return false;
    }
    try {
      array_iter<0, 2> iter(*this, rhs);
      if (!iter.empty()) {
        do {
          const char *const src[2] = {iter.data<0>(), iter.data<1>()};
          ndt::type tp[2] = {iter.get_uniform_dtype<0>(), iter.get_uniform_dtype<1>()};
          const char *arrmeta[2] = {iter.arrmeta<0>(), iter.arrmeta<1>()};
          ndt::type dst_tp = ndt::make_type<bool1>();
          if (not_equal->call(dst_tp, 2, tp, arrmeta, const_cast<char *const *>(src), 0, NULL,
                              std::map<std::string, ndt::type>())
                  .as<bool>()) {
            return false;
          }
        } while (iter.next());
      }
      return true;
    }
    catch (const broadcast_error &) {
      // If there's a broadcast error in a variable-sized dimension, return
      // false for it too
      return false;
    }
  }
}

nd::array nd::array::cast(const ndt::type &tp) const { return nd::assign({*this}, {{"dst_tp", tp}}); }

namespace {
struct cast_dtype_extra {
  cast_dtype_extra(const ndt::type &tp, size_t ru) : replacement_tp(tp), replace_ndim(ru), out_can_view_data(true) {}
  const ndt::type &replacement_tp;
  intptr_t replace_ndim;
  bool out_can_view_data;
};
static void cast_dtype(const ndt::type &dt, intptr_t DYND_UNUSED(arrmeta_offset), void *extra,
                       ndt::type &out_transformed_tp, bool &out_was_transformed)
{
  cast_dtype_extra *e = reinterpret_cast<cast_dtype_extra *>(extra);
  intptr_t replace_ndim = e->replace_ndim;
  if (dt.get_ndim() > replace_ndim) {
    dt.extended()->transform_child_types(&cast_dtype, 0, extra, out_transformed_tp, out_was_transformed);
  }
  else {
    if (replace_ndim > 0) {
      // If the dimension we're replacing doesn't change, then
      // avoid creating the convert type at this level
      if (dt.get_id() == e->replacement_tp.get_id()) {
        bool can_keep_dim = false;
        ndt::type child_dt, child_replacement_tp;
        switch (dt.get_id()) {
        /*
                case cfixed_dim_id: {
                  const cfixed_dim_type *dt_fdd =
           dt.extended<cfixed_dim_type>();
                  const cfixed_dim_type *r_fdd = static_cast<const
           cfixed_dim_type *>(
                      e->replacement_tp.extended());
                  if (dt_fdd->get_fixed_dim_size() ==
           r_fdd->get_fixed_dim_size() &&
                      dt_fdd->get_fixed_stride() == r_fdd->get_fixed_stride()) {
                    can_keep_dim = true;
                    child_dt = dt_fdd->get_element_type();
                    child_replacement_tp = r_fdd->get_element_type();
                  }
                }
        */
        case var_dim_id: {
          const ndt::base_dim_type *dt_budd = dt.extended<ndt::base_dim_type>();
          const ndt::base_dim_type *r_budd = static_cast<const ndt::base_dim_type *>(e->replacement_tp.extended());
          can_keep_dim = true;
          child_dt = dt_budd->get_element_type();
          child_replacement_tp = r_budd->get_element_type();
          break;
        }
        default:
          break;
        }
        if (can_keep_dim) {
          cast_dtype_extra extra_child(child_replacement_tp, replace_ndim - 1);
          dt.extended()->transform_child_types(&cast_dtype, 0, &extra_child, out_transformed_tp, out_was_transformed);
          return;
        }
      }
    }
    throw std::runtime_error("trying to make convert_type");
    //    out_transformed_tp = ndt::convert_type::make(e->replacement_tp, dt);
    // Only flag the transformation if this actually created a convert type
    if (out_transformed_tp.extended() != e->replacement_tp.extended()) {
      out_was_transformed = true;
      e->out_can_view_data = false;
    }
  }
}
} // anonymous namespace

nd::array nd::array::ucast(const ndt::type &scalar_tp, intptr_t replace_ndim) const
{
  // This creates a type which has a convert type for every scalar of different
  // type.
  // The result has the exact same arrmeta and data, so we just have to swap in
  // the new
  // type in a shallow copy.
  ndt::type replaced_tp;
  bool was_transformed = false;
  cast_dtype_extra extra(scalar_tp, replace_ndim);
  cast_dtype(get_type(), 0, &extra, replaced_tp, was_transformed);
  if (was_transformed) {
    return make_array_clone_with_new_type(*this, replaced_tp);
  }
  else {
    return *this;
  }
}

nd::array nd::array::view(const ndt::type &tp) const { return nd::old_view(*this, tp); }

nd::array nd::array::uview(const ndt::type &uniform_dt, intptr_t replace_ndim) const
{
  // Use the view function specifying to replace all dimensions
  return view(get_type().with_replaced_dtype(uniform_dt, replace_ndim));
}

namespace {
struct permute_dims_data {
  intptr_t ndim, i;
  const intptr_t *axes;
  char *arrmeta;
};
static void permute_type_dims(const ndt::type &tp, intptr_t arrmeta_offset, void *extra, ndt::type &out_transformed_tp,
                              bool &out_was_transformed)
{
  permute_dims_data *pdd = reinterpret_cast<permute_dims_data *>(extra);
  intptr_t i = pdd->i;
  if (pdd->axes[i] == i) {
    // Stationary axis
    if (pdd->i == pdd->ndim - 1) {
      // No more perm dimensions left, leave type as is
      out_transformed_tp = tp;
    }
    else {
      if (tp.get_base_id() == dim_kind_id) {
        ++pdd->i;
      }
      tp.extended()->transform_child_types(&permute_type_dims, arrmeta_offset, extra, out_transformed_tp,
                                           out_was_transformed);
      pdd->i = i;
    }
  }
  else {
    // Find the smallest interval of mutually permuted axes
    intptr_t max_i = pdd->axes[i], loop_i = i + 1;
    while (loop_i <= max_i && loop_i < pdd->ndim) {
      if (pdd->axes[loop_i] > max_i) {
        max_i = pdd->axes[loop_i];
      }
      ++loop_i;
    }
    // We must have enough consecutive strided dimensions
    if (tp.get_strided_ndim() < max_i - i + 1) {
      stringstream ss;
      ss << "Cannot permute non-strided dimensions in type " << tp;
      throw invalid_argument(ss.str());
    }
    ndt::type subtp = tp.extended<ndt::base_dim_type>()->get_element_type();
    for (loop_i = i + 1; loop_i <= max_i; ++loop_i) {
      subtp = subtp.extended<ndt::base_dim_type>()->get_element_type();
    }
    intptr_t perm_ndim = max_i - i + 1;
    // If there are more permutation axes left, process the subtype
    if (max_i < pdd->ndim - 1) {
      pdd->i = max_i + 1;
      tp.extended()->transform_child_types(&permute_type_dims,
                                           arrmeta_offset + perm_ndim * sizeof(fixed_dim_type_arrmeta), extra, subtp,
                                           out_was_transformed);
    }
    // Apply the permutation
    dimvector shape(perm_ndim), permuted_shape(perm_ndim);
    shortvector<size_stride_t> perm_arrmeta(perm_ndim);
    size_stride_t *original_arrmeta = reinterpret_cast<size_stride_t *>(pdd->arrmeta + arrmeta_offset);
    memcpy(perm_arrmeta.get(), original_arrmeta, perm_ndim * sizeof(size_stride_t));
    tp.extended()->get_shape(max_i - i + 1, 0, shape.get(), NULL, NULL);
    for (loop_i = i; loop_i <= max_i; ++loop_i) {
      intptr_t srcidx = pdd->axes[loop_i] - i;
      permuted_shape[loop_i - i] = shape[srcidx];
      original_arrmeta[loop_i - i] = perm_arrmeta[srcidx];
    }
    out_transformed_tp = ndt::make_type(perm_ndim, permuted_shape.get(), subtp);
    out_was_transformed = true;
  }
}
} // anonymous namespace

nd::array nd::array::permute(intptr_t ndim, const intptr_t *axes) const
{
  if (ndim > get_ndim()) {
    stringstream ss;
    ss << "Too many dimensions provided for axis permutation, got " << ndim << " for type " << get_type();
    throw invalid_argument(ss.str());
  }
  if (!is_valid_perm(ndim, axes)) {
    stringstream ss;
    ss << "Invalid permutation provided to dynd axis permute: [";
    for (intptr_t i = 0; i < ndim; ++i) {
      ss << axes[i] << (i != ndim - 1 ? " " : "");
    }
    ss << "]";
    throw invalid_argument(ss.str());
  }

  // Make a shallow copy of the arrmeta. When we permute the type,
  // its arrmeta has identical structure, so we can fix it up
  // while we're transforming the type.
  nd::array res(reinterpret_cast<array_preamble *>(
                    shallow_copy_array_memory_block(intrusive_ptr<memory_block_data>(get(), true)).get()),
                true);

  ndt::type transformed_tp;
  bool was_transformed = false;
  permute_dims_data pdd;
  pdd.ndim = ndim;
  pdd.i = 0;
  pdd.axes = axes;
  pdd.arrmeta = res.get()->metadata();
  permute_type_dims(get_type(), 0, &pdd, transformed_tp, was_transformed);

  // We can now substitute our transformed type into
  // the result array
  res.get()->tp = transformed_tp;

  return res;
}

nd::array nd::array::rotate(intptr_t to, intptr_t from) const
{
  if (from < to) {
    intptr_t ndim = to + 1;
    dimvector axes(ndim);
    for (intptr_t i = 0; i < from; ++i) {
      axes[i] = i;
    }
    for (intptr_t i = from; i < to; ++i) {
      axes[i] = i + 1;
    }
    axes[to] = from;

    return permute(ndim, axes.get());
  }

  if (from > to) {
    intptr_t ndim = from + 1;
    dimvector axes(ndim);
    for (intptr_t i = 0; i < to; ++i) {
      axes[i] = i;
    }
    axes[to] = from;
    for (intptr_t i = to + 1; i <= from; ++i) {
      axes[i] = i - 1;
    }

    return permute(ndim, axes.get());
  }

  return *this;
}

nd::array nd::array::transpose() const
{
  intptr_t ndim = get_ndim();
  dimvector axes(ndim);
  for (intptr_t i = 0; i < ndim; ++i) {
    axes[i] = ndim - i - 1;
  }

  return permute(ndim, axes.get());
}

namespace {
struct replace_compatible_dtype_extra {
  replace_compatible_dtype_extra(const ndt::type &tp, intptr_t replace_ndim_)
      : replacement_tp(tp), replace_ndim(replace_ndim_)
  {
  }
  const ndt::type &replacement_tp;
  intptr_t replace_ndim;
};
static void replace_compatible_dtype(const ndt::type &tp, intptr_t DYND_UNUSED(arrmeta_offset), void *extra,
                                     ndt::type &out_transformed_tp, bool &out_was_transformed)
{
  const replace_compatible_dtype_extra *e = reinterpret_cast<const replace_compatible_dtype_extra *>(extra);
  const ndt::type &replacement_tp = e->replacement_tp;
  if (tp.get_ndim() == e->replace_ndim) {
    if (tp != replacement_tp) {
      if (!tp.data_layout_compatible_with(replacement_tp)) {
        stringstream ss;
        ss << "The dynd type " << tp << " is not ";
        ss << " data layout compatible with " << replacement_tp;
        ss << ", so a substitution cannot be made.";
        throw runtime_error(ss.str());
      }
      out_transformed_tp = replacement_tp;
      out_was_transformed = true;
    }
  }
  else {
    tp.extended()->transform_child_types(&replace_compatible_dtype, 0, extra, out_transformed_tp, out_was_transformed);
  }
}
} // anonymous namespace

nd::array nd::array::replace_dtype(const ndt::type &replacement_tp, intptr_t replace_ndim) const
{
  // This creates a type which swaps in the new dtype for
  // the existing one. It raises an error if the data layout
  // is incompatible
  ndt::type replaced_tp;
  bool was_transformed = false;
  replace_compatible_dtype_extra extra(replacement_tp, replace_ndim);
  replace_compatible_dtype(get_type(), 0, &extra, replaced_tp, was_transformed);
  if (was_transformed) {
    return make_array_clone_with_new_type(*this, replaced_tp);
  }
  else {
    return *this;
  }
}

nd::array nd::array::new_axis(intptr_t i, intptr_t new_ndim) const
{
  ndt::type src_tp = get_type();
  ndt::type dst_tp = src_tp.with_new_axis(i, new_ndim);

  // This is taken from view_concrete in view.cpp
  nd::array res(reinterpret_cast<array_preamble *>(make_array_memory_block(dst_tp.get_arrmeta_size()).get()), true);
  res.get()->data = get()->data;
  if (!get()->owner) {
    res.get()->owner = get();
  }
  else {
    res.get()->owner = get_data_memblock();
  }
  res.get()->tp = dst_tp;
  res.get()->flags = get()->flags;

  char *src_arrmeta = const_cast<char *>(get()->metadata());
  char *dst_arrmeta = res.get()->metadata();
  for (intptr_t j = 0; j < i; ++j) {
    dst_tp.extended<ndt::base_dim_type>()->arrmeta_copy_construct_onedim(dst_arrmeta, src_arrmeta,
                                                                         intrusive_ptr<memory_block_data>());
    src_tp = src_tp.get_type_at_dimension(&src_arrmeta, 1);
    dst_tp = dst_tp.get_type_at_dimension(&dst_arrmeta, 1);
  }
  for (intptr_t j = 0; j < new_ndim; ++j) {
    size_stride_t *smd = reinterpret_cast<size_stride_t *>(dst_arrmeta);
    smd->dim_size = 1;
    smd->stride = 0; // Should this not be zero?
    dst_tp = dst_tp.get_type_at_dimension(&dst_arrmeta, 1);
  }
  if (!dst_tp.is_builtin()) {
    dst_tp.extended()->arrmeta_copy_construct(dst_arrmeta, src_arrmeta, intrusive_ptr<memory_block_data>());
  }

  return res;
}

namespace {
static void view_scalar_types(const ndt::type &dt, intptr_t DYND_UNUSED(arrmeta_offset), void *extra,
                              ndt::type &out_transformed_tp, bool &out_was_transformed)
{
  if (dt.is_scalar()) {
    const ndt::type *e = reinterpret_cast<const ndt::type *>(extra);
    // If things aren't simple, use a view_type
    if (dt.get_base_id() == expr_kind_id || dt.get_data_size() != e->get_data_size() || !dt.is_pod() || !e->is_pod()) {
      // Some special cases that have the same memory layouts
      switch (dt.get_id()) {
      case string_id:
      case bytes_id:
        switch (e->get_id()) {
        case string_id:
        case bytes_id:
          // All these types have the same data/arrmeta layout,
          // allow a view whenever the alignment allows it
          if (e->get_data_alignment() <= dt.get_data_alignment()) {
            out_transformed_tp = *e;
            out_was_transformed = true;
            return;
          }
          break;
        default:
          break;
        }
        break;
      default:
        break;
      }
      throw std::runtime_error("creating a view_type");
      //      out_transformed_tp = ndt::view_type::make(*e, dt);
      out_was_transformed = true;
    }
    else {
      out_transformed_tp = *e;
      if (dt != *e) {
        out_was_transformed = true;
      }
    }
  }
  else {
    dt.extended()->transform_child_types(&view_scalar_types, 0, extra, out_transformed_tp, out_was_transformed);
  }
}
} // anonymous namespace

nd::array nd::array::view_scalars(const ndt::type &scalar_tp) const
{
  const ndt::type &array_type = get_type();
  size_t uniform_ndim = array_type.get_ndim();
  // First check if we're dealing with a simple one dimensional block of memory
  // we can reinterpret
  // at will.
  if (uniform_ndim == 1 && array_type.get_id() == fixed_dim_id) {
    const ndt::fixed_dim_type *sad = array_type.extended<ndt::fixed_dim_type>();
    const fixed_dim_type_arrmeta *md = reinterpret_cast<const fixed_dim_type_arrmeta *>(get()->metadata());
    const ndt::type &edt = sad->get_element_type();
    if (edt.is_pod() && (intptr_t)edt.get_data_size() == md->stride &&
        sad->get_element_type().get_base_id() != expr_kind_id) {
      intptr_t nbytes = md->dim_size * edt.get_data_size();
      // Make sure the element size divides into the # of bytes
      if (nbytes % scalar_tp.get_data_size() != 0) {
        std::stringstream ss;
        ss << "cannot view array with " << nbytes << " bytes as type ";
        ss << scalar_tp << ", because its element size " << scalar_tp.get_data_size();
        ss << " doesn't divide evenly into the total array size " << nbytes;
        throw std::runtime_error(ss.str());
      }
      // Create the result array, adjusting the type if the data isn't aligned
      // correctly
      char *data_ptr = get()->data;
      ndt::type result_tp;
      intptr_t dim_size = nbytes / scalar_tp.get_data_size();
      if ((((uintptr_t)data_ptr) & (scalar_tp.get_data_alignment() - 1)) == 0) {
        result_tp = ndt::make_fixed_dim(dim_size, scalar_tp);
      }
      else {
        throw std::runtime_error("creating an unaligned type");
        //        result_tp = ndt::make_fixed_dim(dim_size, make_unaligned(scalar_tp));
      }
      array result(
          reinterpret_cast<array_preamble *>(make_array_memory_block(result_tp.extended()->get_arrmeta_size()).get()),
          true);
      // Copy all the array arrmeta fields
      result.get()->data = get()->data;
      if (get()->owner) {
        result.get()->owner = get()->owner;
      }
      else {
        result.get()->owner = intrusive_ptr<array_preamble>::get();
      }
      result.get()->tp = result_tp;
      result.get()->flags = get()->flags;
      // The result has one strided ndarray field
      fixed_dim_type_arrmeta *result_md = reinterpret_cast<fixed_dim_type_arrmeta *>(result.get()->metadata());
      result_md->dim_size = dim_size;
      result_md->stride = scalar_tp.get_data_size();
      return result;
    }
  }

  // Transform the scalars into view types
  ndt::type viewed_tp;
  bool was_transformed = false;
  view_scalar_types(get_type(), 0, const_cast<void *>(reinterpret_cast<const void *>(&scalar_tp)), viewed_tp,
                    was_transformed);
  return make_array_clone_with_new_type(*this, viewed_tp);
}

void nd::array::debug_print(std::ostream &o, const std::string &indent) const
{
  o << indent << "------ array\n";
  if (intrusive_ptr<array_preamble>::get()) {
    const array_preamble *ndo = get();
    o << " address: " << (void *)intrusive_ptr<array_preamble>::get() << "\n";
    o << " refcount: " << static_cast<long>(ndo->m_use_count) << "\n";
    o << " type:\n";
    o << "  pointer: " << (void *)ndo->tp.extended() << "\n";
    o << "  type: " << get_type() << "\n";
    if (!get_type().is_builtin()) {
      o << "  type refcount: " << get_type().extended()->get_use_count() << "\n";
    }
    o << " arrmeta:\n";
    o << "  flags: " << ndo->flags << " (";
    if (ndo->flags & read_access_flag)
      o << "read_access ";
    if (ndo->flags & write_access_flag)
      o << "write_access ";
    if (ndo->flags & immutable_access_flag)
      o << "immutable ";
    o << ")\n";
    if (!ndo->tp.is_builtin()) {
      o << "  type-specific arrmeta:\n";
      ndo->tp->arrmeta_debug_print(get()->metadata(), o, indent + "   ");
    }
    o << " data:\n";
    o << "   pointer: " << (void *)ndo->data << "\n";
    o << "   reference: " << (void *)ndo->owner.get();
    if (!ndo->owner) {
      o << " (embedded in array memory)\n";
    }
    else {
      o << "\n";
    }
    if (ndo->owner) {
      memory_block_debug_print(ndo->owner.get(), o, "    ");
    }
  }
  else {
    o << indent << "NULL\n";
  }
  o << indent << "------" << endl;
}

std::ostream &nd::operator<<(std::ostream &o, const array &rhs)
{
  if (!rhs.is_null()) {
    o << "array(";
    array v = rhs.eval();
    if (v.get()->tp.is_builtin()) {
      print_builtin_scalar(v.get()->tp.get_id(), o, v.get()->data);
    }
    else {
      stringstream ss;
      v.get()->tp->print_data(ss, v.get()->metadata(), v.get()->data);
      print_indented(o, "      ", ss.str(), true);
    }
    o << ",\n      type=\"" << rhs.get_type() << "\")";
  }
  else {
    o << "array()";
  }
  return o;
}

nd::array nd::as_struct(size_t size, const pair<const char *, array> *pairs)
{
  std::vector<std::string> names(size);
  std::vector<ndt::type> types(size);
  for (size_t i = 0; i < size; ++i) {
    names[i] = pairs[i].first;
    types[i] = pairs[i].second.get_type();
  }

  array res = empty(ndt::struct_type::make(names, types));
  for (size_t i = 0; i < size; ++i) {
    res(i).assign(pairs[i].second);
  }

  return res;
}

nd::array nd::empty_shell(const ndt::type &tp)
{
  if (tp.is_builtin()) {
    char *data_ptr = NULL;
    intptr_t data_size = tp.get_data_size();
    intptr_t data_alignment = tp.get_data_alignment();
    intrusive_ptr<memory_block_data> result(make_array_memory_block(0, data_size, data_alignment, &data_ptr));
    array_preamble *preamble = reinterpret_cast<array_preamble *>(result.get());
    // It's a builtin type id, so no incref
    preamble->tp = tp;
    preamble->data = data_ptr;
    preamble->owner = NULL;
    preamble->flags = nd::read_access_flag | nd::write_access_flag;
    return nd::array(preamble, true);
  }
  else if (!tp.is_symbolic()) {
    char *data_ptr = NULL;
    size_t arrmeta_size = tp.extended()->get_arrmeta_size();
    size_t data_size = tp.extended()->get_default_data_size();
    intrusive_ptr<memory_block_data> result;
    if (tp.get_base_id() != memory_id) {
      // Allocate memory the default way
      result = make_array_memory_block(arrmeta_size, data_size, tp.get_data_alignment(), &data_ptr);
      if (tp.get_flags() & type_flag_zeroinit) {
        memset(data_ptr, 0, data_size);
      }
      if (tp.get_flags() & type_flag_construct) {
        tp.extended()->data_construct(NULL, data_ptr);
      }
    }
    else {
      // Allocate memory based on the memory_kind type
      result = make_array_memory_block(arrmeta_size);
      tp.extended<ndt::base_memory_type>()->data_alloc(&data_ptr, data_size);
      if (tp.get_flags() & type_flag_zeroinit) {
        tp.extended<ndt::base_memory_type>()->data_zeroinit(data_ptr, data_size);
      }
    }
    array_preamble *preamble = reinterpret_cast<array_preamble *>(result.get());
    preamble->tp = tp;
    preamble->data = data_ptr;
    preamble->owner = NULL;
    preamble->flags = nd::read_access_flag | nd::write_access_flag;
    return nd::array(preamble, true);
  }
  else {
    stringstream ss;
    ss << "Cannot create a dynd array with symbolic type " << tp;
    throw type_error(ss.str());
  }
}

nd::array nd::empty(const ndt::type &tp)
{
  // Create an empty shell
  nd::array res = nd::empty_shell(tp);
  // Construct the arrmeta with default settings
  if (tp.get_arrmeta_size() > 0) {
    array_preamble *preamble = res.get();
    preamble->tp->arrmeta_default_construct(reinterpret_cast<char *>(preamble + 1), true);
  }
  return res;
}

nd::array nd::empty_like(const nd::array &rhs, const ndt::type &uniform_tp)
{
  if (rhs.get_ndim() == 0) {
    return nd::empty(uniform_tp);
  }
  else {
    size_t ndim = rhs.get_type().extended()->get_ndim();
    dimvector shape(ndim);
    rhs.get_shape(shape.get());
    array result = empty(make_fixed_dim(ndim, shape.get(), uniform_tp));
    // Reorder strides of output strided dimensions in a KEEPORDER fashion
    if (result.get_type().get_id() == fixed_dim_id) {
      result.get_type().extended<ndt::fixed_dim_type>()->reorder_default_constructed_strides(
          result.get()->metadata(), rhs.get_type(), rhs.get()->metadata());
    }
    return result;
  }
}

nd::array nd::empty_like(const nd::array &rhs)
{
  ndt::type dt;
  if (rhs.get()->tp.is_builtin()) {
    dt = ndt::type(rhs.get()->tp.get_id());
  }
  else {
    dt = rhs.get()->tp->get_canonical_type();
  }

  if (rhs.is_scalar()) {
    return nd::empty(dt);
  }
  else {
    intptr_t ndim = dt.extended()->get_ndim();
    dimvector shape(ndim);
    rhs.get_shape(shape.get());
    nd::array result = empty(make_fixed_dim(ndim, shape.get(), dt.get_dtype()));
    // Reorder strides of output strided dimensions in a KEEPORDER fashion
    if (result.get_type().get_id() == fixed_dim_id) {
      result.get_type().extended<ndt::fixed_dim_type>()->reorder_default_constructed_strides(
          result.get()->metadata(), rhs.get_type(), rhs.get()->metadata());
    }
    return result;
  }
}

nd::array nd::concatenate(const nd::array &x, const nd::array &y)
{
  if (x.get_ndim() != 1 || y.get_ndim() != 1) {
    throw runtime_error("TODO: nd::concatenate is WIP");
  }

  if (x.get_dtype() != y.get_dtype()) {
    throw runtime_error("dtypes must be the same for concatenate");
  }

  nd::array res = nd::empty(x.get_dim_size() + y.get_dim_size(), x.get_dtype());
  res(irange(0, x.get_dim_size())).assign(x);
  res(irange(x.get_dim_size(), res.get_dim_size())).assign(y);

  return res;
}

nd::array nd::reshape(const nd::array &a, const nd::array &shape)
{
  intptr_t ndim = shape.get_dim_size();

  intptr_t old_ndim = a.get_ndim();
  dimvector old_shape(old_ndim);
  a.get_shape(old_shape.get());

  intptr_t old_size = 1;
  for (intptr_t i = 0; i < old_ndim; ++i) {
    old_size *= old_shape[i];
  }
  intptr_t size = 1;
  for (intptr_t i = 0; i < ndim; ++i) {
    size *= shape(i).as<intptr_t>();
  }

  if (old_size != size) {
    stringstream ss;
    ss << "dynd reshape: cannot reshape to a different total number of "
          "elements, from "
       << old_size << " to " << size;
    throw invalid_argument(ss.str());
  }

  dimvector strides(ndim);
  strides[ndim - 1] = a.get_dtype().get_data_size();
  for (intptr_t i = ndim - 2; i >= 0; --i) {
    strides[i] = shape(i + 1).as<intptr_t>() * strides[i + 1];
  }

  dimvector shape_copy(ndim);
  for (intptr_t i = 0; i < ndim; ++i) {
    shape_copy[i] = shape(i).as<intptr_t>();
  }

  return make_strided_array_from_data(a.get_dtype(), ndim, shape_copy.get(), strides.get(), a.get_flags(), a.data(),
                                      intrusive_ptr<memory_block_data>(a.get(), true), NULL);
}

nd::array nd::memmap(const std::string &DYND_UNUSED(filename), intptr_t DYND_UNUSED(begin), intptr_t DYND_UNUSED(end),
                     uint32_t DYND_UNUSED(access))
{
  throw std::runtime_error("nd::memmap is not yet implemented");

  /*
    // ToDo: This was based on the variable-sized bytes type, which changed.

    if (access == 0) {
      access = nd::default_access_flags;
    }

    char *mm_ptr = NULL;
    intptr_t mm_size = 0;
    // Create a memory mapped memblock of the file
    intrusive_ptr<memory_block_data> mm = make_memmap_memory_block(filename, access, &mm_ptr, &mm_size, begin, end);
    // Create a bytes array referring to the data.
    ndt::type dt = ndt::bytes_type::make(1);
    char *data_ptr = 0;
    nd::array result(make_array_memory_block(dt.extended()->get_arrmeta_size(), dt.get_data_size(),
                                             dt.get_data_alignment(), &data_ptr));
    // Set the bytes extents
    reinterpret_cast<bytes *>(data_ptr)->assign(mm_ptr, mm_size);
    // Set the array arrmeta
    array_preamble *ndo = result.get();
    ndo->tp = dt.release();
    ndo->data = data_ptr;
    ndo->owner = NULL;
    ndo->flags = access;
    // Set the bytes arrmeta, telling the system
    // about the memmapped memblock
    ndo_meta->blockref = mm.release();
    return result;
  */
}

nd::array nd::combine_into_tuple(size_t field_count, const array *field_values)
{
  // Make the pointer types
  vector<ndt::type> field_types(field_count);
  for (size_t i = 0; i != field_count; ++i) {
    field_types[i] = ndt::pointer_type::make(field_values[i].get_type());
  }
  // The flags are the intersection of all the input flags
  uint64_t flags = field_values[0].get_flags();
  for (size_t i = 1; i != field_count; ++i) {
    flags &= field_values[i].get_flags();
  }

  ndt::type result_type = ndt::tuple_type::make(field_types);
  const ndt::tuple_type *fsd = result_type.extended<ndt::tuple_type>();
  char *data_ptr = NULL;

  array result(
      reinterpret_cast<array_preamble *>(make_array_memory_block(fsd->get_arrmeta_size(), fsd->get_default_data_size(),
                                                                 fsd->get_data_alignment(), &data_ptr)
                                             .get()),
      true);
  // Set the array properties
  result.get()->tp = result_type;
  result.get()->data = data_ptr;
  result.get()->owner = NULL;
  result.get()->flags = flags;

  // Set the data offsets arrmeta for the tuple type. It's a bunch of pointer
  // types, so the offsets are pretty simple.
  intptr_t *data_offsets = reinterpret_cast<intptr_t *>(result.get()->metadata());
  for (size_t i = 0; i != field_count; ++i) {
    data_offsets[i] = i * sizeof(void *);
  }

  // Copy all the needed arrmeta
  const std::vector<uintptr_t> &arrmeta_offsets = fsd->get_arrmeta_offsets();
  for (size_t i = 0; i != field_count; ++i) {
    pointer_type_arrmeta *pmeta;
    pmeta = reinterpret_cast<pointer_type_arrmeta *>(result.get()->metadata() + arrmeta_offsets[i]);
    pmeta->offset = 0;
    pmeta->blockref = field_values[i].get()->owner ? field_values[i].get()->owner
                                                   : intrusive_ptr<memory_block_data>(field_values[i].get(), true);

    const ndt::type &field_dt = field_values[i].get_type();
    if (field_dt.get_arrmeta_size() > 0) {
      field_dt.extended()->arrmeta_copy_construct(reinterpret_cast<char *>(pmeta + 1),
                                                  field_values[i].get()->metadata(),
                                                  intrusive_ptr<memory_block_data>(field_values[i].get(), true));
    }
  }

  // Set the data pointers
  const char **dp = reinterpret_cast<const char **>(data_ptr);
  for (size_t i = 0; i != field_count; ++i) {
    dp[i] = field_values[i].cdata();
  }
  return result;
}

nd::callable &nd::find_dynamic_function(const char *name) { return callable_registry[name]; }

void dynd::broadcast_input_shapes(intptr_t ninputs, const nd::array *inputs, intptr_t &out_undim, dimvector &out_shape,
                                  shortvector<int> &out_axis_perm)
{
  // Get the number of broadcast dimensions
  intptr_t undim = inputs[0].get_ndim();
  for (intptr_t i = 0; i < ninputs; ++i) {
    intptr_t candidate_undim = inputs[i].get_ndim();
    if (candidate_undim > undim) {
      undim = candidate_undim;
    }
  }

  out_undim = undim;
  out_shape.init(undim);
  out_axis_perm.init(undim);
  intptr_t *shape = out_shape.get();

  // Fill in the broadcast shape
  for (intptr_t k = 0; k < undim; ++k) {
    shape[k] = 1;
  }
  dimvector tmpshape(undim);
  for (intptr_t i = 0; i < ninputs; ++i) {
    intptr_t input_undim = inputs[i].get_ndim();
    inputs[i].get_shape(tmpshape.get());
    intptr_t dimdelta = undim - input_undim;
    for (intptr_t k = dimdelta; k < undim; ++k) {
      intptr_t size = tmpshape[k - dimdelta];
      intptr_t itershape_size = shape[k];
      if (itershape_size == 1) {
        shape[k] = size;
      }
      else if (size < 0) {
        // A negative shape value means variable-sized
        if (itershape_size > 0) {
          shape[k] = -itershape_size;
        }
        else {
          shape[k] = -1;
        }
      }
      else if (itershape_size >= 0) {
        if (size != 1 && itershape_size != size) {
          // cout << "operand " << i << ", comparing size " << itershape_size << " vs " << size << "\n";
          throw broadcast_error(ninputs, inputs);
        }
      }
      else { // itershape_size < 0
        if (itershape_size == -1 && size > 0) {
          shape[k] = -size;
        }
        else if (size > 1 && itershape_size != -size) {
          throw broadcast_error(ninputs, inputs);
        }
      }
    }
  }
  // Fill in the axis permutation
  if (undim > 1) {
    int *axis_perm = out_axis_perm.get();
    // TODO: keeporder behavior, currently always C order
    for (intptr_t i = 0; i < undim; ++i) {
      axis_perm[i] = int(undim - i - 1);
    }
  }
  else if (undim == 1) {
    out_axis_perm[0] = 0;
  }
}

broadcast_error::broadcast_error(const std::string &m) : dynd_exception("broadcast error", m) {}

broadcast_error::~broadcast_error() throw() {}

inline std::string broadcast_error_message(intptr_t dst_ndim, const intptr_t *dst_shape, intptr_t src_ndim,
                                           const intptr_t *src_shape)
{
  stringstream ss;

  ss << "cannot broadcast shape ";
  print_shape(ss, src_ndim, src_shape);
  ss << " to shape ";
  print_shape(ss, dst_ndim, dst_shape);

  return ss.str();
}

broadcast_error::broadcast_error(intptr_t dst_ndim, const intptr_t *dst_shape, intptr_t src_ndim,
                                 const intptr_t *src_shape)
    : dynd_exception("broadcast error", broadcast_error_message(dst_ndim, dst_shape, src_ndim, src_shape))
{
}

inline std::string broadcast_error_message(const nd::array &dst, const nd::array &src)
{
  vector<intptr_t> dst_shape = dst.get_shape(), src_shape = src.get_shape();
  stringstream ss;

  ss << "cannot broadcast dynd array with type ";
  ss << src.get_type() << " and shape ";
  print_shape(ss, src_shape);
  ss << " to type " << dst.get_type() << " and shape ";
  print_shape(ss, dst_shape);

  return ss.str();
}

broadcast_error::broadcast_error(const nd::array &dst, const nd::array &src)
    : dynd_exception("broadcast error", broadcast_error_message(dst, src))
{
}

inline std::string broadcast_error_message(intptr_t ninputs, const nd::array *inputs)
{
  stringstream ss;

  ss << "cannot broadcast input dynd operands with shapes ";
  for (intptr_t i = 0; i < ninputs; ++i) {
    intptr_t undim = inputs[i].get_ndim();
    dimvector shape(undim);
    inputs[i].get_shape(shape.get());
    print_shape(ss, undim, shape.get());
    if (i + 1 != ninputs) {
      ss << " ";
    }
  }

  return ss.str();
}

broadcast_error::broadcast_error(intptr_t ninputs, const nd::array *inputs)
    : dynd_exception("broadcast error", broadcast_error_message(ninputs, inputs))
{
}

inline std::string broadcast_error_message(const ndt::type &dst_tp, const char *dst_arrmeta, const ndt::type &src_tp,
                                           const char *src_arrmeta)
{
  stringstream ss;
  ss << "cannot broadcast input datashape '";
  format_datashape(ss, src_tp, src_arrmeta, NULL, false);
  ss << "' into datashape '";
  format_datashape(ss, dst_tp, dst_arrmeta, NULL, false);
  ss << "'";
  return ss.str();
}

broadcast_error::broadcast_error(const ndt::type &dst_tp, const char *dst_arrmeta, const ndt::type &src_tp,
                                 const char *src_arrmeta)
    : dynd_exception("broadcast error", broadcast_error_message(dst_tp, dst_arrmeta, src_tp, src_arrmeta))
{
}

inline std::string broadcast_error_message(const ndt::type &dst_tp, const char *dst_arrmeta, const char *src_name)
{
  stringstream ss;
  ss << "cannot broadcast input " << src_name << " into datashape '";
  format_datashape(ss, dst_tp, dst_arrmeta, NULL, false);
  ss << "'";
  return ss.str();
}

broadcast_error::broadcast_error(const ndt::type &dst_tp, const char *dst_arrmeta, const char *src_name)
    : dynd_exception("broadcast error", broadcast_error_message(dst_tp, dst_arrmeta, src_name))
{
}

inline std::string broadcast_error_message(intptr_t dst_size, intptr_t src_size, const char *dst_name,
                                           const char *src_name)
{
  stringstream ss;
  ss << "cannot broadcast input " << src_name << " with size " << src_size;
  ss << " into output " << dst_name << " with size " << dst_size;
  return ss.str();
}

broadcast_error::broadcast_error(intptr_t dst_size, intptr_t src_size, const char *dst_name, const char *src_name)
    : dynd_exception("broadcast error", broadcast_error_message(dst_size, src_size, dst_name, src_name))
{
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

nd::array nd::operator<(const array &a0, const array &a1) { return less(a0, a1); }

nd::array nd::operator<=(const array &a0, const array &a1) { return less_equal(a0, a1); }

nd::array nd::operator==(const array &a0, const array &a1) { return equal(a0, a1); }

nd::array nd::operator!=(const array &a0, const array &a1) { return not_equal(a0, a1); }

nd::array nd::operator>=(const array &a0, const array &a1) { return greater_equal(a0, a1); }

nd::array nd::operator>(const array &a0, const array &a1) { return greater(a0, a1); }

nd::array nd::operator+(const array &a0) { return nd::plus(a0); }

nd::array nd::operator-(const array &a0) { return nd::minus(a0); }

nd::array nd::operator!(const array &a0) { return nd::logical_not(a0); }

nd::array nd::operator~(const array &a0) { return nd::bitwise_not(a0); }

nd::array nd::operator+(const array &a0, const array &a1) { return nd::add(a0, a1); }

nd::array nd::operator-(const array &a0, const array &a1) { return nd::subtract(a0, a1); }

nd::array nd::operator*(const array &a0, const array &a1) { return nd::multiply(a0, a1); }

nd::array nd::operator/(const array &a0, const array &a1) { return nd::divide(a0, a1); }

nd::array nd::operator&&(const array &a0, const array &a1) { return nd::logical_and(a0, a1); }

nd::array nd::operator||(const array &a0, const array &a1) { return nd::logical_or(a0, a1); }

nd::array &nd::array::operator+=(const array &rhs)
{
  nd::compound_add({rhs}, {{"dst", *this}});
  return *this;
}

nd::array &nd::array::operator/=(const array &rhs)
{
  nd::compound_div({rhs}, {{"dst", *this}});
  return *this;
}
