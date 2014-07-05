//
// Copyright (C) 2011-14 Mark Wiebe, DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#ifndef _DYND__ARRFUNC_HPP_
#define _DYND__ARRFUNC_HPP_

#include <dynd/config.hpp>
#include <dynd/eval/eval_context.hpp>
#include <dynd/types/base_type.hpp>
#include <dynd/types/funcproto_type.hpp>
#include <dynd/types/arrfunc_type.hpp>
#include <dynd/kernels/ckernel_builder.hpp>
#include <dynd/types/type_pattern_match.hpp>

namespace dynd {

struct arrfunc_type_data;

/**
 * Function prototype for instantiating a ckernel from an
 * arrfunc. To use this function, the
 * caller should first allocate a `ckernel_builder` instance,
 * either from C++ normally or by reserving appropriately aligned/sized
 * data and calling the C function constructor dynd provides. When the
 * data types of the kernel require arrmeta, such as for 'strided'
 * or 'var' dimension types, the arrmeta must be provided as well.
 *
 * \param self  The arrfunc.
 * \param ckb  A ckernel_builder instance where the kernel is placed.
 * \param ckb_offset  The offset into the output ckernel_builder `ckb`
 *                    where the kernel should be placed.
 * \param dst_tp  The destination type of the ckernel to generate. This may be
 *                different from the one in the function prototype, but must
 *                match its pattern.
 * \param dst_arrmeta  The destination arrmeta.
 * \param src_tp  An array of the source types of the ckernel to generate. These may be
 *                different from the ones in the function prototype, but must
 *                match the patterns.
 * \param src_arrmeta  An array of dynd arrmeta pointers,
 *                     corresponding to the source types.
 * \param kernreq  What kind of C function prototype the resulting ckernel should
 *                 follow. Defined by the enum with kernel_request_* values.
 * \param ectx  The evaluation context.
 *
 * \returns  The offset into ``ckb`` immediately after the instantiated ckernel.
 */
typedef intptr_t (*arrfunc_instantiate_t)(
    const arrfunc_type_data *self, dynd::ckernel_builder *ckb,
    intptr_t ckb_offset, const ndt::type &dst_tp, const char *dst_arrmeta,
    const ndt::type *src_tp, const char *const *src_arrmeta, kernel_request_t kernreq,
    const eval::eval_context *ectx);

/**
 * Resolves the destination type for this arrfunc based on the types
 * of the source parameters.
 *
 * \param self  The arrfunc.
 * \param out_dst_tp  To be filled with the destination type.
 * \param src_tp  An array of the source types.
 * \param throw_on_error  If true, should throw when there's an error, if
 *                        false, should return 0 when there's an error.
 *
 * \returns  True on success, false on error (if throw_on_error was false).
 */
typedef int (*arrfunc_resolve_dst_type_t)(const arrfunc_type_data *self,
                                          ndt::type &out_dst_tp,
                                          const ndt::type *src_tp,
                                          int throw_on_error);

/**
 * Returns the shape of the destination array for the provoided inputs
 * and the destination type (which would typically have been produced
 * via the ``resolve_dst_type`` call).
 *
 * \param self  The arrfunc.
 * \param out_shape  This is filled with the shape. It must have size
 *                   ``dst_tp.get_ndim()``.
 * \param dst_tp  The destination type.
 * \param src_tp  Array of source types. It must have the length matching
 *                the number of parameters.
 * \param src_arrmeta  Array of arrmeta corresponding to the source types.
 * \param src_data  Array of data corresponding to the source types/arrmetas.
 *                  This may be an array of NULLs.
 */
typedef void (*arrfunc_resolve_dst_shape_t)(const arrfunc_type_data *self,
                                            intptr_t *out_shape,
                                            const ndt::type &dst_tp,
                                            const ndt::type *src_tp,
                                            const char *const *src_arrmeta,
                                            const char *const *src_data);

/**
 * This is a struct designed for interoperability at
 * the C ABI level. It contains enough information
 * to pass arrfuncs from one library to another
 * with no dependencies between them.
 *
 * The arrfunc can produce a ckernel with with a few
 * variations, like choosing between a single
 * operation and a strided operation, or constructing
 * with different array arrmeta.
 */
struct arrfunc_type_data {
  /**
   * Some memory for the arrfunc to use. If this is not
   * enough space to hold all the data by value, should allocate
   * space on the heap, and free it when free_func is called.
   *
   * On 32-bit platforms, the size of this data is increased by 4
   * so the entire struct is 8-byte aligned.
   */
  char data[4 * 8 + ((sizeof(void *) == 4) ? 4 : 0)];
  /** The function prototype of the arrfunc */
  ndt::type func_proto;
  /**
   * The function which instantiates a ckernel. See the documentation
   * for the function typedef for more details.
   */
  arrfunc_instantiate_t instantiate;
  arrfunc_resolve_dst_type_t resolve_dst_type;
  arrfunc_resolve_dst_shape_t resolve_dst_shape;
  /**
   * A function which deallocates the memory behind data_ptr after
   * freeing any additional resources it might contain.
   */
  void (*free_func)(arrfunc_type_data *self_data_ptr);

  // Default to all NULL, so the destructor works correctly
  inline arrfunc_type_data() : func_proto(), instantiate(0), free_func(0)
  {
    DYND_STATIC_ASSERT((sizeof(arrfunc_type_data) & 7) == 0,
                       "arrfunc_type_data must have size divisible by 8");
  }

  // If it contains an arrfunc, free it
  inline ~arrfunc_type_data()
  {
    if (free_func) {
      free_func(this);
    }
  }

  /**
   * Helper function to reinterpret the data as the specified type.
   */
  template <typename T>
  inline T *get_data_as()
  {
    DYND_STATIC_ASSERT(sizeof(T) <= sizeof(data), "data does not fit");
    DYND_STATIC_ASSERT((int)scalar_align_of<T>::value <=
                           (int)scalar_align_of<uint64_t>::value,
                       "data requires stronger alignment");
    return reinterpret_cast<T *>(data);
  }
  template <typename T>
  inline const T *get_data_as() const
  {
    DYND_STATIC_ASSERT(sizeof(T) <= sizeof(data), "data does not fit");
    DYND_STATIC_ASSERT((int)scalar_align_of<T>::value <=
                           (int)scalar_align_of<uint64_t>::value,
                       "data requires stronger alignment");
    return reinterpret_cast<const T *>(data);
  }

  inline intptr_t get_param_count() const
  {
    return func_proto.tcast<funcproto_type>()->get_param_count();
  }

  inline const ndt::type *get_param_types() const
  {
    return func_proto.tcast<funcproto_type>()->get_param_types_raw();
  }

  inline const ndt::type &get_param_type(intptr_t i) const
  {
    return get_param_types()[i];
  }

  inline const ndt::type &get_return_type() const
  {
    return func_proto.tcast<funcproto_type>()->get_return_type();
  }

  inline ndt::type resolve(const ndt::type *src_tp) const
  {
    if (resolve_dst_type != NULL) {
      ndt::type result;
      resolve_dst_type(this, result, src_tp, true);
      return result;
    } else {
      intptr_t param_count = get_param_count();
      const ndt::type *param_types = get_param_types();
      std::map<nd::string, ndt::type> typevars;
      for (intptr_t i = 0; i != param_count; ++i) {
        if (!ndt::pattern_match(src_tp[i].value_type(), param_types[i],
                                     typevars)) {
          std::stringstream ss;
          ss << "parameter " << (i + 1) << " to arrfunc does not match, ";
          ss << "expected " << param_types[i] << ", received " << src_tp[i];
          throw std::invalid_argument(ss.str());
        }
      }
      // TODO:
      // return ndt::substitute_type_vars(get_return_type(), typevars);
      return get_return_type();
    }
  }
};

namespace nd {
/**
 * Holds a single instance of an arrfunc in an immutable nd::array,
 * providing some more direct convenient interface.
 */
class arrfunc {
  nd::array m_value;

public:
  inline arrfunc() : m_value() {}
  inline arrfunc(const arrfunc &rhs) : m_value(rhs.m_value) {}
  /**
    * Constructor from an nd::array. Validates that the input
    * has "arrfunc" type and is immutable.
    */
  arrfunc(const nd::array &rhs);

  inline arrfunc &operator=(const arrfunc &rhs)
  {
    m_value = rhs.m_value;
    return *this;
  }

  inline bool is_null() const { return m_value.is_null(); }

  inline const arrfunc_type_data *get() const
  {
    return !m_value.is_null() ? reinterpret_cast<const arrfunc_type_data *>(
                                    m_value.get_readonly_originptr())
                              : NULL;
  }

  inline operator nd::array() const { return m_value; }

  /** Implements the general call operator */
  nd::array call(intptr_t arg_count, const nd::array *args,
                 const eval::eval_context *ectx) const;

  /** Convenience call operators */
  inline nd::array operator()() const
  {
    return call(0, NULL, &eval::default_eval_context);
  }
  inline nd::array operator()(const nd::array &a0) const
  {
    return call(1, &a0, &eval::default_eval_context);
  }
  inline nd::array operator()(const nd::array &a0, const nd::array &a1) const
  {
    nd::array args[2] = {a0, a1};
    return call(2, args, &eval::default_eval_context);
  }
  inline nd::array operator()(const nd::array &a0, const nd::array &a1,
                              const nd::array &a2) const
  {
    nd::array args[3] = {a0, a1, a2};
    return call(3, args, &eval::default_eval_context);
  }

  /** Implements the general call operator with output parameter */
  void call_out(intptr_t arg_count, const nd::array *args, const nd::array &out,
                const eval::eval_context *ectx) const;

  inline void call_out(const nd::array &out) const
  {
    call_out(0, NULL, out, &eval::default_eval_context);
  }

  inline void call_out(const nd::array &a0, const nd::array &out) const
  {
    call_out(1, &a0, out, &eval::default_eval_context);
  }

  inline void call_out(const nd::array &a0, const nd::array &a1,
                       const nd::array &out) const
  {
    nd::array args[2] = {a0, a1};
    call_out(2, args, out, &eval::default_eval_context);
  }

  inline void call_out(const nd::array &a0, const nd::array &a1,
                       const nd::array &a2, const nd::array &out) const
  {
    nd::array args[3] = {a0, a1, a2};
    call_out(3, args, out, &eval::default_eval_context);
  }

  inline void call_out(const nd::array &a0, const nd::array &a1,
                       const nd::array &a2, const nd::array &a3, nd::array &out)
      const
  {
    nd::array args[4] = {a0, a1, a2, a3};
    call_out(4, args, out, &eval::default_eval_context);
  }
};
} // namespace nd

/**
 * Creates an arrfunc which does the assignment from
 * data of src_tp to dst_tp.
 *
 * \param dst_tp  The type of the destination.
 * \param src_tp  The type of the source.
 * \param funcproto  The function prototype to generate (must be
 *                   unary_operation_funcproto or expr_operation_funcproto).
 * \param errmode  The error mode to use for the assignment.
 * \param out_af  The output `arrfunc` struct to be populated.
 */
void make_arrfunc_from_assignment(const ndt::type &dst_tp,
                                  const ndt::type &src_tp,
                                  assign_error_mode errmode,
                                  arrfunc_type_data &out_af);

inline nd::arrfunc make_arrfunc_from_assignment(const ndt::type &dst_tp,
                                                const ndt::type &src_tp,
                                                assign_error_mode errmode)
{
    nd::array af = nd::empty(ndt::make_arrfunc());
    make_arrfunc_from_assignment(
        dst_tp, src_tp, errmode,
        *reinterpret_cast<arrfunc_type_data *>(af.get_readwrite_originptr()));
    af.flag_as_immutable();
    return af;
}

/**
 * Creates an arrfunc which does the assignment from
 * data of `tp` to its property `propname`
 *
 * \param tp  The type of the source.
 * \param propname  The name of the property.
 * \param funcproto  The function prototype to generate (must be
 *                   unary_operation_funcproto or expr_operation_funcproto).
 * \param out_af  The output `arrfunc` struct to be populated.
 */
void make_arrfunc_from_property(const ndt::type &tp,
                                const std::string &propname,
                                arrfunc_type_data &out_af);

inline nd::arrfunc make_arrfunc_from_property(const ndt::type &tp,
                                              const std::string &propname)
{
    nd::array af = nd::empty(ndt::make_arrfunc());
    make_arrfunc_from_property(
        tp, propname,
        *reinterpret_cast<arrfunc_type_data *>(af.get_readwrite_originptr()));
    af.flag_as_immutable();
    return af;
}

} // namespace dynd

#endif // _DYND__ARRFUNC_HPP_
