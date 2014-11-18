//
// Copyright (C) 2011-14 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#pragma once

#include <dynd/config.hpp>
#include <dynd/eval/eval_context.hpp>
#include <dynd/types/base_type.hpp>
#include <dynd/types/arrfunc_type.hpp>
#include <dynd/types/arrfunc_old_type.hpp>
#include <dynd/kernels/ckernel_builder.hpp>
#include <dynd/types/struct_type.hpp>
#include <dynd/types/type_pattern_match.hpp>
#include <dynd/types/substitute_typevars.hpp>

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
 * \param af_tp  The function prototype of the arrfunc.
 * \param ckb  A ckernel_builder instance where the kernel is placed.
 * \param ckb_offset  The offset into the output ckernel_builder `ckb`
 *                    where the kernel should be placed.
 * \param dst_tp  The destination type of the ckernel to generate. This may be
 *                different from the one in the function prototype, but must
 *                match its pattern.
 * \param dst_arrmeta  The destination arrmeta.
 * \param src_tp  An array of the source types of the ckernel to generate. These
 *                may be different from the ones in the function prototype, but
 *                must match the patterns.
 * \param src_arrmeta  An array of dynd arrmeta pointers,
 *                     corresponding to the source types.
 * \param kernreq  What kind of C function prototype the resulting ckernel
 *                 should follow. Defined by the enum with kernel_request_*
 *                 values.
 * \param ectx  The evaluation context.
 * \param args  A tuple array of unnamed auxiliary arguments.
 * \param kwds  A struct array of named auxiliary arguments.
 *
 * \returns  The offset into ``ckb`` immediately after the instantiated ckernel.
 */
typedef intptr_t (*arrfunc_instantiate_t)(
    const arrfunc_type_data *self, const arrfunc_type *af_tp,
    dynd::ckernel_builder *ckb, intptr_t ckb_offset, const ndt::type &dst_tp,
    const char *dst_arrmeta, const ndt::type *src_tp,
    const char *const *src_arrmeta, kernel_request_t kernreq,
    const eval::eval_context *ectx, const nd::array &args,
    const nd::array &kwds);

/**
 * Resolves the destination type for this arrfunc based on the types
 * of the source parameters.
 *
 * \param self  The arrfunc.
 * \param af_tp  The function prototype of the arrfunc.
 * \param nsrc  The number of source parameters.
 * \param src_tp  An array of the source types.
 * \param throw_on_error  If true, should throw when there's an error, if
 *                        false, should return 0 when there's an error.
 * \param out_dst_tp  To be filled with the destination type.
 *
 * \returns  True on success, false on error (if throw_on_error was false).
 */
typedef int (*arrfunc_resolve_dst_type_t)(
    const arrfunc_type_data *self, const arrfunc_type *af_tp, intptr_t nsrc,
    const ndt::type *src_tp, int throw_on_error, ndt::type &out_dst_tp,
    const nd::array &args, const nd::array &kwds);

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
   * On 32-bit platforms, if the size changes, it may be
   * necessary to use
   * char data[4 * 8 + ((sizeof(void *) == 4) ? 4 : 0)];
   * to ensure the total struct size is divisible by 64.
   */
  char data[4 * 8 + ((sizeof(void *) == 4) ? 4 : 0)];
  /**
   * The function which instantiates a ckernel. See the documentation
   * for the function typedef for more details.
   */
  arrfunc_instantiate_t instantiate;
  arrfunc_resolve_dst_type_t resolve_dst_type;
  /**
   * A function which deallocates the memory behind data_ptr after
   * freeing any additional resources it might contain.
   */
  void (*free_func)(arrfunc_type_data *self_data_ptr);

  // Default to all NULL, so the destructor works correctly
  arrfunc_type_data() : instantiate(0), free_func(0)
  {
    DYND_STATIC_ASSERT((sizeof(arrfunc_type_data) & 7) == 0,
                       "arrfunc_type_data must have size divisible by 8");
  }

  // If it contains an arrfunc, free it
  ~arrfunc_type_data()
  {
    if (free_func) {
      free_func(this);
    }
  }

  /**
   * Helper function to reinterpret the data as the specified type.
   */
  template <typename T>
  T *get_data_as()
  {
    if (sizeof(T) > sizeof(data)) {
      throw std::runtime_error("data does not fit");
    }
    if ((int)scalar_align_of<T>::value >
        (int)scalar_align_of<uint64_t>::value) {
      throw std::runtime_error("data requires stronger alignment");
    }
    return reinterpret_cast<T *>(data);
  }
  template <typename T>
  const T *get_data_as() const
  {
    if (sizeof(T) > sizeof(data)) {
      throw std::runtime_error("data does not fit");
    }
    if ((int)scalar_align_of<T>::value >
        (int)scalar_align_of<uint64_t>::value) {
      throw std::runtime_error("data requires stronger alignment");
    }
    return reinterpret_cast<const T *>(data);
  }

  ndt::type resolve(const arrfunc_type *af_tp, intptr_t nsrc,
                    const ndt::type *src_tp, const nd::array &args,
                    const nd::array kwds) const
  {
    if (resolve_dst_type != NULL) {
      ndt::type result;
      resolve_dst_type(this, af_tp, nsrc, src_tp, true, result, args, kwds);
      return result;
    }
    else {
      if (nsrc != af_tp->get_npos()) {
        std::stringstream ss;
        ss << "arrfunc expected " << af_tp->get_npos()
           << " parameters, but received " << nsrc;
        throw std::invalid_argument(ss.str());
      }
      const ndt::type *param_types = af_tp->get_arg_types_raw();
      std::map<nd::string, ndt::type> typevars;
      for (intptr_t i = 0; i != nsrc; ++i) {
        if (!ndt::pattern_match(src_tp[i].value_type(), param_types[i],
                                typevars)) {
          std::stringstream ss;
          ss << "parameter " << (i + 1) << " to arrfunc does not match, ";
          ss << "expected " << param_types[i] << ", received " << src_tp[i];
          throw std::invalid_argument(ss.str());
        }
      }
      return ndt::substitute(af_tp->get_return_type(), typevars, true);
    }
  }
};

class kwds {
  nd::array m_kwds;

public:
  kwds() {}

  template <typename A0>
  kwds(const std::string &name0, const A0 &a0)
      : m_kwds(pack(name0, a0))
  {
  }

  template <typename A0, typename A1>
  kwds(const std::string &name0, const A0 &a0, const std::string &name1,
       const A1 &a1)
      : m_kwds(pack(name0, a0, name1, a1))
  {
  }

  template <typename A0, typename A1, typename A2>
  kwds(const std::string &name0, const A0 &a0, const std::string &name1,
       const A1 &a1, const std::string &name2, const A2 &a2)
      : m_kwds(pack(name0, a0, name1, a1, name2, a2))
  {
  }

  template <typename A0, typename A1, typename A2, typename A3>
  kwds(const std::string &name0, const A0 &a0, const std::string &name1,
       const A1 &a1, const std::string &name2, const A2 &a2,
       const std::string &name3, const A3 &a3)
      : m_kwds(pack(name0, a0, name1, a1, name2, a2, name3, a3))
  {
  }

  template <typename A0, typename A1, typename A2, typename A3, typename A4>
  kwds(const std::string &name0, const A0 &a0, const std::string &name1,
       const A1 &a1, const std::string &name2, const A2 &a2,
       const std::string &name3, const A3 &a3, const std::string &name4,
       const A4 &a4)
      : m_kwds(pack(name0, a0, name1, a1, name2, a2, name3, a3, name4, a4))
  {
  }
  const nd::array &get() const { return m_kwds; }
};

namespace nd {
/**
 * Holds a single instance of an arrfunc in an immutable nd::array,
 * providing some more direct convenient interface.
 */
class arrfunc {
  nd::array m_value;

public:
  arrfunc() : m_value() {}
  arrfunc(const arrfunc &rhs) : m_value(rhs.m_value) {}
  /**
    * Constructor from an nd::array. Validates that the input
    * has "arrfunc" type and is immutable.
    */
  arrfunc(const nd::array &rhs);

  arrfunc &operator=(const arrfunc &rhs)
  {
    m_value = rhs.m_value;
    return *this;
  }

  bool is_null() const { return m_value.is_null(); }

  const arrfunc_type_data *get() const
  {
    return !m_value.is_null() ? reinterpret_cast<const arrfunc_type_data *>(
                                    m_value.get_readonly_originptr())
                              : NULL;
  }

  const arrfunc_type *get_type() const
  {
    return !m_value.is_null() ? m_value.get_type().extended<arrfunc_type>()
                              : NULL;
  }

  const ndt::type &get_array_type() const { return m_value.get_type(); }

  operator nd::array() const { return m_value; }

  void swap(nd::arrfunc &rhs) { m_value.swap(rhs.m_value); }

  /** Implements the general call operator */
  nd::array call(intptr_t arg_count, const nd::array *args, const kwds &kwds,
                 const eval::eval_context *ectx) const;
  nd::array call(intptr_t arg_count, const nd::array *args,
                 const eval::eval_context *ectx) const
  {
    return call(arg_count, args, dynd::kwds(), ectx);
  }

  /** Convenience call operators */
  nd::array operator()(const kwds &kwds = dynd::kwds()) const
  {
    return call(0, NULL, kwds, &eval::default_eval_context);
  }
  nd::array operator()(const nd::array &a0,
                       const kwds &kwds = dynd::kwds()) const
  {
    return call(1, &a0, kwds, &eval::default_eval_context);
  }
  nd::array operator()(const nd::array &a0, const nd::array &a1,
                       const kwds &kwds = dynd::kwds()) const
  {
    nd::array args[2] = {a0, a1};
    return call(2, args, kwds, &eval::default_eval_context);
  }
  nd::array operator()(const nd::array &a0, const nd::array &a1,
                       const nd::array &a2,
                       const kwds &kwds = dynd::kwds()) const
  {
    nd::array args[3] = {a0, a1, a2};
    return call(3, args, kwds, &eval::default_eval_context);
  }

  /** Implements the general call operator with output parameter */
  void call_out(intptr_t arg_count, const nd::array *args, const kwds &kwds,
                const nd::array &out, const eval::eval_context *ectx) const;
  void call_out(intptr_t arg_count, const nd::array *args, const nd::array &out,
                const eval::eval_context *ectx) const
  {
    call_out(arg_count, args, dynd::kwds(), out, ectx);
  }

  /** Convenience call operators with output parameter */
  void call_out(const nd::array &out) const
  {
    call_out(0, NULL, out, &eval::default_eval_context);
  }
  void call_out(const nd::array &a0, const nd::array &out,
                const kwds &kwds = dynd::kwds()) const
  {
    call_out(1, &a0, kwds, out, &eval::default_eval_context);
  }
  void call_out(const nd::array &a0, const nd::array &a1, const nd::array &out,
                const kwds &kwds = dynd::kwds()) const
  {
    nd::array args[2] = {a0, a1};
    call_out(2, args, kwds, out, &eval::default_eval_context);
  }
  void call_out(const nd::array &a0, const nd::array &a1, const nd::array &a2,
                const nd::array &out, const kwds &kwds = dynd::kwds()) const
  {
    nd::array args[3] = {a0, a1, a2};
    call_out(3, args, kwds, out, &eval::default_eval_context);
  }
  void call_out(const nd::array &a0, const nd::array &a1, const nd::array &a2,
                const nd::array &a3, nd::array &out,
                const kwds &kwds = dynd::kwds()) const
  {
    nd::array args[4] = {a0, a1, a2, a3};
    call_out(4, args, kwds, out, &eval::default_eval_context);
  }
};

/**
 * This is a helper class for creating static nd::arrfunc instances
 * whose lifetime is managed by init/cleanup functions. When declared
 * as a global static variable, because it is a POD type, this will begin with
 * the value NULL. It can generally be treated just like an nd::arrfunc, though
 * its internals are not protected from meddling.
 */
struct pod_arrfunc {
  memory_block_data *m_memblock;

  operator const nd::arrfunc &()
  {
    return *reinterpret_cast<const nd::arrfunc *>(&m_memblock);
  }

  const arrfunc_type_data *get() const
  {
    return reinterpret_cast<const nd::arrfunc *>(&m_memblock)->get();
  }

  const arrfunc_type *get_type() const
  {
    return reinterpret_cast<const nd::arrfunc *>(&m_memblock)->get_type();
  }

  void init(const nd::arrfunc &rhs)
  {
    m_memblock = nd::array(rhs).get_memblock().get();
    memory_block_incref(m_memblock);
  }

  void cleanup()
  {
    if (m_memblock) {
      memory_block_decref(m_memblock);
      m_memblock = NULL;
    }
  }
};

inline arrfunc make_arrfunc(ndt::type af_tp, arrfunc_instantiate_t instantiate)
{
  array af = empty(af_tp);
  arrfunc_type_data *out_af = reinterpret_cast<arrfunc_type_data *>(af.get_readwrite_originptr());
  out_af->instantiate = instantiate;
  af.flag_as_immutable();

  return af;
}

template <typename T>
arrfunc make_arrfunc(ndt::type af_tp, arrfunc_instantiate_t instantiate, T &&data)
{
  array af = empty(af_tp);
  arrfunc_type_data *out_af = reinterpret_cast<arrfunc_type_data *>(af.get_readwrite_originptr());
  out_af->instantiate = instantiate;
  *out_af->get_data_as<T>() = data;
  af.flag_as_immutable();

  return af;
}

} // namespace nd

/**
 * Creates an arrfunc which does the assignment from
 * data of src_tp to dst_tp.
 *
 * \param dst_tp  The type of the destination.
 * \param src_tp  The type of the source.
 * \param errmode  The error mode to use for the assignment.
 */
nd::arrfunc make_arrfunc_from_assignment(const ndt::type &dst_tp,
                                         const ndt::type &src_tp,
                                         assign_error_mode errmode);

/**
 * Creates an arrfunc which does the assignment from
 * data of `tp` to its property `propname`
 *
 * \param tp  The type of the source.
 * \param propname  The name of the property.
 */
nd::arrfunc make_arrfunc_from_property(const ndt::type &tp,
                                       const std::string &propname);

} // namespace dynd
