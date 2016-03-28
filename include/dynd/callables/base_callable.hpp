//
// Copyright (C) 2011-15 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#pragma once

#include <atomic>
#include <map>

#include <dynd/kernels/kernel_prefix.hpp>
#include <dynd/array.hpp>
#include <dynd/types/substitute_typevars.hpp>

namespace dynd {
namespace nd {

  class call_stack;

  enum callable_property {
    none = 0x00000000,
    left_associative = 0x00000001,
    right_associative = 0x00000002,
    commutative = 0x00000004
  };

  inline callable_property operator|(callable_property a, callable_property b)
  {
    return static_cast<callable_property>(static_cast<int>(a) | static_cast<int>(b));
  }

  /**
   * This is a struct designed for interoperability at
   * the C ABI level. It contains enough information
   * to pass callable from one library to another
   * with no dependencies between them.
   *
   * The callable can produce a ckernel with with a few
   * variations, like choosing between a single
   * operation and a strided operation, or constructing
   * with different array arrmeta.
   */
  class DYND_API base_callable {
  protected:
    std::atomic_long m_use_count;
    ndt::type m_tp;
    bool m_new_style; // whether or not this callable is operating in the new "resolve" framework

  public:
    base_callable(const ndt::type &tp) : m_use_count(0), m_tp(tp), m_new_style(false) {}

    // non-copyable
    base_callable(const base_callable &) = delete;

    virtual ~base_callable();

    const ndt::type &get_type() const { return m_tp; }

    virtual void resolve(call_stack &DYND_UNUSED(stack), size_t DYND_UNUSED(nkwd), const array *DYND_UNUSED(kwds),
                         const std::map<std::string, ndt::type> &DYND_UNUSED(tp_vars)){};

    virtual void new_instantiate(char *DYND_UNUSED(data), kernel_builder *DYND_UNUSED(ckb),
                                 const ndt::type &DYND_UNUSED(dst_tp), const char *DYND_UNUSED(dst_arrmeta),
                                 intptr_t DYND_UNUSED(nsrc), const ndt::type *DYND_UNUSED(src_tp),
                                 const char *const *DYND_UNUSED(src_arrmeta), kernel_request_t DYND_UNUSED(kernreq),
                                 intptr_t DYND_UNUSED(nkwd), const array *DYND_UNUSED(kwds))
    {
    }

    virtual array alloc(const ndt::type *dst_tp) const { return empty(*dst_tp); }

    /**
     * Resolves any missing keyword arguments for this callable based on
     * the types of the positional arguments and the available keywords arguments.
     *
     * \param self    The callable.
     * \param self_tp The function prototype of the callable.
     * \param nsrc    The number of positional arguments.
     * \param src_tp  An array of the source types.
     * \param kwds    An array of the.
     */
    virtual char *data_init(const ndt::type &DYND_UNUSED(dst_tp), intptr_t DYND_UNUSED(nsrc),
                            const ndt::type *DYND_UNUSED(src_tp), intptr_t DYND_UNUSED(nkwd),
                            const array *DYND_UNUSED(kwds),
                            const std::map<std::string, ndt::type> &DYND_UNUSED(tp_vars))
    {
      return NULL;
    }

    /**
     * Resolves the destination type for this callable based on the types
     * of the source parameters.
     *
     * \param self  The callable.
     * \param af_tp  The function prototype of the callable.
     * \param dst_tp  To be filled with the destination type.
     * \param nsrc  The number of source parameters.
     * \param src_tp  An array of the source types.
     */
    virtual void resolve_dst_type(char *DYND_UNUSED(data), ndt::type &dst_tp, intptr_t DYND_UNUSED(nsrc),
                                  const ndt::type *DYND_UNUSED(src_tp), intptr_t DYND_UNUSED(nkwd),
                                  const array *DYND_UNUSED(kwds), const std::map<std::string, ndt::type> &tp_vars)
    {
      dst_tp = ndt::substitute(dst_tp, tp_vars, true);
    }

    /**
     * Function prototype for instantiating a kernel from an
     * callable. To use this function, the
     * caller should first allocate a `ckernel_builder` instance,
     * either from C++ normally or by reserving appropriately aligned/sized
     * data and calling the C function constructor dynd provides. When the
     * data types of the kernel require arrmeta, such as for 'strided'
     * or 'var' dimension types, the arrmeta must be provided as well.
     *
     * \param self  The callable.
     * \param self_tp  The function prototype of the callable.
     * \param ckb  A ckernel_builder instance where the kernel is placed.
     * \param ckb_offset  The offset into the output ckernel_builder `ckb`
     *                    where the kernel should be placed.
     * \param dst_tp  The destination type of the ckernel to generate. This may be
     *                different from the one in the function prototype, but must
     *                match its pattern.
     * \param dst_arrmeta  The destination arrmeta.
     * \param nsrc  The number of source arrays.
     * \param src_tp  An array of the source types of the ckernel to generate. These
     *                may be different from the ones in the function prototype, but
     *                must match the patterns.
     * \param src_arrmeta  An array of dynd arrmeta pointers,
     *                     corresponding to the source types.
     * \param kernreq  What kind of C function prototype the resulting ckernel
     *                 should follow. Defined by the enum with kernel_request_*
     *                 values.
     * \param kwds  A struct array of named auxiliary arguments.
     */
    virtual void instantiate(char *data, kernel_builder *ckb, const ndt::type &dst_tp, const char *dst_arrmeta,
                             intptr_t nsrc, const ndt::type *src_tp, const char *const *src_arrmeta,
                             kernel_request_t kernreq, intptr_t nkwd, const array *kwds,
                             const std::map<std::string, ndt::type> &tp_vars) = 0;

    virtual void overload(const ndt::type &DYND_UNUSED(ret_tp), intptr_t DYND_UNUSED(narg),
                          const ndt::type *DYND_UNUSED(arg_tp), const callable &DYND_UNUSED(value))
    {
      throw std::runtime_error("callable is not overloadable");
    }

    virtual const callable &specialize(const ndt::type &DYND_UNUSED(ret_tp), intptr_t DYND_UNUSED(narg),
                                       const ndt::type *DYND_UNUSED(arg_tp))
    {
      throw std::runtime_error("callable is not specializable");
    }

    array call(ndt::type &dst_tp, intptr_t nsrc, const ndt::type *src_tp, const char *const *src_arrmeta,
               char *const *src_data, intptr_t nkwd, const array *kwds,
               const std::map<std::string, ndt::type> &tp_vars);

    array call(ndt::type &dst_tp, intptr_t nsrc, const ndt::type *src_tp, const char *const *src_arrmeta,
               const array *src_data, intptr_t nkwd, const array *kwds,
               const std::map<std::string, ndt::type> &tp_vars);

    void call(const ndt::type &dst_tp, const char *dst_arrmeta, array *dst_data, intptr_t nsrc, const ndt::type *src_tp,
              const char *const *src_arrmeta, const array *src_data, intptr_t nkwd, const array *kwds,
              const std::map<std::string, ndt::type> &tp_vars);

    void call(const ndt::type &dst_tp, const char *dst_arrmeta, char *dst_data, intptr_t nsrc, const ndt::type *src_tp,
              const char *const *src_arrmeta, char *const *src_data, intptr_t nkwd, const array *kwds,
              const std::map<std::string, ndt::type> &tp_vars);

    friend void intrusive_ptr_retain(base_callable *ptr);
    friend void intrusive_ptr_release(base_callable *ptr);
    friend long intrusive_ptr_use_count(base_callable *ptr);
  };

  inline void intrusive_ptr_retain(base_callable *ptr) { ++ptr->m_use_count; }

  inline void intrusive_ptr_release(base_callable *ptr)
  {
    if (--ptr->m_use_count == 0) {
      delete ptr;
    }
  }

  inline long intrusive_ptr_use_count(base_callable *ptr) { return ptr->m_use_count; }

} // namespace dynd::nd
} // namespace dynd
