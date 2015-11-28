//
// Copyright (C) 2011-15 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#pragma once

#include <atomic>
#include <map>

#include <dynd/array.hpp>

namespace dynd {
namespace nd {

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
  typedef char *(*callable_data_init_t)(char *static_data, const ndt::type &dst_tp, intptr_t nsrc,
                                        const ndt::type *src_tp, intptr_t nkwd, const array *kwds,
                                        const std::map<std::string, ndt::type> &tp_vars);

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
  typedef void (*callable_resolve_dst_type_t)(char *static_data, char *data, ndt::type &dst_tp, intptr_t nsrc,
                                              const ndt::type *src_tp, intptr_t nkwd, const array *kwds,
                                              const std::map<std::string, ndt::type> &tp_vars);

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
   * \param ectx  The evaluation context.
   * \param kwds  A struct array of named auxiliary arguments.
   *
   * \returns  The offset into ``ckb`` immediately after the instantiated ckernel.
   */
  typedef intptr_t (*callable_instantiate_t)(char *static_data, char *data, void *ckb, intptr_t ckb_offset,
                                             const ndt::type &dst_tp, const char *dst_arrmeta, intptr_t nsrc,
                                             const ndt::type *src_tp, const char *const *src_arrmeta,
                                             kernel_request_t kernreq, const eval::eval_context *ectx, intptr_t nkwd,
                                             const array *kwds, const std::map<std::string, ndt::type> &tp_vars);

  /**
   * A function which deallocates the memory behind data_ptr after
   * freeing any additional resources it might contain.
   */
  typedef void (*callable_static_data_free_t)(char *static_data);

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

  struct single_t {
    volatile expr_single_t func;
    const char *ir;

    single_t() = default;

    single_t(volatile expr_single_t func, const volatile char *ir) : func(func), ir(const_cast<const char *>(ir))
    {
    }
  };

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
  struct DYND_API base_callable {
    char buffer[4];

    std::atomic_long use_count;
    ndt::type tp;
    kernel_request_t kernreq;
    kernel_targets_t targets;
    const char *ir;
    callable_data_init_t data_init;
    callable_resolve_dst_type_t resolve_dst_type;
    callable_instantiate_t instantiate;

    base_callable() : use_count(0), data_init(NULL), resolve_dst_type(NULL), instantiate(NULL)
    {
    }

    base_callable(const ndt::type &tp, kernel_targets_t targets)
        : use_count(0), tp(tp), kernreq(kernel_request_single), targets(targets), data_init(&ckernel_prefix::data_init),
          resolve_dst_type(NULL), instantiate(&ckernel_prefix::instantiate)
    {
      new (static_data()) kernel_targets_t(targets);
    }

    base_callable(const ndt::type &tp, kernel_request_t kernreq, kernel_targets_t targets, const volatile char *ir,
                  callable_data_init_t data_init, callable_resolve_dst_type_t resolve_dst_type,
                  callable_instantiate_t instantiate)
        : use_count(0), tp(tp), kernreq(kernreq), targets(targets), ir(const_cast<const char *>(ir)),
          data_init(data_init), resolve_dst_type(resolve_dst_type), instantiate(instantiate)
    {
    }

    // non-copyable
    base_callable(const base_callable &) = delete;

    virtual ~base_callable()
    {
    }

    char *static_data()
    {
      return reinterpret_cast<char *>(this + 1);
    }

    const char *static_data() const
    {
      return reinterpret_cast<const char *>(this + 1);
    }

    array operator()(ndt::type &dst_tp, intptr_t nsrc, const ndt::type *src_tp, const char *const *src_arrmeta,
                     char *const *src_data, intptr_t nkwd, const array *kwds,
                     const std::map<std::string, ndt::type> &tp_vars);

    array operator()(ndt::type &dst_tp, intptr_t nsrc, const ndt::type *src_tp, const char *const *src_arrmeta,
                     array *const *src_data, intptr_t nkwd, const array *kwds,
                     const std::map<std::string, ndt::type> &tp_vars);

    void operator()(const ndt::type &dst_tp, const char *dst_arrmeta, char *dst_data, intptr_t nsrc,
                    const ndt::type *src_tp, const char *const *src_arrmeta, char *const *src_data, intptr_t nkwd,
                    const array *kwds, const std::map<std::string, ndt::type> &tp_vars);

    void operator()(const ndt::type &dst_tp, const char *dst_arrmeta, char *dst_data, intptr_t nsrc,
                    const ndt::type *src_tp, const char *const *src_arrmeta, array *const *src_data, intptr_t nkwd,
                    const array *kwds, const std::map<std::string, ndt::type> &tp_vars);

    static void *operator new(size_t size, size_t static_data_size = 0)
    {
      return ::operator new(size + static_data_size);
    }

    static void operator delete(void *ptr)
    {
      ::operator delete(ptr);
    }

    static void operator delete(void *ptr, size_t DYND_UNUSED(static_data_size))
    {
      ::operator delete(ptr);
    }
  };

  static_assert((sizeof(base_callable) & 7) == 0, "base_callable must have size divisible by 8");

  inline void intrusive_ptr_retain(base_callable *ptr)
  {
    ++ptr->use_count;
  }

  inline void intrusive_ptr_release(base_callable *ptr)
  {
    if (--ptr->use_count == 0) {
      delete ptr;
    }
  }

  inline long intrusive_ptr_use_count(base_callable *ptr)
  {
    return ptr->use_count;
  }

} // namespace dynd::nd
} // namespace dynd
