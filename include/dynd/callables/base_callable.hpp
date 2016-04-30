//
// Copyright (C) 2011-16 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#pragma once

#include <atomic>
#include <map>
#include <typeinfo>

#include <dynd/array.hpp>
#include <dynd/callables/call_graph.hpp>
#include <dynd/kernels/kernel_prefix.hpp>
#include <dynd/types/callable_type.hpp>
#include <dynd/types/substitute_typevars.hpp>

namespace dynd {
namespace nd {

  class call_graph;

  enum callable_property {
    none = 0x00000000,
    left_associative = 0x00000001,
    right_associative = 0x00000002,
    commutative = 0x00000004
  };

  inline callable_property operator|(callable_property a, callable_property b) {
    return static_cast<callable_property>(static_cast<int>(a) | static_cast<int>(b));
  }

  enum callable_flags_t {
    // A symbolic name instead of just "0"
    callable_flag_none = 0x00000000,
    // This callable cannot be instantiated
    callable_flag_abstract = 0x00000001,
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
  class DYND_API base_callable {
  protected:
    std::atomic_long m_use_count;
    ndt::type m_tp;

  public:
    base_callable(const ndt::type &tp) : m_use_count(0), m_tp(tp) {}

    // non-copyable
    base_callable(const base_callable &) = delete;

    virtual ~base_callable();

    const ndt::type &get_type() const { return m_tp; }

    const ndt::type &get_ret_type() const { return m_tp.extended<ndt::callable_type>()->get_return_type(); }

    size_t get_narg() const { return m_tp.extended<ndt::callable_type>()->get_npos(); }

    const std::vector<ndt::type> &get_arg_types() const {
      return m_tp.extended<ndt::callable_type>()->get_argument_types();
    }

    bool is_arg_variadic() const { return m_tp.extended<ndt::callable_type>()->is_arg_variadic(); }

    size_t get_nkwd() const { return m_tp.extended<ndt::callable_type>()->get_nkwd(); }

    const std::vector<std::pair<ndt::type, std::string>> &get_kwd_types() const {
      return m_tp.extended<ndt::callable_type>()->get_named_kwd_types();
    }

    const std::vector<intptr_t> &get_option_kwd_indices() const {
      return m_tp.extended<ndt::callable_type>()->get_option_kwd_indices();
    }

    intptr_t get_kwd_index(const std::string &name) const {
return m_tp.extended<ndt::callable_type>()->get_kwd_index(name);
 }

    bool is_kwd_variadic() const { return m_tp.extended<ndt::callable_type>()->is_kwd_variadic(); }

    /**
     * Function prototype for instantiating a kernel from an
     * callable. To use this function, the
     * caller should first allocate a `ckernel_builder` instance,
     * either from C++ normally or by reserving appropriately aligned/sized
     * data and calling the C function constructor dynd provides. When the
     * data types of the kernel require arrmeta, such as for 'strided'
     * or 'var' dimension types, the arrmeta must be provided as well.
     *
     * \param caller  The calling callable.
     * \param dst_tp  The destination type of the ckernel to generate. This may be
     *                different from the one in the function prototype, but must
     *                match its pattern.
     * \param nsrc  The number of source arrays.
     * \param src_tp  An array of the source types of the ckernel to generate. These
     *                may be different from the ones in the function prototype, but
     *                must match the patterns.
     * \param kwds  A struct array of named auxiliary arguments.
     */
    virtual ndt::type resolve(base_callable *caller, char *data, call_graph &cg, const ndt::type &res_tp, size_t narg,
                              const ndt::type *arg_tp, size_t nkwd, const array *kwds,
                              const std::map<std::string, ndt::type> &tp_vars) = 0;

    //    virtual void resolve() {}

    virtual array alloc(const ndt::type *dst_tp) const { return empty(*dst_tp); }

    virtual void overload(const ndt::type &DYND_UNUSED(ret_tp), intptr_t DYND_UNUSED(narg),
                          const ndt::type *DYND_UNUSED(arg_tp), const callable &DYND_UNUSED(value)) {
      throw std::runtime_error("callable is not overloadable");
    }

    virtual const callable &specialize(const ndt::type &DYND_UNUSED(ret_tp), intptr_t DYND_UNUSED(narg),
                                       const ndt::type *DYND_UNUSED(arg_tp)) {
      throw std::runtime_error("callable is not specializable");
    }

    array call(ndt::type &dst_tp, size_t nsrc, const ndt::type *src_tp, const char *const *src_arrmeta,
               char *const *src_data, size_t nkwd, const array *kwds, const std::map<std::string, ndt::type> &tp_vars);

    array call(ndt::type &dst_tp, size_t nsrc, const ndt::type *src_tp, const char *const *src_arrmeta,
               const array *src_data, size_t nkwd, const array *kwds, const std::map<std::string, ndt::type> &tp_vars);

    void call(const ndt::type &dst_tp, const char *dst_arrmeta, array *dst_data, size_t nsrc, const ndt::type *src_tp,
              const char *const *src_arrmeta, const array *src_data, size_t nkwd, const array *kwds,
              const std::map<std::string, ndt::type> &tp_vars);

    void call(const ndt::type &dst_tp, const char *dst_arrmeta, char *dst_data, size_t nsrc, const ndt::type *src_tp,
              const char *const *src_arrmeta, char *const *src_data, size_t nkwd, const array *kwds,
              const std::map<std::string, ndt::type> &tp_vars);

    friend void intrusive_ptr_retain(base_callable *ptr);
    friend void intrusive_ptr_release(base_callable *ptr);
    friend long intrusive_ptr_use_count(base_callable *ptr);
  };

  inline void intrusive_ptr_retain(base_callable *ptr) { ++ptr->m_use_count; }

  inline void intrusive_ptr_release(base_callable *ptr) {
    if (--ptr->m_use_count == 0) {
      delete ptr;
    }
  }

  inline long intrusive_ptr_use_count(base_callable *ptr) { return ptr->m_use_count; }

} // namespace dynd::nd
} // namespace dynd
