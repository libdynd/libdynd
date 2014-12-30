//
// Copyright (C) 2011-14 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <sstream>

#include <dynd/array.hpp>
#include <dynd/type_promotion.hpp>
#include <dynd/kernels/expr_kernel_generator.hpp>
#include <dynd/kernels/elwise_expr_kernels.hpp>
#include <dynd/func/elwise.hpp>
#include <dynd/shape_tools.hpp>
#include <dynd/types/var_dim_type.hpp>
#include <dynd/types/expr_type.hpp>
#include <dynd/types/string_type.hpp>
#include <dynd/kernels/string_algorithm_kernels.hpp>

using namespace std;
using namespace dynd;

namespace dynd {
namespace kernels {
  template <typename T>
  struct subtract_ck : expr_ck<subtract_ck<T>, kernel_request_cuda_host_device, 2> {
    DYND_CUDA_HOST_DEVICE void single(char *dst, char *const *src)
    {
      *reinterpret_cast<T *>(dst) =
          *reinterpret_cast<T *>(src[0]) - *reinterpret_cast<T *>(src[1]);
    }

    DYND_CUDA_HOST_DEVICE void strided(char *dst, intptr_t dst_stride, char *const *src,
                 const intptr_t *src_stride, size_t count)
    {
      char *src0 = src[0], *src1 = src[1];
      intptr_t src0_stride = src_stride[0], src1_stride = src_stride[1];
      for (size_t i = 0; i != count; ++i) {
        *reinterpret_cast<T *>(dst) =
            *reinterpret_cast<T *>(src0) - *reinterpret_cast<T *>(src1);
        dst += dst_stride;
        src0 += src0_stride;
        src1 += src1_stride;
      }
    }
  };
} // namespace kernels
} // namespace dynd

namespace {

template <class OP>
struct binary_kernel
    : kernels::expr_ck<binary_kernel<OP>, kernel_request_host, 2> {

  void single(char *dst, char *const *src)
  {
    typedef typename OP::type T;
    T s0, s1, r;

    s0 = *reinterpret_cast<T *>(src[0]);
    s1 = *reinterpret_cast<T *>(src[1]);

    r = OP::operate(s0, s1);

    *reinterpret_cast<T *>(dst) = r;
  }

  void strided(char *dst, intptr_t dst_stride, char *const *src,
               const intptr_t *src_stride, size_t count)
  {
    typedef typename OP::type T;
    char *src0 = src[0], *src1 = src[1];
    intptr_t src0_stride = src_stride[0], src1_stride = src_stride[1];

    for (size_t i = 0; i != count; ++i) {
      T s0, s1, r;
      s0 = *reinterpret_cast<T *>(src0);
      s1 = *reinterpret_cast<T *>(src1);

      r = OP::operate(s0, s1);

      *reinterpret_cast<T *>(dst) = r;

      dst += dst_stride;
      src0 += src0_stride;
      src1 += src1_stride;
    }
  }
};

template <class extra_type>
class arithmetic_op_kernel_generator : public expr_kernel_generator {
  ndt::type m_rdt, m_op1dt, m_op2dt;
  expr_operation_pair m_op_pair;
  const char *m_name;

public:
  arithmetic_op_kernel_generator(const ndt::type &rdt, const ndt::type &op1dt,
                                 const ndt::type &op2dt,
                                 const expr_operation_pair &op_pair,
                                 const char *name)
      : expr_kernel_generator(true), m_rdt(rdt), m_op1dt(op1dt), m_op2dt(op2dt),
        m_op_pair(op_pair), m_name(name)
  {
  }

  virtual ~arithmetic_op_kernel_generator() {}

  size_t make_expr_kernel(void *ckb, intptr_t ckb_offset,
                          const ndt::type &dst_tp, const char *dst_arrmeta,
                          size_t src_count, const ndt::type *src_tp,
                          const char *const *src_arrmeta,
                          kernel_request_t kernreq,
                          const eval::eval_context *ectx) const
  {
    if (src_count != 2) {
      stringstream ss;
      ss << "The " << m_name << " kernel requires 2 src operands, ";
      ss << "received " << src_count;
      throw runtime_error(ss.str());
    }
    if (dst_tp != m_rdt || src_tp[0] != m_op1dt || src_tp[1] != m_op2dt) {
      // If the types don't match the ones for this generator,
      // call the elementwise dimension handler to handle one dimension
      // or handle input/output buffering, giving 'this' as the next
      // kernel generator to call
      return make_elwise_dimension_expr_kernel(
          ckb, ckb_offset, dst_tp, dst_arrmeta, src_count, src_tp, src_arrmeta,
          kernreq, ectx, this);
    }
    // This is a leaf kernel, so no additional allocation is needed
    extra_type *e = reinterpret_cast<ckernel_builder<kernel_request_host> *>(
                        ckb)->alloc_ck_leaf<extra_type>(ckb_offset);
    switch (kernreq) {
    case kernel_request_single:
      e->base().set_function(m_op_pair.single);
      break;
    case kernel_request_strided:
      e->base().set_function(m_op_pair.strided);
      break;
    default: {
      stringstream ss;
      ss << "generic_kernel_generator: unrecognized request " << (int)kernreq;
      throw runtime_error(ss.str());
    }
    }
    e->init(2, dst_arrmeta, (const char **)src_arrmeta);
    return ckb_offset;
  }

  void print_type(std::ostream &o) const { o << m_name << "(op0, op1)"; }
};
} // anonymous namespace

namespace {
template <class T>
struct addition {
  typedef T type;
  static inline T operate(T x, T y) { return x + y; }
};

template <class T>
struct subtraction {
  typedef T type;
  static inline T operate(T x, T y) { return x - y; }
};

template <class T>
struct multiplication {
  typedef T type;
  static inline T operate(T x, T y) { return x * y; }
};

template <class T>
struct division {
  typedef T type;
  static inline T operate(T x, T y) { return x / y; }
};
} // anonymous namespace

#ifdef DYND_HAS_INT128
#define DYND_INT128_BINARY_OP_PAIR(operation)                                  \
  {                                                                            \
    &binary_kernel<operation<dynd_int128>>::single_wrapper,                    \
        &binary_kernel<operation<dynd_int128>>::strided_wrapper                \
  }
#else
#define DYND_INT128_BINARY_OP_PAIR(operation)                                  \
  {                                                                            \
    NULL, NULL                                                                 \
  }
#endif

#ifdef DYND_HAS_UINT128
#define DYND_UINT128_BINARY_OP_PAIR(operation)                                 \
  {                                                                            \
    &binary_kernel<operation<dynd_uint128>>::single_wrapper,                   \
        &binary_kernel<operation<dynd_uint128>>::strided_wrapper               \
  }
#else
#define DYND_UINT128_BINARY_OP_PAIR(operation)                                 \
  {                                                                            \
    NULL, NULL                                                                 \
  }
#endif

#ifdef DYND_HAS_FLOAT128
#define DYND_FLOAT128_BINARY_OP_PAIR(operation)                                \
  {                                                                            \
    &binary_kernel<operation<dynd_float128>>::single_wrapper,                  \
        &binary_kernel<operation<dynd_float128>>::strided_wrapper              \
  }
#else
#define DYND_FLOAT128_BINARY_OP_PAIR(operation)                                \
  {                                                                            \
    NULL, NULL                                                                 \
  }
#endif

#define DYND_BUILTIN_DTYPE_BINARY_OP_TABLE(operation)                          \
  {                                                                            \
    {                                                                          \
      &binary_kernel<operation<int32_t>>::single_wrapper,                      \
          &binary_kernel<operation<int32_t>>::strided_wrapper                  \
    }                                                                          \
    , {&binary_kernel<operation<int64_t>>::single_wrapper,                     \
       &binary_kernel<operation<int64_t>>::strided_wrapper},                   \
        DYND_INT128_BINARY_OP_PAIR(operation),                                 \
        {&binary_kernel<operation<int32_t>>::single_wrapper,                   \
         &binary_kernel<operation<uint32_t>>::strided_wrapper},                \
        {&binary_kernel<operation<uint64_t>>::single_wrapper,                  \
         &binary_kernel<operation<uint64_t>>::strided_wrapper},                \
        DYND_UINT128_BINARY_OP_PAIR(operation),                                \
        {&binary_kernel<operation<float>>::single_wrapper,                     \
         &binary_kernel<operation<float>>::strided_wrapper},                   \
        {&binary_kernel<operation<double>>::single_wrapper,                    \
         &binary_kernel<operation<double>>::strided_wrapper},                  \
        DYND_FLOAT128_BINARY_OP_PAIR(operation),                               \
        {&binary_kernel<operation<dynd_complex<float>>>::single_wrapper,       \
         &binary_kernel<operation<dynd_complex<float>>>::strided_wrapper},     \
    {                                                                          \
      &binary_kernel<operation<dynd_complex<double>>>::single_wrapper,         \
          &binary_kernel<operation<dynd_complex<double>>>::strided_wrapper     \
    }                                                                          \
  }

#define DYND_BUILTIN_DTYPE_BINARY_OP_TABLE_DEFS(operation)                     \
  static const expr_operation_pair operation##_table[11] =                     \
      DYND_BUILTIN_DTYPE_BINARY_OP_TABLE(operation);

DYND_BUILTIN_DTYPE_BINARY_OP_TABLE_DEFS(addition);
// DYND_BUILTIN_DTYPE_BINARY_OP_TABLE_DEFS(subtraction);

DYND_BUILTIN_DTYPE_BINARY_OP_TABLE_DEFS(multiplication);
DYND_BUILTIN_DTYPE_BINARY_OP_TABLE_DEFS(division);

// These operators are declared in nd::array.hpp

static kernels::create_t subtract_table[builtin_type_id_count - 2] = {
    &kernels::create<kernels::subtract_ck<bool>>,
    &kernels::create<kernels::subtract_ck<int8_t>>,
    &kernels::create<kernels::subtract_ck<int16_t>>,
    &kernels::create<kernels::subtract_ck<int32_t>>,
    &kernels::create<kernels::subtract_ck<int64_t>>,
    NULL,
    &kernels::create<kernels::subtract_ck<uint8_t>>,
    &kernels::create<kernels::subtract_ck<uint16_t>>,
    &kernels::create<kernels::subtract_ck<uint32_t>>,
    &kernels::create<kernels::subtract_ck<uint64_t>>,
    NULL,
    NULL,
    &kernels::create<kernels::subtract_ck<float>>,
    &kernels::create<kernels::subtract_ck<double>>,
    NULL,
    &kernels::create<kernels::subtract_ck<dynd_complex<float>>>,
    &kernels::create<kernels::subtract_ck<dynd_complex<double>>>,
};

int resolve_dst_type_subtract(const arrfunc_type_data *DYND_UNUSED(self),
                              const arrfunc_type *DYND_UNUSED(self_tp),
                              intptr_t DYND_UNUSED(nsrc),
                              const ndt::type *src_tp,
                              int DYND_UNUSED(throw_on_error),
                              ndt::type &out_dst_tp,
                              const nd::array &DYND_UNUSED(kwds))
{
  out_dst_tp = promote_types_arithmetic(src_tp[0].without_memory_type(), src_tp[1].without_memory_type());
  if (src_tp[0].get_kind() == memory_kind) {
    out_dst_tp = src_tp[0].extended<base_memory_type>()->with_replaced_storage_type(out_dst_tp);
  }

  return 1;
}

intptr_t instantiate_subtract(const arrfunc_type_data *DYND_UNUSED(self),
                              const arrfunc_type *DYND_UNUSED(self_tp),
                              void *ckb, intptr_t ckb_offset,
                              const ndt::type &dst_tp,
                              const char *DYND_UNUSED(dst_arrmeta),
                              const ndt::type *DYND_UNUSED(src_tp),
                              const char *const *DYND_UNUSED(src_arrmeta),
                              kernel_request_t kernreq,
                              const eval::eval_context *DYND_UNUSED(ectx),
                              const nd::array &DYND_UNUSED(kwds))
{
  kernels::create_t create =
      subtract_table[dst_tp.get_type_id() - bool_type_id];

  create(ckb, kernreq, ckb_offset);
  return ckb_offset;
}

// Get the table index by compressing the type_id's we do implement
static int compress_builtin_type_id[builtin_type_id_count] = {
    -1, -1,       // uninitialized, bool
    -1, -1, 0, 1, // int8, ..., int64
    2,            // int128
    -1, -1, 3, 4, // uint8, ..., uint64,
    5,            // uint128
    -1, 6,  7,    // float16, ..., float64
    8,            // float128
    9,  10,       // complex[float32], complex[float64]
    -1};

template <class KD>
nd::array apply_binary_operator(const nd::array *ops, const ndt::type &rdt,
                                const ndt::type &op1dt, const ndt::type &op2dt,
                                expr_operation_pair expr_ops, const char *name)
{
  if (expr_ops.single == NULL) {
    stringstream ss;
    ss << "Operator " << name << " is not supported for dynd types ";
    ss << op1dt << " and " << op2dt;
    throw runtime_error(ss.str());
  }

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
      incremental_broadcast(ndim, result_shape.get(), ndim_i, tmp_shape.get());
    }
  }

  // Assemble the destination value type
  ndt::type result_vdt = ndt::make_type(ndim, result_shape.get(), rdt);

  // Create the result
  nd::array ops_as_dt[2] = {ops[0].ucast(op1dt), ops[1].ucast(op2dt)};
  nd::array result = combine_into_tuple(2, ops_as_dt);
  // Because the expr type's operand is the result's type,
  // we can swap it in as the type
  ndt::type edt = ndt::make_expr(result_vdt, result.get_type(),
                                 new arithmetic_op_kernel_generator<KD>(
                                     rdt, op1dt, op2dt, expr_ops, name));
  edt.swap(result.get_ndo()->m_type);
  return result;
}

namespace {
struct ckernel_prefix_with_init : public ckernel_prefix {
  template <class R, class S, class T>
  inline void init(R, S, T)
  {
  }
};
} // anonymous namespace

nd::array nd::operator+(const nd::array &op1, const nd::array &op2)
{
  nd::array ops[2] = {op1, op2};
  expr_operation_pair func_ptr;
  ndt::type op1dt = op1.get_dtype().value_type();
  ndt::type op2dt = op2.get_dtype().value_type();
  if (op1dt.is_builtin() && op1dt.is_builtin()) {
    ndt::type rdt = promote_types_arithmetic(op1dt, op2dt);
    int table_index = compress_builtin_type_id[rdt.get_type_id()];
    if (table_index >= 0) {
      func_ptr = addition_table[table_index];
    } else {
      func_ptr.single = NULL;
      func_ptr.strided = NULL;
    }

    // The signature is (T, T) -> T, so we don't use the original types
    return apply_binary_operator<ckernel_prefix_with_init>(
               ops, rdt, rdt, rdt, func_ptr, "addition").eval();
  } else if (op1dt.get_kind() == string_kind &&
             op2dt.get_kind() == string_kind) {
    ndt::type rdt = ndt::make_string();
    func_ptr.single = &kernels::string_concatenation_kernel::single;
    func_ptr.strided = &kernels::string_concatenation_kernel::strided;
    // The signature is (string, string) -> string, so we don't use the original
    // types
    // NOTE: Using a different name for string concatenation in the generated
    // expression
    nd::array tmp = apply_binary_operator<kernels::string_concatenation_kernel>(
        ops, rdt, rdt, rdt, func_ptr, "string_concat");

    return tmp.eval();
  } else {
    stringstream ss;
    ss << "Addition is not supported for dynd types ";
    ss << op1dt << " and " << op2dt;
    throw runtime_error(ss.str());
  }
}

namespace dynd {
namespace decl {
  namespace nd {
    class sub : public arrfunc<sub> {
    public:
      static dynd::nd::arrfunc make()
      {
        arrfunc_type_data child(&instantiate_subtract, NULL,
                                &resolve_dst_type_subtract, NULL);
        dynd::nd::arrfunc child_af(&child, ndt::type("(Any, Any) -> Any"));

        return elwise::bind("func", child_af);
      }
    };
  }
}
}

decl::nd::sub sub;

nd::array nd::operator-(const nd::array &op1, const nd::array &op2)
{
  return sub(op1, op2);
}

nd::array nd::operator*(const nd::array &op1, const nd::array &op2)
{
  ndt::type rdt;
  expr_operation_pair func_ptr;
  ndt::type op1dt = op1.get_dtype().value_type();
  ndt::type op2dt = op2.get_dtype().value_type();
  if (op1dt.is_builtin() && op1dt.is_builtin()) {
    rdt = promote_types_arithmetic(op1dt, op2dt);
    int table_index = compress_builtin_type_id[rdt.get_type_id()];
    if (table_index >= 0) {
      func_ptr = multiplication_table[table_index];
    }
  }

  nd::array ops[2] = {op1, op2};
  return apply_binary_operator<ckernel_prefix_with_init>(
             ops, rdt, rdt, rdt, func_ptr, "multiplication").eval();
}

nd::array nd::operator/(const nd::array &op1, const nd::array &op2)
{
  ndt::type rdt;
  expr_operation_pair func_ptr;
  ndt::type op1dt = op1.get_dtype().value_type();
  ndt::type op2dt = op2.get_dtype().value_type();
  if (op1dt.is_builtin() && op1dt.is_builtin()) {
    rdt = promote_types_arithmetic(op1dt, op2dt);
    int table_index = compress_builtin_type_id[rdt.get_type_id()];
    if (table_index >= 0) {
      func_ptr = division_table[table_index];
    }
  }

  nd::array ops[2] = {op1, op2};
  return apply_binary_operator<ckernel_prefix_with_init>(
             ops, rdt, rdt, rdt, func_ptr, "division").eval();
}
