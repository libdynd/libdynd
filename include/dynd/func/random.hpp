#pragma once

#include <chrono>
#include <memory>
#include <random>

#include <dynd/kernels/expr_kernels.hpp>
#include <dynd/func/arrfunc.hpp>

namespace dynd {

typedef type_sequence<int32_t, int64_t> integral_types;
typedef type_sequence<float, double> real_types;

typedef type_sequence<int32_t, int64_t, float, double> numeric_types;

template <typename R>
class uniform_distribution;

template <>
class uniform_distribution<int32_t>
    : public std::uniform_int_distribution<int32_t> {
public:
  using std::uniform_int_distribution<int32_t>::uniform_int_distribution;
};

template <>
class uniform_distribution<int64_t>
    : public std::uniform_int_distribution<int64_t> {
public:
  using std::uniform_int_distribution<int64_t>::uniform_int_distribution;
};

template <>
class uniform_distribution<float>
    : public std::uniform_real_distribution<float> {
public:
  using std::uniform_real_distribution<float>::uniform_real_distribution;
};

template <>
class uniform_distribution<double>
    : public std::uniform_real_distribution<double> {
public:
  using std::uniform_real_distribution<double>::uniform_real_distribution;
};

namespace kernels {
  template <typename... T>
  struct uniform_ck;

  template <typename G, typename R>
  struct uniform_ck<G, R> : expr_ck<uniform_ck<G, R>, kernel_request_host, 0> {
    typedef uniform_ck self_type;

    G &g;
    uniform_distribution<R> d;

    uniform_ck(G *g, R a, R b) : g(*g), d(a, b) {}

    void single(char *dst, char *const *DYND_UNUSED(src))
    {
      *reinterpret_cast<R *>(dst) = d(g);
    }

    static ndt::type make_type()
    {
      std::map<nd::string, ndt::type> tp_vars;
      tp_vars["R"] = ndt::make_type<R>();

      return ndt::substitute(ndt::type("(a: ?R, b: ?R) -> R"), tp_vars, true);
    }

    static void resolve_option_values(
        const arrfunc_type_data *DYND_UNUSED(self),
        const arrfunc_type *DYND_UNUSED(self_tp), intptr_t DYND_UNUSED(nsrc),
        const ndt::type *DYND_UNUSED(src_tp), nd::array &kwds,
        const std::map<dynd::nd::string, ndt::type> &DYND_UNUSED(tp_vars))
    {
      nd::array a = kwds.p("a");
      if (a.is_missing()) {
        a.val_assign(0);
      }

      nd::array b = kwds.p("b");
      if (b.is_missing()) {
        b.val_assign(std::numeric_limits<R>::max());
      }
    }

    static intptr_t instantiate(
        const arrfunc_type_data *self, const arrfunc_type *DYND_UNUSED(self_tp),
        void *ckb, intptr_t ckb_offset, const ndt::type &DYND_UNUSED(dst_tp),
        const char *DYND_UNUSED(dst_arrmeta),
        const ndt::type *DYND_UNUSED(src_tp),
        const char *const *DYND_UNUSED(src_arrmeta), kernel_request_t kernreq,
        const eval::eval_context *DYND_UNUSED(ectx), const nd::array &kwds,
        const std::map<nd::string, ndt::type> &DYND_UNUSED(tp_vars))
    {
      std::shared_ptr<G> g = *self->get_data_as<std::shared_ptr<G>>();
      R a = kwds.p("a").as<R>();
      R b = kwds.p("b").as<R>();

      self_type::create(ckb, kernreq, ckb_offset, g.get(), a, b);
      return ckb_offset;
    }
  };

} // namespace dynd::kernels

namespace nd {
  namespace decl {
    namespace random {

      struct uniform : arrfunc<uniform> {
        static nd::arrfunc as_arrfunc();
      };

    } // namespace dynd::nd::decl::random
  }   // namespace dynd::nd::decl

  namespace random {

    extern decl::random::uniform uniform;

  } // namespace dynd::nd::random

} // namespace dynd::nd
} // namespace dynd