#pragma once

#include <chrono>
#include <random>

#include <dynd/kernels/expr_kernels.hpp>
#include <dynd/func/arrfunc.hpp>
#include <dynd/func/multidispatch.hpp>

namespace dynd {
namespace nd {
  namespace random {
    arrfunc default_random_engine();

    arrfunc minstd_rand();
    arrfunc minstd_rand(std::minstd_rand::result_type seed);
  } // namespace dynd::nd::random
} // namespace dynd::nd

namespace kernels {
  template <kernel_request_t kernreq, typename... T>
  struct uniform_ck;

  template <typename G, typename R>
  struct uniform_ck<kernel_request_host, G, R>
      : expr_ck<uniform_ck<kernel_request_host, G, R>, kernel_request_host, 0> {
    typedef uniform_ck self_type;

    G &g;
    std::uniform_int_distribution<R> d;

    uniform_ck(G &g, R a, R b) : g(g), d(a, b){};

    void single(char *dst, char *const *DYND_UNUSED(src))
    {
      *reinterpret_cast<R *>(dst) = d(g);
    }
  };

} // namespace dynd::kernels

namespace decl {
  namespace nd {

    class uniform : public arrfunc<uniform> {
    public:
      template <typename R>
      static void resolve_option_values(
          const arrfunc_type_data *DYND_UNUSED(self),
          const arrfunc_type *DYND_UNUSED(self_tp), intptr_t DYND_UNUSED(nsrc),
          const ndt::type *DYND_UNUSED(src_tp), dynd::nd::array &kwds)
      {
        dynd::nd::array a = kwds.p("a");
        if (a.is_missing()) {
          a.val_assign(0);
        }

        dynd::nd::array b = kwds.p("b");
        if (b.is_missing()) {
          b.val_assign(std::numeric_limits<R>::max());
        }
      }

      static void resolve_option_values(const arrfunc_type_data *self,
                                        const arrfunc_type *self_tp,
                                        intptr_t nsrc, const ndt::type *src_tp,
                                        dynd::nd::array &kwds)
      {
        ndt::type tp = kwds.p("tp").f("dereference").as<ndt::type>();
        switch (tp.get_type_id()) {
        case int32_type_id:
          resolve_option_values<int32_t>(self, self_tp, nsrc, src_tp, kwds);
          break;
        case int64_type_id:
          resolve_option_values<int64_t>(self, self_tp, nsrc, src_tp, kwds);
          break;
        default:
          throw std::runtime_error("error");
        }
      }

      template <typename R>
      static intptr_t instantiate(const arrfunc_type_data *DYND_UNUSED(self),
                                  const arrfunc_type *DYND_UNUSED(self_tp),
                                  void *ckb, intptr_t ckb_offset,
                                  const ndt::type &DYND_UNUSED(dst_tp),
                                  const char *DYND_UNUSED(dst_arrmeta),
                                  const ndt::type *DYND_UNUSED(src_tp),
                                  const char *const *DYND_UNUSED(src_arrmeta),
                                  kernel_request_t kernreq,
                                  const eval::eval_context *DYND_UNUSED(ectx),
                                  const dynd::nd::array &kwds)
      {
        R a = kwds.p("a").as<R>();
        R b = kwds.p("b").as<R>();

        dynd::nd::array gen = kwds.p("engine");
        if (gen.is_missing()) {
          static std::default_random_engine rng = std::default_random_engine();
          typedef kernels::uniform_ck<kernel_request_host,
                                      std::default_random_engine, R> self_type;
          self_type::create(ckb, kernreq, ckb_offset, rng, a, b);
        } else {
        }

        return ckb_offset;
      }

      static intptr_t
      instantiate(const arrfunc_type_data *self, const arrfunc_type *self_tp,
                  void *ckb, intptr_t ckb_offset, const ndt::type &dst_tp,
                  const char *dst_arrmeta, const ndt::type *src_tp,
                  const char *const *src_arrmeta, kernel_request_t kernreq,
                  const eval::eval_context *ectx, const dynd::nd::array &kwds)
      {
        ndt::type tp = kwds.p("tp").f("dereference").as<ndt::type>();
        switch (tp.get_type_id()) {
        case int32_type_id:
          return instantiate<int32_t>(self, self_tp, ckb, ckb_offset, dst_tp,
                                      dst_arrmeta, src_tp, src_arrmeta, kernreq,
                                      ectx, kwds);
        case int64_type_id:
          return instantiate<int64_t>(self, self_tp, ckb, ckb_offset, dst_tp,
                                      dst_arrmeta, src_tp, src_arrmeta, kernreq,
                                      ectx, kwds);
        default:
          throw std::runtime_error("error");
        }
      }

      static dynd::nd::arrfunc make()
      {
        return dynd::nd::arrfunc(
            &instantiate, &resolve_option_values, NULL,
            ndt::type("(tp: type | R, a: ?R, b: ?R, engine: ?void) -> R"));
      }
    };

  } // namespace dynd::decl::nd
} // namespace dynd::decl

namespace nd {

  extern decl::nd::uniform uniform;

} // namespace dynd::nd

} // namespace dynd