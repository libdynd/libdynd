#pragma once

#include <chrono>
#include <memory>
#include <random>

#include <dynd/kernels/expr_kernels.hpp>
#include <dynd/func/arrfunc.hpp>

namespace dynd {

typedef type_sequence<int32_t, int64_t> integral_types;

namespace kernels {
  template <typename... T>
  struct uniform_ck;

  template <typename G, typename R>
  struct uniform_ck<G, R> : expr_ck<uniform_ck<G, R>, kernel_request_host, 0> {
    typedef uniform_ck self_type;

    G &g;
    std::uniform_int_distribution<R> d;

    uniform_ck(G &g, R a, R b) : g(g), d(a, b){};

    void single(char *dst, char *const *DYND_UNUSED(src))
    {
      *reinterpret_cast<R *>(dst) = d(g);
    }

    static ndt::type make_type()
    {
      return ndt::make_arrfunc(
          ndt::make_tuple(),
          ndt::make_struct(ndt::make_option(ndt::make_type<R>()), "a",
                           ndt::make_option(ndt::make_type<R>()), "b"),
          ndt::make_type<R>());
    }

    static void resolve_option_values(
        const arrfunc_type_data *DYND_UNUSED(self),
        const arrfunc_type *DYND_UNUSED(self_tp), intptr_t DYND_UNUSED(nsrc),
        const ndt::type *DYND_UNUSED(src_tp), dynd::nd::array &kwds,
        const std::map<dynd::nd::string, ndt::type> &DYND_UNUSED(tp_vars))
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

    static intptr_t instantiate(
        const arrfunc_type_data *self, const arrfunc_type *DYND_UNUSED(self_tp),
        void *ckb, intptr_t ckb_offset, const ndt::type &DYND_UNUSED(dst_tp),
        const char *DYND_UNUSED(dst_arrmeta),
        const ndt::type *DYND_UNUSED(src_tp),
        const char *const *DYND_UNUSED(src_arrmeta), kernel_request_t kernreq,
        const eval::eval_context *DYND_UNUSED(ectx),
        const dynd::nd::array &kwds,
        const std::map<dynd::nd::string, ndt::type> &DYND_UNUSED(tp_vars))
    {
      std::shared_ptr<G> g = *self->get_data_as<std::shared_ptr<G>>();
      R a = kwds.p("a").as<R>();
      R b = kwds.p("b").as<R>();

      self_type::create(ckb, kernreq, ckb_offset, *g, a, b);
      return ckb_offset;
    }
  };

} // namespace dynd::kernels

namespace nd {
  namespace decl {

    class uniform : public arrfunc<uniform> {
    public:
      static nd::arrfunc make()
      {
        return as_arrfunc<kernels::uniform_ck, std::default_random_engine,
                          integral_types>(
            ndt::type("(a: ?R, b: ?R, tp: type | R) -> R"),
            std::shared_ptr<std::default_random_engine>(
                new std::default_random_engine()));
      }
    };
  } // namespace dynd::nd::decl

  extern decl::uniform uniform;

} // namespace dynd::nd

} // namespace dynd