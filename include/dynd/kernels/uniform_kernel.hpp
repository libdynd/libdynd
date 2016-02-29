//
// Copyright (C) 2011-15 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#pragma once

#include <dynd/kernels/base_kernel.hpp>

namespace dynd {

inline std::shared_ptr<std::default_random_engine> &get_random_device()
{
  static std::random_device random_device;
  static std::shared_ptr<std::default_random_engine> g(new std::default_random_engine(random_device()));

  return g;
}

namespace nd {
  namespace random {
    namespace detail {

      template <type_id_t ResID, type_id_t ResBaseID, typename GeneratorType>
      struct uniform_kernel;

      template <type_id_t ResID, typename GeneratorType>
      struct uniform_kernel<ResID, int_kind_id, GeneratorType>
          : base_strided_kernel<uniform_kernel<ResID, int_kind_id, GeneratorType>, 0> {
        typedef typename type_of<ResID>::type R;

        GeneratorType &g;
        std::uniform_int_distribution<R> d;

        uniform_kernel(GeneratorType *g, R a, R b) : g(*g), d(a, b) {}

        void single(char *dst, char *const *DYND_UNUSED(src)) { *reinterpret_cast<R *>(dst) = d(g); }

        /*
            static void
            data_init(const arrfunc_type_data *DYND_UNUSED(self),
                      const ndt::arrfunc_type *DYND_UNUSED(self_tp),
                      const char *DYND_UNUSED(static_data),
                      size_t DYND_UNUSED(data_size), char *DYND_UNUSED(data),
                      intptr_t DYND_UNUSED(nsrc), const ndt::type
           *DYND_UNUSED(src_tp),
                      nd::array &kwds,
                      const std::map<std::string, ndt::type>
           &DYND_UNUSED(tp_vars))
            {
              nd::array a = kwds.p("a");
              if (a.is_na()) {
                a.val_assign(0);
              }

              nd::array b = kwds.p("b");
              if (b.is_na()) {
                b.val_assign(std::numeric_limits<R>::max());
              }
            }
        */

        static void instantiate(char *DYND_UNUSED(static_data), char *DYND_UNUSED(data), kernel_builder *ckb,
                                const ndt::type &DYND_UNUSED(dst_tp), const char *DYND_UNUSED(dst_arrmeta),
                                intptr_t DYND_UNUSED(nsrc), const ndt::type *DYND_UNUSED(src_tp),
                                const char *const *DYND_UNUSED(src_arrmeta), kernel_request_t kernreq,
                                intptr_t DYND_UNUSED(nkwd), const nd::array *kwds,
                                const std::map<std::string, ndt::type> &DYND_UNUSED(tp_vars))
        {
          std::shared_ptr<GeneratorType> g = get_random_device();

          R a;
          if (kwds[0].is_na()) {
            a = 0;
          }
          else {
            a = kwds[0].as<R>();
          }

          R b;
          if (kwds[1].is_na()) {
            b = std::numeric_limits<R>::max();
          }
          else {
            b = kwds[1].as<R>();
          }

          ckb->emplace_back<uniform_kernel>(kernreq, g.get(), a, b);
        }
      };

      template <type_id_t ResID, typename GeneratorType>
      struct uniform_kernel<ResID, uint_kind_id, GeneratorType> : uniform_kernel<ResID, int_kind_id, GeneratorType> {
      };

      template <type_id_t ResID, typename GeneratorType>
      struct uniform_kernel<ResID, float_kind_id, GeneratorType>
          : base_strided_kernel<uniform_kernel<ResID, float_kind_id, GeneratorType>, 0> {
        typedef typename type_of<ResID>::type R;

        GeneratorType &g;
        std::uniform_real_distribution<R> d;

        uniform_kernel(GeneratorType *g, R a, R b) : g(*g), d(a, b) {}

        void single(char *dst, char *const *DYND_UNUSED(src)) { *reinterpret_cast<R *>(dst) = d(g); }

        /*
            static void
            data_init(const arrfunc_type_data *DYND_UNUSED(self),
                      const ndt::arrfunc_type *DYND_UNUSED(self_tp),
                      const char *DYND_UNUSED(static_data),
                      size_t DYND_UNUSED(data_size), char *DYND_UNUSED(data),
                      intptr_t DYND_UNUSED(nsrc), const ndt::type
           *DYND_UNUSED(src_tp),
                      nd::array &kwds,
                      const std::map<std::string, ndt::type>
           &DYND_UNUSED(tp_vars))
            {
              nd::array a = kwds.p("a");
              if (a.is_na()) {
                a.val_assign(0);
              }

              nd::array b = kwds.p("b");
              if (b.is_na()) {
                b.val_assign(1);
              }
            }
        */

        static void instantiate(char *DYND_UNUSED(static_data), char *DYND_UNUSED(data), kernel_builder *ckb,
                                const ndt::type &DYND_UNUSED(dst_tp), const char *DYND_UNUSED(dst_arrmeta),
                                intptr_t DYND_UNUSED(nsrc), const ndt::type *DYND_UNUSED(src_tp),
                                const char *const *DYND_UNUSED(src_arrmeta), kernel_request_t kernreq,
                                intptr_t DYND_UNUSED(nkwd), const nd::array *kwds,
                                const std::map<std::string, ndt::type> &DYND_UNUSED(tp_vars))
        {
          std::shared_ptr<GeneratorType> g = get_random_device();

          R a;
          if (kwds[0].is_na()) {
            a = 0;
          }
          else {
            a = kwds[0].as<R>();
          }

          R b;
          if (kwds[1].is_na()) {
            b = 1;
          }
          else {
            b = kwds[1].as<R>();
          }

          ckb->emplace_back<uniform_kernel>(kernreq, g.get(), a, b);
        }
      };

      template <type_id_t ResID, typename GeneratorType>
      struct uniform_kernel<ResID, complex_kind_id, GeneratorType>
          : base_strided_kernel<uniform_kernel<ResID, complex_kind_id, GeneratorType>, 0> {
        typedef typename type_of<ResID>::type R;

        GeneratorType &g;
        std::uniform_real_distribution<typename R::value_type> d_real;
        std::uniform_real_distribution<typename R::value_type> d_imag;

        uniform_kernel(GeneratorType *g, R a, R b) : g(*g), d_real(a.real(), b.real()), d_imag(a.imag(), b.imag()) {}

        void single(char *dst, char *const *DYND_UNUSED(src)) { *reinterpret_cast<R *>(dst) = R(d_real(g), d_imag(g)); }

        /*
            static void
            data_init(const arrfunc_type_data *DYND_UNUSED(self),
                      const ndt::arrfunc_type *DYND_UNUSED(self_tp),
                      const char *DYND_UNUSED(static_data),
                      size_t DYND_UNUSED(data_size), char *DYND_UNUSED(data),
                      intptr_t DYND_UNUSED(nsrc), const ndt::type
           *DYND_UNUSED(src_tp),
                      nd::array &kwds,
                      const std::map<std::string, ndt::type>
           &DYND_UNUSED(tp_vars))
            {
              nd::array a = kwds.p("a");
              if (a.is_na()) {
                a.val_assign(R(0, 0));
              }

              nd::array b = kwds.p("b");
              if (b.is_na()) {
                b.val_assign(R(1, 1));
              }
            }
        */

        static void instantiate(char *DYND_UNUSED(static_data), char *DYND_UNUSED(data), kernel_builder *ckb,
                                const ndt::type &DYND_UNUSED(dst_tp), const char *DYND_UNUSED(dst_arrmeta),
                                intptr_t DYND_UNUSED(nsrc), const ndt::type *DYND_UNUSED(src_tp),
                                const char *const *DYND_UNUSED(src_arrmeta), kernel_request_t kernreq,
                                intptr_t DYND_UNUSED(nkwd), const nd::array *kwds,
                                const std::map<std::string, ndt::type> &DYND_UNUSED(tp_vars))
        {
          std::shared_ptr<GeneratorType> g = get_random_device();

          R a;
          if (kwds[0].is_na()) {
            a = R(0, 0);
          }
          else {
            a = kwds[0].as<R>();
          }

          R b;
          if (kwds[1].is_na()) {
            b = R(1, 1);
          }
          else {
            b = kwds[1].as<R>();
          }

          ckb->emplace_back<uniform_kernel>(kernreq, g.get(), a, b);
        }
      };

    } // namespace dynd::nd::random::detail

    template <type_id_t ResID, typename GeneratorType>
    using uniform_kernel = detail::uniform_kernel<ResID, base_id_of<ResID>::value, GeneratorType>;

  } // namespace dynd::nd::random
} // namespace dynd::nd

namespace ndt {

  template <type_id_t DstTypeID, typename GeneratorType>
  struct traits<nd::random::uniform_kernel<DstTypeID, GeneratorType>> {
    typedef typename dynd::type_of<DstTypeID>::type R;

    static type equivalent()
    {
      std::map<std::string, ndt::type> tp_vars;
      tp_vars["R"] = ndt::make_type<R>();

      return ndt::substitute(ndt::type("(a: ?R, b: ?R) -> R"), tp_vars, true);
    }
  };

} // namespace dynd::ndt
} // namespace dynd
