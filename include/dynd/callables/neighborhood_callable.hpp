//
// Copyright (C) 2011-15 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#pragma once

#include <dynd/callables/base_instantiable_callable.hpp>
#include <dynd/kernels/neighborhood.hpp>

namespace dynd {
namespace nd {
  namespace functional {

    template <int N>
    class neighborhood_callable : public base_callable {
      callable m_child;
      callable m_boundary_child;

    public:
      struct data_type {
        ndt::type child_src_tp;
        size_stride_t child_src_arrmeta[5];

        const ndt::type *src_tp;
        const char *src_arrmeta;

        intptr_t ndim;
        intptr_t *shape;
        int *offset;
        std::shared_ptr<bool> out_of_bounds;

        data_type(const ndt::type *src_tp, intptr_t ndim, int *shape, int *offset)
            : src_tp(src_tp), src_arrmeta(NULL), ndim(ndim), offset(offset), out_of_bounds(std::make_shared<bool>())
        {
          this->shape = new intptr_t[ndim];
          for (int i = 0; i < ndim; ++i) {
            this->shape[i] = shape[i];
          }

          /*
                    child_src_arrmeta = new char[ndim * sizeof(size_stride_t)];
                    for (int i = 0; i < ndim; ++i) {
                      (reinterpret_cast<size_stride_t *>(child_src_arrmeta) + i)->dim_size = shape[i];
                      (reinterpret_cast<size_stride_t *>(child_src_arrmeta) + i)->stride = sizeof(int);
                    }
          */

          if (ndim == 1) {
            child_src_arrmeta[0].dim_size = shape[0];
            child_src_arrmeta[0].stride = sizeof(int);
          }
          else if (ndim == 2) {
            child_src_arrmeta[0].dim_size = shape[0];
            child_src_arrmeta[0].stride = 4 * sizeof(int);
            child_src_arrmeta[1].dim_size = shape[1];
            child_src_arrmeta[1].stride = sizeof(int);
          }
          else if (ndim == 3) {
            child_src_arrmeta[0].dim_size = shape[0];
            child_src_arrmeta[0].stride = 4 * 4 * sizeof(int);
            child_src_arrmeta[1].dim_size = shape[1];
            child_src_arrmeta[1].stride = 4 * sizeof(int);
            child_src_arrmeta[2].dim_size = shape[2];
            child_src_arrmeta[2].stride = sizeof(int);
          }
        }

        ~data_type()
        {
          //          delete[] child_src_arrmeta;
        }
      };

      neighborhood_callable(const ndt::type &tp, const callable &child, const callable &boundary_child)
          : base_callable(tp), m_child(child), m_boundary_child(boundary_child)
      {
      }

      char *data_init(const ndt::type &DYND_UNUSED(dst_tp), intptr_t DYND_UNUSED(nsrc), const ndt::type *src_tp,
                      intptr_t DYND_UNUSED(nkwd), const array *kwds,
                      const std::map<std::string, ndt::type> &DYND_UNUSED(tp_vars))
      {
        char *data = reinterpret_cast<char *>(
            new data_type(src_tp, kwds[0].get_dim_size(), reinterpret_cast<int *>(kwds[0].data()),
                          kwds[1].is_na() ? NULL : reinterpret_cast<int *>(kwds[1].data())));

        reinterpret_cast<data_type *>(data)->child_src_tp =
            ndt::substitute_shape(m_child.get_arg_type(0), reinterpret_cast<data_type *>(data)->ndim,
                                  reinterpret_cast<data_type *>(data)->shape);

        return data;
      }

      void resolve_dst_type(char *DYND_UNUSED(data), ndt::type &dst_tp, intptr_t DYND_UNUSED(nsrc),
                            const ndt::type *src_tp, intptr_t DYND_UNUSED(nkwd), const array *DYND_UNUSED(kwds),
                            const std::map<std::string, ndt::type> &DYND_UNUSED(tp_vars))
      {
        // swap in the input dimension values for the Fixed**N
        intptr_t ndim = src_tp[0].get_ndim();
        dimvector shape(ndim);
        src_tp[0].extended()->get_shape(ndim, 0, shape.get(), NULL, NULL);
        dst_tp = ndt::substitute_shape(dst_tp, ndim, shape.get());
      }

      void instantiate(char *data, kernel_builder *ckb, const ndt::type &dst_tp, const char *dst_arrmeta, intptr_t nsrc,
                       const ndt::type *src_tp, const char *const *src_arrmeta, kernel_request_t kernreq, intptr_t nkwd,
                       const nd::array *kwds, const std::map<std::string, ndt::type> &tp_vars)
      {
        intptr_t neighborhood_offset = ckb->size();
        ckb->emplace_back<neighborhood_kernel<N>>(
            kernreq, reinterpret_cast<const fixed_dim_type_arrmeta *>(dst_arrmeta)->stride,
            reinterpret_cast<const fixed_dim_type_arrmeta *>(src_arrmeta[0])->dim_size,
            reinterpret_cast<const fixed_dim_type_arrmeta *>(src_arrmeta[0])->stride,
            reinterpret_cast<data_type *>(data)->shape[0],
            (reinterpret_cast<data_type *>(data)->offset == NULL) ? 0 : reinterpret_cast<data_type *>(data)->offset[0],
            reinterpret_cast<data_type *>(data)->out_of_bounds);

        const ndt::type &child_dst_tp = dst_tp.extended<ndt::fixed_dim_type>()->get_element_type();
        const char *child_dst_arrmeta = dst_arrmeta + sizeof(fixed_dim_type_arrmeta);

        reinterpret_cast<data_type *>(data)->ndim -= 1;
        reinterpret_cast<data_type *>(data)->shape += 1;
        if (reinterpret_cast<data_type *>(data)->offset != NULL) {
          reinterpret_cast<data_type *>(data)->offset += 1;
        }

        if (reinterpret_cast<data_type *>(data)->ndim == 0) {
          const callable &child = m_child;
          const callable &boundary_child = m_boundary_child;

          const char *child_src_arrmeta =
              reinterpret_cast<char *>(reinterpret_cast<data_type *>(data)->child_src_arrmeta);
          child.get()->instantiate(NULL, ckb, child_dst_tp, child_dst_arrmeta, nsrc,
                                   &reinterpret_cast<data_type *>(data)->child_src_tp, &child_src_arrmeta,
                                   kernel_request_single, nkwd - 3, kwds + 3, tp_vars);
          ckb->get_at<neighborhood_kernel<N>>(neighborhood_offset)->boundary_child_offset =
              ckb->size() - neighborhood_offset;

          boundary_child.get()->instantiate(NULL, ckb, child_dst_tp, child_dst_arrmeta, 0, NULL, NULL,
                                            kernel_request_single, nkwd - 3, kwds + 3, tp_vars);

          delete reinterpret_cast<data_type *>(data);
          return;
        }

        ndt::type child_src_tp[N];
        const char *child_src_arrmeta[N];
        for (int i = 0; i < N; ++i) {
          child_src_tp[i] = src_tp[i].extended<ndt::fixed_dim_type>()->get_element_type();
          child_src_arrmeta[i] = src_arrmeta[i] + sizeof(fixed_dim_type_arrmeta);
        }

        ckb->get_at<neighborhood_kernel<N>>(neighborhood_offset)->boundary_child_offset =
            sizeof(neighborhood_kernel<N>);

        return instantiate(data, ckb, child_dst_tp, child_dst_arrmeta, nsrc, child_src_tp, child_src_arrmeta,
                           kernel_request_single, nkwd, kwds, tp_vars);
      }
    };

  } // namespace dynd::nd::functional
} // namespace dynd::nd
} // namespace dynd
