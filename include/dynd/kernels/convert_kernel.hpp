//
// Copyright (C) 2011-15 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#pragma once

#include <dynd/func/assignment.hpp>

namespace dynd {
namespace nd {
  namespace functional {

    /**
     * Given a buffer array of type "N * T" which was
     * created by nd::empty, resets it so it can be used
     * as a buffer again.
     *
     * NOTE: If the array is not of type "N * T" and default
     *       initialized by nd::empty, undefined behavior will result.
     *
     */
    inline void reset_strided_buffer_array(const nd::array &buf)
    {
      const ndt::type &buf_tp = buf.get_type();
      uint32_t flags = buf_tp.extended()->get_flags();
      if (flags & (type_flag_blockref | type_flag_zeroinit | type_flag_destructor)) {
        char *buf_arrmeta = buf.get()->metadata();
        char *buf_data = buf.data();
        buf_tp.extended()->arrmeta_reset_buffers(buf.get()->metadata());
        fixed_dim_type_arrmeta *am = reinterpret_cast<fixed_dim_type_arrmeta *>(buf_arrmeta);
        if (flags & type_flag_destructor) {
          buf_tp.extended()->data_destruct(buf_arrmeta, buf_data);
        }
        memset(buf_data, 0, am->dim_size * am->stride);
      }
    }

    class DYND_API buffer_storage {
      char *m_storage;
      char *m_arrmeta;
      ndt::type m_type;
      intptr_t m_stride;

      void internal_allocate()
      {
        if (m_type.get_id() != uninitialized_id) {
          m_stride = m_type.get_data_size();
          m_storage = new char[DYND_BUFFER_CHUNK_SIZE * m_stride];
          m_arrmeta = NULL;
          size_t metasize = m_type.is_builtin() ? 0 : m_type.extended()->get_arrmeta_size();
          if (metasize != 0) {
            try {
              m_arrmeta = new char[metasize];
              m_type.extended()->arrmeta_default_construct(m_arrmeta, true);
            }
            catch (...) {
              delete[] m_storage;
              delete[] m_arrmeta;
              throw;
            }
          }
        }
      }

    public:
      buffer_storage() : m_storage(NULL), m_arrmeta(NULL), m_type() {}

      buffer_storage(const buffer_storage &rhs) : m_storage(NULL), m_arrmeta(NULL), m_type(rhs.m_type)
      {
        internal_allocate();
      }

      buffer_storage(const ndt::type &tp) : m_storage(NULL), m_arrmeta(NULL), m_type(tp) { internal_allocate(); }

      ~buffer_storage()
      {
        if (m_storage && m_type.get_flags() & type_flag_destructor) {
          m_type.extended()->data_destruct_strided(m_arrmeta, m_storage, m_stride, DYND_BUFFER_CHUNK_SIZE);
        }
        delete[] m_storage;
        if (m_arrmeta) {
          m_type.extended()->arrmeta_destruct(m_arrmeta);
          delete[] m_arrmeta;
        }
      }

      // Assignment copies the same type
      buffer_storage &operator=(const buffer_storage &rhs)
      {
        allocate(rhs.m_type);
        return *this;
      }

      void allocate(const ndt::type &tp)
      {
        delete[] m_storage;
        m_storage = 0;
        if (m_arrmeta) {
          m_type.extended()->arrmeta_destruct(m_arrmeta);
          delete[] m_arrmeta;
          m_arrmeta = NULL;
        }
        m_type = tp;
        internal_allocate();
      }

      bool is_null() const { return m_storage == NULL; }

      intptr_t get_stride() const { return m_stride; }

      const ndt::type &get_type() const { return m_type; }

      char *const &get_storage() const { return m_storage; }

      const char *get_arrmeta() const { return m_arrmeta; }

      void reset_arrmeta()
      {
        if (m_arrmeta && !m_type.is_builtin()) {
          m_type.extended()->arrmeta_reset_buffers(m_arrmeta);
        }
      }
    };

    /**
     * Instantiate an callable, adding buffers for any inputs where the types
     * don't match.
     */
    struct convert_kernel : base_kernel<convert_kernel> {
      typedef callable static_data_type;

      intptr_t narg;
      std::vector<intptr_t> m_src_buf_ck_offsets;
      std::vector<buffer_storage> m_bufs;

      convert_kernel(intptr_t narg) : narg(narg), m_src_buf_ck_offsets(this->narg), m_bufs(this->narg) {}

      void call(array *dst, const array *src)
      {
        std::vector<char *> src_data(narg);
        for (int i = 0; i < narg; ++i) {
          src_data[i] = const_cast<char *>(src[i].cdata());
        }
        this->single(const_cast<char *>(dst->cdata()), src_data.data());
      }

      void single(char *dst, char *const *src)
      {
        std::vector<char *> buf_src(narg);
        for (intptr_t i = 0; i < narg; ++i) {
          if (!m_bufs[i].is_null()) {
            m_bufs[i].reset_arrmeta();
            kernel_prefix *ck = get_child(m_src_buf_ck_offsets[i]);
            ck->single(m_bufs[i].get_storage(), &src[i]);
            buf_src[i] = m_bufs[i].get_storage();
          }
          else {
            buf_src[i] = src[i];
          }
        }
        kernel_prefix *child = get_child();
        child->single(dst, &buf_src[0]);
      }

      void strided(char *dst, intptr_t dst_stride, char *const *src, const intptr_t *src_stride, size_t count)
      {
        std::vector<char *> buf_src(narg);
        std::vector<intptr_t> buf_stride(narg);
        kernel_prefix *child = get_child();
        kernel_strided_t child_fn = child->get_function<kernel_strided_t>();

        for (intptr_t i = 0; i < narg; ++i) {
          if (!m_bufs[i].is_null()) {
            buf_src[i] = m_bufs[i].get_storage();
            buf_stride[i] = m_bufs[i].get_stride();
          }
          else {
            buf_src[i] = src[i];
            buf_stride[i] = src_stride[i];
          }
        }

        while (count > 0) {
          size_t chunk_size = std::min(count, (size_t)DYND_BUFFER_CHUNK_SIZE);
          for (intptr_t i = 0; i < narg; ++i) {
            if (!m_bufs[i].is_null()) {
              m_bufs[i].reset_arrmeta();
              kernel_prefix *ck = get_child(m_src_buf_ck_offsets[i]);
              kernel_strided_t ck_fn = ck->get_function<kernel_strided_t>();
              ck_fn(ck, m_bufs[i].get_storage(), m_bufs[i].get_stride(), &src[i], &src_stride[i], chunk_size);
            }
          }
          child_fn(child, dst, dst_stride, &buf_src[0], &buf_stride[0], chunk_size);
          for (intptr_t i = 0; i < narg; ++i) {
            if (!m_bufs[i].is_null()) {
              m_bufs[i].reset_arrmeta();
              kernel_prefix *ck = get_child(m_src_buf_ck_offsets[i]);
              kernel_strided_t ck_fn = ck->get_function<kernel_strided_t>();
              ck_fn(ck, m_bufs[i].get_storage(), buf_stride[i], &src[i], &src_stride[i], chunk_size);
            }
            else {
              buf_src[i] += chunk_size * buf_stride[i];
            }
          }
          count -= chunk_size;
        }
      }

      static void instantiate(char *static_data, char *DYND_UNUSED(data), kernel_builder *ckb, const ndt::type &dst_tp,
                              const char *dst_arrmeta, intptr_t nsrc, const ndt::type *src_tp,
                              const char *const *src_arrmeta, kernel_request_t kernreq, intptr_t nkwd,
                              const array *kwds, const std::map<std::string, ndt::type> &tp_vars)
      {
        intptr_t ckb_offset = ckb->size();
        callable &af = *reinterpret_cast<callable *>(static_data);
        const std::vector<ndt::type> &src_tp_for_af = af.get_type()->get_pos_types();

        intptr_t root_ckb_offset = ckb_offset;
        ckb->emplace_back<convert_kernel>(kernreq, nsrc);
        ckb_offset = ckb->size();
        std::vector<const char *> buffered_arrmeta(nsrc);
        convert_kernel *self = ckb->get_at<convert_kernel>(root_ckb_offset);
        for (intptr_t i = 0; i < nsrc; ++i) {
          if (src_tp[i] == src_tp_for_af[i]) {
            buffered_arrmeta[i] = src_arrmeta[i];
          }
          else {
            self->m_bufs[i].allocate(src_tp_for_af[i]);
            buffered_arrmeta[i] = self->m_bufs[i].get_arrmeta();
          }
        }
        // Instantiate the callable being buffered
        af.get()->instantiate(af.get()->static_data(), NULL, ckb, dst_tp, dst_arrmeta, nsrc, src_tp_for_af.data(),
                              &buffered_arrmeta[0], kernreq | kernel_request_data_only, nkwd, kwds, tp_vars);
        ckb_offset = ckb->size();
        reinterpret_cast<kernel_builder *>(ckb)->reserve(ckb_offset + sizeof(kernel_prefix));
        self = reinterpret_cast<kernel_builder *>(ckb)->get_at<convert_kernel>(root_ckb_offset);
        // Instantiate assignments for all the buffered operands
        for (intptr_t i = 0; i < nsrc; ++i) {
          if (!self->m_bufs[i].is_null()) {
            self->m_src_buf_ck_offsets[i] = ckb_offset - root_ckb_offset;
            nd::array error_mode = eval::default_eval_context.errmode;
            assign::get()->instantiate(assign::get()->static_data(), NULL, ckb, src_tp_for_af[i],
                                       self->m_bufs[i].get_arrmeta(), 1, src_tp + i, src_arrmeta + i,
                                       kernreq | kernel_request_data_only, 1, &error_mode, tp_vars);
            ckb_offset = ckb->size();
            reinterpret_cast<kernel_builder *>(ckb)->reserve(ckb_offset + sizeof(kernel_prefix));
            if (i < nsrc - 1) {
              self = reinterpret_cast<kernel_builder *>(ckb)->get_at<convert_kernel>(root_ckb_offset);
            }
          }
        }
      }
    };

  } // namespace dynd::nd::functional
} // namespace dynd::nd
} // namespace dynd
