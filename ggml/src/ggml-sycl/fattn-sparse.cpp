#include "fattn.hpp"
#include "fattn-sparse.hpp"

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <cmath>
#include <vector>

static constexpr int64_t SPARSE_FA_PAD       = 256;
static constexpr int64_t SPARSE_FA_MIN_RATIO = 2;

extern int g_ggml_sycl_enable_sparse_fa;
extern int g_ggml_sycl_debug_sparse_fa;
extern int g_ggml_sycl_sparse_fa_margin;

static int sparse_fa_enabled(void) {
    return g_ggml_sycl_enable_sparse_fa;
}

static int sparse_fa_debug(void) {
    return g_ggml_sycl_debug_sparse_fa;
}

// slack above n_kv_max; callers may exceed the hint by a few always-attended positions
static int sparse_fa_margin(void) {
    return g_ggml_sycl_sparse_fa_margin;
}

// largest query batch gathered (speculative verify batches); the selected sets of the rows are unioned
static int sparse_fa_max_q(void) {
    static const int v = ggml_sycl_get_env("GGML_SYCL_SPARSE_FA_MAX_Q", 8);
    return v;
}

// rows added on top of the FATTN_KQ_STRIDE-padded count. A non-zero value keeps the gathered n_kv off
// the stride multiple, so the dense kernels take the same (non-GQA-tiled) variant they use for almost
// every ungathered decode call.
static int sparse_fa_odd(void) {
    static const int v = ggml_sycl_get_env("GGML_SYCL_SPARSE_FA_ODD", 16);
    return v;
}

static int sparse_fa_sync(void) {
    static const int v = ggml_sycl_get_env("GGML_SYCL_SPARSE_FA_SYNC", 0);
    return v;
}

// Unordered output is fine: softmax over the selected set is permutation invariant.
static void sparse_fa_compact_mask(sycl::queue * stream,
                                   const sycl::half * __restrict__ mask,
                                   int32_t * __restrict__ indices,
                                   int32_t * __restrict__ count,
                                   const int64_t n_kv,
                                   const int64_t n_kv_g,
                                   const int64_t n_rows,
                                   const size_t mask_s1) {
    constexpr size_t WG = 256;
    const size_t global = (size_t) GGML_PAD(n_kv, (int64_t) WG);

    stream->parallel_for(
        sycl::nd_range<1>(sycl::range<1>(global), sycl::range<1>(WG)),
        [=](sycl::nd_item<1> item) {
            const int64_t i = (int64_t) item.get_global_id(0);
            if (i >= n_kv) {
                return;
            }
            bool any = false;
            for (int64_t r = 0; r < n_rows; ++r) {
                any = any || sycl::isfinite((float) mask[(size_t) r * mask_s1 + (size_t) i]);
            }
            if (!any) {
                return;
            }

            sycl::atomic_ref<int32_t,
                             sycl::memory_order::relaxed,
                             sycl::memory_scope::device,
                             sycl::access::address_space::global_space> ctr(*count);

            const int32_t pos = ctr.fetch_add(1);
            if (pos >= 0 && pos < (int32_t) n_kv_g) {
                indices[pos] = (int32_t) i;
            }
        });
}

// Rows along ne[0] are contiguous for every type used as a KV cache, so this is
// a plain byte copy and needs no per-type code. Padding slots are zeroed.
static void sparse_fa_gather_rows(sycl::queue * stream,
                                  const uint8_t * __restrict__ src,
                                  uint8_t * __restrict__ dst,
                                  const int32_t * __restrict__ indices,
                                  const int32_t * __restrict__ count,
                                  const size_t row_size,
                                  const size_t src_nb1,
                                  const size_t src_nb2,
                                  const int64_t n_kv_g,
                                  const int64_t n_head,
                                  const int64_t n_kv) {
    GGML_ASSERT(row_size % sizeof(uint32_t) == 0);
    const size_t words = row_size / sizeof(uint32_t);
    // one work-item copies a run of words: far fewer items than one per word
    const size_t cw     = words % 16 == 0 ? 16 : 1;
    const size_t chunks = words / cw;

    stream->parallel_for(
        sycl::range<3>((size_t) n_head, (size_t) n_kv_g, chunks),
        [=](sycl::id<3> id) {
            const int64_t h    = (int64_t) id[0];
            const int64_t slot = (int64_t) id[1];
            const size_t  w0   = id[2] * cw;

            uint32_t * dst_row =
                (uint32_t *) (dst + ((size_t) (h * n_kv_g + slot)) * row_size);

            const int32_t idx = slot < (int64_t) *count ? indices[slot] : -1;
            if (idx < 0 || (int64_t) idx >= n_kv) {
                for (size_t w = w0; w < w0 + cw; ++w) {
                    dst_row[w] = 0;
                }
                return;
            }

            const uint32_t * src_row =
                (const uint32_t *) (src + (size_t) idx * src_nb1 +
                                    (size_t) h * src_nb2);
            for (size_t w = w0; w < w0 + cw; ++w) {
                dst_row[w] = src_row[w];
            }
        });
}

static void sparse_fa_gather_mask(sycl::queue * stream,
                                  const sycl::half * __restrict__ mask,
                                  sycl::half * __restrict__ mask_g,
                                  const int32_t * __restrict__ indices,
                                  const int32_t * __restrict__ count,
                                  const int64_t n_kv_g,
                                  const int64_t n_rows,
                                  const size_t mask_s1,
                                  const int64_t n_kv) {
    stream->parallel_for(
        sycl::range<2>((size_t) n_rows, (size_t) n_kv_g),
        [=](sycl::id<2> id) {
            const int64_t r    = (int64_t) id[0];
            const int64_t slot = (int64_t) id[1];

            sycl::half v = sycl::half(-INFINITY);
            const int32_t idx = slot < (int64_t) *count ? indices[slot] : -1;
            if (idx >= 0 && (int64_t) idx < n_kv) {
                v = mask[(size_t) r * mask_s1 + (size_t) idx];
            }
            mask_g[(size_t) r * n_kv_g + slot] = v;
        });
}

static bool sparse_fa_applicable(const ggml_tensor * dst, int64_t & n_kv_g_out) {
    const ggml_tensor * Q    = dst->src[0];
    const ggml_tensor * K    = dst->src[1];
    const ggml_tensor * V    = dst->src[2];
    const ggml_tensor * mask = dst->src[3];

    if (!Q || !K || !V || !mask) {
        return false;
    }

    const int32_t n_kv_max = ggml_get_op_params_i32(dst, 4);
    if (n_kv_max <= 0) {
        return false;
    }

    float max_bias      = 0.0f;
    float logit_softcap = 0.0f;
    memcpy(&max_bias,      (const float *) dst->op_params + 1, sizeof(float));
    memcpy(&logit_softcap, (const float *) dst->op_params + 2, sizeof(float));
    if (max_bias != 0.0f || logit_softcap != 0.0f) {
        return false;
    }

    // decode and small speculative-verify batches only; prefill amortises the scan already
    const int64_t n_q = Q->ne[1];
    if (n_q < 1 || n_q > sparse_fa_max_q() || mask->ne[1] < n_q) {
        return false;
    }
    if (K->ne[3] != 1 || V->ne[3] != 1 || mask->ne[2] != 1 || mask->ne[3] != 1) {
        return false;
    }
    if (mask->type != GGML_TYPE_F16 || mask->ne[0] < K->ne[1]) {
        return false;
    }
    if (K->ne[2] != V->ne[2]) {
        return false;
    }

    // nb[1] may stride over heads (interleaved cache); only ne[0] must be contiguous
    if (K->nb[0] != ggml_type_size(K->type) || V->nb[0] != ggml_type_size(V->type)) {
        return false;
    }

    const size_t k_row = ggml_row_size(K->type, K->ne[0]);
    const size_t v_row = ggml_row_size(V->type, V->ne[0]);
    if (k_row % sizeof(uint32_t) || v_row % sizeof(uint32_t)) {
        return false;
    }

    // every query row selects at most n_kv_max positions; size for the union over the batch
    const int64_t n_kv_g = GGML_PAD(n_q * (int64_t) n_kv_max + sparse_fa_margin(), SPARSE_FA_PAD) + sparse_fa_odd();
    if (n_kv_g * SPARSE_FA_MIN_RATIO > K->ne[1]) {
        return false;
    }

    n_kv_g_out = n_kv_g;
    return true;
}

bool ggml_sycl_flash_attn_ext_sparse(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    int64_t n_kv_g = 0;
    if (!sparse_fa_enabled() || !sparse_fa_applicable(dst, n_kv_g)) {
        return false;
    }

    ggml_tensor * K    = dst->src[1];
    ggml_tensor * V    = dst->src[2];
    ggml_tensor * mask = dst->src[3];

    const int64_t n_kv     = K->ne[1];
    const int64_t n_head_k = K->ne[2];
    const int64_t n_rows_m = mask->ne[1];
    const int64_t n_q      = dst->src[0]->ne[1];
    const size_t  mask_s1  = mask->nb[1] / sizeof(sycl::half);

    // attention over a shared cache passes the same tensor as K and V: gather it once
    const bool v_is_k = V == K || (V->data == K->data && V->type == K->type && V->ne[0] == K->ne[0] &&
                                   V->ne[2] == K->ne[2] && V->nb[1] == K->nb[1] && V->nb[2] == K->nb[2]);

    const size_t k_row = ggml_row_size(K->type, K->ne[0]);
    const size_t v_row = ggml_row_size(V->type, V->ne[0]);

    dpct::queue_ptr stream = ctx.stream();

    // some kernel variants only exist for row counts on the FATTN_KQ_STRIDE multiple: drop the odd rows for them
    if (sparse_fa_odd() != 0) {
        ggml_tensor K_p = *K;
        K_p.ne[1] = n_kv_g;
        K_p.nb[1] = k_row;
        K_p.nb[2] = (size_t) n_kv_g * k_row;
        K_p.nb[3] = (size_t) n_head_k * n_kv_g * k_row;
        ggml_tensor V_p = *V;
        V_p.ne[1] = n_kv_g;
        V_p.nb[1] = v_row;
        V_p.nb[2] = (size_t) n_kv_g * v_row;
        V_p.nb[3] = (size_t) V->ne[2] * n_kv_g * v_row;
        ggml_tensor M_p = *mask;
        M_p.ne[0] = n_kv_g;
        M_p.nb[1] = (size_t) n_kv_g * sizeof(sycl::half);
        M_p.nb[2] = M_p.nb[1] * mask->ne[1];
        M_p.nb[3] = M_p.nb[2];
        ggml_tensor dst_p = *dst;
        dst_p.src[1] = &K_p;
        dst_p.src[2] = &V_p;
        dst_p.src[3] = &M_p;
        dst_p.op_params[4] = 0;
        if (!ggml_sycl_flash_attn_ext_supported(ctx.device, &dst_p)) {
            n_kv_g -= sparse_fa_odd();
        }
    }

    ggml_sycl_pool_alloc<int32_t>    idx_alloc(ctx.pool(), (size_t) n_kv_g);
    ggml_sycl_pool_alloc<int32_t>    cnt_alloc(ctx.pool(), 1);
    ggml_sycl_pool_alloc<uint8_t>    k_alloc(ctx.pool(), (size_t) n_head_k * n_kv_g * k_row);
    ggml_sycl_pool_alloc<uint8_t>    v_alloc(ctx.pool());
    if (!v_is_k) {
        v_alloc.alloc((size_t) n_head_k * n_kv_g * v_row);
    }
    ggml_sycl_pool_alloc<sycl::half> m_alloc(ctx.pool(), (size_t) n_rows_m * n_kv_g);

    int32_t *    d_idx  = idx_alloc.get();
    int32_t *    d_cnt  = cnt_alloc.get();
    uint8_t *    d_K    = k_alloc.get();
    uint8_t *    d_V    = v_is_k ? d_K : v_alloc.get();
    sycl::half * d_mask = m_alloc.get();

    // Reset the counter and the index list from a kernel, not with queue::memset: a USM memset is not
    // reliably ordered against the kernels that follow it on this stack, and a late reset of the counter
    // corrupts the compaction (short counts, out-of-range slots).
    stream->parallel_for(sycl::range<1>((size_t) n_kv_g), [=](sycl::id<1> id) {
        d_idx[id[0]] = -1;
        if (id[0] == 0) {
            *d_cnt = 0;
        }
    });

    if (sparse_fa_debug() >= 2) {
        // host recount of the selected positions, before any gather kernel runs
        std::vector<sycl::half> h_mask((size_t) n_kv * (size_t) n_q);
        for (int64_t r = 0; r < n_q; ++r) {
            SYCL_CHECK(CHECK_TRY_ERROR(stream->memcpy(h_mask.data() + r * n_kv, (const sycl::half *) mask->data + r * mask_s1,
                                                      (size_t) n_kv * sizeof(sycl::half))));
        }
        SYCL_CHECK(CHECK_TRY_ERROR(stream->wait()));
        int64_t h_fin = 0, first = -1, last = -1;
        for (int64_t i = 0; i < n_kv; ++i) {
            bool any = false;
            for (int64_t r = 0; r < n_q; ++r) { any = any || std::isfinite((float) h_mask[r * n_kv + i]); }
            if (any) { h_fin++; if (first < 0) first = i; last = i; }
        }
        fprintf(stderr, "[FA-SPARSE2] dev=%d K=%s ne=[%lld,%lld,%lld,%lld] nb=[%zu,%zu,%zu] V%sK mask ne=[%lld,%lld,%lld,%lld] nb1=%zu host_finite=%lld first=%lld last=%lld\n",
                ctx.device, ggml_type_name(K->type), (long long) K->ne[0], (long long) K->ne[1], (long long) K->ne[2], (long long) K->ne[3],
                K->nb[1], K->nb[2], K->nb[3], v_is_k ? "==" : "!=", (long long) mask->ne[0], (long long) mask->ne[1], (long long) mask->ne[2],
                (long long) mask->ne[3], mask->nb[1], (long long) h_fin, (long long) first, (long long) last);
    }

    sparse_fa_compact_mask(stream, (const sycl::half *) mask->data,
                           d_idx, d_cnt, n_kv, n_kv_g, n_q, mask_s1);

    sparse_fa_gather_rows(stream, (const uint8_t *) K->data, d_K, d_idx, d_cnt,
                          k_row, K->nb[1], K->nb[2], n_kv_g, n_head_k, n_kv);

    if (!v_is_k) {
        sparse_fa_gather_rows(stream, (const uint8_t *) V->data, d_V, d_idx, d_cnt,
                              v_row, V->nb[1], V->nb[2], n_kv_g, n_head_k, n_kv);
    }

    sparse_fa_gather_mask(stream, (const sycl::half *) mask->data, d_mask,
                          d_idx, d_cnt, n_kv_g, n_rows_m, mask_s1, n_kv);

    if (sparse_fa_debug()) {
        int32_t h_cnt = 0;
        SYCL_CHECK(CHECK_TRY_ERROR(stream->memcpy(&h_cnt, d_cnt, sizeof(int32_t))));
        SYCL_CHECK(CHECK_TRY_ERROR(stream->wait()));
        fprintf(stderr, "[FA-SPARSE] n_q=%lld n_kv=%lld n_kv_max=%d n_kv_g=%lld finite=%d%s\n",
                (long long) n_q, (long long) n_kv, ggml_get_op_params_i32(dst, 4),
                (long long) n_kv_g, (int) h_cnt,
                h_cnt > (int32_t) n_kv_g ? "  OVERFLOW" : "");
    }

    // shallow copies retargeted at the gathered buffers; kernels are unchanged
    ggml_tensor K_g = *K;
    K_g.data      = d_K;
    K_g.ne[1]     = n_kv_g;
    K_g.nb[1]     = k_row;
    K_g.nb[2]     = (size_t) n_kv_g * k_row;
    K_g.nb[3]     = (size_t) n_head_k * n_kv_g * k_row;
    K_g.view_src  = nullptr;
    K_g.view_offs = 0;

    ggml_tensor V_g = *V;
    V_g.data      = d_V;
    V_g.ne[1]     = n_kv_g;
    V_g.nb[1]     = v_row;
    V_g.nb[2]     = (size_t) n_kv_g * v_row;
    V_g.nb[3]     = (size_t) V->ne[2] * n_kv_g * v_row;
    V_g.view_src  = nullptr;
    V_g.view_offs = 0;

    ggml_tensor M_g = *mask;
    M_g.data      = d_mask;
    M_g.ne[0]     = n_kv_g;
    M_g.nb[1]     = (size_t) n_kv_g * sizeof(sycl::half);
    M_g.nb[2]     = M_g.nb[1] * mask->ne[1];
    M_g.nb[3]     = M_g.nb[2];
    M_g.view_src  = nullptr;
    M_g.view_offs = 0;

    ggml_tensor dst_g = *dst;
    dst_g.src[1] = &K_g;
    dst_g.src[2] = &V_g;
    dst_g.src[3] = &M_g;
    dst_g.op_params[4] = 0;   // avoid re-entering this path

    ggml_sycl_flash_attn_ext(ctx, &dst_g);

    if (sparse_fa_sync()) {
        SYCL_CHECK(CHECK_TRY_ERROR(stream->wait()));
    }

    return true;
}
