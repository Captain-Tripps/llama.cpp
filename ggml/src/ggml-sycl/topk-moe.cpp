#include <algorithm>
#include <cstdio>
#include <atomic>
#include <cfloat>
#include <initializer_list>
#include <vector>

#include "ggml.h"
#include "ggml-impl.h"
#include "ggml-backend-impl.h"
#include "topk-moe.hpp"

// SYCL port of ggml-cuda/topk-moe.cu. The kernel is a translation of the CUDA no-bias, no-PDL
// path of topk_moe_cuda; the fusion-detection helpers below are ported near-verbatim from
// ggml-cuda.cu (pure graph / pointer inspection, backend-agnostic). Bias is not implemented here:
// if a routing bias is detected, the fusion is declined and the eager path runs unchanged.

struct ggml_sycl_topk_moe_args {
    bool sigmoid{};
    bool softmax{};
    bool delayed_softmax{};
    bool prob_bias{};
    bool norm{};
    bool scale{};
};

struct topk_moe_config {
    bool use_sigmoid;
    bool with_norm;
    bool delayed_softmax;
};

// warp-local softmax used for both the pre-top-k logits and the post-top-k delayed path
template <int experts_per_thread, bool use_limit>
static inline void softmax_warp_inplace(float (&vals)[experts_per_thread], const int limit, const int lane) {
    float max_val = -INFINITY;
#pragma unroll
    for (int i = 0; i < experts_per_thread; i++) {
        const int  idx    = lane + i * WARP_SIZE;
        const bool active = !use_limit || (idx < limit);
        if (active) {
            max_val = sycl::fmax(max_val, vals[i]);
        }
    }
    max_val = warp_reduce_max<WARP_SIZE>(max_val);

    float sum = 0.f;
#pragma unroll
    for (int i = 0; i < experts_per_thread; i++) {
        const int  idx    = lane + i * WARP_SIZE;
        const bool active = !use_limit || (idx < limit);
        if (active) {
            const float val = sycl::exp(vals[i] - max_val);
            vals[i]         = val;
            sum += val;
        } else {
            vals[i] = 0.f;
        }
    }
    sum = warp_reduce_sum<WARP_SIZE>(sum);

    const float inv_sum = 1.0f / sum;
#pragma unroll
    for (int i = 0; i < experts_per_thread; i++) {
        const int idx = lane + i * WARP_SIZE;
        if (!use_limit || idx < limit) {
            vals[i] *= inv_sum;
        }
    }
}

template <int experts_per_thread, bool use_limit>
static inline void sigmoid_warp_inplace(float (&vals)[experts_per_thread], const int limit, const int lane) {
#pragma unroll
    for (int i = 0; i < experts_per_thread; i++) {
        const int  idx    = lane + i * WARP_SIZE;
        const bool active = !use_limit || (idx < limit);
        vals[i]           = active ? 1.f / (1.f + sycl::exp(-vals[i])) : -INFINITY;
    }
}

/*
    This kernel does the following:
    1. optionally softmax/sigmoid over the logits per token [n_experts, n_tokens]
    2. argmax reduce over the top-k (n_experts_used) logits
    3. write weights + ids to global memory
    4. optionally normalize the weights or apply softmax over the selected logits

    It is intended as a fusion of the softmax->top-k->get_rows pipeline for MoE models.
    One sub-group handles one row/token, mirroring topk_moe_cuda's one-warp-per-row layout.
*/
template <int n_experts, bool single_wg = false>
static void topk_moe_kernel(const float * __restrict__ logits,
                            float * __restrict__       weights,
                            int32_t * __restrict__     ids,
                            const int                  n_rows,
                            const int                  n_expert_used,
                            const float                clamp_val,
                            const float                scale_val,
                            const topk_moe_config       config) {
    auto      item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<1>();
    // single_wg: every row of a small batch is one sub-group of ONE work-group, so a group barrier after the logits
    // load makes the kernel safe when weights / ids alias the logits buffer (every row is read before any write)
    const int row       = single_wg ? (int) item_ct1.get_sub_group().get_group_linear_id() : (int) item_ct1.get_group(0);
    if (!single_wg && row >= n_rows) {
        return;
    }
    const int lane = single_wg ? (int) item_ct1.get_sub_group().get_local_linear_id() : (int) item_ct1.get_local_id(0);

    logits  += (size_t) n_experts * row;
    weights += (size_t) n_expert_used * row;
    ids     += (size_t) n_experts * row;  // ids row stride is n_experts (matches the argsort tensor)

    constexpr int experts_per_thread = (n_experts > WARP_SIZE) ? n_experts / WARP_SIZE : 1;

    float wt[experts_per_thread];
#pragma unroll
    for (int i = 0; i < experts_per_thread; i++) {
        wt[i] = -INFINITY;
    }
#pragma unroll
    for (int i = 0; i < n_experts; i += WARP_SIZE) {
        const int expert  = i + lane;
        wt[i / WARP_SIZE] = (n_experts % WARP_SIZE == 0 || expert < n_experts) ? logits[expert] : -INFINITY;
    }

    if (single_wg) {
        sycl::group_barrier(item_ct1.get_group());
    }

    if (!config.delayed_softmax) {
        if (config.use_sigmoid) {
            sigmoid_warp_inplace<experts_per_thread, false>(wt, n_experts, lane);
        } else {
            softmax_warp_inplace<experts_per_thread, false>(wt, n_experts, lane);
        }
    }

    // Sanitize NaN to -FLT_MAX so the iterative argmax produces unique expert IDs. NaN comparisons
    // always return false, which would cause the same expert to be selected repeatedly.
#pragma unroll
    for (int i = 0; i < experts_per_thread; i++) {
        if (sycl::isnan(wt[i])) {
            wt[i] = -FLT_MAX;
        }
    }

    // each thread now holds either a portion of the softmax distribution or the raw logits. Do the
    // argmax reduce over n_expert_used, each time marking the selected expert as -inf to exclude it
    // from the next iteration.

    float wt_sum = 0.f;
    float output_weights[experts_per_thread];
#pragma unroll
    for (int i = 0; i < experts_per_thread; i++) {
        output_weights[i] = 0.f;
    }

    const sycl::sub_group sg = item_ct1.get_sub_group();

    for (int k = 0; k < n_expert_used; k++) {
        float max_val    = wt[0];
        int   max_expert = lane;
#pragma unroll
        for (int i = 1; i < experts_per_thread; i++) {
            const int expert = lane + i * WARP_SIZE;
            if ((n_experts % WARP_SIZE == 0 || expert < n_experts) && wt[i] > max_val) {
                max_val    = wt[i];
                max_expert = expert;
            }
        }
#pragma unroll
        for (int mask = WARP_SIZE / 2; mask > 0; mask >>= 1) {
            const float val    = dpct::permute_sub_group_by_xor(sg, max_val, mask);
            const int   expert = dpct::permute_sub_group_by_xor(sg, max_expert, mask);
            if (val > max_val || (val == max_val && expert < max_expert)) {
                max_val    = val;
                max_expert = expert;
            }
        }

        if ((max_expert & (WARP_SIZE - 1)) == lane) {
            wt[max_expert / WARP_SIZE] = -INFINITY;
        }
        if ((k & (WARP_SIZE - 1)) == lane) {
            output_weights[k / WARP_SIZE] = max_val;
        }
        if ((max_expert & (WARP_SIZE - 1)) == lane) {
            ids[k] = max_expert;
            if (config.with_norm) {
                wt_sum += max_val;
            }
        }
    }

    if (config.with_norm) {
        wt_sum          = warp_reduce_sum<WARP_SIZE>(wt_sum);
        wt_sum          = sycl::fmax(wt_sum, clamp_val);
        const float inv = 1.0f / wt_sum;
#pragma unroll
        for (int i = 0; i < experts_per_thread; i++) {
            output_weights[i] *= inv;
        }
    }

    if (config.delayed_softmax) {
        softmax_warp_inplace<experts_per_thread, true>(output_weights, n_expert_used, lane);
    }

#pragma unroll
    for (int i = 0; i < experts_per_thread; i++) {
        const int idx = i * WARP_SIZE + lane;
        if (idx < n_expert_used) {
            weights[idx] = output_weights[i] * scale_val;
        }
    }
}

template <int n_experts>
static void launch_topk_moe(queue_ptr stream, const float * logits, float * weights, int32_t * ids, int n_rows,
                            int n_expert_used, float clamp_val, float scale_val, const topk_moe_config & config) {
    // small batches (decode / MTP verify): one work-group, all rows read before any write (alias-safe)
    static const bool single_wg_enabled = ggml_sycl_get_env("GGML_SYCL_TOPK_SINGLE_WG", 1) != 0;
    if (single_wg_enabled && n_rows >= 1 && n_rows <= 8) {
        const sycl::range<1> wg((size_t) n_rows * WARP_SIZE);
        stream->parallel_for(sycl::nd_range<1>(wg, wg),
                             [=](sycl::nd_item<1> item_ct1) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                                 topk_moe_kernel<n_experts, true>(logits, weights, ids, n_rows, n_expert_used, clamp_val,
                                                                  scale_val, config);
                                 GGML_UNUSED(item_ct1);
                             });
        return;
    }
    const sycl::range<1> block_dims(WARP_SIZE);
    const sycl::range<1> block_nums(n_rows);
    stream->parallel_for(sycl::nd_range<1>(block_nums * block_dims, block_dims),
                         [=](sycl::nd_item<1> item_ct1) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                             topk_moe_kernel<n_experts>(logits, weights, ids, n_rows, n_expert_used, clamp_val,
                                                        scale_val, config);
                             GGML_UNUSED(item_ct1);
                         });
}

static void ggml_sycl_op_topk_moe(ggml_backend_sycl_context &     ctx,
                                  const ggml_tensor *             logits,
                                  ggml_tensor *                   weights,
                                  ggml_tensor *                   ids,
                                  const ggml_tensor *             clamp,
                                  const ggml_tensor *             scale,
                                  const ggml_sycl_topk_moe_args & args) {
    GGML_ASSERT(logits->type  == GGML_TYPE_F32);
    GGML_ASSERT(weights->type == GGML_TYPE_F32);
    GGML_ASSERT(ids->type     == GGML_TYPE_I32);

    const int n_experts     = logits->ne[0];
    const int n_rows        = logits->ne[1];
    const int n_expert_used = weights->ne[1];

    GGML_ASSERT(ids->nb[1] / ggml_type_size(ids->type) == (size_t) n_experts);

    const float * logits_d  = (const float *) logits->data;
    float *       weights_d = (float *) weights->data;
    int32_t *     ids_d     = (int32_t *) ids->data;

    const bool  with_norm = clamp != nullptr;
    const float clamp_val = clamp ? ggml_get_op_params_f32(clamp, 0) : -INFINITY;
    const float scale_val = scale ? ggml_get_op_params_f32(scale, 0) : 1.0f;

    topk_moe_config config;
    config.use_sigmoid     = args.sigmoid;
    config.with_norm       = with_norm;
    config.delayed_softmax = args.delayed_softmax;

    queue_ptr stream = ctx.stream();
    ggml_sycl_set_device(ctx.device);

    switch (n_experts) {
        case 1:
            launch_topk_moe<1>(stream, logits_d, weights_d, ids_d, n_rows, n_expert_used, clamp_val, scale_val,
                               config);
            break;
        case 2:
            launch_topk_moe<2>(stream, logits_d, weights_d, ids_d, n_rows, n_expert_used, clamp_val, scale_val,
                               config);
            break;
        case 4:
            launch_topk_moe<4>(stream, logits_d, weights_d, ids_d, n_rows, n_expert_used, clamp_val, scale_val,
                               config);
            break;
        case 8:
            launch_topk_moe<8>(stream, logits_d, weights_d, ids_d, n_rows, n_expert_used, clamp_val, scale_val,
                               config);
            break;
        case 16:
            launch_topk_moe<16>(stream, logits_d, weights_d, ids_d, n_rows, n_expert_used, clamp_val, scale_val,
                                config);
            break;
        case 32:
            launch_topk_moe<32>(stream, logits_d, weights_d, ids_d, n_rows, n_expert_used, clamp_val, scale_val,
                                config);
            break;
        case 64:
            launch_topk_moe<64>(stream, logits_d, weights_d, ids_d, n_rows, n_expert_used, clamp_val, scale_val,
                                config);
            break;
        case 128:
            launch_topk_moe<128>(stream, logits_d, weights_d, ids_d, n_rows, n_expert_used, clamp_val, scale_val,
                                 config);
            break;
        case 256:
            launch_topk_moe<256>(stream, logits_d, weights_d, ids_d, n_rows, n_expert_used, clamp_val, scale_val,
                                 config);
            break;
        case 512:
            launch_topk_moe<512>(stream, logits_d, weights_d, ids_d, n_rows, n_expert_used, clamp_val, scale_val,
                                 config);
            break;
        default:
            GGML_ASSERT(false && "fatal error");
            break;
    }
}

static bool ggml_sycl_should_use_topk_moe(const ggml_tensor * gating_op, const ggml_tensor * weights,
                                          const ggml_tensor * logits, const ggml_tensor * ids) {
    const int n_expert = ids->nb[1] / ids->nb[0];
    if ((n_expert & (n_expert - 1)) != 0 || n_expert > 512) {
        return false;
    }

    if (!ggml_is_contiguous(weights) || !ggml_is_contiguous(logits)) {
        return false;
    }

    if (gating_op->op == GGML_OP_SOFT_MAX) {
        float scale    = 1.0f;
        float max_bias = 0.0f;

        memcpy(&scale, (const float *) gating_op->op_params + 0, sizeof(float));
        memcpy(&max_bias, (const float *) gating_op->op_params + 1, sizeof(float));

        if (!ggml_is_contiguous(gating_op->src[0])) {
            return false;
        }
        if (scale != 1.0f || max_bias != 0.0f) {
            return false;
        }
        // don't fuse when masks or sinks are present
        if (gating_op->src[1] || gating_op->src[2]) {
            return false;
        }
    } else if (gating_op->op == GGML_OP_UNARY) {
        if (ggml_get_unary_op(gating_op) != GGML_UNARY_OP_SIGMOID) {
            return false;
        }
    }

    return true;
}

// ported from ggml_cuda_topk_moe_fusion - pure graph inspection, backend-agnostic
static bool ggml_sycl_topk_moe_fusion(const ggml_cgraph * cgraph, int node_idx, ggml_sycl_topk_moe_args & args) {
    args = ggml_sycl_topk_moe_args{};

    const int      n_nodes = cgraph->n_nodes;
    ggml_tensor ** nodes   = cgraph->nodes;

    if (nodes[node_idx]->op == GGML_OP_SOFT_MAX) {
        args.softmax = true;
    }

    if (nodes[node_idx]->op == GGML_OP_UNARY) {
        if (ggml_get_unary_op(nodes[node_idx]) != GGML_UNARY_OP_SIGMOID) {
            return false;
        }
        args.sigmoid = true;
    }

    if (nodes[node_idx]->op == GGML_OP_ARGSORT) {
        args.delayed_softmax = true;
    }

    node_idx++;

    if (args.sigmoid || args.softmax) {
        // SOFTMAX -> RESHAPE
        if (node_idx >= n_nodes || nodes[node_idx]->op != GGML_OP_RESHAPE ||
            nodes[node_idx]->src[0] != nodes[node_idx - 1]) {
            return false;
        }
        ggml_tensor * probs_reshaped = nodes[node_idx];
        node_idx++;

        if (node_idx >= n_nodes) {
            return false;
        }

        // src of bias add is the unreshaped probs (-2 instead of -1)
        if (nodes[node_idx]->op == GGML_OP_ADD && nodes[node_idx]->src[0] == nodes[node_idx - 2]) {
            args.prob_bias = true;
            node_idx++;
        }
        // RESHAPE/ADD -> ARGSORT
        if (node_idx >= n_nodes || nodes[node_idx]->op != GGML_OP_ARGSORT) {
            return false;
        }

        if (args.prob_bias && nodes[node_idx]->src[0] != nodes[node_idx - 1]) {
            return false;
        } else if (!args.prob_bias && nodes[node_idx]->src[0] != nodes[node_idx - 2]) {
            return false;
        }

        node_idx++;

        // ARGSORT -> VIEW
        if (node_idx >= n_nodes || nodes[node_idx]->op != GGML_OP_VIEW ||
            nodes[node_idx]->src[0] != nodes[node_idx - 1]) {
            return false;
        }
        node_idx++;

        if (node_idx >= n_nodes || nodes[node_idx]->op != GGML_OP_GET_ROWS) {
            return false;
        }

        // GET_ROWS
        if (nodes[node_idx]->src[0] != probs_reshaped || nodes[node_idx]->src[1] != nodes[node_idx - 1]) {
            return false;
        }
        node_idx++;
    } else if (args.delayed_softmax) {
        if (node_idx - 2 < 0) {
            return false;
        }
        ggml_tensor * probs_reshaped = nodes[node_idx - 2];

        // VIEW -> ARGSORT
        if (node_idx >= n_nodes || nodes[node_idx]->op != GGML_OP_VIEW ||
            nodes[node_idx]->src[0] != nodes[node_idx - 1]) {
            return false;
        }
        node_idx++;

        // GET_ROWS
        if (node_idx >= n_nodes || nodes[node_idx]->src[1] != nodes[node_idx - 1] ||
            nodes[node_idx]->src[0] != probs_reshaped) {
            return false;
        }
        node_idx++;

        static const std::vector<ggml_op> remaining_ops = { GGML_OP_RESHAPE, GGML_OP_SOFT_MAX, GGML_OP_RESHAPE };

        for (const ggml_op op : remaining_ops) {
            if (node_idx >= n_nodes || nodes[node_idx]->op != op || nodes[node_idx]->src[0] != nodes[node_idx - 1]) {
                return false;
            }
            node_idx++;
        }
    }

    // at this point we can check for norm + scale; everything is now at least valid up to the norm
    if (node_idx >= n_nodes) {
        return true;
    }

    if (nodes[node_idx]->op == GGML_OP_RESHAPE) {
        // check RESHAPE -> SUM_ROWS -> CLAMP -> DIV -> RESHAPE
        static const std::vector<ggml_op> norm_ops = { GGML_OP_RESHAPE, GGML_OP_SUM_ROWS, GGML_OP_CLAMP };

        args.norm = true;
        for (const ggml_op op : norm_ops) {
            if (nodes[node_idx]->op == op && nodes[node_idx]->src[0] == nodes[node_idx - 1]) {
                node_idx++;
            } else {
                args.norm = false;
                return true;
            }
        }

        // DIV <- CLAMP, RESHAPE
        if (nodes[node_idx]->op != GGML_OP_DIV || nodes[node_idx]->src[1] != nodes[node_idx - 1] ||
            nodes[node_idx]->src[0] != nodes[node_idx - 3]) {
            args.norm = false;
            return true;
        }
        node_idx++;

        if (nodes[node_idx]->op != GGML_OP_RESHAPE || nodes[node_idx]->src[0] != nodes[node_idx - 1]) {
            args.norm = false;
            return true;
        }
        node_idx++;
    }

    if (nodes[node_idx]->op == GGML_OP_SCALE && nodes[node_idx]->src[0] == nodes[node_idx - 1]) {
        args.scale = true;
    }

    return true;
}

// returns whether the write (out) nodes overwrite the read nodes in operation
// ported from ggml_cuda_check_fusion_memory_ranges - pure pointer/range inspection
static bool ggml_sycl_check_fusion_memory_ranges(const ggml_cgraph * cgraph, const int node_idx,
                                                 const int node_count, const int * out_nodes, const int out_count,
                                                 const bool is_topk_moe = false) {
    auto nodes_overlap = [&](const ggml_tensor * a, const ggml_tensor * b) {
        const int64_t a_start = (int64_t) a->data;
        const int64_t a_end   = a_start + ggml_backend_buft_get_alloc_size(a->buffer->buft, a);

        const int64_t b_start = (int64_t) b->data;
        const int64_t b_end   = b_start + ggml_backend_buft_get_alloc_size(b->buffer->buft, b);

        if ((b_start <= a_start && a_start < b_end) || (a_start <= b_start && b_start < a_end)) {
            return true;
        }

        return false;
    };

    bool is_ok = true;
    // exception for topk-moe, as each row is read entirely before writing
    if (ggml_nrows(cgraph->nodes[node_idx]) == 1 && is_topk_moe) {
        return true;
    }

    for (int i = 0; i < out_count; ++i) {
        const ggml_tensor * dst = cgraph->nodes[out_nodes[i]];

        for (int j = node_idx; j < node_idx + node_count; ++j) {
            // loop over all srcs of all nodes in the fusion. If the src overlaps the destination and
            // the src is not an intermediate node that's being elided, then disable fusion.
            for (int src_idx = 0; src_idx < GGML_MAX_SRC; ++src_idx) {
                const ggml_tensor * src = cgraph->nodes[j]->src[src_idx];

                if (!src || src->op == GGML_OP_NONE) {
                    continue;
                }

                if (nodes_overlap(dst, src)) {
                    bool found = false;

                    for (int k = node_idx; k < j; ++k) {
                        if (cgraph->nodes[k] == src) {
                            found = true;
                            break;
                        }
                    }

                    if (!found) {
                        is_ok = false;
                        break;
                    }
                }
            }
        }
    }

    return is_ok;
}

// ---------------------------------------------------------------------------------------------------------------
// DeepSeek-V4 routing chain in one launch (one sub-group per token row):
//   probs = sqrt(softplus(logits)); sel = probs + bias; ids = top-k(sel); w = probs[ids];
//   w = clamp(sum(w), lo, hi) normalization; w = w / sum * scale.
// Graph order (build_moe_ffn, gating SQRT_SOFTPLUS with exp_probs_b):
//   UNARY(SOFTPLUS) SQRT RESHAPE ADD(bias) ARGSORT VIEW(ids) GET_ROWS [RESHAPE SUM_ROWS CLAMP DIV RESHAPE] [SCALE]
// Same softplus formula as element_wise.cpp (max(x,0) + log1p(exp(-|x|))). GGML_SYCL_DISABLE_DSV4_ROUTE_FUSION=1 disables it,
// GGML_SYCL_DSV4_ROUTE_DEBUG=1 prints the node sequence the first few times the matcher looks at a softplus node.
// ---------------------------------------------------------------------------------------------------------------
template <int n_experts>
static void dsv4_moe_route_kernel(const float * __restrict__ logits, const float * __restrict__ bias,
                                  float * __restrict__ weights, int32_t * __restrict__ ids,
                                  const int n_rows, const int n_expert_used, const bool with_norm,
                                  const float clamp_lo, const float clamp_hi, const float scale_val, const int div_mode) {
    auto      item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<1>();
    const int row      = item_ct1.get_group(0);
    if (row >= n_rows) {
        return;
    }
    const int lane = item_ct1.get_local_id(0);

    logits  += (size_t) n_experts * row;
    weights += (size_t) n_expert_used * row;
    ids     += (size_t) n_experts * row;  // ids row stride is n_experts (the view of the argsort tensor)

    constexpr int experts_per_thread = (n_experts > WARP_SIZE) ? n_experts / WARP_SIZE : 1;

    float pr[experts_per_thread];   // unbiased probabilities (the output weights)
    float wt[experts_per_thread];   // biased scores (the selection key)
#pragma unroll
    for (int i = 0; i < experts_per_thread; i++) {
        const int expert = i * WARP_SIZE + lane;
        if (n_experts % WARP_SIZE == 0 || expert < n_experts) {
            const float x  = logits[expert];
            const float sp = sycl::fmax(x, 0.0f) + sycl::log1p(sycl::exp(-sycl::fabs(x)));
            const float p  = sycl::sqrt(sp);
            pr[i] = p;
            wt[i] = p + bias[expert];
        } else {
            pr[i] = 0.0f;
            wt[i] = -INFINITY;
        }
        if (sycl::isnan(wt[i])) {
            wt[i] = -FLT_MAX;
        }
    }

    const sycl::sub_group sg = item_ct1.get_sub_group();

    float output_weights[experts_per_thread];
#pragma unroll
    for (int i = 0; i < experts_per_thread; i++) {
        output_weights[i] = 0.0f;
    }
    float wsum = 0.0f;

    for (int k = 0; k < n_expert_used; k++) {
        float max_val    = wt[0];
        int   max_expert = lane;
#pragma unroll
        for (int i = 1; i < experts_per_thread; i++) {
            const int expert = lane + i * WARP_SIZE;
            if ((n_experts % WARP_SIZE == 0 || expert < n_experts) && wt[i] > max_val) {
                max_val    = wt[i];
                max_expert = expert;
            }
        }
#pragma unroll
        for (int mask = WARP_SIZE / 2; mask > 0; mask >>= 1) {
            const float val    = dpct::permute_sub_group_by_xor(sg, max_val, mask);
            const int   expert = dpct::permute_sub_group_by_xor(sg, max_expert, mask);
            if (val > max_val || (val == max_val && expert < max_expert)) {
                max_val    = val;
                max_expert = expert;
            }
        }

        const int owner = max_expert & (WARP_SIZE - 1);
        const int slot  = max_expert / WARP_SIZE;
        float     pv    = pr[slot];
        pv              = sycl::select_from_group(sg, pv, (uint32_t) owner);
        if (owner == lane) {
            wt[slot] = -INFINITY;
            ids[k]   = max_expert;
        }
        if ((k & (WARP_SIZE - 1)) == lane) {
            output_weights[k / WARP_SIZE] = pv;
        }
        wsum += pv;   // identical on every lane, summed in selection order
    }

    if (with_norm) {
        // sum exactly like k_sum_rows_f32: one value per lane (others 0), xor butterfly, lane 0's result
        float tot = wsum;
        if (n_expert_used <= WARP_SIZE) {
            float v = (lane < n_expert_used) ? output_weights[0] : 0.0f;
#pragma unroll
            for (int mask = WARP_SIZE / 2; mask > 0; mask >>= 1) {
                v += dpct::permute_sub_group_by_xor(sg, v, mask);
            }
            tot = sycl::select_from_group(sg, v, (uint32_t) 0);
        }
        const float s = sycl::fmin(sycl::fmax(tot, clamp_lo), clamp_hi);
#pragma unroll
        for (int i = 0; i < experts_per_thread; i++) {
            output_weights[i] = div_mode == 1 ? output_weights[i] * (1.0f / s)
                              : div_mode == 2 ? sycl::native::divide(output_weights[i], s)
                                              : output_weights[i] / s;
        }
    }
#pragma unroll
    for (int i = 0; i < experts_per_thread; i++) {
        const int idx = i * WARP_SIZE + lane;
        if (idx < n_expert_used) {
            weights[idx] = output_weights[i] * scale_val;
        }
    }
}

template <int n_experts>
static void launch_dsv4_moe_route(queue_ptr stream, const float * logits, const float * bias, float * weights,
                                  int32_t * ids, int n_rows, int n_expert_used, bool with_norm, float clamp_lo,
                                  float clamp_hi, float scale_val, int div_mode) {
    const sycl::range<1> block_dims(WARP_SIZE);
    const sycl::range<1> block_nums(n_rows);
    stream->parallel_for(sycl::nd_range<1>(block_nums * block_dims, block_dims),
                         [=](sycl::nd_item<1> item_ct1) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                             dsv4_moe_route_kernel<n_experts>(logits, bias, weights, ids, n_rows, n_expert_used,
                                                              with_norm, clamp_lo, clamp_hi, scale_val, div_mode);
                             GGML_UNUSED(item_ct1);
                         });
}

static int ggml_sycl_fuse_dsv4_moe_route(ggml_backend_sycl_context & ctx, ggml_cgraph * cgraph, int i) {
    static const bool disabled = ggml_sycl_get_env("GGML_SYCL_DISABLE_DSV4_ROUTE_FUSION", 0) != 0;
    static const bool debug    = ggml_sycl_get_env("GGML_SYCL_DSV4_ROUTE_DEBUG", 0) != 0;
    static std::atomic<int> debug_count{0};
    ggml_tensor ** nodes = cgraph->nodes;
    const int      n     = cgraph->n_nodes;
    if (disabled || nodes[i]->op != GGML_OP_UNARY || ggml_get_unary_op(nodes[i]) != GGML_UNARY_OP_SOFTPLUS) {
        return 0;
    }
    if (debug && debug_count.fetch_add(1) < 3) {
        fprintf(stderr, "[SYCL] dsv4-route: softplus at node %d, following ops:", i);
        for (int j = i; j < std::min(n, i + 16); ++j) {
            fprintf(stderr, " %s", ggml_op_name(nodes[j]->op));
        }
        fprintf(stderr, "\n");
    }
    auto fail = [&](int why) {
        if (debug && debug_count.load() < 12) {
            fprintf(stderr, "[SYCL] dsv4-route: no match at node %d (why=%d)\n", i, why);
        }
        return 0;
    };
    if (i + 12 >= n) { return fail(1); }
    ggml_tensor * sp = nodes[i];
    ggml_tensor * sq = nodes[i + 1];
    ggml_tensor * rs = nodes[i + 2];
    ggml_tensor * ad = nodes[i + 3];
    ggml_tensor * as = nodes[i + 4];
    ggml_tensor * vw = nodes[i + 5];
    ggml_tensor * gr = nodes[i + 6];
    if (sq->op != GGML_OP_SQRT || sq->src[0] != sp || rs->op != GGML_OP_RESHAPE || rs->src[0] != sq ||
        ad->op != GGML_OP_ADD || ad->src[0] != sq || as->op != GGML_OP_ARGSORT || as->src[0] != ad ||
        vw->op != GGML_OP_VIEW || vw->src[0] != as || gr->op != GGML_OP_GET_ROWS || gr->src[0] != rs ||
        gr->src[1] != vw) {
        return fail(2);
    }
    const ggml_tensor * logits = sp->src[0];
    const ggml_tensor * bias   = ad->src[1];
    const int64_t n_expert = logits->ne[0];
    if (logits->type != GGML_TYPE_F32 || !ggml_is_contiguous(logits) || sp->type != GGML_TYPE_F32 ||
        bias == nullptr || bias->type != GGML_TYPE_F32 || !ggml_is_contiguous(bias) ||
        ggml_nelements(bias) != n_expert || (n_expert & (n_expert - 1)) != 0 || n_expert > 512 || n_expert < 8 ||
        vw->type != GGML_TYPE_I32 || vw->nb[0] != sizeof(int32_t) || vw->nb[1] != (size_t) n_expert * sizeof(int32_t)) {
        return fail(3);
    }
    // optional normalization and scale
    int cnt = 7;
    bool with_norm = false;
    const ggml_tensor * clamp = nullptr;
    const ggml_tensor * scale = nullptr;
    if (nodes[i + 7]->op == GGML_OP_RESHAPE && nodes[i + 7]->src[0] == gr && nodes[i + 8]->op == GGML_OP_SUM_ROWS &&
        nodes[i + 8]->src[0] == nodes[i + 7] && nodes[i + 9]->op == GGML_OP_CLAMP && nodes[i + 9]->src[0] == nodes[i + 8] &&
        nodes[i + 10]->op == GGML_OP_DIV && nodes[i + 10]->src[0] == nodes[i + 7] && nodes[i + 10]->src[1] == nodes[i + 9] &&
        nodes[i + 11]->op == GGML_OP_RESHAPE && nodes[i + 11]->src[0] == nodes[i + 10]) {
        with_norm = true;
        clamp     = nodes[i + 9];
        cnt       = 12;
        if (i + 12 < n && nodes[i + 12]->op == GGML_OP_SCALE && nodes[i + 12]->src[0] == nodes[i + 11]) {
            scale = nodes[i + 12];
            cnt   = 13;
        }
    } else if (nodes[i + 7]->op == GGML_OP_SCALE && nodes[i + 7]->src[0] == gr) {
        scale = nodes[i + 7];
        cnt   = 8;
    }
    ggml_tensor * weights = nodes[i + cnt - 1];
    std::vector<ggml_op> ops = { GGML_OP_UNARY, GGML_OP_SQRT, GGML_OP_RESHAPE, GGML_OP_ADD, GGML_OP_ARGSORT, GGML_OP_VIEW, GGML_OP_GET_ROWS };
    if (with_norm) {
        ops.insert(ops.end(), { GGML_OP_RESHAPE, GGML_OP_SUM_ROWS, GGML_OP_CLAMP, GGML_OP_DIV, GGML_OP_RESHAPE });
    }
    if (scale) {
        ops.push_back(GGML_OP_SCALE);
    }
    GGML_ASSERT((int) ops.size() == cnt);
    int out_nodes[2] = { i + 5, i + cnt - 1 };
    if (weights->type != GGML_TYPE_F32 || !ggml_is_contiguous(weights) || weights->ne[0] != 1) {
        return fail(4);
    }
    if (!ggml_can_fuse_subgraph(cgraph, i, ops.size(), ops.data(), out_nodes, 2)) {
        return fail(5);
    }
    if (!ggml_sycl_check_fusion_memory_ranges(cgraph, i, (int) ops.size(), out_nodes, 2, /*is_topk_moe=*/true)) {
        return fail(6);
    }
    const int n_rows        = (int) logits->ne[1];
    const int n_expert_used = (int) weights->ne[1];
    const float clamp_lo = clamp ? ggml_get_op_params_f32(clamp, 0) : -INFINITY;
    const float clamp_hi = clamp ? ggml_get_op_params_f32(clamp, 1) : INFINITY;
    const float scale_val = scale ? ggml_get_op_params_f32(scale, 0) : 1.0f;
    if (scale && ggml_get_op_params_f32(scale, 1) != 0.0f) {
        return fail(7);   // a scale bias is not implemented
    }
    const float * logits_d  = (const float *) logits->data;
    const float * bias_d    = (const float *) bias->data;
    float *       weights_d = (float *) weights->data;
    int32_t *     ids_d     = (int32_t *) vw->data;
    static const int div_mode = ggml_sycl_get_env("GGML_SYCL_DSV4_ROUTE_DIV", 0);
    queue_ptr stream = ctx.stream();
    ggml_sycl_set_device(ctx.device);
    switch (n_expert) {
        case 8:   launch_dsv4_moe_route<8>(stream, logits_d, bias_d, weights_d, ids_d, n_rows, n_expert_used, with_norm, clamp_lo, clamp_hi, scale_val, div_mode); break;
        case 16:  launch_dsv4_moe_route<16>(stream, logits_d, bias_d, weights_d, ids_d, n_rows, n_expert_used, with_norm, clamp_lo, clamp_hi, scale_val, div_mode); break;
        case 32:  launch_dsv4_moe_route<32>(stream, logits_d, bias_d, weights_d, ids_d, n_rows, n_expert_used, with_norm, clamp_lo, clamp_hi, scale_val, div_mode); break;
        case 64:  launch_dsv4_moe_route<64>(stream, logits_d, bias_d, weights_d, ids_d, n_rows, n_expert_used, with_norm, clamp_lo, clamp_hi, scale_val, div_mode); break;
        case 128: launch_dsv4_moe_route<128>(stream, logits_d, bias_d, weights_d, ids_d, n_rows, n_expert_used, with_norm, clamp_lo, clamp_hi, scale_val, div_mode); break;
        case 256: launch_dsv4_moe_route<256>(stream, logits_d, bias_d, weights_d, ids_d, n_rows, n_expert_used, with_norm, clamp_lo, clamp_hi, scale_val, div_mode); break;
        case 512: launch_dsv4_moe_route<512>(stream, logits_d, bias_d, weights_d, ids_d, n_rows, n_expert_used, with_norm, clamp_lo, clamp_hi, scale_val, div_mode); break;
        default:  return fail(8);
    }
    return cnt - 1;
}

int ggml_sycl_fuse(ggml_backend_sycl_context & ctx, ggml_cgraph * cgraph, int i) {
    if (!g_ggml_sycl_enable_fusion) {
        return 0;
    }

    if (const int skipped = ggml_sycl_fuse_dsv4_moe_route(ctx, cgraph, i)) {
        return skipped;
    }
    return ggml_sycl_fuse_topk_moe(ctx, cgraph, i);
}

int ggml_sycl_fuse_topk_moe(ggml_backend_sycl_context & ctx, ggml_cgraph * cgraph, int i) {
    ggml_tensor * node = cgraph->nodes[i];

    if (node->op != GGML_OP_UNARY && node->op != GGML_OP_SOFT_MAX && node->op != GGML_OP_ARGSORT) {
        return 0;
    }

    ggml_sycl_topk_moe_args args;
    if (!ggml_sycl_topk_moe_fusion(cgraph, i, args)) {
        return 0;
    }

    // this kernel implements the no-bias path only; decline anything with a routing bias
    if (args.prob_bias) {
        return 0;
    }

    const ggml_tensor * logits  = node->src[0];
    ggml_tensor *       weights = nullptr;
    ggml_tensor *       ids     = nullptr;
    const ggml_tensor * clamp   = nullptr;
    const ggml_tensor * scale   = nullptr;

    std::vector<ggml_op> ops;
    int                  out_nodes[2];

    if (!args.delayed_softmax) {
        const ggml_op gating_op = args.sigmoid ? GGML_OP_UNARY : GGML_OP_SOFT_MAX;
        ops.insert(ops.end(), { gating_op, GGML_OP_RESHAPE, GGML_OP_ARGSORT, GGML_OP_VIEW, GGML_OP_GET_ROWS });
        out_nodes[0] = i + 3;
        ids          = cgraph->nodes[i + 3];

        if (args.norm) {
            ops.insert(ops.end(), { GGML_OP_RESHAPE, GGML_OP_SUM_ROWS, GGML_OP_CLAMP, GGML_OP_DIV, GGML_OP_RESHAPE });
            clamp = cgraph->nodes[i + (int) ops.size() - 3];
        }
        if (args.scale) {
            ops.insert(ops.end(), { GGML_OP_SCALE });
            scale = cgraph->nodes[i + (int) ops.size() - 1];
        }

        weights      = cgraph->nodes[i + (int) ops.size() - 1];
        out_nodes[1] = i + (int) ops.size() - 1;

        static const bool topk_dbg = ggml_sycl_get_env("GGML_SYCL_TOPK_DEBUG", 0) != 0;
        static std::atomic<int> topk_dbg_n{0};
        const bool ok_sub = ggml_can_fuse_subgraph(cgraph, i, ops.size(), ops.data(), out_nodes, 2);
        const bool ok_use = ok_sub && ggml_sycl_should_use_topk_moe(node, weights, logits, ids);
        bool       ok_mem = ok_use && ggml_sycl_check_fusion_memory_ranges(cgraph, i, (int) ops.size(), out_nodes, 2, /*is_topk_moe=*/true);
        // the single-work-group kernel reads every row before writing any, so aliased outputs are safe for small batches
        static const bool single_wg_enabled = ggml_sycl_get_env("GGML_SYCL_TOPK_SINGLE_WG", 1) != 0;
        const bool small_batch = single_wg_enabled && logits->ne[1] >= 1 && logits->ne[1] <= 8;
        if (topk_dbg && topk_dbg_n.fetch_add(1) < 24) {
            fprintf(stderr, "[TOPK] node=%s rows=%lld norm=%d scale=%d can_fuse_subgraph=%d should_use=%d mem_ranges=%d small_batch=%d\n",
                    node->name, (long long) logits->ne[1], (int) args.norm, (int) args.scale, (int) ok_sub, (int) ok_use,
                    (int) ok_mem, (int) small_batch);
        }
        if (ok_use && !ok_mem && small_batch) {
            ok_mem = true;
        }
        if (ok_use && ok_mem) {
            ggml_sycl_op_topk_moe(ctx, logits, weights, ids, clamp, scale, args);
            return (int) ops.size() - 1;
        }
    } else if (!args.norm && !args.prob_bias) {
        // gpt-oss style: argsort -> view -> get_rows -> reshape -> softmax -> reshape, no norm/bias
        ops.insert(ops.end(),
                   { GGML_OP_ARGSORT, GGML_OP_VIEW, GGML_OP_GET_ROWS, GGML_OP_RESHAPE, GGML_OP_SOFT_MAX,
                     GGML_OP_RESHAPE });
        weights                     = cgraph->nodes[i + 5];
        ids                         = cgraph->nodes[i + 1];
        const ggml_tensor * softmax = cgraph->nodes[i + 4];
        out_nodes[0]                = i + 1;
        out_nodes[1]                = i + 5;

        if (ggml_can_fuse_subgraph(cgraph, i, ops.size(), ops.data(), out_nodes, 2) &&
            ggml_sycl_should_use_topk_moe(softmax, weights, logits, ids) &&
            ggml_sycl_check_fusion_memory_ranges(cgraph, i, (int) ops.size(), out_nodes, 2, /*is_topk_moe=*/true)) {
            ggml_sycl_op_topk_moe(ctx, logits, weights, ids, clamp, scale, args);
            return (int) ops.size() - 1;
        }
    }

    return 0;
}
