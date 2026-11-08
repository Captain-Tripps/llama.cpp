//
// MIT license
// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: MIT
//

//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//

#include <algorithm>
#include <array>
#include <assert.h>
#include <atomic>
#include <cctype>
#include <map>
#include <string>
#include <cinttypes>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <cstdlib>
#include <float.h>
#include <limits>
#include <optional>
#include <stdint.h>
#include <stdio.h>
#include <vector>
#include <cmath>
#include <iostream>
#include <fstream>
#include <stdio.h>
#include <stdlib.h>
#include <regex>
#include <csignal>
#include <exception>
#include <stdexcept>
#include <system_error>
#if defined(_WIN32)
#    ifndef WIN32_LEAN_AND_MEAN
#        define WIN32_LEAN_AND_MEAN
#    endif
#    ifndef NOMINMAX
#        define NOMINMAX
#    endif
#    include <windows.h>
#endif

#include <sycl/sycl.hpp>
#include <sycl/backend.hpp>
#ifdef GGML_SYCL_SUPPORT_LEVEL_ZERO_API
#include <level_zero/ze_api.h>
#include <level_zero/zes_api.h>
#endif
#if defined(GGML_SYCL_GRAPH) && SYCL_EXT_ONEAPI_ASYNC_MEMORY_ALLOC
#    include <sycl/ext/oneapi/experimental/async_alloc/async_alloc.hpp>
#endif
#if SYCL_EXT_ONEAPI_VIRTUAL_MEM
#    include <sycl/ext/oneapi/virtual_mem/physical_mem.hpp>
#    include <sycl/ext/oneapi/virtual_mem/virtual_mem.hpp>
#    define GGML_SYCL_SUPPORT_VMM
#endif
#include <sycl/half_type.hpp>
#include <sycl/ext/oneapi/matrix/matrix.hpp>

namespace sycl_mx = sycl::ext::oneapi::experimental::matrix;

#include "ggml.h"
#include "ggml-sycl.h"
#include "ggml-impl.h"
#include "ggml-backend-impl.h"

#include "ggml-sycl/add-id.hpp"
#include "ggml-sycl/backend.hpp"
#include "ggml-sycl/common.hpp"
#include "ggml-sycl/element_wise.hpp"
#include "ggml-sycl/fwht.hpp"
#include "ggml-sycl/fused-gemm.hpp"
#include "ggml-sycl/gemm.hpp"
#include "ggml-sycl/getrows.hpp"
#include "ggml-sycl/mem.hpp"
#include "ggml-sycl/norm.hpp"
#include "ggml-sycl/presets.hpp"
#include "ggml-sycl/quantize.hpp"
#include "ggml-sycl/repeat_back.hpp"
#include "ggml-sycl/set_rows.hpp"
#include "ggml-sycl/set.hpp"
#include "ggml-sycl/dsv4-hc.hpp"
#include "ggml-sycl/lightning-indexer.hpp"
#include "ggml-sycl/conv2d.hpp"
#include "ggml-sycl/conv2d-dw.hpp"
#include "ggml-sycl/conv2d-transpose.hpp"
#include "ggml-sycl/ssm_conv.hpp"
#include "ggml-sycl/sycl_hw.hpp"
#include "ggml-sycl/ssm_scan.hpp"
#include "ggml-sycl/fill.hpp"
#include "ggml-sycl/cumsum.hpp"
#include "ggml-sycl/diag.hpp"
#include "ggml-sycl/opt-step.hpp"
#include "ggml-sycl/solve_tri.hpp"
#include "ggml-sycl/gated_delta_net.hpp"
#include "ggml-sycl/pool.hpp"
#include "ggml-sycl/cross_entropy_loss.hpp"

#define MEM_SIZE_2M	0x00200000
#define MEM_SIZE_1G	0x40000000


// ---- cross-card pipeline (GGML_SYCL_PIPELINE=1, default off) ------------------------------------------------------
// With a layer split only one card works at a time, because the host thread runs each card's part of a prompt batch
// to completion before it can prepare the next batch. With the switch on, every card gets a worker thread and a FIFO
// that acts as that card's command stream: a prompt-batch graph is copied and queued instead of executed, so the
// scheduler can go on to the next batch while the cards work. Everything else that touches a card's memory is either
// queued behind the pending work (writes) or waits for it (reads), which keeps the order the scheduler relies on.
// Decode-sized graphs are never queued: they run inline as before (after the FIFO has drained).
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <memory>
#include <mutex>
#include <vector>
#include <deque>
#include <functional>
#include <thread>
#include <unordered_map>

namespace sycl_pipe {

struct worker {
    std::thread                        th;
    std::mutex                         m;
    std::condition_variable            cv;
    std::condition_variable            cv_idle;
    std::deque<std::function<void()>>  q;
    bool                               busy    = false;
    bool                               started = false;
    int                                graphs  = 0;   // batch graphs queued or running (guarded by m)
};

static worker            g_workers[GGML_SYCL_MAX_DEVICES];
static std::mutex        g_start_mutex;
static std::atomic<bool> g_started{false};
static std::atomic<int>  g_registered{0};
static thread_local int  t_worker_device = -1;   // device whose FIFO this thread serves, -1 on other threads

// the submission lock (see common.hpp)
static std::mutex        g_submit_mutex;
static thread_local int  t_submit_depth = 0;
static thread_local bool t_submit_held  = false;
// diagnostics (GGML_SYCL_PIPELINE_TRACE): time spent waiting for the lock / running without it, and number of releases
static thread_local int64_t t_lock_wait_us = 0;
static thread_local int64_t t_unlocked_us  = 0;
static thread_local int64_t t_unlock_at    = 0;
static thread_local int64_t t_releases     = 0;

// held by every backend entry point and every queued job while it issues work
struct submit_scope {
    submit_scope() {
        if (t_submit_depth++ == 0 && g_started.load(std::memory_order_acquire)) {
            const int64_t w0 = ggml_time_us();
            g_submit_mutex.lock();
            t_lock_wait_us += ggml_time_us() - w0;
            t_submit_held = true;
        }
    }
    ~submit_scope() {
        if (--t_submit_depth == 0 && t_submit_held) {
            t_submit_held = false;
            g_submit_mutex.unlock();
        }
    }
    submit_scope(const submit_scope &) = delete;
    submit_scope & operator=(const submit_scope &) = delete;
};

static bool enabled() {
    static const bool v = ggml_sycl_get_env("GGML_SYCL_PIPELINE", 0) != 0;
    return v;
}

static int min_tokens() {
    static const int v = std::max(2, ggml_sycl_get_env("GGML_SYCL_PIPELINE_MIN_TOKENS", 64));
    return v;
}

// where each device's graph loop is (node index, op); read by the stall report below
static std::atomic<int> g_cur_node[GGML_SYCL_MAX_DEVICES];
static std::atomic<int> g_cur_op[GGML_SYCL_MAX_DEVICES];

static bool trace() {
    static const bool v = ggml_sycl_get_env("GGML_SYCL_PIPELINE_TRACE", 0) != 0;
    return v;
}

// ---- crash reporter ---------------------------------------------------------------------------------------------
static void crash_report(const char * why) {
    fprintf(stderr, "[SYCL-CRASH] %s on thread of device %d (-1 = not a pipeline worker)\n", why, t_worker_device);
    if (std::exception_ptr e = std::current_exception()) {
        try {
            std::rethrow_exception(e);
        } catch (const sycl::exception & ex) {
            fprintf(stderr, "[SYCL-CRASH] sycl::exception: %s (code %d)\n", ex.what(), (int) ex.code().value());
        } catch (const std::system_error & ex) {
            fprintf(stderr, "[SYCL-CRASH] std::system_error: %s (code %d)\n", ex.what(), (int) ex.code().value());
        } catch (const std::exception & ex) {
            fprintf(stderr, "[SYCL-CRASH] std::exception: %s\n", ex.what());
        } catch (...) {
            fprintf(stderr, "[SYCL-CRASH] unknown exception type\n");
        }
    } else {
        fprintf(stderr, "[SYCL-CRASH] no C++ exception in flight\n");
    }
    for (int i = 0; i < ggml_sycl_info().device_count; ++i) {
        fprintf(stderr, "[SYCL-CRASH] dev%d: node %d %s\n", i, g_cur_node[i].load(), ggml_op_name((ggml_op) g_cur_op[i].load()));
    }
#if defined(_WIN32)
    void * frames[62];
    const USHORT n = CaptureStackBackTrace(0, 62, frames, nullptr);
    for (USHORT i = 0; i < n; ++i) {
        HMODULE mod = nullptr;
        char    path[MAX_PATH] = "?";
        if (GetModuleHandleExA(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS | GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT, (LPCSTR) frames[i], &mod) && mod) {
            GetModuleFileNameA(mod, path, MAX_PATH);
        }
        const char * base = strrchr(path, (char) 92);
        fprintf(stderr, "[SYCL-CRASH]   #%02d %s+0x%llx\n", (int) i, base ? base + 1 : path,
                (unsigned long long) ((const char *) frames[i] - (const char *) mod));
    }
#endif
    fflush(stderr);
}

static void print_stack(const char * tag, int max_frames) {
#if defined(_WIN32)
    void * frames[62];
    const USHORT n = CaptureStackBackTrace(1, (ULONG) std::min(max_frames, 62), frames, nullptr);
    for (USHORT i = 0; i < n; ++i) {
        HMODULE mod = nullptr;
        char    path[MAX_PATH] = "?";
        if (GetModuleHandleExA(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS | GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT, (LPCSTR) frames[i], &mod) && mod) {
            GetModuleFileNameA(mod, path, MAX_PATH);
        }
        const char * base = strrchr(path, (char) 92);
        fprintf(stderr, "%s   #%02d %s+0x%llx\n", tag, (int) i, base ? base + 1 : path, (unsigned long long) ((const char *) frames[i] - (const char *) mod));
    }
#else
    (void) tag; (void) max_frames;
#endif
}

static void crash_on_abort(int) {
    signal(SIGABRT, SIG_DFL);
    crash_report("abort()");
}

static void crash_on_terminate() {
    crash_report("std::terminate");
    signal(SIGABRT, SIG_DFL);
    std::abort();
}

static void crash_install() {
    signal(SIGABRT, crash_on_abort);
    std::set_terminate(crash_on_terminate);   // per thread on this runtime: called on every pipeline thread
}

static void worker_main(int device) {
    t_worker_device = device;
    crash_install();
    {
        // the per-thread device table is a plain map: register this thread while nothing else runs
        std::lock_guard<std::mutex> lock(g_start_mutex);
        dpct::select_device(device);
    }
    g_registered.fetch_add(1);
    // (jobs take the submission lock themselves)
    worker & w = g_workers[device];
    for (;;) {
        std::function<void()> job;
        {
            std::unique_lock<std::mutex> lock(w.m);
            w.cv.wait(lock, [&] { return !w.q.empty(); });
            job = std::move(w.q.front());
            w.q.pop_front();
            w.busy = true;
        }
        try {
            job();
        } catch (...) {
            crash_report("exception escaped a queued job");
            signal(SIGABRT, SIG_DFL);
            std::abort();
        }
        {
            std::lock_guard<std::mutex> lock(w.m);
            w.busy = false;
            w.cv_idle.notify_all();
        }
    }
}

// start one worker per device, once; returns after every worker has registered itself
static void ensure_started() {
    if (g_started.load(std::memory_order_acquire)) {
        return;
    }
    const int n = ggml_sycl_info().device_count;
    crash_install();
    {
        std::lock_guard<std::mutex> lock(g_start_mutex);
        if (g_started.load(std::memory_order_relaxed)) {
            return;
        }
        for (int i = 0; i < n; ++i) {
            g_workers[i].started = true;
            g_workers[i].th = std::thread(worker_main, i);
            g_workers[i].th.detach();
        }
    }
    // wait until every worker has registered its device for its thread
    while (g_registered.load() < n) {
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    g_started.store(true, std::memory_order_release);
    GGML_LOG_INFO("%s: cross-card pipeline enabled: %d worker threads, batch graphs of >= %d tokens are queued\n", __func__, n, min_tokens());
}

static bool active(int device) {
    return g_started.load(std::memory_order_acquire) && device >= 0 && device < GGML_SYCL_MAX_DEVICES && g_workers[device].started;
}

// work queued or running on this device's FIFO (always false on the FIFO's own thread: its jobs run in order already)
static bool pending(int device) {
    if (!active(device) || t_worker_device == device) {
        return false;
    }
    worker & w = g_workers[device];
    std::lock_guard<std::mutex> lock(w.m);
    return w.busy || !w.q.empty();
}

static void drain(int device) {
    if (!active(device) || t_worker_device == device) {
        return;
    }
    worker & w = g_workers[device];
    if (trace() && t_worker_device < 0 && pending(device)) {
        fprintf(stderr, "[SYCL-PIPE] host waits for device %d t=%lld\n", device, (long long) ggml_time_us());
        static std::atomic<int> n_stacks{0};
        static const bool stacks = ggml_sycl_get_env("GGML_SYCL_PIPELINE_TRACE", 0) >= 2;
        if (stacks && n_stacks.fetch_add(1) < 24) {
            print_stack("[SYCL-PIPE-STACK]", 14);
        }
    }
    ggml_sycl_pipe_wait_scope ws_;
    std::unique_lock<std::mutex> lock(w.m);
    int    last_sig   = -1;
    size_t last_queue = (size_t) -1;
    while (!w.cv_idle.wait_for(lock, std::chrono::seconds(20), [&] { return !w.busy && w.q.empty(); })) {
        // stall report: only when nothing moved in 20 s (a long queue that is being worked off is not a stall)
        int sig = 0;
        for (int i = 0; i < ggml_sycl_info().device_count; ++i) {
            sig = sig * 31 + g_cur_node[i].load() * 7 + g_cur_op[i].load();
        }
        const bool moved = sig != last_sig || w.q.size() != last_queue;
        last_sig   = sig;
        last_queue = w.q.size();
        if (moved) {
            continue;
        }
        fprintf(stderr, "[SYCL-PIPE] still waiting for device %d (queued=%zu busy=%d);", device, w.q.size(), (int) w.busy);
        for (int i = 0; i < ggml_sycl_info().device_count; ++i) {
            fprintf(stderr, " dev%d: node %d %s;", i, g_cur_node[i].load(), ggml_op_name((ggml_op) g_cur_op[i].load()));
        }
        fprintf(stderr, "\n");
    }
}

static void enqueue(int device, std::function<void()> job, const char * label = "job") {
    worker & w = g_workers[device];
    if (trace()) {
        const int64_t tq = ggml_time_us();
        job = [job, label, device, tq] {
            const int64_t t0 = ggml_time_us();
            job();
            const int64_t t1 = ggml_time_us();
            if (t1 - t0 > 30000) {
                fprintf(stderr, "[SYCL-PIPE] slow job device=%d label=%s queued=%lld start=%lld end=%lld\n", device, label,
                        (long long) tq, (long long) t0, (long long) t1);
            }
        };
    }
    {
        std::lock_guard<std::mutex> lock(w.m);
        w.q.push_back(std::move(job));
    }
    w.cv.notify_one();
}

// a private copy of one split graph: the scheduler reuses the memory of the tensor structs for the next graph
struct graph_copy {
    std::deque<ggml_tensor>                                tensors;
    std::unordered_map<const ggml_tensor *, ggml_tensor *> map;
    std::vector<ggml_tensor *>                             nodes;
    std::vector<int32_t>                                   use_counts;
    std::vector<ggml_bitset_t>                             used;
    std::vector<ggml_tensor *>                             keys;
    ggml_cgraph                                            g;

    ggml_tensor * clone(const ggml_tensor * t) {
        if (t == nullptr) {
            return nullptr;
        }
        auto it = map.find(t);
        if (it != map.end()) {
            return it->second;
        }
        tensors.push_back(*t);
        ggml_tensor * n = &tensors.back();
        map.emplace(t, n);
        for (int j = 0; j < GGML_MAX_SRC; ++j) {
            n->src[j] = clone(t->src[j]);
        }
        n->view_src = clone(t->view_src);
        return n;
    }

    explicit graph_copy(const ggml_cgraph * cgraph) {
        nodes.reserve(cgraph->n_nodes);
        for (int i = 0; i < cgraph->n_nodes; ++i) {
            nodes.push_back(clone(cgraph->nodes[i]));
        }
        g           = *cgraph;
        g.size      = cgraph->n_nodes;
        g.n_leafs   = 0;
        g.nodes     = nodes.data();
        g.grads     = nullptr;
        g.grad_accs = nullptr;
        g.leafs     = nullptr;

        // use counts are looked up through the visited set by tensor address: rebuild it for the copies
        const size_t hsize = map.size() * 2 + 17;
        used.assign(ggml_bitset_size(hsize), 0);
        keys.assign(hsize, nullptr);
        use_counts.assign(hsize, 0);
        g.visited_hash_set.size = hsize;
        g.visited_hash_set.used = used.data();
        g.visited_hash_set.keys = keys.data();
        g.use_counts            = use_counts.data();
        // every copied tensor that the original set knows (nodes of this split and the tensors they read): the
        // fusion checks ask for the use count of sources too, and a missing entry reads as "unused"
        if (cgraph->use_counts != nullptr) {
            for (const auto & kv : map) {
                const size_t opos = ggml_hash_find(&cgraph->visited_hash_set, kv.first);
                if (opos == GGML_HASHSET_FULL || !ggml_bitset_get(cgraph->visited_hash_set.used, opos)) {
                    continue;
                }
                const size_t pos = ggml_hash_find_or_insert(&g.visited_hash_set, kv.second);
                use_counts[pos]  = cgraph->use_counts[opos];
            }
        }
    }
};

// a prompt batch: the first matrix product of the split has at least min_tokens() columns
static bool is_batch_graph(const ggml_cgraph * cgraph) {
    const ggml_tensor * first_mm = nullptr;
    for (int i = 0; i < cgraph->n_nodes; ++i) {
        const ggml_tensor * node = cgraph->nodes[i];
        if (node->op == GGML_OP_MUL_MAT_ID && node->src[2] != nullptr) {
            // ids are [n_expert_used, n_tokens]
            return node->src[2]->ne[1] >= min_tokens();
        }
        if (node->op == GGML_OP_MUL_MAT && first_mm == nullptr) {
            first_mm = node;
        }
    }
    return first_mm != nullptr && first_mm->ne[1] >= min_tokens() && first_mm->ne[2] == 1 && first_mm->ne[3] == 1;
}

// events of the scheduler: complete when the FIFO has reached the point where they were recorded
struct event_state {
    std::mutex              m;
    std::condition_variable cv;
    bool                    done = false;
};
static std::mutex                                                          g_event_mutex;
static std::unordered_map<ggml_backend_event_t, std::shared_ptr<event_state>> g_events;

static void event_record(int device, ggml_backend_event_t event) {
    if (!active(device)) {
        return;
    }
    auto st = std::make_shared<event_state>();
    {
        std::lock_guard<std::mutex> lock(g_event_mutex);
        g_events[event] = st;
    }
    if (pending(device)) {
        enqueue(device, [st] {
            { std::lock_guard<std::mutex> lock(st->m); st->done = true; }
            st->cv.notify_all();
        }, "event");
    } else {
        st->done = true;
    }
}

static void event_wait(ggml_backend_event_t event) {
    // GGML_SYCL_PIPELINE_EVENT_WAIT=1 restores the blocking wait (stage 1 behaviour)
    static const bool blocking = ggml_sycl_get_env("GGML_SYCL_PIPELINE_EVENT_WAIT", 0) != 0;
    if (!blocking) {
        return;
    }
    std::shared_ptr<event_state> st;
    {
        std::lock_guard<std::mutex> lock(g_event_mutex);
        auto it = g_events.find(event);
        if (it == g_events.end()) {
            return;
        }
        st = it->second;
    }
    ggml_sycl_pipe_wait_scope ws_;
    std::unique_lock<std::mutex> lock(st->m);
    st->cv.wait(lock, [&] { return st->done; });
}

static void event_forget(ggml_backend_event_t event) {
    std::lock_guard<std::mutex> lock(g_event_mutex);
    g_events.erase(event);
}

// Bounded queues: wait (without the submission lock) while this card already has max_queued() batch graphs unfinished,
// then count the one about to be queued.
static int max_queued() {
    static const int v = std::max(1, ggml_sycl_get_env("GGML_SYCL_PIPELINE_MAX_QUEUED", 2));
    return v;
}

static void graph_admit(int device) {
    worker & w = g_workers[device];
    ggml_sycl_pipe_wait_scope ws_;
    std::unique_lock<std::mutex> lock(w.m);
    const int limit = max_queued();
    if (w.graphs >= limit && trace()) {
        fprintf(stderr, "[SYCL-PIPE] host holds back: device %d has %d batch graphs unfinished t=%lld\n", device, w.graphs, (long long) ggml_time_us());
    }
    w.cv_idle.wait(lock, [&] { return w.graphs < limit; });
    w.graphs++;
}

static void graph_finished(int device) {
    worker & w = g_workers[device];
    std::lock_guard<std::mutex> lock(w.m);
    w.graphs--;
}

// Host copies of writes that are still queued, by target address. A later card-to-card copy of the same tensor can take
// the bytes from here: they are what the tensor will hold once the write has run. Entries of a card are dropped when a
// graph is handed to that card (the graph may reuse the memory for something else).
struct held_write {
    std::shared_ptr<std::vector<char>> data;
    int                                device;
};
static std::mutex                                     g_held_mutex;
static std::unordered_map<const char *, held_write>   g_held;

static bool held_enabled() {
    static const bool v = ggml_sycl_get_env("GGML_SYCL_PIPELINE_HELD", 1) != 0;
    return v;
}

static void held_add(int device, const char * base, size_t offset, const std::shared_ptr<std::vector<char>> & data) {
    std::lock_guard<std::mutex> lock(g_held_mutex);
    if (offset != 0 || !held_enabled()) {
        g_held.erase(base);   // a partial write: the earlier full copy no longer describes the tensor
        return;
    }
    g_held[base] = { data, device };
}

static void held_done(const char * target, const std::shared_ptr<std::vector<char>> & data) {
    std::lock_guard<std::mutex> lock(g_held_mutex);
    auto it = g_held.find(target);
    if (it != g_held.end() && it->second.data == data) {
        g_held.erase(it);
    }
}

static void held_forget_device(int device) {
    std::lock_guard<std::mutex> lock(g_held_mutex);
    for (auto it = g_held.begin(); it != g_held.end();) {
        it = it->second.device == device ? g_held.erase(it) : std::next(it);
    }
}

static void held_report(const char * what, const ggml_tensor * src, int dev_src, int dev_dst) {
    if (!trace()) {
        return;
    }
    size_t held_size = 0;
    int    held_dev  = -1;
    {
        std::lock_guard<std::mutex> lock(g_held_mutex);
        auto it = g_held.find((const char *) src->data);
        if (it != g_held.end()) {
            held_size = it->second.data->size();
            held_dev  = it->second.device;
        }
    }
    fprintf(stderr, "[SYCL-PIPE] copy %s '%s' op=%s view=%d dev%d->dev%d nbytes=%zu held_size=%zu held_dev=%d src_pending=%d" "\n", what, src->name,
            ggml_op_name(src->op), (int) (src->view_src != nullptr), dev_src, dev_dst, ggml_nbytes(src), held_size, held_dev, (int) pending(dev_src));
}

static std::shared_ptr<std::vector<char>> held_find(int device, const char * src, size_t nbytes) {
    std::lock_guard<std::mutex> lock(g_held_mutex);
    auto it = g_held.find(src);
    if (it == g_held.end() || it->second.device != device || it->second.data->size() != nbytes) {
        return nullptr;
    }
    return it->second.data;
}

// Copy device memory from one card to another without blocking the caller. The read is queued on the source card
// (behind the work that produces the data), the write on the destination card (ahead of the graph that consumes it);
// the write waits for the read through `st`. The bytes travel through host memory, as the blocking copy does.
struct copy_state {
    std::mutex              m;
    std::condition_variable cv;
    bool                    ready = false;
    std::vector<char>       data;
};

static void copy_async(int dev_src, const char * src, int dev_dst, char * dst, size_t nbytes) {
    if (auto held = held_find(dev_src, src, nbytes)) {
        // the source is a host write that card dev_src has not run yet: no need to wait for that card
        auto write_held = [held, dev_dst, dst] {
            submit_scope ss_;
            auto q = &(dpct::dev_mgr::instance().get_device(dev_dst).default_queue());
            { ggml_sycl_pipe_wait_scope ws_; SYCL_CHECK(CHECK_TRY_ERROR(q->wait())); }
            sycl::event ev;
            SYCL_CHECK(CHECK_TRY_ERROR(ev = q->memcpy(dst, held->data(), held->size())));
            { ggml_sycl_pipe_wait_scope ws_; SYCL_CHECK(CHECK_TRY_ERROR(ev.wait())); }
        };
        if (pending(dev_dst)) {
            enqueue(dev_dst, write_held, "cpy-held");
        } else {
            write_held();
        }
        return;
    }
    auto st = std::make_shared<copy_state>();
    auto read = [st, dev_src, src, nbytes] {
        {
            submit_scope ss_;
            auto q = &(dpct::dev_mgr::instance().get_device(dev_src).default_queue());
            st->data.resize(nbytes);
            { ggml_sycl_pipe_wait_scope ws_; SYCL_CHECK(CHECK_TRY_ERROR(q->wait())); }
            sycl::event ev;
            SYCL_CHECK(CHECK_TRY_ERROR(ev = q->memcpy(st->data.data(), src, nbytes)));
            { ggml_sycl_pipe_wait_scope ws_; SYCL_CHECK(CHECK_TRY_ERROR(ev.wait())); }
        }
        { std::lock_guard<std::mutex> lock(st->m); st->ready = true; }
        st->cv.notify_all();
    };
    auto write = [st, dev_dst, dst, nbytes] {
        {
            std::unique_lock<std::mutex> lock(st->m);
            st->cv.wait(lock, [&] { return st->ready; });
        }
        submit_scope ss_;
        auto q = &(dpct::dev_mgr::instance().get_device(dev_dst).default_queue());
        { ggml_sycl_pipe_wait_scope ws_; SYCL_CHECK(CHECK_TRY_ERROR(q->wait())); }
        sycl::event ev;
        SYCL_CHECK(CHECK_TRY_ERROR(ev = q->memcpy(dst, st->data.data(), nbytes)));
        { ggml_sycl_pipe_wait_scope ws_; SYCL_CHECK(CHECK_TRY_ERROR(ev.wait())); }
    };

    const bool src_busy = pending(dev_src);
    if (src_busy) {
        enqueue(dev_src, read, "cpy-read");
    } else {
        read();   // the caller already holds the submission lock
    }
    if (src_busy || pending(dev_dst)) {
        enqueue(dev_dst, write, "cpy-write");
    } else {
        write();
    }
}

} // namespace sycl_pipe

bool ggml_sycl_pipe_unlock() {
    if (!sycl_pipe::t_submit_held) {
        return false;
    }
    sycl_pipe::t_submit_held = false;
    sycl_pipe::t_unlock_at   = ggml_time_us();
    sycl_pipe::t_releases++;
    sycl_pipe::g_submit_mutex.unlock();
    return true;
}

void ggml_sycl_pipe_relock() {
    const int64_t w0 = ggml_time_us();
    sycl_pipe::t_unlocked_us += w0 - sycl_pipe::t_unlock_at;
    sycl_pipe::g_submit_mutex.lock();
    sycl_pipe::t_lock_wait_us += ggml_time_us() - w0;
    sycl_pipe::t_submit_held = true;
}

static bool g_sycl_loaded = false;
int g_ggml_sycl_debug = 0;
int g_ggml_sycl_dev_debug = 0;
int g_ggml_sycl_enable_optimize = 1;
int g_ggml_sycl_enable_graph = 0;
int g_ggml_sycl_enable_dnn = 1;
int g_ggml_sycl_fa_onednn = 1;
int g_ggml_sycl_fa_onednn_max_kv = 0;
int g_ggml_sycl_enable_mkl_fa = 1;
int g_ggml_sycl_memtrace = 0;
int g_ggml_sycl_memtrace_step = 64;
int g_ggml_sycl_enable_vmm = 1;
int g_ggml_sycl_enable_fusion = 1;
int g_ggml_sycl_enable_esimd = 1;
int g_ggml_sycl_mmvq_wide = 1;
int g_ggml_sycl_enable_xmx_esimd = 1;
int g_ggml_sycl_xmx_max_cols = GGML_SYCL_XMX_MAX_COLS_DEFAULT;
int g_ggml_sycl_xmx_glu_max_cols = GGML_SYCL_XMX_GLU_MAX_COLS;
int g_ggml_sycl_prioritize_dmmv = 0;
int g_ggml_sycl_xmx_gather_types = GGML_SYCL_XMX_GATHER_TYPES_DEFAULT;
int g_ggml_sycl_xmx_gather_shapes = GGML_SYCL_XMX_GATHER_SHAPES_DEFAULT;
int g_ggml_sycl_dynamic_precision = GGML_SYCL_DYNAMIC_PRECISION_DEFAULT;
int g_ggml_sycl_dynamic_required_precision = GGML_SYCL_DYNAMIC_PRECISION_F32;
static const char * ggml_sycl_dynamic_precision_names[] = { "F16", "BF16", "TF32", "F32" };

// value of a GGML_SYCL_DYNAMIC_PRECISION-style variable; def if unset or invalid
static int ggml_sycl_get_env_precision(const char * name, int def) {
    const char * env = getenv(name);
    if (!env) {
        return def;
    }
    std::string mode(env);
    for (char & c : mode) {
        c = (char) std::toupper((unsigned char) c);
    }
    for (int i = GGML_SYCL_DYNAMIC_PRECISION_F16; i <= GGML_SYCL_DYNAMIC_PRECISION_F32; i++) {
        if (mode == ggml_sycl_dynamic_precision_names[i]) {
            return i;
        }
    }
    GGML_LOG_WARN("%s: unknown %s=%s, using %s\n", __func__, name, env, ggml_sycl_dynamic_precision_names[def]);
    return def;
}
int g_ggml_sycl_use_async_mem_op = 0;
int g_ggml_sycl_use_async_mem_op_requested = 1;
int g_ggml_sycl_use_level_zero_api = 0;
int g_ggml_sycl_enable_flash_attention = 1;
int g_ggml_sycl_dev2dev_memcpy = DEV2DEV_MEMCPY_SYCL;
int g_ggml_sycl_usm_system = 0;
int g_ggml_sycl_enable_host_pinned_mem = 1;
int g_ggml_sycl_host_pinned_mem_2g = 0;
int g_ggml_sycl_upload_staging_slots = 4;
int g_ggml_sycl_get_mem_api = MEMORY_API_TYPE_LEVEL_ZERO;
int g_ggml_sycl_enable_sparse_fa = 0;
int g_ggml_sycl_debug_sparse_fa = 0;
int g_ggml_sycl_sparse_fa_margin = 256;

#if GGML_SYCL_DNNL
// ask oneDNN which matmul it picks for a small f16 problem
static bool ggml_sycl_dnnl_detect_optimized_gemm(int device) {
    using dt = dnnl::memory::data_type;

    try {
        const dnnl::memory::dims dims    = { 1, 64, 64 };
        const dnnl::memory::dims strides = { 64 * 64, 64, 1 };

        dnnl::primitive_attr attr;
        attr.set_scratchpad_mode(dnnl::scratchpad_mode::user);

        const auto & q   = dpct::dev_mgr::instance().get_device(device).default_queue();
        const auto   eng = dnnl::sycl_interop::make_engine(q.get_device(), q.get_context());
        const auto   pd  = dnnl::matmul::primitive_desc(eng,
                                                        dnnl::memory::desc(dims, dt::f16, strides),
                                                        dnnl::memory::desc(dims, dt::f16, strides),
                                                        dnnl::memory::desc(dims, dt::f32, strides), attr);

        const std::string impl = pd.impl_info_str();
        if (impl.find("ref") == std::string::npos) {
            return true;
        }

        GGML_LOG_WARN("%s: oneDNN has no optimized matmul for device %d (picks %s), using SYCL kernels\n",
                      __func__, device, impl.c_str());
    } catch (const std::exception & e) {
        GGML_LOG_WARN("%s: oneDNN matmul probe failed on device %d (%s), using SYCL kernels\n",
                      __func__, device, e.what());
    }

    return false;
}
#endif

static ggml_sycl_device_info ggml_sycl_init() {
    GGML_SYCL_DEBUG("[SYCL] call ggml_sycl_init\n");
    ggml_sycl_device_info info = {};

    // Do not hard crash when there exists no SYCL devices.
    // We want to allow the user to use non-SYCL tools when SYCL is compiled (such as llama-quantize)
    try {
        info.device_count = dpct::dev_mgr::instance().device_count();
    } catch (sycl::exception const &exc) {
        GGML_LOG_INFO("%s: no SYCL device available: %s\n", __func__, exc.what());
        info.device_count = 0;
    }
    if (info.device_count == 0) {
        GGML_LOG_ERROR("%s: failed to initialize: %s\n", GGML_SYCL_NAME, __func__);
        return info;
    }

    GGML_ASSERT(info.device_count <= GGML_SYCL_MAX_DEVICES);

    int64_t total_vram = 0;
/* This is a bit misleading;  reserved for later */
// #if defined(SYCL_USE_XMX)
//     GGML_LOG_INFO("%s: SYCL_USE_XMX: yes\n", __func__);
// #else
//     GGML_LOG_INFO("%s: SYCL_USE_XMX: no\n", __func__);
// #endif
    for (int i = 0; i < info.device_count; ++i) {
        dpct::device_info prop;
        auto & device = dpct::dev_mgr::instance().get_device(i);

        SYCL_CHECK(CHECK_TRY_ERROR(dpct::get_device_info(
            prop, device)));

#if !defined(GGML_SYCL_SUPPORT_VMM)
        info.devices[i].vmm = 0;
#else
        info.devices[i].vmm = device.has(sycl::aspect::ext_oneapi_virtual_mem);
        if (info.devices[i].vmm) {
            // NB: SYCL's get_mem_granularity always returns the _minimum_ granularity,
            // but the L0 API requires a larger page size for allocs above 2 MiB and
            // rejects non-multiples with UR_RESULT_ERROR_INVALID_VALUE [sic].
            // Here we clamp it to 2 MiB for simplicity, but other devices may require
            // calling zeVirtualMemQueryPageSize or yet unexposed public API.
            const size_t physical_page = 2ull << 20; // 2 MiB
            info.devices[i].vmm_granularity = std::max<size_t>(
                sycl::ext::oneapi::experimental::get_mem_granularity(
                    device, sycl::context(device)),
                physical_page);
        }
#endif

        info.default_tensor_split[i] = total_vram;
        total_vram += prop.get_global_mem_size();

        info.devices[i].cc =
            100 * prop.get_major_version() + 10 * prop.get_minor_version();
        info.devices[i].nsm = prop.get_max_compute_units() / 16; //16: Number of Xe Cores
        info.devices[i].opt_feature.reorder = device.ext_oneapi_architecture_is(syclex::arch_category::intel_gpu);
#if GGML_SYCL_DNNL
        info.devices[i].opt_feature.onednn_optimized_gemm = ggml_sycl_dnnl_detect_optimized_gemm(i);
#endif
        info.devices[i].smpbo = prop.get_local_mem_size();
        info.devices[i].warp_size = WARP_SIZE;
        info.devices[i].usm_system_support = device.has(sycl::aspect::usm_system_allocations);

        info.max_work_group_sizes[i] = prop.get_max_work_group_size();
        info.devices[i].max_wg_per_cu = info.max_work_group_sizes[i] / prop.get_max_compute_units();
        info.devices[i].hw_info = get_device_hw_info(&device);

        // Only check GPU devices; CPU devices use OpenCL and would otherwise
        // disable Level Zero for the GPUs on systems without ONEAPI_DEVICE_SELECTOR set.
        if (device.is_gpu() && device.default_queue().get_backend() != sycl::backend::ext_oneapi_level_zero) {
            GGML_LOG_WARN("SYCL GPU device %d does not use Level Zero backend, disabling Level Zero memory API\n", i);
            info.ext_oneapi_level_zero = false;
        }

#ifdef GGML_SYCL_SUPPORT_LEVEL_ZERO_API
        if (info.ext_oneapi_level_zero && device.is_gpu() && device.default_queue().get_backend() == sycl::backend::ext_oneapi_level_zero) {
            ze_device_handle_t ze_dev = sycl::get_native<sycl::backend::ext_oneapi_level_zero>(device.default_queue().get_device());
            ze_device_properties_t props = {};
            props.stype = ZE_STRUCTURE_TYPE_DEVICE_PROPERTIES;
            ze_result_t r = zeDeviceGetProperties(ze_dev, &props);
            if (r == ZE_RESULT_SUCCESS) {
                info.devices[i].l0_device_type_valid = true;
                info.devices[i].l0_discrete_gpu = !(props.flags & ZE_DEVICE_PROPERTY_FLAG_INTEGRATED);
            }
        }
#endif
    }

    for (int id = 0; id < info.device_count; ++id) {
        info.default_tensor_split[id] /= total_vram;
    }

#ifdef GGML_SYCL_SUPPORT_LEVEL_ZERO_API
    //update g_ggml_sycl_use_level_zero_api according to the device support
    g_ggml_sycl_use_level_zero_api =
        info.ext_oneapi_level_zero && g_ggml_sycl_use_level_zero_api;
#endif

    return info;
}

const ggml_sycl_device_info & ggml_sycl_info() {
    static ggml_sycl_device_info info = ggml_sycl_init();
    return info;
}

static void print_device_detail(int id, sycl::device &device, std::string device_type) {

    dpct::device_info prop;
    SYCL_CHECK(CHECK_TRY_ERROR(
        dpct::get_device_info(prop, device)));

    std::string version;
    version += std::to_string(prop.get_major_version());
    version += ".";
    version += std::to_string(prop.get_minor_version());

    device_type = std::regex_replace(device_type, std::regex("ext_oneapi_"), "");
    std::string name = std::string(prop.get_name());
    name = std::regex_replace(name, std::regex("\\(R\\)"), "");
    name = std::regex_replace(name, std::regex("\\(TM\\)"), "");

    auto global_mem_size = prop.get_global_mem_size()/1000000;
    GGML_LOG_INFO("|%2d|%19s|%39s|%7s|%7d|%8d|%5d|%6luM|%21s|\n", id, device_type.c_str(),
            name.c_str(), version.c_str(), prop.get_max_compute_units(),
            prop.get_max_work_group_size(), prop.get_max_sub_group_size(),
            global_mem_size, device.get_info<sycl::info::device::driver_version>().c_str());
}

static void print_device_opt_feature(int device_count) {
    GGML_LOG_INFO("SYCL Optimization Feature:\n");
    GGML_LOG_INFO(
        "|ID|        Device Type|Reorder|\n");
    GGML_LOG_INFO(
        "|--|-------------------|-------|\n");
    std::map<std::string, size_t> DeviceNums;
    for (int id = 0; id < device_count; ++id) {
      sycl::device device = dpct::dev_mgr::instance().get_device(id);
      std::string backend_type = get_device_backend_and_type(device);
      int type_id = DeviceNums[backend_type]++;
      std::stringstream device_type;
      device_type << "[" << backend_type << ":" << std::to_string(type_id)
                  << "]";
      std::string device_type_s = device_type.str();
      device_type_s = std::regex_replace(device_type_s, std::regex("ext_oneapi_"), "");
      GGML_LOG_INFO("|%2d|%19s|%7s|\n", id, device_type_s.c_str(),
        ggml_sycl_info().devices[id].opt_feature.reorder ? "Y": "N");
    }

}
void ggml_backend_sycl_print_sycl_devices() {
    GGML_SYCL_DEBUG("[SYCL] call ggml_backend_sycl_print_sycl_devices\n");
    int device_count = dpct::dev_mgr::instance().device_count();
    std::map<std::string, size_t> DeviceNums;
    GGML_LOG_INFO("Found %d SYCL devices:\n", device_count);

    GGML_LOG_INFO(
        "|  |                   |                                       |      "
        " |Max    |        |Max  |Global |                     |\n");
    GGML_LOG_INFO(
        "|  |                   |                                       |      "
        " |compute|Max work|sub  |mem    |                     |\n");
    GGML_LOG_INFO(
        "|ID|        Device Type|                                   "
        "Name|Version|units  |group   |group|size   |       Driver version|\n");
    GGML_LOG_INFO(
        "|--|-------------------|---------------------------------------|------"
        "-|-------|--------|-----|-------|---------------------|\n");

    for (int id = 0; id < device_count; ++id) {
      sycl::device device = dpct::dev_mgr::instance().get_device(id);
      std::string backend_type = get_device_backend_and_type(device);
      int type_id = DeviceNums[backend_type]++;
      std::stringstream device_type;
      device_type << "[" << backend_type << ":" << std::to_string(type_id)
                  << "]";
      print_device_detail(id, device, device_type.str());
    }

    print_device_opt_feature(device_count);
}

static const char* dev2dev_int2str(int dev2dev) {
    if (dev2dev == DEV2DEV_MEMCPY_SYCL) {
        return "SYCL API";
    } else if (dev2dev == DEV2DEV_MEMCPY_L0) {
        return "Level Zero API";
    } else if (dev2dev == DEV2DEV_MEMCPY_FORWARD) {
        return "Host Forward";
    } else {
        return "Unknown";
    }
}

/*
* There are several entry APIs to be called as first function in SYCL backend in different cases.
* It's the first internal function to be called by them in SYCL backend.
* This function is used to do initialize work for the SYCL backend and set the global variables.
*/
#ifdef GGML_SYCL_SUPPORT_LEVEL_ZERO_API
static ze_result_t init_zes() {
    ze_result_t res = zesInit(0);
    if (res != ZE_RESULT_SUCCESS) {
        GGML_SYCL_DEBUG("Warning: [%s] zesInit failed with code %d. Sysman free-memory query be unavailable.\n",
                            __func__, (int) res);
    }
    return res;
}

static ze_result_t get_zes_init_res() {
    static ze_result_t zes_init_res = init_zes();
    GGML_SYCL_DEBUG("[SYCL] call %s: zesInit result: %d\n", __func__, (int) zes_init_res);
    return zes_init_res;
}
#endif

static void initialize_sycl_begining() {
#ifdef GGML_SYCL_SUPPORT_LEVEL_ZERO_API
    //must be called in initialization stage, before any other Level Zero API calls
    GGML_SYCL_DEBUG("[SYCL] call %s\n", __func__);
    get_zes_init_res();
#endif
}

static void ggml_check_sycl() try {
    GGML_SYCL_DEBUG("[SYCL] ggml_check_sycl()\n");
    static bool initialized = false;

    if (!initialized) {
        initialize_sycl_begining();

        g_ggml_sycl_debug = ggml_sycl_get_env("GGML_SYCL_DEBUG", 0);
        g_ggml_sycl_dev_debug = ggml_sycl_get_env("GGML_SYCL_DEV_DEBUG", 0);
        g_ggml_sycl_enable_optimize = ggml_sycl_get_env("GGML_SYCL_ENABLE_OPT", 1);
        g_ggml_sycl_enable_graph = ggml_sycl_get_env("GGML_SYCL_ENABLE_GRAPH", 0);
        g_ggml_sycl_enable_dnn = ggml_sycl_get_env("GGML_SYCL_ENABLE_DNN", 1);
        g_ggml_sycl_fa_onednn = ggml_sycl_get_env("GGML_SYCL_FA_ONEDNN", 1);
        g_ggml_sycl_fa_onednn_max_kv = ggml_sycl_get_env("GGML_SYCL_FA_ONEDNN_MAX_KV", 0);
        g_ggml_sycl_enable_mkl_fa = ggml_sycl_get_env("GGML_SYCL_ENABLE_MKL_FA", 1);
        g_ggml_sycl_memtrace = ggml_sycl_get_env("GGML_SYCL_MEMTRACE", 0);
        g_ggml_sycl_memtrace_step = ggml_sycl_get_env("GGML_SYCL_MEMTRACE_STEP", 64);
        g_ggml_sycl_enable_vmm = ggml_sycl_get_env("GGML_SYCL_ENABLE_VMM", 1);
        g_ggml_sycl_enable_fusion = ggml_sycl_get_env("GGML_SYCL_ENABLE_FUSION", 1);
        g_ggml_sycl_enable_esimd = ggml_sycl_get_env("GGML_SYCL_ENABLE_ESIMD", 1);
        g_ggml_sycl_mmvq_wide = ggml_sycl_get_env("GGML_SYCL_MMVQ_WIDE", 1);
        g_ggml_sycl_enable_xmx_esimd = ggml_sycl_get_env("GGML_SYCL_ENABLE_XMX_ESIMD", 1);
        g_ggml_sycl_xmx_max_cols = std::max(0, ggml_sycl_get_env("GGML_SYCL_XMX_MAX_COLS", GGML_SYCL_XMX_MAX_COLS_DEFAULT));
        g_ggml_sycl_xmx_glu_max_cols =
            std::clamp(ggml_sycl_get_env("GGML_SYCL_XMX_GLU_MAX_COLS", GGML_SYCL_XMX_GLU_MAX_COLS), 0, GGML_SYCL_XMX_GLU_MAX_COLS);
        g_ggml_sycl_prioritize_dmmv = ggml_sycl_get_env("GGML_SYCL_PRIORITIZE_DMMV", 0);
        g_ggml_sycl_xmx_gather_types = ggml_sycl_get_env("GGML_SYCL_XMX_GATHER_TYPES", GGML_SYCL_XMX_GATHER_TYPES_DEFAULT);
        g_ggml_sycl_xmx_gather_shapes = ggml_sycl_get_env("GGML_SYCL_XMX_GATHER_SHAPES", GGML_SYCL_XMX_GATHER_SHAPES_DEFAULT);
        g_ggml_sycl_dynamic_precision =
            ggml_sycl_get_env_precision("GGML_SYCL_DYNAMIC_PRECISION", GGML_SYCL_DYNAMIC_PRECISION_DEFAULT);
        g_ggml_sycl_dynamic_required_precision =
            ggml_sycl_get_env_precision("GGML_SYCL_DYNAMIC_REQUIRED_PRECISION", GGML_SYCL_DYNAMIC_PRECISION_F32);

#ifdef GGML_SYCL_SUPPORT_LEVEL_ZERO_API
        g_ggml_sycl_use_level_zero_api = ggml_sycl_get_env("GGML_SYCL_USE_LEVEL_ZERO_API", 1);
#else
        g_ggml_sycl_use_level_zero_api = 0;
#endif
        g_ggml_sycl_dev2dev_memcpy = ggml_sycl_get_env("GGML_SYCL_DEV2DEV_MEMCPY", DEV2DEV_MEMCPY_SYCL);
        g_ggml_sycl_get_mem_api = ggml_sycl_get_env("GGML_SYCL_GET_MEM_API", MEMORY_API_TYPE_LEVEL_ZERO);
        if (g_ggml_sycl_use_level_zero_api == 0) {
            g_ggml_sycl_dev2dev_memcpy = DEV2DEV_MEMCPY_SYCL;
            g_ggml_sycl_get_mem_api = MEMORY_API_TYPE_SYCL;
        }

#ifdef SYCL_FLASH_ATTN
        g_ggml_sycl_enable_flash_attention = ggml_sycl_get_env("GGML_SYCL_ENABLE_FLASH_ATTN", 1);
#else
        g_ggml_sycl_enable_flash_attention = 0;
#endif

        g_ggml_sycl_usm_system = ggml_sycl_get_env("GGML_SYCL_USM_SYSTEM", 0);
        g_ggml_sycl_enable_host_pinned_mem =
            ggml_sycl_get_env("GGML_SYCL_ENABLE_HOST_PINNED_MEM", 1);

        g_ggml_sycl_host_pinned_mem_2g =
            ggml_sycl_get_env("GGML_SYCL_HOST_PINNED_MEM_2G", 0) & g_ggml_sycl_enable_host_pinned_mem;
        g_ggml_sycl_upload_staging_slots = std::max(0, ggml_sycl_get_env("GGML_SYCL_UPLOAD_STAGING_SLOTS", 4));

        g_ggml_sycl_enable_sparse_fa = ggml_sycl_get_env("GGML_SYCL_SPARSE_FA", 0);
        g_ggml_sycl_debug_sparse_fa = ggml_sycl_get_env("GGML_SYCL_SPARSE_FA_DEBUG", 0);
        g_ggml_sycl_sparse_fa_margin = ggml_sycl_get_env("GGML_SYCL_SPARSE_FA_MARGIN", 256);

        GGML_SYCL_DEBUG("[SYCL] call ggml_check_sycl\n");

        GGML_LOG_INFO("Build with Macros:\n");
#if defined(GGML_SYCL_DNNL)
        GGML_LOG_INFO("  GGML_SYCL_DNNL: yes\n");
#else
        GGML_LOG_INFO("  GGML_SYCL_DNNL: no\n");
#endif

#if defined(GGML_SYCL_F16)
        GGML_LOG_INFO("  GGML_SYCL_F16: yes\n");
#else
        GGML_LOG_INFO("  GGML_SYCL_F16: no\n");
#endif

#if defined(GGML_SYCL_FORCE_MMQ)
        GGML_LOG_INFO("  GGML_SYCL_FORCE_MMQ: yes\n");
#else
        GGML_LOG_INFO("  GGML_SYCL_FORCE_MMQ: no\n");
#endif

#if defined(GGML_SYCL_GRAPH)
        GGML_LOG_INFO("  GGML_SYCL_GRAPH: yes\n");
#else
        GGML_LOG_INFO("  GGML_SYCL_GRAPH: no\n");
#endif

#if defined(GGML_SYCL_SUPPORT_LEVEL_ZERO_API)
        GGML_LOG_INFO("  GGML_SYCL_SUPPORT_LEVEL_ZERO_API: yes\n");
#else
        GGML_LOG_INFO("  GGML_SYCL_SUPPORT_LEVEL_ZERO_API: no\n");
#endif
#if defined(GGML_SYCL_SUPPORT_VMM)
        GGML_LOG_INFO("  GGML_SYCL_SUPPORT_VMM: yes\n");
#else
        GGML_LOG_INFO("  GGML_SYCL_SUPPORT_VMM: no\n");
#endif

        //Print the running environment variables for SYCL backend
        GGML_LOG_INFO("Running with Environment Variables:\n");
        GGML_LOG_INFO("  GGML_SYCL_DEBUG: %d\n", g_ggml_sycl_debug);
        GGML_LOG_INFO("  GGML_SYCL_DEV_DEBUG: %d\n", g_ggml_sycl_dev_debug);

#ifdef GGML_SYCL_SUPPORT_LEVEL_ZERO_API
        GGML_LOG_INFO("  GGML_SYCL_DEV2DEV_MEMCPY: %d (%s)\n", g_ggml_sycl_dev2dev_memcpy, dev2dev_int2str(g_ggml_sycl_dev2dev_memcpy));
        GGML_LOG_INFO("  GGML_SYCL_GET_MEM_API: %d (%s)\n", g_ggml_sycl_get_mem_api, mem_api_int2str(g_ggml_sycl_get_mem_api));
#else
        GGML_LOG_INFO("  GGML_SYCL_DEV2DEV_MEMCPY: %d (%s), enable to SYCL API since missing GGML_SYCL_SUPPORT_LEVEL_ZERO_API\n",
                      g_ggml_sycl_dev2dev_memcpy, dev2dev_int2str(g_ggml_sycl_dev2dev_memcpy));
        GGML_LOG_INFO("  GGML_SYCL_GET_MEM_API: %d (%s), enable to SYCL API since missing GGML_SYCL_SUPPORT_LEVEL_ZERO_API\n",
                      g_ggml_sycl_get_mem_api, mem_api_int2str(g_ggml_sycl_get_mem_api));
#endif

#if defined(GGML_SYCL_DNNL)
        GGML_LOG_INFO("  GGML_SYCL_ENABLE_DNN: %d\n", g_ggml_sycl_enable_dnn);
        GGML_LOG_INFO("  GGML_SYCL_FA_ONEDNN: %d\n", g_ggml_sycl_fa_onednn);
#else
        GGML_LOG_INFO("  GGML_SYCL_ENABLE_DNN: DNN disabled by compile flag\n");
        GGML_LOG_INFO("  GGML_SYCL_FA_ONEDNN: %d\n", g_ggml_sycl_fa_onednn);
#endif
        GGML_LOG_INFO("  GGML_SYCL_FA_ONEDNN_MAX_KV: %d\n", g_ggml_sycl_fa_onednn_max_kv);
        GGML_LOG_INFO("  GGML_SYCL_ENABLE_MKL_FA: %d\n", g_ggml_sycl_enable_mkl_fa);
        GGML_LOG_INFO("  GGML_SYCL_MEMTRACE: %d\n", g_ggml_sycl_memtrace);
        GGML_LOG_INFO("  GGML_SYCL_MEMTRACE_STEP: %d\n", g_ggml_sycl_memtrace_step);
#ifdef SYCL_FLASH_ATTN
        GGML_LOG_INFO("  GGML_SYCL_ENABLE_FLASH_ATTN: %d\n", g_ggml_sycl_enable_flash_attention);
#else
        GGML_LOG_INFO("  GGML_SYCL_ENABLE_FLASH_ATTN: %d disabled by compile flag\n",
            g_ggml_sycl_enable_flash_attention);
#endif

#ifdef GGML_SYCL_GRAPH
        GGML_LOG_INFO("  GGML_SYCL_ENABLE_GRAPH: %d\n", g_ggml_sycl_enable_graph);
#else
        GGML_LOG_INFO("  GGML_SYCL_ENABLE_GRAPH: graph disabled by compile flag\n");
#endif

        GGML_LOG_INFO("  GGML_SYCL_ENABLE_OPT: %d\n", g_ggml_sycl_enable_optimize);
        GGML_LOG_INFO("  GGML_SYCL_XMX_GATHER_TYPES: %d\n", g_ggml_sycl_xmx_gather_types);
        GGML_LOG_INFO("  GGML_SYCL_XMX_GATHER_SHAPES: %d\n", g_ggml_sycl_xmx_gather_shapes);
        GGML_LOG_INFO("  GGML_SYCL_DYNAMIC_PRECISION: %s\n",
                      ggml_sycl_dynamic_precision_names[g_ggml_sycl_dynamic_precision]);
        GGML_LOG_INFO("  GGML_SYCL_DYNAMIC_REQUIRED_PRECISION: %s\n",
                      ggml_sycl_dynamic_precision_names[g_ggml_sycl_dynamic_required_precision]);

#if defined(GGML_SYCL_SUPPORT_VMM)
        GGML_LOG_INFO("  GGML_SYCL_ENABLE_VMM: %d\n", g_ggml_sycl_enable_vmm);
#else
        GGML_LOG_INFO("  GGML_SYCL_ENABLE_VMM: virtual memory extension is not available\n");
#endif

        GGML_LOG_INFO("  GGML_SYCL_ENABLE_FUSION: %d\n", g_ggml_sycl_enable_fusion);
        GGML_LOG_INFO("  GGML_SYCL_UPLOAD_STAGING_SLOTS: %d\n", g_ggml_sycl_upload_staging_slots);

#if defined(__INTEL_LLVM_COMPILER)
        GGML_LOG_INFO("  GGML_SYCL_ENABLE_ESIMD: %d\n", g_ggml_sycl_enable_esimd);
#else
        GGML_LOG_INFO("  GGML_SYCL_ENABLE_ESIMD: %d disabled by compile flag\n", g_ggml_sycl_enable_esimd);
#endif

        GGML_LOG_INFO("  GGML_SYCL_MMVQ_WIDE: %d\n", g_ggml_sycl_mmvq_wide);

        GGML_LOG_INFO("  GGML_SYCL_ENABLE_XMX_ESIMD: %d\n", g_ggml_sycl_enable_xmx_esimd);
        GGML_LOG_INFO("  GGML_SYCL_XMX_MAX_COLS: %d\n", g_ggml_sycl_xmx_max_cols);
        GGML_LOG_INFO("  GGML_SYCL_XMX_GLU_MAX_COLS: %d\n", g_ggml_sycl_xmx_glu_max_cols);

        GGML_LOG_INFO("  GGML_SYCL_PRIORITIZE_DMMV: %d\n", g_ggml_sycl_prioritize_dmmv);

        g_ggml_sycl_use_async_mem_op_requested = ggml_sycl_get_env("GGML_SYCL_USE_ASYNC_MEM_OP", 1);
        GGML_LOG_INFO("  GGML_SYCL_USE_ASYNC_MEM_OP: %d\n", g_ggml_sycl_use_async_mem_op_requested);

#ifdef GGML_SYCL_SUPPORT_LEVEL_ZERO_API
        GGML_LOG_INFO("  GGML_SYCL_USE_LEVEL_ZERO_API: %d\n", g_ggml_sycl_use_level_zero_api);
#else
        GGML_LOG_INFO("  GGML_SYCL_USE_LEVEL_ZERO_API: Disable Level Zero API usage by compile flag\n");
#endif

        GGML_LOG_INFO("  GGML_SYCL_USM_SYSTEM: %d\n", g_ggml_sycl_usm_system);
        GGML_LOG_INFO("  GGML_SYCL_ENABLE_HOST_PINNED_MEM: %d\n", g_ggml_sycl_enable_host_pinned_mem);
        GGML_LOG_INFO("  GGML_SYCL_HOST_PINNED_MEM_2G: %d\n", g_ggml_sycl_host_pinned_mem_2g);

        GGML_LOG_INFO("  GGML_SYCL_SPARSE_FA: %d\n", g_ggml_sycl_enable_sparse_fa);
        GGML_LOG_INFO("  GGML_SYCL_SPARSE_FA_DEBUG: %d\n", g_ggml_sycl_debug_sparse_fa);
        GGML_LOG_INFO("  GGML_SYCL_SPARSE_FA_MARGIN: %d\n", g_ggml_sycl_sparse_fa_margin);

/* NOT REMOVE, keep it for next optimize for XMX.
#if defined(SYCL_USE_XMX)
        fprintf(stderr, "%s: SYCL_USE_XMX: yes\n", __func__);
#else
        fprintf(stderr, "%s: SYCL_USE_XMX: no\n", __func__);
#endif
*/
        // Async USM allocation/free is also useful outside the graph path: it avoids the host waits in the reorder
        // staging path while preserving queue ordering semantics. Graph support still depends on the extension being
        // available, but it no longer needs to control the non-graph fast path.
#if defined(GGML_SYCL_GRAPH) && SYCL_EXT_ONEAPI_ASYNC_MEMORY_ALLOC
        g_ggml_sycl_use_async_mem_op = g_ggml_sycl_use_async_mem_op_requested || g_ggml_sycl_enable_graph;
        if (g_ggml_sycl_use_async_mem_op) {
            for (unsigned int i = 0; i < dpct::dev_mgr::instance().device_count(); ++i) {
                if (!dpct::dev_mgr::instance().get_device(i).has(sycl::aspect::ext_oneapi_async_memory_alloc)) {
                    g_ggml_sycl_use_async_mem_op = 0;
                    break;
                }
            }
        }
#endif
        if (CHECK_TRY_ERROR(g_all_sycl_device_count =
                            dpct::dev_mgr::instance().device_count()) != 0) {
            initialized = true;
            g_sycl_loaded = false;
            return;
        }
        GGML_ASSERT(g_all_sycl_device_count <= GGML_SYCL_MAX_DEVICES);

        initialized = true;
        g_sycl_loaded = true;
        ggml_backend_sycl_print_sycl_devices();
    }
}
catch (sycl::exception const &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
  std::exit(1);
}

/*
device_index: device index from 0 to n (continue numbers).
    It is used for device select/set in SYCL backend internal data structure.
*/
inline void check_allow_gpu_index(const int device_index) {
  if (device_index >= ggml_sycl_info().device_count) {
    char error_buf[256];
    snprintf(
        error_buf,
        sizeof(error_buf),
        "%s error: device_index:%d is out of range: [0-%d]",
        __func__,
        device_index,
        ggml_sycl_info().device_count - 1);
    GGML_LOG_ERROR("%s\n", error_buf);
    assert(false);
  }
}

GGML_API void ggml_backend_sycl_get_gpu_list(int *id_list, int max_len) try {
    GGML_SYCL_DEBUG("[SYCL] call ggml_backend_sycl_get_gpu_list\n");
    for(int i=0;i<max_len;i++) id_list[i] = -1;

    for (int i=0;i< ggml_sycl_info().device_count;i++){
        if (i>=max_len) break;
        id_list[i] = i;
    }
    return;
}
catch (sycl::exception const &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
  std::exit(1);
}

inline void free_aligned_mem_host(void * memblock) {
#ifdef _WIN32
    _aligned_free(memblock);
#else
    free(memblock);
#endif
}

// sycl buffer

struct ggml_backend_sycl_buffer_context {
    // pinned staging for uploads; the host fills one slot while the previous one transfers
    static constexpr size_t staging_slot_size = 8*1024*1024;

    struct host_staging {
        void * data = nullptr;
        std::vector<sycl::event> events;
        std::vector<bool> submitted;
        int next = 0;
    };

    int device;
    void * dev_ptr = nullptr;
    queue_ptr stream;
    std::string name;
    optimize_feature opt_feature;
    std::vector<ggml_tensor_extra_gpu *> tensor_extras;
    host_staging staging;
    bool is_usm_system;

    ggml_backend_sycl_buffer_context(int device, void * dev_ptr, queue_ptr stream, bool is_usm_system) :
        device(device), dev_ptr(dev_ptr), stream(stream), is_usm_system(is_usm_system) {
            check_allow_gpu_index(device);
            name = (GGML_SYCL_NAME + std::to_string(device));
            opt_feature = ggml_sycl_info().devices[device].opt_feature;
        }

    // waits for every queued upload, then releases the pinned block
    void drop_host_staging() {
        for (size_t i = 0; i < staging.submitted.size(); ++i) {
            if (staging.submitted[i]) {
                staging.events[i].wait_and_throw();
                staging.submitted[i] = false;
            }
        }
        if (staging.data != nullptr) {
            sycl::free(staging.data, *stream);
            staging.data = nullptr;
        }
    }

    ~ggml_backend_sycl_buffer_context() {
        drop_host_staging();
        if (dev_ptr != nullptr) {
            ggml_sycl_set_device(device);
            if (is_usm_system)
                free_aligned_mem_host(dev_ptr);
            else
                SYCL_CHECK(CHECK_TRY_ERROR(ggml_sycl_free_device(dev_ptr, *stream)));
        }

        //release extra used by tensors
        for (ggml_tensor_extra_gpu * extra : tensor_extras) {
            release_extra_gpu(extra);
        }

    }
};

static const char * ggml_backend_sycl_buffer_type_get_name(ggml_backend_buffer_type_t buft);

static bool ggml_backend_buffer_is_sycl(ggml_backend_buffer_t buffer) {
    return buffer->buft->iface.get_name == ggml_backend_sycl_buffer_type_get_name;
}

static void
ggml_backend_sycl_buffer_free_buffer(ggml_backend_buffer_t buffer) try {
    sycl_pipe::submit_scope pipe_ss_;
    ggml_backend_sycl_buffer_context * ctx = ( ggml_backend_sycl_buffer_context *)buffer->context;
    sycl_pipe::drain(ctx->device);
    ggml_sycl_set_device(ctx->device);

    delete ctx;
}
catch (sycl::exception const &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
  std::exit(1);
}

static void * ggml_backend_sycl_buffer_get_base(ggml_backend_buffer_t buffer) {
    ggml_backend_sycl_buffer_context * ctx = ( ggml_backend_sycl_buffer_context *)buffer->context;
    return ctx->dev_ptr;
}

static enum ggml_status
ggml_backend_sycl_buffer_init_tensor(ggml_backend_buffer_t buffer,
                                     ggml_tensor *tensor) try {
    sycl_pipe::submit_scope pipe_ss_;
    GGML_SYCL_DEBUG("[SYCL] call %s", __func__);
    GGML_SYCL_DEBUG("%s", debug_get_tensor_str(": tensor", tensor, "\n").c_str());
    ggml_backend_sycl_buffer_context * ctx = (ggml_backend_sycl_buffer_context *)buffer->context;

    if (tensor->view_src != NULL) {
        assert(tensor->view_src->buffer->buft == buffer->buft);
        return GGML_STATUS_SUCCESS;
    }

    if (g_ggml_sycl_enable_optimize) {
        // set reorder extra buffer based on supported type
        switch (tensor->type) {
            case GGML_TYPE_Q4_0:
            case GGML_TYPE_Q8_0:
            case GGML_TYPE_Q2_K:
            case GGML_TYPE_Q3_K:
            case GGML_TYPE_Q4_K:
            case GGML_TYPE_Q5_K:
            case GGML_TYPE_Q6_K:
            case GGML_TYPE_IQ3_XXS:
            case GGML_TYPE_IQ3_S:
            case GGML_TYPE_MXFP4:{
                ggml_tensor_extra_gpu * extra = new ggml_tensor_extra_gpu{};
                tensor->extra                 = extra;
                ctx->tensor_extras.push_back(extra);
                break;
            }
            default:
                break;
        }
    }

    if (ggml_is_quantized(tensor->type)) {
        // initialize padding to 0 to avoid possible NaN values
        size_t original_size = ggml_nbytes(tensor);
        size_t padded_size = ggml_backend_buft_get_alloc_size(buffer->buft, tensor);

        if (padded_size > original_size && tensor->view_src == nullptr) {
            SYCL_CHECK(CHECK_TRY_ERROR(ctx->stream->memset(
                (char *)tensor->data + original_size, 0,
                padded_size - original_size).wait()));
        }
    }
    return GGML_STATUS_SUCCESS;
}
catch (sycl::exception const &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
  std::exit(1);
}

static void ggml_backend_sycl_buffer_set_tensor(ggml_backend_buffer_t buffer,
                                                ggml_tensor *tensor,
                                                const void *data, size_t offset,
                                                size_t size) try {
    sycl_pipe::submit_scope pipe_ss_;
    GGML_SYCL_DEBUG("[SYCL] call %s", __func__);
    GGML_SYCL_DEBUG("%s", debug_get_tensor_str(": tensor", tensor).c_str());
    GGML_SYCL_DEBUG(" size=%zu offset=%zu\n", size, offset);
    ggml_backend_sycl_buffer_context * ctx = ( ggml_backend_sycl_buffer_context *)buffer->context;
    if (sycl_pipe::pending(ctx->device)) {
        // the card is still working on queued batches: this write must land after them, with its own copy of the data
        auto   held   = std::make_shared<std::vector<char>>((const char *) data, (const char *) data + size);
        char * target = (char *) tensor->data + offset;
        const int device = ctx->device;
        sycl_pipe::held_add(device, (const char *) tensor->data, offset, held);
        sycl_pipe::enqueue(device, [held, target, device] {
            {
                sycl_pipe::submit_scope ss_;
                auto q = &(dpct::dev_mgr::instance().get_device(device).default_queue());
                { ggml_sycl_pipe_wait_scope ws_; SYCL_CHECK(CHECK_TRY_ERROR(q->wait())); }
                sycl::event ev;
                SYCL_CHECK(CHECK_TRY_ERROR(ev = q->memcpy(target, held->data(), held->size())));
                { ggml_sycl_pipe_wait_scope ws_; SYCL_CHECK(CHECK_TRY_ERROR(ev.wait())); }
            }
            sycl_pipe::held_done(target, held);
        }, "buf-set");
        return;
    }
    ggml_sycl_set_device(ctx->device);

    // copy through pinned memory so the device never reads mmap()ed pages directly
    // chunks pipeline on the in-order compute queue, so no drain per tensor is needed
    const int n_slots = g_ggml_sycl_upload_staging_slots;
    if (n_slots > 0 && ctx->staging.data == nullptr) {
        ctx->staging.data = sycl::malloc_host(n_slots * ctx->staging_slot_size, *ctx->stream);
        if (ctx->staging.data != nullptr) {
            ctx->staging.events.resize(n_slots);
            ctx->staging.submitted.assign(n_slots, false);
        }
    }
    if (ctx->staging.data != nullptr) {
        queue_ptr    stream    = ctx->stream;
        char *       dst       = (char *) tensor->data + offset;
        const char * src       = (const char *) data;
        size_t       remaining = size;
        while (remaining > 0) {
            const size_t chunk = std::min(remaining, ctx->staging_slot_size);
            const int    slot  = ctx->staging.next;
            ctx->staging.next = (ctx->staging.next + 1) % (int) ctx->staging.submitted.size();
            if (ctx->staging.submitted[slot]) {
                ctx->staging.events[slot].wait_and_throw();
            }
            void * stage = (char *) ctx->staging.data + slot * ctx->staging_slot_size;
            memcpy(stage, src, chunk);
            ctx->staging.events[slot] = stream->memcpy(dst, stage, chunk);
            ctx->staging.submitted[slot] = true;
            src       += chunk;
            dst       += chunk;
            remaining -= chunk;
        }
        return;
    }

    auto stream = &(dpct::dev_mgr::instance().get_device(ctx->device).default_queue());
    SYCL_CHECK(CHECK_TRY_ERROR(dpct::dev_mgr::instance().get_device(ctx->device).queues_wait_and_throw()));
#ifndef _WIN32
    // Note: Use host buffer to save the data from mmap(), then copy to device. It's workaround for mmap() issue on PVC GPU.
    // This function will be called during load model from disk. Use memory buffer replace dynamic won't save more time and brings potential memory leak risk here.
    char * host_buf = (char *) malloc(size);
    memcpy(host_buf, data, size);
    SYCL_CHECK(CHECK_TRY_ERROR((*stream).memcpy((char *) tensor->data + offset, host_buf, size).wait()));
    free(host_buf);
#else
    SYCL_CHECK(CHECK_TRY_ERROR((*stream).memcpy((char *) tensor->data + offset, data, size).wait()));
#endif
}
catch (sycl::exception const &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
  std::exit(1);
}

// Strided upload for the tensor-split (meta) backend, which writes each device's slice of a split
// weight as n_copies rows. Without this the generic fallback issued one set_tensor per row, each
// draining the device queues and blocking on a tiny memcpy; a split 3-D MoE expert tensor is
// ne1*ne2 rows, so tensor-parallel loads made millions of synchronous copies.
static void ggml_backend_sycl_buffer_set_tensor_2d(ggml_backend_buffer_t buffer, ggml_tensor * tensor,
                                                   const void * data, size_t offset, size_t size,
                                                   size_t n_copies, size_t stride_tensor,
                                                   size_t stride_data) try {
    sycl_pipe::submit_scope pipe_ss_;
    ggml_backend_sycl_buffer_context * ctx = (ggml_backend_sycl_buffer_context *) buffer->context;
    sycl_pipe::drain(ctx->device);
    ggml_sycl_set_device(ctx->device);
    auto stream = &(dpct::dev_mgr::instance().get_device(ctx->device).default_queue());
    SYCL_CHECK(CHECK_TRY_ERROR(dpct::dev_mgr::instance().get_device(ctx->device).queues_wait_and_throw()));

    char *       dst = (char *) tensor->data + offset;
    const char * src = (const char *) data;

    if (stride_tensor == size) {
        // Contiguous destination: gather the strided source rows into bounded host batches.
        const size_t max_batch_bytes = 64u << 20;
        const size_t rows_per_batch  = std::max<size_t>(1, max_batch_bytes / std::max<size_t>(size, 1));
        std::vector<char> host_buf(std::min(n_copies, rows_per_batch) * size);
        for (size_t i0 = 0; i0 < n_copies; i0 += rows_per_batch) {
            const size_t n = std::min(rows_per_batch, n_copies - i0);
            for (size_t i = 0; i < n; ++i) {
                memcpy(host_buf.data() + i * size, src + (i0 + i) * stride_data, size);
            }
            SYCL_CHECK(CHECK_TRY_ERROR(stream->memcpy(dst + i0 * size, host_buf.data(), n * size).wait()));
        }
        return;
    }

    // Strided destination: queue every row, then wait once.
    for (size_t i = 0; i < n_copies; ++i) {
        stream->memcpy(dst + i * stride_tensor, src + i * stride_data, size);
    }
    { ggml_sycl_pipe_wait_scope ws_; SYCL_CHECK(CHECK_TRY_ERROR(stream->wait())); }
}
catch (sycl::exception const &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
  std::exit(1);
}

static void ggml_backend_sycl_buffer_get_tensor(ggml_backend_buffer_t buffer,
                                                const ggml_tensor *tensor,
                                                void *data, size_t offset,
                                                size_t size) try {
    sycl_pipe::submit_scope pipe_ss_;
    GGML_SYCL_DEBUG("[SYCL] call %s", __func__);
    GGML_SYCL_DEBUG("%s", debug_get_tensor_str(": tensor", tensor).c_str());
    GGML_SYCL_DEBUG(" size=%zu offset=%zu\n", size, offset);
    ggml_backend_sycl_buffer_context * ctx = ( ggml_backend_sycl_buffer_context *)buffer->context;

    sycl_pipe::drain(ctx->device);
    ggml_sycl_set_device(ctx->device);
    auto stream = dpct::dev_mgr::instance().get_device(ctx->device).default_queue();

    // Drain the queue before reading: a USM copy is not reliably ordered against the kernels queued before it
    // on this stack, so a read issued right after a graph (the backend-sampled draft tokens are a few ints)
    // could return the previous contents of the buffer. GGML_SYCL_GET_TENSOR_WAIT=0 restores the old behaviour.
    static const bool wait_first = ggml_sycl_get_env("GGML_SYCL_GET_TENSOR_WAIT", 1) != 0;
    if (wait_first) {
        SYCL_CHECK(CHECK_TRY_ERROR(dpct::dev_mgr::instance().get_device(ctx->device).queues_wait_and_throw()));
    }

    SYCL_CHECK(CHECK_TRY_ERROR(
        stream.memcpy(data, (const char *)tensor->data + offset, size)
            .wait()));
}
catch (sycl::exception const &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
  std::exit(1);
}

#ifdef GGML_SYCL_SUPPORT_LEVEL_ZERO_API
static bool ggml_sycl_is_l0_discrete_gpu(int device) {
    return ggml_sycl_info().devices[device].l0_discrete_gpu;
}
#endif

static void memcpy_host_forward(sycl::queue &q_dst, sycl::queue &q_src, void *ptr_dst,
                         const void *ptr_src, size_t size) {
    char *host_buf = (char *)malloc(size);
    if (host_buf == nullptr) {
        GGML_LOG_ERROR("%s: can't allocate %zu Bytes of memory for host staging\n", __func__, size);
        GGML_ABORT("fatal error");
    }
    q_src.memcpy(host_buf, (const char *)ptr_src, size).wait();
    q_dst.memcpy((char *)ptr_dst, host_buf, size).wait();
    free(host_buf);
}

static void dev2dev_memcpy(int device_dst, sycl::queue &q_dst, int device_src, sycl::queue &q_src, void *ptr_dst,
                    const void *ptr_src, size_t size) {

#ifdef GGML_SYCL_SUPPORT_LEVEL_ZERO_API
    if (g_ggml_sycl_dev2dev_memcpy == DEV2DEV_MEMCPY_L0) {
        // Use Level Zero direct copy for dGPU-to-dGPU transfers.
        const bool l0_copy_supported =
            ggml_sycl_is_l0_discrete_gpu(device_dst) && ggml_sycl_is_l0_discrete_gpu(device_src);
        if (g_ggml_sycl_use_level_zero_api && l0_copy_supported) {
            auto ze_ctx = sycl::get_native<sycl::backend::ext_oneapi_level_zero>(q_dst.get_context());
            auto ze_dev = sycl::get_native<sycl::backend::ext_oneapi_level_zero>(q_dst.get_device());
            ze_command_queue_desc_t cq_desc = {ZE_STRUCTURE_TYPE_COMMAND_QUEUE_DESC, nullptr, 0, 0,
                                            0, ZE_COMMAND_QUEUE_MODE_SYNCHRONOUS, ZE_COMMAND_QUEUE_PRIORITY_NORMAL};
            ze_command_list_handle_t cl;
            ze_result_t r = zeCommandListCreateImmediate(ze_ctx, ze_dev, &cq_desc, &cl);
            if (r == ZE_RESULT_SUCCESS) {
                GGML_SYCL_DEBUG("[SYCL] dev2dev memcpy by L0\n");
                r = zeCommandListAppendMemoryCopy(cl, ptr_dst, ptr_src, size, nullptr, 0, nullptr);
                zeCommandListDestroy(cl);
                if (r == ZE_RESULT_SUCCESS) {
                    return;
                }
            }
        }
    }
#endif

    if (g_ggml_sycl_dev2dev_memcpy == DEV2DEV_MEMCPY_SYCL) {
        if (q_dst.get_device().ext_oneapi_can_access_peer(q_src.get_device(),
                                                          sycl::ext::oneapi::peer_access::access_supported)) {
            GGML_SYCL_DEBUG("[SYCL] dev2dev memcpy by SYCL\n");
            q_dst.get_device().ext_oneapi_enable_peer_access(q_src.get_device());
            SYCL_CHECK(CHECK_TRY_ERROR(q_dst.memcpy(ptr_dst, ptr_src, size).wait()));
            return;
        }
    }

    // Host-staged copy
    if(g_ggml_sycl_dev2dev_memcpy == DEV2DEV_MEMCPY_FORWARD) {
        GGML_SYCL_DEBUG("[SYCL] dev2dev memcpy by host forward for setting GGML_SYCL_DEV2DEV_MEMCPY=2\n");
    } else {
        GGML_SYCL_DEBUG("[SYCL] dev2dev memcpy by host forward for SYCL/L0 fallback\n");
    }
    memcpy_host_forward(q_dst, q_src, ptr_dst, ptr_src, size);
}

static bool
ggml_backend_sycl_buffer_cpy_tensor(ggml_backend_buffer_t buffer,
                                    const ggml_tensor *src,
                                    ggml_tensor *dst) try {
    sycl_pipe::submit_scope pipe_ss_;
    bool is_cpy_supported = ggml_backend_buffer_is_sycl(src->buffer);
    GGML_SYCL_DEBUG("[SYCL] call %s", __func__);
    GGML_SYCL_DEBUG("%s", debug_get_tensor_str(": dst", dst).c_str());
    GGML_SYCL_DEBUG("%s", debug_get_tensor_str(" src", src).c_str());
    GGML_SYCL_DEBUG(" is_cpy_supported=%d\n", is_cpy_supported);
    if (is_cpy_supported) {
        ggml_backend_sycl_buffer_context * src_ctx = (ggml_backend_sycl_buffer_context *)src->buffer->context;
        ggml_backend_sycl_buffer_context * dst_ctx = (ggml_backend_sycl_buffer_context *)dst->buffer->context;

        if (sycl_pipe::active(dst_ctx->device) && sycl_pipe::t_worker_device < 0 &&
            (sycl_pipe::pending(src_ctx->device) || sycl_pipe::pending(dst_ctx->device))) {
            // queued behind whatever both cards still have to do; the caller is not blocked
            sycl_pipe::held_report("buf", src, src_ctx->device, dst_ctx->device);
            sycl_pipe::copy_async(src_ctx->device, (const char *) src->data, dst_ctx->device, (char *) dst->data, ggml_nbytes(src));
            return true;
        }

        ggml_sycl_set_device(src_ctx->device);
        /*
        DPCT1009:198: SYCL uses exceptions to report errors and does not use the
        error codes. The original code was commented out and a warning string
        was inserted. You need to rewrite this code.
        */
        SYCL_CHECK(CHECK_TRY_ERROR(
            dpct::dev_mgr::instance().get_device(src_ctx->device).queues_wait_and_throw()));
        ggml_sycl_set_device(dst_ctx->device);
        /*
        DPCT1009:199: SYCL uses exceptions to report errors and does not use the
        error codes. The original code was commented out and a warning string
        was inserted. You need to rewrite this code.
        */
        SYCL_CHECK(CHECK_TRY_ERROR(
            dpct::dev_mgr::instance().get_device(dst_ctx->device).queues_wait_and_throw()));
        /*
        DPCT1009:200: SYCL uses exceptions to report errors and does not use the
        error codes. The original code was commented out and a warning string
        was inserted. You need to rewrite this code.
        */

        queue_ptr stream_dst = dst_ctx->stream;
        queue_ptr stream_src = src_ctx->stream;
        size_t size = ggml_nbytes(src);

        //todo. it's dirty solutino to walkaroud known issue:device2device cross GPUs.
        dev2dev_memcpy(dst_ctx->device, *stream_dst, src_ctx->device, *stream_src, dst->data, src->data, size);

//todo, it's known issue：error in device2device cross GPUs. reused when the issue is fixed. DON"T remove
#if 0
        SYCL_CHECK(CHECK_TRY_ERROR((*stream).memcpy(
            (char *)dst->data, (const char *)src->data, size).wait()));

        /*
        DPCT1009:201: SYCL uses exceptions to report errors and does not use the
        error codes. The original code was commented out and a warning string
        was inserted. You need to rewrite this code.
        */
        SYCL_CHECK(CHECK_TRY_ERROR(
            dpct::dev_mgr::instance().get_device(dst_ctx->device).queues_wait_and_throw()));
#endif
        return true;
    }
    return false;
    GGML_UNUSED(buffer);
} catch (const sycl::exception & exc) {
    std::cerr << exc.what() << "Exception caught at file:" << __FILE__ << ", line:" << __LINE__ << std::endl;
    std::exit(1);
}

static void ggml_backend_sycl_buffer_clear(ggml_backend_buffer_t buffer,
                                           uint8_t value) try {
    sycl_pipe::submit_scope pipe_ss_;
    GGML_SYCL_DEBUG("[SYCL] call %s: size=%zu\n", __func__, buffer->size);
    ggml_backend_sycl_buffer_context * ctx = (ggml_backend_sycl_buffer_context *) buffer->context;

    sycl_pipe::drain(ctx->device);
    ggml_sycl_set_device(ctx->device);
    queue_ptr stream = ctx->stream;
    SYCL_CHECK(
        CHECK_TRY_ERROR(dpct::get_current_device().queues_wait_and_throw()));

    constexpr size_t MAX_CHUNK = 2ULL << 30;  // 2 GiB
    for (size_t off = 0; off < buffer->size; off += MAX_CHUNK) {
        size_t chunk = std::min(buffer->size - off, MAX_CHUNK);
        SYCL_CHECK(CHECK_TRY_ERROR(
            (*stream)
                .memset(static_cast<char*>(ctx->dev_ptr) + off, value, chunk)
                .wait()
        ));
    }
}
catch (sycl::exception const &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
  std::exit(1);
}

static void ggml_backend_sycl_buffer_memset_tensor(ggml_backend_buffer_t buffer, ggml_tensor * tensor, uint8_t value,
                                                   size_t offset, size_t size) {
    sycl_pipe::submit_scope pipe_ss_;
    GGML_SYCL_DEBUG("[SYCL] call %s", __func__);
    GGML_SYCL_DEBUG("%s", debug_get_tensor_str(": tensor", tensor).c_str());
    GGML_SYCL_DEBUG(" size=%zu offset=%zu value=%u\n", size, offset, value);
    ggml_backend_sycl_buffer_context * ctx = (ggml_backend_sycl_buffer_context *) buffer->context;
    sycl_pipe::drain(ctx->device);
    SYCL_CHECK(ggml_sycl_set_device(ctx->device));
    auto stream = &(dpct::dev_mgr::instance().get_device(ctx->device).default_queue());
    if (size == 0) {
        return;  // Nothing to do
    }
    if (tensor->data == nullptr) {
        GGML_ABORT("Error: Tensor data pointer is null.\n");
    }
    void * target_ptr = static_cast<char *>(tensor->data) + offset;
    SYCL_CHECK(CHECK_TRY_ERROR((*stream).memset(target_ptr, value, size)));
    { ggml_sycl_pipe_wait_scope ws_; SYCL_CHECK(CHECK_TRY_ERROR((*stream).wait())); }
}

static void ggml_backend_sycl_buffer_reset(ggml_backend_buffer_t buffer) {
    GGML_SYCL_DEBUG("[SYCL] call %s\n", __func__);
    if (buffer == nullptr) {
        return;
    }

    ggml_backend_sycl_buffer_context * ctx = (ggml_backend_sycl_buffer_context *) buffer->context;

    if (ctx != nullptr) {
        for (ggml_tensor_extra_gpu * extra : ctx->tensor_extras) {
            release_extra_gpu(extra);
        }
        ctx->tensor_extras.clear();  // reset the tensor_extras vector
    }
}

static const ggml_backend_buffer_i ggml_backend_sycl_buffer_interface = {
    /* .free_buffer     = */ ggml_backend_sycl_buffer_free_buffer,
    /* .get_base        = */ ggml_backend_sycl_buffer_get_base,
    /* .init_tensor     = */ ggml_backend_sycl_buffer_init_tensor,
    /* .memset_tensor   = */ ggml_backend_sycl_buffer_memset_tensor,
    /* .set_tensor      = */ ggml_backend_sycl_buffer_set_tensor,
    /* .get_tensor      = */ ggml_backend_sycl_buffer_get_tensor,
    /* .set_tensor_2d   = */ ggml_backend_sycl_buffer_set_tensor_2d,
    /* .get_tensor_2d   = */ NULL,
    /* .cpy_tensor      = */ ggml_backend_sycl_buffer_cpy_tensor,
    /* .clear           = */ ggml_backend_sycl_buffer_clear,
    /* .reset           = */ ggml_backend_sycl_buffer_reset,
};

// sycl buffer type
struct ggml_backend_sycl_buffer_type_context {
    int device;
    std::string name;

    // each buffer type has its own stream
    queue_ptr stream = nullptr;
};

static const char * ggml_backend_sycl_buffer_type_get_name(ggml_backend_buffer_type_t buft) {
    ggml_backend_sycl_buffer_type_context * ctx = (ggml_backend_sycl_buffer_type_context *)buft->context;

    return ctx->name.c_str();
}

static bool check_usm_system(int device, size_t size) {
    bool use_usm_system = g_ggml_sycl_usm_system && size >= ((size_t)4 * MEM_SIZE_1G);

    if (use_usm_system && !ggml_sycl_info().devices[device].usm_system_support) {
        GGML_LOG_INFO("Device does not support USM system allocations\n");
        use_usm_system = false;
    }

    return use_usm_system;
}

inline void * aligned_malloc_host(size_t alignment, size_t size) {
#ifdef _WIN32
    return _aligned_malloc(size, alignment);
#else
    return aligned_alloc(alignment, size);
#endif
}

static ggml_backend_buffer_t
ggml_backend_sycl_buffer_type_alloc_buffer(ggml_backend_buffer_type_t buft,
                                           size_t size) try {
    GGML_SYCL_DEBUG("[SYCL] call %s: size=%zu\n", __func__, size);
    ggml_check_sycl();

    ggml_backend_sycl_buffer_type_context * buft_ctx = (ggml_backend_sycl_buffer_type_context *)buft->context;
    ggml_sycl_set_device(buft_ctx->device);
    const queue_ptr stream = buft_ctx->stream;
    size = std::max(size, (size_t)1); // syclMalloc returns null for size 0
    /*
    Alignment below ensures best performance. While in theory it could lead to
    wasting memory, this is acceptable because in practice only few buffers are
    allocated and even less exceed the minimum size accepted here for USM system
    allocations.
     */
    size_t alignment = MEM_SIZE_2M;
    size_t aligned_size = ((size + alignment - 1) / alignment) * alignment;
    bool use_usm_system = check_usm_system(buft_ctx->device, aligned_size);

    void * dev_ptr;
    if (use_usm_system) {
        GGML_SYCL_DEBUG("[SYCL] allocating %zu Bytes with USM system\n", size);
        dev_ptr = (void *)aligned_malloc_host(alignment, aligned_size);
        if (!dev_ptr) {
            GGML_LOG_ERROR("%s: can't allocate %zu Bytes of memory on host\n", __func__, size);
            return nullptr;
        }
    } else {
        SYCL_CHECK(CHECK_TRY_ERROR(dev_ptr = (void *)ggml_sycl_malloc_device(size, *stream, GGML_SYCL_MEM_BUFFER)));
        if (!dev_ptr) {
          GGML_LOG_ERROR("%s: can't allocate %zu Bytes of memory on device\n", __func__, size);
          return nullptr;
        }
    }
    ggml_backend_sycl_buffer_context * ctx = new  ggml_backend_sycl_buffer_context(buft_ctx->device, dev_ptr, buft_ctx->stream, use_usm_system);
    return ggml_backend_buffer_init(buft, ggml_backend_sycl_buffer_interface, ctx, size);
}
catch (sycl::exception const &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
  std::exit(1);
}

static size_t ggml_backend_sycl_buffer_type_get_alignment(ggml_backend_buffer_type_t buft) {
    return SYCL_BUFFER_ALIGNMENT;
    GGML_UNUSED(buft);
}

static bool is_bmg_g31_arch(int device) {
    return ggml_sycl_info().devices[device].hw_info.arch == gpu_arch::intel_gpu_bmg_g31;
}

static size_t ggml_backend_sycl_buffer_type_get_max_size(ggml_backend_buffer_type_t buft) {
    ggml_backend_sycl_buffer_type_context * ctx = (ggml_backend_sycl_buffer_type_context *)buft->context;
    int device = ctx->device;
    // use the buft device, the current device may differ on multi-GPU systems
    size_t max_alloc_size = dpct::get_device(device).get_max_mem_alloc_size();
    if (g_ggml_sycl_host_pinned_mem_2g) {
        return std::min(max_alloc_size, (size_t) 2LL*1024*1024*1024);
    } else {
        if(is_bmg_g31_arch(device)) {
            //Todo, it's workaround for BMG-G31, which has a known issue with large allocations.
            //The max alloc size is reduced to 60% of the reported max alloc size.
            //remove it after https://github.com/intel/compute-runtime/issues/998 is fixed.
            max_alloc_size = max_alloc_size*0.6;
        }
        return max_alloc_size;

    }
    GGML_UNUSED(buft);
}

static size_t ggml_backend_sycl_buffer_type_get_alloc_size(ggml_backend_buffer_type_t buft, const ggml_tensor * tensor) {
    // Reserve the additional scratch so it's visible to the graph allocator
    size_t size = tensor->op == GGML_OP_FLASH_ATTN_EXT
        ? ggml_sycl_flash_attn_ext_get_alloc_size(tensor)
        : ggml_nbytes(tensor);
    int64_t ne0 = tensor->ne[0];

    if (ggml_is_quantized(tensor->type)) {
        if (ne0 % MATRIX_ROW_PADDING != 0) {
            size += ggml_row_size(tensor->type, MATRIX_ROW_PADDING - ne0 % MATRIX_ROW_PADDING);
        }
    }

    return size;

    GGML_UNUSED(buft);
}

static const ggml_backend_buffer_type_i ggml_backend_sycl_buffer_type_interface = {
    /* .get_name            = */ ggml_backend_sycl_buffer_type_get_name,
    /* .alloc_buffer        = */ ggml_backend_sycl_buffer_type_alloc_buffer,
    /* .alloc_buffer_n      = */ NULL,
    /* .get_alignment       = */ ggml_backend_sycl_buffer_type_get_alignment,
    /* .get_max_size        = */ ggml_backend_sycl_buffer_type_get_max_size,
    /* .get_alloc_size      = */ ggml_backend_sycl_buffer_type_get_alloc_size,
    /* .get_alloc_size_n    = */ NULL,
    /* .is_host             = */ NULL,
};

ggml_backend_buffer_type_t ggml_backend_sycl_buffer_type(int device) {
    static std::mutex mutex;
    std::lock_guard<std::mutex> lock(mutex);


    auto dev_count = ggml_backend_sycl_get_device_count();

    if (device>=dev_count or device<0) {
        GGML_LOG_ERROR("ggml_backend_sycl_buffer_type error: device_index:%d is out of range [0, %d], miss to call ggml_backend_sycl_set_single_device()\n",
            device, dev_count-1);
        GGML_ASSERT(device<dev_count);
    }
    static struct ggml_backend_buffer_type ggml_backend_sycl_buffer_types[GGML_SYCL_MAX_DEVICES];

    static bool ggml_backend_sycl_buffer_type_initialized = false;

    if (!ggml_backend_sycl_buffer_type_initialized) {
        for (int i = 0; i < dev_count; i++) {
            auto & device_i = dpct::dev_mgr::instance().get_device(i);
            queue_ptr stream = &(device_i.default_queue());
            ggml_backend_sycl_buffer_types[i] = {
                /* .iface    = */ ggml_backend_sycl_buffer_type_interface,
                /* .device   = */ ggml_backend_reg_dev_get(ggml_backend_sycl_reg(), i),
                /* .context  = */ new ggml_backend_sycl_buffer_type_context{i, GGML_SYCL_NAME + std::to_string(i), stream},
            };
        }
        ggml_backend_sycl_buffer_type_initialized = true;
    }
    return &ggml_backend_sycl_buffer_types[device];
}

static ggml_backend_buffer_type_t ggml_backend_sycl_buffer_type(ggml_backend_sycl_context * ctx) {
    GGML_SYCL_DEBUG("[SYCL] call ggml_backend_sycl_buffer_type\n");

    int device = ctx->device;
    if (device>=ggml_sycl_info().device_count or device<0) {
        GGML_LOG_ERROR("ggml_backend_sycl_buffer_type error: device_index:%d is out of range [0, %d], miss to call ggml_backend_sycl_set_single_device()\n",
            device, ggml_sycl_info().device_count-1);
        GGML_ASSERT(device<ggml_sycl_info().device_count);
    }
    static struct ggml_backend_buffer_type ggml_backend_sycl_buffer_types[GGML_SYCL_MAX_DEVICES];

    static bool ggml_backend_sycl_buffer_type_initialized = false;

    if (!ggml_backend_sycl_buffer_type_initialized) {
        for (int i = 0; i < ggml_sycl_info().device_count; i++) {
            ggml_backend_sycl_buffer_types[i] = {
                /* .iface    = */ ggml_backend_sycl_buffer_type_interface,
                /* .device   = */ nullptr,
                /* .context  = */ new ggml_backend_sycl_buffer_type_context{i, GGML_SYCL_NAME + std::to_string(i), ctx->stream(i, 0)},
            };
        }
        ggml_backend_sycl_buffer_type_initialized = true;
    }
    return &ggml_backend_sycl_buffer_types[device];
}

// sycl split buffer

static int64_t get_row_rounding(ggml_type type, const std::array<float, GGML_SYCL_MAX_DEVICES> & tensor_split) {
    int64_t min_compute_capability = INT_MAX;
    int64_t max_compute_capability = INT_MIN;
    for (int i = 0; i < ggml_sycl_info().device_count; ++i) {
        if (tensor_split[i] < (i + 1 < ggml_sycl_info().device_count ? tensor_split[i + 1] : 1.0f)) {
            if (min_compute_capability > ggml_sycl_info().devices[i].cc) {
                min_compute_capability = ggml_sycl_info().devices[i].cc;
            }
            if (max_compute_capability < ggml_sycl_info().devices[i].cc) {
                max_compute_capability = ggml_sycl_info().devices[i].cc;
            }
        }
    }

    switch(type) {
        case GGML_TYPE_Q1_0:
        case GGML_TYPE_Q4_0:
        case GGML_TYPE_Q4_1:
            return max_compute_capability >= VER_GEN9 ? 128 : 64;
        case GGML_TYPE_Q5_0:
        case GGML_TYPE_Q5_1:
        case GGML_TYPE_Q8_0:
            return 64;
        case GGML_TYPE_F16:
        case GGML_TYPE_F32:
            return 1;
        case GGML_TYPE_Q2_K:
        case GGML_TYPE_Q3_K:
        case GGML_TYPE_Q4_K:
        case GGML_TYPE_Q5_K:
        case GGML_TYPE_IQ2_XXS:
        case GGML_TYPE_IQ2_XS:
        case GGML_TYPE_IQ2_S:
        case GGML_TYPE_IQ1_S:
        case GGML_TYPE_IQ1_M:
        case GGML_TYPE_IQ3_XXS:
        case GGML_TYPE_IQ4_XS:
        case GGML_TYPE_IQ4_NL:
            return max_compute_capability >= VER_GEN9 ? 128 : 64;
        case GGML_TYPE_IQ3_S:
            return max_compute_capability >= VER_GEN9 ? 128 : 64;
        case GGML_TYPE_Q6_K:
            return 64;
        default:
            GGML_ABORT("fatal error");
    }
}

static void get_row_split(int64_t * row_low, int64_t * row_high, const ggml_tensor * tensor, const std::array<float, GGML_SYCL_MAX_DEVICES> & tensor_split, int id) {
    const int64_t nrows = ggml_nrows(tensor);
    const int64_t rounding = get_row_rounding(tensor->type, tensor_split);

    *row_low = id == 0 ? 0 : nrows*tensor_split[id];
    *row_low -= *row_low % rounding;
    if (id == ggml_sycl_info().device_count - 1) {
        *row_high = nrows;
    } else {
        *row_high = nrows*tensor_split[id + 1];
        *row_high -= *row_high % rounding;
    }
}

static size_t ggml_nbytes_split(const struct ggml_tensor * tensor, int nrows_split) {
    static_assert(GGML_MAX_DIMS == 4, "GGML_MAX_DIMS is not 4 - update this function");

    return nrows_split*ggml_row_size(tensor->type, tensor->ne[0]);
}

struct ggml_backend_sycl_split_buffer_type_context {
    std::array<float, GGML_SYCL_MAX_DEVICES> tensor_split;
};

struct ggml_backend_sycl_split_buffer_context {
    ~ggml_backend_sycl_split_buffer_context() try {
        for (ggml_tensor_extra_gpu * extra : tensor_extras) {
            release_extra_gpu(extra, streams);
        }
    }
    catch (sycl::exception const &exc) {
      std::cerr << exc.what() << "Exception caught at file:" << __FILE__
                << ", line:" << __LINE__ << std::endl;
      std::exit(1);
    }

    std::vector<ggml_tensor_extra_gpu *> tensor_extras;
    std::vector<queue_ptr> streams;
};

static void ggml_backend_sycl_split_buffer_free_buffer(ggml_backend_buffer_t buffer) {
    ggml_backend_sycl_split_buffer_context * ctx = (ggml_backend_sycl_split_buffer_context *)buffer->context;
    delete ctx;
}

static void * ggml_backend_sycl_split_buffer_get_base(ggml_backend_buffer_t buffer) {
    // the pointers are stored in the tensor extras, this is just a dummy address and never dereferenced
    return (void *)0x1000;

    GGML_UNUSED(buffer);
}

static enum ggml_status
ggml_backend_sycl_split_buffer_init_tensor(ggml_backend_buffer_t buffer,
                                           ggml_tensor *tensor) try {
    GGML_SYCL_DEBUG("[SYCL] call %s", __func__);
    GGML_SYCL_DEBUG("%s", debug_get_tensor_str(": tensor", tensor, "\n").c_str());
    GGML_ASSERT(tensor->view_src == nullptr); // views of split tensors are not supported

    ggml_backend_sycl_split_buffer_context * ctx = (ggml_backend_sycl_split_buffer_context *)buffer->context;
    ggml_backend_sycl_split_buffer_type_context * buft_ctx = (ggml_backend_sycl_split_buffer_type_context *)buffer->buft->context;

    const int64_t ne0 = tensor->ne[0];

    ggml_tensor_extra_gpu * extra = new ggml_tensor_extra_gpu{};

    ctx->tensor_extras.push_back(extra);
    // consumers index the queues by device id, so size the vector once with one queue per device
    if (ctx->streams.empty()) {
        for (int i = 0; i < ggml_sycl_info().device_count; ++i) {
            ctx->streams.push_back(&(dpct::get_device(i).default_queue()));
        }
    }

    for (int i = 0; i < ggml_sycl_info().device_count; ++i) {
        int64_t row_low, row_high;
        get_row_split(&row_low, &row_high, tensor, buft_ctx->tensor_split, i);

        int64_t nrows_split = row_high - row_low;
        if (nrows_split == 0) {
            continue;
        }

        size_t size = ggml_nbytes_split(tensor, nrows_split);
        const size_t original_size = size;

        // pad last row to a multiple of 512 elements to avoid out-of-bounds memory accesses
        if (ne0 % MATRIX_ROW_PADDING != 0) {
            size += ggml_row_size(tensor->type, MATRIX_ROW_PADDING - ne0 % MATRIX_ROW_PADDING);
        }

        ggml_sycl_set_device(i);
        const queue_ptr stream = ctx->streams[i];
        char * buf;
        SYCL_CHECK(CHECK_TRY_ERROR(buf = (char *)ggml_sycl_malloc_device(size, *stream, GGML_SYCL_MEM_BUFFER)));
        if (!buf) {
            char err_buf[1024];
            snprintf(err_buf, 1023, "%s: can't allocate %zu Bytes of memory on device\n", __func__, size);
            throw std::runtime_error(err_buf);
        }
        // set padding to 0 to avoid possible NaN values
        if (size > original_size) {
            /*
            DPCT1009:209: SYCL uses exceptions to report errors and does not use
            the error codes. The original code was commented out and a warning
            string was inserted. You need to rewrite this code.
            */
            SYCL_CHECK(CHECK_TRY_ERROR(
                (*stream)
                    .memset(buf + original_size, 0, size - original_size)
                    .wait()));
        }

        extra->data_device[i] = buf;

        for (int64_t is = 0; is < GGML_SYCL_MAX_STREAMS; ++is) {
            /*
            DPCT1009:210: SYCL uses exceptions to report errors and does not use
            the error codes. The original code was commented out and a warning
            string was inserted. You need to rewrite this code.
            */
            SYCL_CHECK(
                CHECK_TRY_ERROR(extra->events[i][is] = new sycl::event()));
        }
    }
    tensor->extra = extra;
    return GGML_STATUS_SUCCESS;
}
catch (sycl::exception const &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
  std::exit(1);
}

static void
ggml_backend_sycl_split_buffer_set_tensor(ggml_backend_buffer_t buffer,
                                          ggml_tensor *tensor, const void *data,
                                          size_t offset, size_t size) try {
    GGML_SYCL_DEBUG("[SYCL] call %s", __func__);
    GGML_SYCL_DEBUG("%s", debug_get_tensor_str(": tensor", tensor).c_str());
    GGML_SYCL_DEBUG(" size=%zu offset=%zu\n", size, offset);
    // split tensors must always be set in their entirety at once
    GGML_ASSERT(offset == 0);
    GGML_ASSERT(size == ggml_nbytes(tensor));

    ggml_backend_sycl_split_buffer_context * ctx = (ggml_backend_sycl_split_buffer_context *)buffer->context;
    ggml_backend_sycl_split_buffer_type_context * buft_ctx = (ggml_backend_sycl_split_buffer_type_context *)buffer->buft->context;

    const int64_t ne0 = tensor->ne[0];
    const size_t nb1 = tensor->nb[1];
    ggml_tensor_extra_gpu * extra = (ggml_tensor_extra_gpu *)tensor->extra;

    for (int i = 0; i < ggml_sycl_info().device_count; ++i) {
        int64_t row_low, row_high;
        get_row_split(&row_low, &row_high, tensor, buft_ctx->tensor_split, i);

        int64_t nrows_split = row_high - row_low;
        if (nrows_split == 0) {
            continue;
        }

        const size_t offset_split = row_low*nb1;
        size_t size = ggml_nbytes_split(tensor, nrows_split);
        const size_t original_size = size;

        // pad last row to a multiple of 512 elements to avoid out-of-bounds memory accesses
        if (ne0 % MATRIX_ROW_PADDING != 0) {
            size += ggml_row_size(tensor->type, MATRIX_ROW_PADDING - ne0 % MATRIX_ROW_PADDING);
        }

        const char * buf_host = (const char *)data + offset_split;
        /*
        DPCT1009:211: SYCL uses exceptions to report errors and does not use the
        error codes. The original code was commented out and a warning string
        was inserted. You need to rewrite this code.
        */
        ggml_sycl_set_device(i);
        const queue_ptr stream = ctx->streams[i];
        SYCL_CHECK(CHECK_TRY_ERROR(
            (*stream)
                .memcpy(extra->data_device[i], buf_host, original_size)
                .wait()));
    }
}
catch (sycl::exception const &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
  std::exit(1);
}

static void
ggml_backend_sycl_split_buffer_get_tensor(ggml_backend_buffer_t buffer,
                                          const ggml_tensor *tensor, void *data,
                                          size_t offset, size_t size) try {
    GGML_SYCL_DEBUG("[SYCL] call %s", __func__);
    GGML_SYCL_DEBUG("%s", debug_get_tensor_str(": tensor", tensor).c_str());
    GGML_SYCL_DEBUG(" size=%zu offset=%zu\n", size, offset);
    // split tensors must always be set in their entirety at once
    GGML_ASSERT(offset == 0);
    GGML_ASSERT(size == ggml_nbytes(tensor));

    ggml_backend_sycl_split_buffer_context * ctx = (ggml_backend_sycl_split_buffer_context *)buffer->context;
    ggml_backend_sycl_split_buffer_type_context * buft_ctx = (ggml_backend_sycl_split_buffer_type_context *)buffer->buft->context;

    const int64_t ne0 = tensor->ne[0];
    const size_t nb1 = tensor->nb[1];
    ggml_tensor_extra_gpu * extra = (ggml_tensor_extra_gpu *)tensor->extra;

    for (int i = 0; i < ggml_sycl_info().device_count; ++i) {
        int64_t row_low, row_high;
        get_row_split(&row_low, &row_high, tensor, buft_ctx->tensor_split, i);

        int64_t nrows_split = row_high - row_low;
        if (nrows_split == 0) {
            continue;
        }

        const size_t offset_split = row_low*nb1;
        size_t size = ggml_nbytes_split(tensor, nrows_split);
        const size_t original_size = size;

        // pad last row to a multiple of 512 elements to avoid out-of-bounds memory accesses
        if (ne0 % MATRIX_ROW_PADDING != 0) {
            size += ggml_row_size(tensor->type, MATRIX_ROW_PADDING - ne0 % MATRIX_ROW_PADDING);
        }

        char * buf_host = (char *)data + offset_split;
        /*
        DPCT1009:212: SYCL uses exceptions to report errors and does not use the
        error codes. The original code was commented out and a warning string
        was inserted. You need to rewrite this code.
        */
        ggml_sycl_set_device(i);
        const queue_ptr stream = ctx->streams[i];
        SYCL_CHECK(CHECK_TRY_ERROR(
            (*stream)
                .memcpy(buf_host, extra->data_device[i], original_size)
                .wait()));
    }
}
catch (sycl::exception const &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
  std::exit(1);
}

static void ggml_backend_sycl_split_buffer_clear(ggml_backend_buffer_t buffer, uint8_t value) {
    GGML_UNUSED(buffer);
    GGML_UNUSED(value);
}

static struct ggml_backend_buffer_i ggml_backend_sycl_split_buffer_interface = {
    /* .free_buffer     = */ ggml_backend_sycl_split_buffer_free_buffer,
    /* .get_base        = */ ggml_backend_sycl_split_buffer_get_base,
    /* .init_tensor     = */ ggml_backend_sycl_split_buffer_init_tensor,
    /* .memset_tensor   = */ NULL,
    /* .set_tensor      = */ ggml_backend_sycl_split_buffer_set_tensor,
    /* .get_tensor      = */ ggml_backend_sycl_split_buffer_get_tensor,
    /* .set_tensor_2d   = */ NULL,
    /* .get_tensor_2d   = */ NULL,
    /* .cpy_tensor      = */ NULL,
    /* .clear           = */ ggml_backend_sycl_split_buffer_clear,
    /* .reset           = */ NULL,
};

// sycl split buffer type

static const char * ggml_backend_sycl_split_buffer_type_get_name(ggml_backend_buffer_type_t buft) {
    return GGML_SYCL_NAME "_Split";

    GGML_UNUSED(buft);
}

static bool ggml_backend_buffer_is_sycl_split(ggml_backend_buffer_t buffer) {
   return buffer->buft->iface.get_name == ggml_backend_sycl_split_buffer_type_get_name;
}

static ggml_backend_buffer_t ggml_backend_sycl_split_buffer_type_alloc_buffer(ggml_backend_buffer_type_t buft, size_t size) {
    // since we don't know the exact split after rounding, we cannot allocate the device buffers at this point
    // instead, we allocate them for each tensor separately in init_tensor
    // however, the size still represents the maximum cumulative size of all the device buffers after the tensors are allocated,
    // as returned by get_alloc_size. this limit is enforced during tensor allocation by ggml-alloc, so it must be correct.
    ggml_backend_sycl_split_buffer_context * ctx = new ggml_backend_sycl_split_buffer_context();

    return ggml_backend_buffer_init(buft, ggml_backend_sycl_split_buffer_interface, ctx, size);
}

static size_t ggml_backend_sycl_split_buffer_type_get_alignment(ggml_backend_buffer_type_t buft) {
    return SYCL_BUFFER_ALIGNMENT;
    GGML_UNUSED(buft);
}

static size_t ggml_backend_sycl_split_buffer_type_get_alloc_size(ggml_backend_buffer_type_t buft, const ggml_tensor * tensor) {
    ggml_backend_sycl_split_buffer_type_context * ctx = (ggml_backend_sycl_split_buffer_type_context *)buft->context;

    size_t total_size = 0;

    const int64_t ne0 = tensor->ne[0];

    for (int i = 0; i < ggml_sycl_info().device_count; ++i) {
        int64_t row_low, row_high;
        get_row_split(&row_low, &row_high, tensor, ctx->tensor_split, i);

        int64_t nrows_split = row_high - row_low;
        if (nrows_split == 0) {
            continue;
        }

        total_size += ggml_nbytes_split(tensor, nrows_split);

        // pad last row to a multiple of 512 elements to avoid out-of-bounds memory accesses
        if (ne0 % MATRIX_ROW_PADDING != 0) {
            total_size += ggml_row_size(tensor->type, MATRIX_ROW_PADDING - ne0 % MATRIX_ROW_PADDING);
        }
    }

    return total_size;
}

static bool ggml_backend_sycl_split_buffer_type_is_host(ggml_backend_buffer_type_t buft) {
    return false;

    GGML_UNUSED(buft);
}

static ggml_backend_buffer_type_i ggml_backend_sycl_split_buffer_type_interface = {
    /* .get_name            = */ ggml_backend_sycl_split_buffer_type_get_name,
    /* .alloc_buffer        = */ ggml_backend_sycl_split_buffer_type_alloc_buffer,
    /* .alloc_buffer_n      = */ NULL,
    /* .get_alignment       = */ ggml_backend_sycl_split_buffer_type_get_alignment,
    /* .get_max_size        = */ NULL, // defaults to SIZE_MAX
    /* .get_alloc_size      = */ ggml_backend_sycl_split_buffer_type_get_alloc_size,
    /* .get_alloc_size_n    = */ NULL,
    /* .is_host             = */ ggml_backend_sycl_split_buffer_type_is_host,
};

ggml_backend_buffer_type_t ggml_backend_sycl_split_buffer_type(int main_device, const float * tensor_split) {
    GGML_SYCL_DEBUG("[SYCL] call ggml_backend_sycl_split_buffer_type\n");

    GGML_UNUSED(main_device);

    static std::mutex mutex;
    std::lock_guard<std::mutex> lock(mutex);

    ggml_check_sycl();
    // FIXME: this is not thread safe
    static std::map<std::array<float, GGML_SYCL_MAX_DEVICES>, struct ggml_backend_buffer_type> buft_map;

    std::array<float, GGML_SYCL_MAX_DEVICES> tensor_split_arr = {};

    bool all_zero = tensor_split == nullptr || std::all_of(tensor_split, tensor_split + GGML_SYCL_MAX_DEVICES, [](float x) { return x == 0.0f; });
    if (all_zero) {
        tensor_split_arr = ggml_sycl_info().default_tensor_split;
    } else {
        float split_sum = 0.0f;
        for (int i = 0; i < ggml_sycl_info().device_count; ++i) {
            tensor_split_arr[i] = split_sum;
            split_sum += tensor_split[i];
        }
        for (int i = 0; i < ggml_sycl_info().device_count; ++i) {
            tensor_split_arr[i] /= split_sum;
        }
    }

    auto it = buft_map.find(tensor_split_arr);
    if (it != buft_map.end()) {
        return &it->second;
    }

    struct ggml_backend_buffer_type buft {
        /* .iface   = */ ggml_backend_sycl_split_buffer_type_interface,
        /* .device  = */ ggml_backend_reg_dev_get(ggml_backend_sycl_reg(), 0),
        /* .context = */ new ggml_backend_sycl_split_buffer_type_context{tensor_split_arr},
    };

    auto result = buft_map.emplace(tensor_split_arr, buft);
    return &result.first->second;
}

// host buffer type

struct ggml_backend_sycl_device_context {
    int device;
    std::string name;
    std::string description;
    int op_offload_min_batch_size;
};

static const char * ggml_backend_sycl_host_buffer_type_name(ggml_backend_buffer_type_t buft) {
    return GGML_SYCL_NAME "_Host";

    GGML_UNUSED(buft);
}

static int ggml_backend_sycl_host_buffer_type_device(ggml_backend_buffer_type_t buft) {
    return static_cast<const ggml_backend_sycl_device_context *>(buft->device->context)->device;
}

//host pinned memory
static void * ggml_backend_sycl_host_malloc(int device, size_t size) {
    GGML_SYCL_DEBUG("[SYCL] call ggml_backend_sycl_host_malloc of size %.2f MiB on device %d\n", size / 1024.0 / 1024.0, device);
    void * ptr = nullptr;
    try {
        ggml_check_sycl();
        // USM host memory is page-locked and device-accessible by construction
        auto & q = dpct::dev_mgr::instance().get_device(device).default_queue();
        ptr = sycl::malloc_host(size, q, sycl::property_list{});
    } catch (...) {
        ptr = nullptr;
    }
    if (ptr == nullptr) {
        GGML_LOG_WARN("%s: failed to allocate %.2f MiB of pinned memory\n", __func__,
                size / 1024.0 / 1024.0);
    }

    return ptr;
}

static void ggml_backend_sycl_host_buffer_free_buffer(ggml_backend_buffer_t buffer) {
    if (buffer->context == nullptr) {
        return;
    }
    if (g_ggml_sycl_enable_host_pinned_mem) {
        const int device = ggml_backend_sycl_host_buffer_type_device(buffer->buft);
        auto & q = dpct::dev_mgr::instance().get_device(device).default_queue();
        SYCL_CHECK(CHECK_TRY_ERROR(sycl::free(buffer->context, q)));
    } else {
        free_aligned_mem_host((void *) buffer->context);
    }
}

static ggml_backend_buffer_t ggml_backend_sycl_host_buffer_type_alloc_buffer(ggml_backend_buffer_type_t buft, size_t size) {
    void * ptr = g_ggml_sycl_enable_host_pinned_mem ?
                     ggml_backend_sycl_host_malloc(ggml_backend_sycl_host_buffer_type_device(buft), size) :
                     aligned_malloc_host(TENSOR_ALIGNMENT, size);
    if (ptr == nullptr) {
        // fallback to cpu buffer
        return ggml_backend_buft_alloc_buffer(ggml_backend_cpu_buffer_type(), size);
    }

    // FIXME: this is a hack to avoid having to implement a new buffer type
    ggml_backend_buffer_t buffer = ggml_backend_cpu_buffer_from_ptr(ptr, size);
    buffer->buft = buft;
    buffer->iface.free_buffer = ggml_backend_sycl_host_buffer_free_buffer;

    return buffer;
}

static size_t ggml_backend_sycl_host_buffer_type_get_max_size(ggml_backend_buffer_type_t buft) {

    if (g_ggml_sycl_enable_host_pinned_mem) {
        const int device = ggml_backend_sycl_host_buffer_type_device(buft);
        size_t max_alloc_size = dpct::dev_mgr::instance().get_device(device).get_max_mem_alloc_size();
        if (g_ggml_sycl_host_pinned_mem_2g) {
            return std::min(max_alloc_size, (size_t) 2LL*1024*1024*1024);
        } else {
            return max_alloc_size;
        }
    } else {
        return SIZE_MAX;
    }
}

static ggml_backend_buffer_type_t ggml_backend_sycl_host_buffer_type_for_device(int device) {
    GGML_SYCL_DEBUG("[SYCL] call ggml_backend_sycl_host_buffer_type_for_device on device %d\n", device);

    // the vector is never resized after this, so the returned pointers stay valid
    static std::vector<ggml_backend_buffer_type> buffer_types_host = [] {
        std::vector<ggml_backend_buffer_type> bufts(ggml_backend_sycl_get_device_count());
        for (size_t i = 0; i < bufts.size(); i++) {
            bufts[i] = {
                /* .iface    = */ {
                    /* .get_name            = */ ggml_backend_sycl_host_buffer_type_name,
                    /* .alloc_buffer        = */ ggml_backend_sycl_host_buffer_type_alloc_buffer,
                    /* .alloc_buffer_n      = */ NULL,
                    /* .get_alignment       = */ ggml_backend_cpu_buffer_type()->iface.get_alignment,
                    /* .get_max_size        = */ ggml_backend_sycl_host_buffer_type_get_max_size,
                    /* .get_alloc_size      = */ ggml_backend_cpu_buffer_type()->iface.get_alloc_size,
                    /* .get_alloc_size_n    = */ NULL,
                    /* .is_host             = */ ggml_backend_cpu_buffer_type()->iface.is_host,
                },
                /* .device   = */ ggml_backend_reg_dev_get(ggml_backend_sycl_reg(), i),
                /* .context  = */ nullptr,
            };
        }
        return bufts;
    }();

    GGML_ASSERT(device >= 0 && device < (int) buffer_types_host.size());

    return &buffer_types_host[device];
}

// TODO: this function is unused and is a temporary hack to avoid breaking changes
ggml_backend_buffer_type_t ggml_backend_sycl_host_buffer_type() {
    return ggml_backend_sycl_host_buffer_type_for_device(0);
}

// buffer pool for sycl (legacy)
struct ggml_sycl_pool_leg : public ggml_sycl_pool {
    static const int MAX_SYCL_BUFFERS = 256;

    int device;
    queue_ptr qptr;
    struct ggml_sycl_buffer {
        void * ptr = nullptr;
        size_t size = 0;
    };

    ggml_sycl_buffer buffer_pool[MAX_SYCL_BUFFERS] = {};
    size_t pool_size = 0;

    explicit ggml_sycl_pool_leg(queue_ptr qptr_, int device_) : device(device_), qptr(qptr_) {}

    ~ggml_sycl_pool_leg() {
#ifdef DEBUG_SYCL_POOL
        int    n_cached    = 0;
        size_t bytes_cached = 0;
        for (int i = 0; i < MAX_SYCL_BUFFERS; ++i) {
            if (buffer_pool[i].ptr != nullptr) {
                ++n_cached;
                bytes_cached += buffer_pool[i].size;
            }
        }
        GGML_LOG_INFO("%s: %d buffers, cached = %.2f MiB\n", __func__,
                      n_cached, bytes_cached / 1024.0 / 1024.0);
        const auto slots = format_slots_in_alloc_order();
        if (!slots.empty()) {
            GGML_LOG_INFO("%s: slots MiB: %s\n", __func__, slots.c_str());
        }
#endif

        for (int i = 0; i < MAX_SYCL_BUFFERS; ++i) {
            ggml_sycl_buffer & b = buffer_pool[i];
            if (b.ptr != nullptr) {
                SYCL_CHECK(CHECK_TRY_ERROR(ggml_sycl_free_device(b.ptr, *qptr)));
                pool_size -= b.size;
            }
        }
        GGML_ASSERT(pool_size == 0);
    }

#ifdef DEBUG_SYCL_POOL
    std::string format_slots_in_alloc_order() const {
        std::string line;
        char buf[32];
        bool first = true;
        for (int i = 0; i < MAX_SYCL_BUFFERS; ++i) {
            if (buffer_pool[i].ptr == nullptr) {
                continue;
            }
            if (!first) {
                line += '/';
            }
            first = false;
            snprintf(buf, sizeof(buf), "%.2f", buffer_pool[i].size / 1024.0 / 1024.0);
            line += buf;
        }
        return line;
    }
#endif

    void * alloc(size_t size, size_t * actual_size) override {
#ifdef DEBUG_sycl_MALLOC
        int nnz = 0;
        size_t max_size = 0;
#endif
        size_t best_diff = 1ull << 36;
        int ibest = -1;
        for (int i = 0; i < MAX_SYCL_BUFFERS; ++i) {
            ggml_sycl_buffer& b = buffer_pool[i];
            if (b.ptr != nullptr) {
#ifdef DEBUG_sycl_MALLOC
                ++nnz;
                if (b.size > max_size) max_size = b.size;
#endif
                if (b.size >= size) {
                    size_t diff = b.size - size;
                    if (diff < best_diff) {
                        best_diff = diff;
                        ibest = i;
                        if (!best_diff) {
                            void * ptr = b.ptr;
                            *actual_size = b.size;
                            b.ptr = nullptr;
                            b.size = 0;
                            return ptr;
                        }
                    }
                }
            }
        }
        if (ibest >= 0) {
            ggml_sycl_buffer& b = buffer_pool[ibest];
            void * ptr = b.ptr;
            *actual_size = b.size;
            b.ptr = nullptr;
            b.size = 0;
            return ptr;
        }
        void * ptr;
        size_t look_ahead_size = (size_t) (1.05 * size);

        SYCL_CHECK(CHECK_TRY_ERROR(ptr = (void *)ggml_sycl_malloc_device(look_ahead_size, *qptr, GGML_SYCL_MEM_POOL_LEG)));
        if (!ptr) {
            GGML_LOG_ERROR("%s: can't allocate %zu Bytes of memory on device/GPU\n", __func__, look_ahead_size);
            return nullptr;
        }

        *actual_size = look_ahead_size;
        pool_size += look_ahead_size;

#ifdef DEBUG_SYCL_MALLOC
        GGML_LOG_DEBUG("%s[%d]: %d buffers, max_size = %u MB, pool_size = %u MB, requested %u MB\n", __func__, id, nnz,
                (uint32_t)(max_size/1024/1024), (uint32_t)(g_sycl_pool_size[id]/1024/1024), (uint32_t)(size/1024/1024));
#endif

        // GGML_SYCL_DEBUG("ggml_sycl_pool_malloc_leg look_ahead_size=%zu, return %p\n", look_ahead_size, ptr);
        return ptr;
    }

    void free(void * ptr, size_t size) override {
        for (int i = 0; i < MAX_SYCL_BUFFERS; ++i) {
            ggml_sycl_buffer& b = buffer_pool[i];
            if (b.ptr == nullptr) {
                b.ptr = ptr;
                b.size = size;
                return;
            }
        }
        GGML_LOG_WARN("WARNING: sycl buffer pool full, increase MAX_sycl_BUFFERS\n");
        SYCL_CHECK(CHECK_TRY_ERROR(ggml_sycl_free_device(ptr, *qptr)));
        pool_size -= size;
    }
};

// pool with virtual memory management
#if defined(GGML_SYCL_SUPPORT_VMM)
struct ggml_sycl_pool_vmm : public ggml_sycl_pool {
    static const size_t SYCL_POOL_VMM_MAX_SIZE = 1ull << 35; // 32 GB

    int           device;
    sycl::context ctx;
    sycl::device  dev;

    uintptr_t pool_addr = 0;
    size_t    pool_used = 0;
    size_t    pool_size = 0;
    size_t    granularity;

    // physical_mem owns the commits (unlike cuMemMap)
    struct mapping {
        sycl::ext::oneapi::experimental::physical_mem phys;
        void * map_ptr;
    };
    std::vector<mapping> mappings;

    explicit ggml_sycl_pool_vmm(queue_ptr qptr_, int device_) :
        device(device_),
        ctx(qptr_->get_context()),
        dev(qptr_->get_device()),
        granularity(ggml_sycl_info().devices[device_].vmm_granularity) {
    }

    ~ggml_sycl_pool_vmm() {
        if (pool_addr == 0) {
            return;
        }

        // Per spec, unmap must (a) match the exact (ptr, size) of an earlier
        // physical_mem::map() call and (b) precede destruction of the
        // physical_mem objects (their dtors won't unmap).
        for (auto & m : mappings) {
            SYCL_CHECK(CHECK_TRY_ERROR(sycl::ext::oneapi::experimental::unmap(
                m.map_ptr, m.phys.size(), ctx)));
        }
        SYCL_CHECK(CHECK_TRY_ERROR(sycl::ext::oneapi::experimental::free_virtual_mem(
            pool_addr, SYCL_POOL_VMM_MAX_SIZE, ctx)));
    }

    void * alloc(size_t size, size_t * actual_size) override {
        // round up the allocation size to the alignment to ensure that all allocations are aligned for all data types
        size = GGML_PAD(size, SYCL_BUFFER_ALIGNMENT);

        size_t avail = pool_size - pool_used;

        if (size > avail) {
            // round up to the next multiple of the granularity
            size_t reserve_size = GGML_PAD(size - avail, granularity);

            GGML_ASSERT(pool_size + reserve_size <= SYCL_POOL_VMM_MAX_SIZE);

            if (ggml_sycl_memtrace_enabled()) {
                GGML_LOG_INFO(GGML_SYCL_MEMTRACE_TAG " pool_vmm[%d] committing %5zu MiB (pool %5zu -> %5zu MiB)\n",
                              device, reserve_size / (1024 * 1024), pool_size / (1024 * 1024),
                              (pool_size + reserve_size) / (1024 * 1024));
                ggml_sycl_memtrace_report("before pool_vmm commit");
            }

            // allocate more physical memory
            std::optional<sycl::ext::oneapi::experimental::physical_mem> phys;
            SYCL_CHECK(CHECK_TRY_ERROR(phys.emplace(dev, ctx, reserve_size)));

            // reserve virtual address space (if not already reserved)
            if (pool_addr == 0) {
                SYCL_CHECK(CHECK_TRY_ERROR(
                    pool_addr = sycl::ext::oneapi::experimental::reserve_virtual_mem(
                        SYCL_POOL_VMM_MAX_SIZE, ctx)));
            }

            // map at the end of the pool
            void * map_ptr = nullptr;
            SYCL_CHECK(CHECK_TRY_ERROR(
                map_ptr = phys->map(pool_addr + pool_size, reserve_size,
                                    sycl::ext::oneapi::experimental::address_access_mode::read_write)));

            // stash these so we could unmap this exact range in dtor
            mappings.push_back({
                std::move(*phys),
                map_ptr,
            });

            // add to the pool
            pool_size += reserve_size;
            ggml_sycl_memtrace_add(GGML_SYCL_MEM_POOL_VMM, map_ptr, reserve_size);

#ifdef DEBUG_SYCL_MALLOC
            GGML_LOG_INFO("sycl pool[%d]: size increased to %llu MB (reserved %llu MB)\n",
                          device, (unsigned long long) (pool_size/1024/1024),
                          (unsigned long long) (reserve_size/1024/1024));
#endif
        }

        GGML_ASSERT(pool_addr != 0);

        void * ptr = reinterpret_cast<void *>(pool_addr + pool_used);
        *actual_size = size;
        pool_used += size;

#ifdef DEBUG_SYCL_MALLOC
        GGML_LOG_INFO("sycl pool[%d]: allocated %llu bytes at %p\n", device, (unsigned long long) size, ptr);
#endif

        return ptr;
    }

    void free(void * ptr, size_t size) override {
#ifdef DEBUG_SYCL_MALLOC
        GGML_LOG_INFO("sycl pool[%d]: freed %llu bytes at %p\n", device, (unsigned long long) size, ptr);
#endif

        pool_used -= size;

        // all deallocations must be in reverse order of the allocations
        GGML_ASSERT(ptr == reinterpret_cast<void *>(pool_addr + pool_used));
    }
};
#endif // defined(GGML_SYCL_SUPPORT_VMM)

struct ggml_sycl_pool_host : public ggml_sycl_pool {
    queue_ptr qptr;
    int       device;

    struct ggml_sycl_buffer {
        void * ptr  = nullptr;
        size_t size = 0;
    };

    // Set arbitrarly to 64
    static constexpr int          MAX_POOL_SIZE{ 64 };
    std::vector<ggml_sycl_buffer> buffer_pool = std::vector<ggml_sycl_buffer>(MAX_POOL_SIZE);
    size_t                        pool_size   = 0;
    int                           counter     = 0;

    explicit ggml_sycl_pool_host(queue_ptr qptr_, int device_) : qptr(qptr_), device(device_) {}

    ~ggml_sycl_pool_host() {
        for (int i = 0; i < MAX_POOL_SIZE; ++i) {
            ggml_sycl_buffer & b = buffer_pool[i];
            if (b.ptr != nullptr) {
                SYCL_CHECK(CHECK_TRY_ERROR(sycl::free(b.ptr, *qptr)));
                b.ptr = nullptr;
                pool_size -= b.size;
                b.size = 0;
            }
        }
    }

    void * alloc(size_t size, size_t * actual_size) override {
        if (counter == MAX_POOL_SIZE) {
            // the pool wrapped around, no slot is reserved for this allocation
            void * ptr = nullptr;
            SYCL_CHECK(CHECK_TRY_ERROR(ptr = (void *) sycl::malloc_host(size, *qptr)));
            if (!ptr) {
                GGML_LOG_ERROR("%s: can't allocate %zu Bytes of memory on host\n", __func__, size);
                return nullptr;
            }
            pool_size    += size;
            *actual_size = size;
            counter      = 0;
            return ptr;
        }
        ggml_sycl_buffer & b = buffer_pool[counter];
        ++counter;

        if (b.ptr != nullptr) {
            if (b.size >= size) {
                // hand out the cached buffer, free() puts it back into the pool
                void * ptr = b.ptr;
                *actual_size = b.size;
                b.ptr = nullptr;
                b.size = 0;
                return ptr;
            }
            // the cached buffer is too small, replace it
            SYCL_CHECK(CHECK_TRY_ERROR(sycl::free(b.ptr, *qptr)));
            pool_size -= b.size;
            b.ptr = nullptr;
            b.size = 0;
        }

        void * ptr;
        SYCL_CHECK(CHECK_TRY_ERROR(ptr = (void *) sycl::malloc_host(size, *qptr)));
        if (!ptr) {
            GGML_LOG_ERROR("%s: can't allocate %zu Bytes of memory on host\n", __func__, size);
            return nullptr;
        }
        pool_size    += size;
        *actual_size = size;
        return ptr;
    }

    void free(void * ptr, size_t size) override {
        // if the pool is not completed add the pointer to it in place of the first nullptr found.
        // Otherwise free the buffer, all slots are cached.
        for (int i = 0; i < MAX_POOL_SIZE; ++i) {
            ggml_sycl_buffer & b = buffer_pool[i];
            if (b.ptr == nullptr) {
                b.ptr  = ptr;
                b.size = size;
                return;
            }
        }
        SYCL_CHECK(CHECK_TRY_ERROR(sycl::free(ptr, *qptr)));
        pool_size -= size;
    }
};

std::unique_ptr<ggml_sycl_pool> ggml_backend_sycl_context::new_pool_for_host(queue_ptr qptr, int device) {
    // return pool for the host to speed up memory management
    return std::unique_ptr<ggml_sycl_pool>(new ggml_sycl_pool_host(qptr, device));
}

std::unique_ptr<ggml_sycl_pool> ggml_backend_sycl_context::new_pool_for_device(queue_ptr qptr, int device) {
#if defined(GGML_SYCL_SUPPORT_VMM)
    if (g_ggml_sycl_enable_vmm && ggml_sycl_info().devices[device].vmm) {
        return std::unique_ptr<ggml_sycl_pool>(new ggml_sycl_pool_vmm(qptr, device));
    }
#endif // defined(GGML_SYCL_SUPPORT_VMM)
    return std::unique_ptr<ggml_sycl_pool>(new ggml_sycl_pool_leg(qptr, device));
}

/// kernels
typedef void (*ggml_sycl_op_mul_mat_t)(
    ggml_backend_sycl_context & ctx,
    const ggml_tensor *src0, const ggml_tensor *src1, ggml_tensor *dst,
    const char *src0_dd_i, const float *src1_ddf_i, const char *src1_ddq_i,
    float *dst_dd_i, const int64_t row_low, const int64_t row_high,
    const int64_t src1_ncols, const int64_t src1_padded_row_size,
    const queue_ptr &stream);



static void mul_mat_p021_f16_f32(
    const void * __restrict__ vx, const float * __restrict__ y, float * __restrict__ dst,
    const int ncols_x, const int nrows_x, const int nchannels_x, const int nchannels_y,
    const sycl::nd_item<3> &item_ct1) {

    const sycl::half *x = (const sycl::half *)vx;

    const int row_x = item_ct1.get_local_range(1) * item_ct1.get_group(1) +
                      item_ct1.get_local_id(1);
    const int channel = item_ct1.get_local_range(0) * item_ct1.get_group(0) +
                        item_ct1.get_local_id(0);
    const int channel_x = channel / (nchannels_y / nchannels_x);

    const int nrows_y = ncols_x;
    const int nrows_dst = nrows_x;
    const int row_dst = row_x;

    float tmp = 0.0f;

    for (int col_x0 = 0; col_x0 < ncols_x;
         col_x0 += item_ct1.get_local_range(2)) {
        const int col_x = col_x0 + item_ct1.get_local_id(2);

        if (col_x >= ncols_x) {
            break;
        }

        // x is transposed and permuted
        const int ix = row_x*nchannels_x*ncols_x + channel_x*ncols_x + col_x;
        const float xi =
            sycl::vec<sycl::half, 1>(x[ix])
                .convert<float, sycl::rounding_mode::automatic>()[0];

        const int row_y = col_x;


        // y is not transposed but permuted
        const int iy = channel*nrows_y + row_y;

        tmp += xi * y[iy];
    }

    // dst is not transposed and not permuted
    const int idst = channel*nrows_dst + row_dst;

    // sum up partial sums and write back result
#pragma unroll
    for (int mask = WARP_SIZE / 2; mask > 0; mask >>= 1) {
        tmp +=
            dpct::permute_sub_group_by_xor(item_ct1.get_sub_group(), tmp, mask);
    }

    if (item_ct1.get_local_id(2) == 0) {
        dst[idst] = tmp;
    }
}

static void mul_mat_vec_nc_f16_f32( // nc == non-contiguous
    const void * __restrict__ vx, const float * __restrict__ y, float * __restrict__ dst, const int ncols_x, const int nrows_x,
    const int row_stride_x, const int channel_stride_x,const int channel_stride_y, const int channel_x_divisor,
    const sycl::nd_item<3> &item_ct1) {

    const sycl::half *x = (const sycl::half *)vx;

    const int row_x = item_ct1.get_local_range(1) * item_ct1.get_group(1) +
                      item_ct1.get_local_id(1);
    const int channel = item_ct1.get_local_range(0) * item_ct1.get_group(0) +
                        item_ct1.get_local_id(0);
    const int channel_x = channel / channel_x_divisor;

    const int nrows_dst = nrows_x;
    const int row_dst   = row_x;

    const int idst = channel*nrows_dst + row_dst;

    float tmp = 0.0f;

    for (int col_x0 = 0; col_x0 < ncols_x;
         col_x0 += item_ct1.get_local_range(2)) {
        const int col_x = col_x0 + item_ct1.get_local_id(2);

        if (col_x >= ncols_x) {
            break;
        }

        const int row_y = col_x;

        const int ix = channel_x*channel_stride_x + row_x*row_stride_x + col_x;
        const int iy = channel * channel_stride_y + row_y;

        const float xi =
            sycl::vec<sycl::half, 1>(x[ix])
                .convert<float, sycl::rounding_mode::automatic>()[0];

        tmp += xi * y[iy];
    }

    // sum up partial sums and write back result
#pragma unroll
    for (int mask = WARP_SIZE / 2; mask > 0; mask >>= 1) {
        tmp +=
            dpct::permute_sub_group_by_xor(item_ct1.get_sub_group(), tmp, mask);
    }

    if (item_ct1.get_local_id(2) == 0) {
        dst[idst] = tmp;
    }
}

static void k_sum_rows_f32(const float * x, float * dst, const int ncols,
                           const sycl::nd_item<3> &item_ct1) {
    const int row = item_ct1.get_group(1);
    const int col = item_ct1.get_local_id(2);

    float sum = 0.0f;
    for (int i = col; i < ncols; i += item_ct1.get_local_range(2)) {
        sum += x[row * ncols + i];
    }

    sum = warp_reduce_sum(sum, item_ct1);

    if (col == 0) {
        dst[row] = sum;
    }
}


template<typename T>
static inline void ggml_sycl_swap(T & a, T & b) {
    T tmp = a;
    a = b;
    b = tmp;
}

template <ggml_sort_order order>
__dpct_inline__ static void
k_argsort_f32_i32(const float *x, int *dst, const int ncols, int ncols_pad,
                  const int tasks_per_thread, const sycl::nd_item<3> &item_ct1,
                  uint8_t *dpct_local) {
    // bitonic sort
    int col_index =  item_ct1.get_local_id(2);
    int row = item_ct1.get_group(1);

    for (int i = 0; i < tasks_per_thread; i++) {
        int col = col_index * tasks_per_thread + i;
        if (col >= ncols_pad) {
            return;
        }
    }

    const float * x_row = x + row * ncols;
    auto dst_row = (int *)dpct_local;

    // initialize indices
    for (int i=0;i<tasks_per_thread;i++){
        int col = col_index*tasks_per_thread+i;
        dst_row[col] = col;
    }

    item_ct1.barrier(sycl::access::fence_space::local_space);

    for (int k = 2; k <= ncols_pad; k *= 2) {
        for (int j = k / 2; j > 0; j /= 2) {
            for (int i = 0; i < tasks_per_thread; i++) {
                int col = col_index * tasks_per_thread + i;
                int ixj = col ^ j;
                if (ixj > col) {
                    if ((col & k) == 0) {
                        if (dst_row[col] >= ncols ||
                            (dst_row[ixj] < ncols &&
                             (order == GGML_SORT_ORDER_ASC
                                  ? x_row[dst_row[col]] > x_row[dst_row[ixj]]
                                  : x_row[dst_row[col]] <
                                        x_row[dst_row[ixj]]))) {
                            ggml_sycl_swap(dst_row[col], dst_row[ixj]);
                        }
                    } else {
                        if (dst_row[ixj] >= ncols ||
                            (dst_row[col] < ncols &&
                             (order == GGML_SORT_ORDER_ASC
                                  ? x_row[dst_row[col]] < x_row[dst_row[ixj]]
                                  : x_row[dst_row[col]] >
                                        x_row[dst_row[ixj]]))) {
                            ggml_sycl_swap(dst_row[col], dst_row[ixj]);
                        }
                    }
                }
                item_ct1.barrier(sycl::access::fence_space::local_space);
            }
        }
    }

    // copy the result to dst without the padding
    for (int i = 0; i < tasks_per_thread; i++) {
        int col = col_index * tasks_per_thread + i;
        if (col < ncols) {
            dst[row * ncols + col] = dst_row[col];
        }
    }
}

static void diag_mask_inf_f32(const float * x, float * dst, const int ncols, const int rows_per_channel, const int n_past,
                              const sycl::nd_item<3> &item_ct1) {
    const int col = item_ct1.get_local_range(1) * item_ct1.get_group(1) +
                    item_ct1.get_local_id(1);
    const int row = item_ct1.get_local_range(2) * item_ct1.get_group(2) +
                    item_ct1.get_local_id(2);

    if (col >= ncols) {
        return;
    }

    const int i = row*ncols + col;
    //dst[i] = col > (n_past + row % rows_per_channel) ? -INFINITY : x[i];
    //dst[i] = x[i] - (col > n_past + row % rows_per_channel) * INT_MAX; // equivalent within rounding error but slightly faster on GPU
    dst[i] = x[i] - (col > n_past + row % rows_per_channel) * FLT_MAX;
}

static void scale_f32(const float * x, float * dst, const float scale, const float bias, const int k,
                      const sycl::nd_item<3> &item_ct1) {
    const int i = item_ct1.get_local_range(2) * item_ct1.get_group(2) +
                  item_ct1.get_local_id(2);

    if (i >= k) {
        return;
    }

    dst[i] = scale * x[i] + bias;
}


static void ggml_mul_mat_p021_f16_f32_sycl(const void *vx, const float *y,
                                           float *dst, const int ncols_x,
                                           const int nrows_x,
                                           const int nchannels_x,
                                           const int nchannels_y,
                                           queue_ptr stream) {

    const sycl::range<3> block_nums(nchannels_y, nrows_x, 1);
    const sycl::range<3> block_dims(1, 1, WARP_SIZE);
    {
        dpct::has_capability_or_fail(stream->get_device(),
                                     {sycl::aspect::fp16});

        stream->parallel_for(
            sycl::nd_range<3>(block_nums * block_dims, block_dims),
            [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                mul_mat_p021_f16_f32(vx, y, dst, ncols_x, nrows_x, nchannels_x,
                                     nchannels_y, item_ct1);
            });
    }
}

static void ggml_mul_mat_vec_nc_f16_f32_sycl(
    const void *vx, const float *y, float *dst, const int ncols_x,
    const int nrows_x, const int row_stride_x, const int nchannels_x,
    const int nchannels_y, const int channel_stride_x, const int channel_stride_y, queue_ptr stream) {

    const sycl::range<3> block_nums(nchannels_y, nrows_x, 1);
    const sycl::range<3> block_dims(1, 1, WARP_SIZE);
    {
        dpct::has_capability_or_fail(stream->get_device(),
                                     {sycl::aspect::fp16});

        stream->parallel_for(
            sycl::nd_range<3>(block_nums * block_dims, block_dims),
            [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                mul_mat_vec_nc_f16_f32(vx, y, dst, ncols_x, nrows_x,
                                       row_stride_x, channel_stride_x, channel_stride_y,
                                       nchannels_y / nchannels_x, item_ct1);
            });
    }
}



static void scale_f32_sycl(const float *x, float *dst, const float scale, const float bias,
                           const int k, queue_ptr stream) {
    const int num_blocks = (k + SYCL_SCALE_BLOCK_SIZE - 1) / SYCL_SCALE_BLOCK_SIZE;
    stream->parallel_for(
        sycl::nd_range<3>(sycl::range<3>(1, 1, num_blocks) *
                              sycl::range<3>(1, 1, SYCL_SCALE_BLOCK_SIZE),
                          sycl::range<3>(1, 1, SYCL_SCALE_BLOCK_SIZE)),
        [=](sycl::nd_item<3> item_ct1) {
            scale_f32(x, dst, scale, bias, k, item_ct1);
        });
}


static void sum_rows_f32_sycl(const float *x, float *dst, const int ncols,
                              const int nrows, queue_ptr stream) {
    const sycl::range<3> block_dims(1, 1, WARP_SIZE);
    const sycl::range<3> block_nums(1, nrows, 1);
    stream->parallel_for(sycl::nd_range<3>(block_nums * block_dims, block_dims),
                         [=](sycl::nd_item<3> item_ct1)
                             [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                                 k_sum_rows_f32(x, dst, ncols, item_ct1);
                             });
}

static int next_power_of_2(int x) {
    int n = 1;
    while (n < x) {
        n *= 2;
    }
    return n;
}

static void init_argsort_indices_padded(
        int * idx,
        const int nrows,
        const int ncols_pad,
        const sycl::nd_item<1> & item_ct1) {
    const size_t gid = item_ct1.get_local_range(0) * item_ct1.get_group(0) + item_ct1.get_local_id(0);
    const size_t total = (size_t) nrows * (size_t) ncols_pad;

    if (gid >= total) {
        return;
    }

    idx[gid] = (int) (gid % (size_t) ncols_pad);
}

template <ggml_sort_order order>
static void argsort_f32_i32_global_pass(const float *            x,
                                        int *                    idx,
                                        const int                ncols,
                                        const int                nrows,
                                        const int                ncols_pad,
                                        const int                j,
                                        const int                k,
                                        const sycl::nd_item<1> & item_ct1) {
    const size_t gid   = item_ct1.get_local_range(0) * item_ct1.get_group(0) + item_ct1.get_local_id(0);
    const size_t total = (size_t) nrows * (size_t) ncols_pad;

    if (gid >= total) {
        return;
    }

    const int row = (int) (gid / (size_t) ncols_pad);
    const int col = (int) (gid % (size_t) ncols_pad);
    const int ixj = col ^ j;

    if (ixj <= col || ixj >= ncols_pad) {
        return;
    }

    const size_t base  = (size_t) row * (size_t) ncols_pad;
    const size_t pos_a = base + (size_t) col;
    const size_t pos_b = base + (size_t) ixj;

    const int a = idx[pos_a];
    const int b = idx[pos_b];

    bool do_swap = false;

    if ((col & k) == 0) {
        if (a >= ncols ||
            (b < ncols &&
             (order == GGML_SORT_ORDER_ASC ?
                  x[(size_t) row * (size_t) ncols + (size_t) a] > x[(size_t) row * (size_t) ncols + (size_t) b] :
                  x[(size_t) row * (size_t) ncols + (size_t) a] < x[(size_t) row * (size_t) ncols + (size_t) b]))) {
            do_swap = true;
        }
    } else {
        if (b >= ncols ||
            (a < ncols &&
             (order == GGML_SORT_ORDER_ASC ?
                  x[(size_t) row * (size_t) ncols + (size_t) a] < x[(size_t) row * (size_t) ncols + (size_t) b] :
                  x[(size_t) row * (size_t) ncols + (size_t) a] > x[(size_t) row * (size_t) ncols + (size_t) b]))) {
            do_swap = true;
        }
    }

    if (do_swap) {
        idx[pos_a] = b;
        idx[pos_b] = a;
    }
}

static void copy_argsort_indices_unpadded(const int *              idx_padded,
                                          int *                    dst,
                                          const int                nrows,
                                          const int                ncols,
                                          const int                ncols_pad,
                                          const sycl::nd_item<1> & item_ct1) {
    const size_t gid   = item_ct1.get_local_range(0) * item_ct1.get_group(0) + item_ct1.get_local_id(0);
    const size_t total = (size_t) nrows * (size_t) ncols;

    if (gid >= total) {
        return;
    }

    const int row = (int) (gid / (size_t) ncols);
    const int col = (int) (gid % (size_t) ncols);

    dst[(size_t) row * (size_t) ncols + (size_t) col] = idx_padded[(size_t) row * (size_t) ncols_pad + (size_t) col];
}

static void argsort_f32_i32_sycl(const float *x, int *dst, const int ncols,
                                 const int nrows, ggml_sort_order order,
                                 queue_ptr stream, int device, ggml_sycl_pool & pool) {
    // bitonic sort requires ncols to be power of 2
    const int ncols_pad = next_power_of_2(ncols);
    const size_t shared_mem = (size_t) ncols_pad * sizeof(int);
    const size_t smpbo = ggml_sycl_info().devices[device].smpbo;

    if (shared_mem > smpbo) {
        ggml_sycl_pool_alloc<int> idx_padded_alloc(pool, (size_t) nrows * (size_t) ncols_pad);
        int *                     idx_padded = idx_padded_alloc.get();

        constexpr size_t block_size     = 256;
        const size_t     total_padded   = (size_t) nrows * (size_t) ncols_pad;
        const size_t     nblocks_padded = (total_padded + block_size - 1) / block_size;

        stream->parallel_for(
            sycl::nd_range<1>(sycl::range<1>(nblocks_padded * block_size), sycl::range<1>(block_size)),
            [=](sycl::nd_item<1> item_ct1) { init_argsort_indices_padded(idx_padded, nrows, ncols_pad, item_ct1); });

        for (int k = 2; k <= ncols_pad; k *= 2) {
            for (int j = k / 2; j > 0; j /= 2) {
                if (order == GGML_SORT_ORDER_ASC) {
                    stream->parallel_for(
                        sycl::nd_range<1>(sycl::range<1>(nblocks_padded * block_size), sycl::range<1>(block_size)),
                        [=](sycl::nd_item<1> item_ct1) {
                            argsort_f32_i32_global_pass<GGML_SORT_ORDER_ASC>(x, idx_padded, ncols, nrows, ncols_pad, j,
                                                                             k, item_ct1);
                        });
                } else if (order == GGML_SORT_ORDER_DESC) {
                    stream->parallel_for(
                        sycl::nd_range<1>(sycl::range<1>(nblocks_padded * block_size), sycl::range<1>(block_size)),
                        [=](sycl::nd_item<1> item_ct1) {
                            argsort_f32_i32_global_pass<GGML_SORT_ORDER_DESC>(x, idx_padded, ncols, nrows, ncols_pad, j,
                                                                              k, item_ct1);
                        });
                } else {
                    GGML_ABORT("invalid sort order");
                }
            }
        }

        const size_t total   = (size_t) nrows * (size_t) ncols;
        const size_t nblocks = (total + block_size - 1) / block_size;
        stream->parallel_for(sycl::nd_range<1>(sycl::range<1>(nblocks * block_size), sycl::range<1>(block_size)),
                             [=](sycl::nd_item<1> item_ct1) {
                                 copy_argsort_indices_unpadded(idx_padded, dst, nrows, ncols, ncols_pad, item_ct1);
                             });

        return;
    }

    int nth = 1;
    int max_block_size = ggml_sycl_info().max_work_group_sizes[device];
    while (nth < ncols_pad && nth < max_block_size)
        nth *= 2;
    if (nth > max_block_size)
        nth = max_block_size;

    const int tasks_per_thread = ncols_pad / nth;

    const sycl::range<3> block_dims(1, 1, nth);
    const sycl::range<3> block_nums(1, nrows, 1);

    if (order == GGML_SORT_ORDER_ASC) {
        stream->submit([&](sycl::handler &cgh) {
            sycl::local_accessor<uint8_t, 1> dpct_local_acc_ct1(
                sycl::range<1>(shared_mem), cgh);

            cgh.parallel_for(
                sycl::nd_range<3>(block_nums * block_dims, block_dims),
                [=](sycl::nd_item<3> item_ct1) {
                    k_argsort_f32_i32<GGML_SORT_ORDER_ASC>(
                        x, dst, ncols, ncols_pad, tasks_per_thread, item_ct1,
                        dpct_local_acc_ct1
                            .get_multi_ptr<sycl::access::decorated::no>()
                            .get());
                });
        });
    } else if (order == GGML_SORT_ORDER_DESC) {
        stream->submit([&](sycl::handler &cgh) {
            sycl::local_accessor<uint8_t, 1> dpct_local_acc_ct1(
                sycl::range<1>(shared_mem), cgh);

            cgh.parallel_for(
                sycl::nd_range<3>(block_nums * block_dims, block_dims),
                [=](sycl::nd_item<3> item_ct1) {
                    k_argsort_f32_i32<GGML_SORT_ORDER_DESC>(
                        x, dst, ncols, ncols_pad, tasks_per_thread, item_ct1,
                        dpct_local_acc_ct1
                            .get_multi_ptr<sycl::access::decorated::no>()
                            .get());
                });
        });
    } else {
        GGML_ABORT("fatal error");
    }
}

// Scan and block merge, shared by every launch shape below so a partitioned row uses the
// same insertion order as an unpartitioned one.
//
// src_map  != nullptr: report src_map[col] instead of col, so a merge pass can carry the
//                      original column index through.
// out_vals != nullptr: also emit the k winning values, for a later merge pass.
// swap01:              emit in the output order the single-pass path uses.
static void top_k_scan_merge_f32(
    const float *   src_vals,
    const int32_t * src_map,
    const int       begin,
    const int       end,
    const int       k,
    const int       block_size,
    float *         shared_vals,
    int *           shared_idx,
    float *         out_vals,
    int32_t *       out_idx,
    const bool      swap01,
    const sycl::nd_item<1> & item_ct1
) {
    const int tid = item_ct1.get_local_id(0);

    // The running top-k lives in SLM (shared local memory) rather than a private array:
    // an array indexed by a runtime position cannot be register-allocated, so a private
    // one lands in scratch, i.e. device memory, and insertion is this kernel's dominant
    // cost.
    //
    // Lane-strided (lv[i * block_size]) rather than lane-blocked (lv[i]) so a given i is
    // contiguous across lanes; a k-strided layout would put every lane of a shift step in
    // the same SLM bank.
    float * lv = shared_vals + tid;
    int * li = shared_idx + tid;

    for (int i = 0; i < k; i++) {
        lv[i * block_size] = -FLT_MAX;
        li[i * block_size] = -1;
    }

    // The k-th best, cached in a register. The reject test is taken for the large
    // majority of elements scanned, and in that case touches no memory.
    float kth = -FLT_MAX;

    for (int col = begin + tid; col < end; col += block_size) {
        float val = src_vals[col];

        if (val > kth) {
            int pos = k - 1;
            while (pos > 0 && val > lv[(pos - 1) * block_size]) {
                pos--;
            }

            for (int i = k - 1; i > pos; i--) {
                lv[i * block_size] = lv[(i - 1) * block_size];
                li[i * block_size] = li[(i - 1) * block_size];
            }
            lv[pos * block_size] = val;
            li[pos * block_size] = src_map ? src_map[col] : col;

            kth = lv[(k - 1) * block_size];
        }
    }

    item_ct1.barrier(sycl::access::fence_space::local_space);

    if (tid != 0) {
        return;
    }

    // Same treatment for the merge accumulator, past the per-lane region.
    float * fv = shared_vals + (size_t) k * block_size;
    int * fi = shared_idx + (size_t) k * block_size;

    for (int i = 0; i < k; i++) {
        fv[i] = -FLT_MAX;
        fi[i] = -1;
    }

    float fkth = -FLT_MAX;

    // Candidates are visited in the same (t, i) order as before, so tie-breaking is
    // unchanged.
    for (int t = 0; t < block_size; t++) {
        for (int i = 0; i < k; i++) {
            float val = shared_vals[i * block_size + t];

            if (val <= fkth) {
                // Lane t's list is sorted descending, so once one of its entries loses
                // to the k-th best, every later entry loses too. fkth only rises, so
                // that stays true for the rest of the merge. This turns the merge from
                // block_size*k steps into roughly block_size plus the candidates
                // accepted.
                break;
            }

            int idx = shared_idx[i * block_size + t];

            int pos = k - 1;
            while (pos > 0 && val > fv[pos - 1]) {
                pos--;
            }

            for (int j = k - 1; j > pos; j--) {
                fv[j] = fv[j - 1];
                fi[j] = fi[j - 1];
            }
            fv[pos] = val;
            fi[pos] = idx;

            fkth = fv[k - 1];
        }
    }

    if (out_vals) {
        for (int i = 0; i < k; i++) {
            out_vals[i] = fv[i];
        }
    }

    for (int i = 0; i < k; i++) {
        out_idx[i] = fi[i];
    }

    if (swap01 && k > 1) {
        int32_t temp = out_idx[0];
        out_idx[0] = out_idx[1];
        out_idx[1] = temp;
    }
}

static void top_k_f32_sycl(
    ggml_backend_sycl_context & ctx,
    const float * src,
    int32_t * dst_indices,
    const int64_t ncols,
    const int64_t nrows,
    const int k,
    dpct::queue_ptr main_stream
) {
    // A row is scanned by exactly one work-group, so a vocabulary-sized row leaves the
    // rest of the device idle. What the scan is short of is memory requests in flight,
    // not bandwidth or per-request latency, so lanes in flight is the lever: split the
    // row across independent work-groups, have each emit its partition's top-k, and
    // merge those nsplit*k candidates in a second launch.
    //
    // split_block trades parallelism against SLM residency. Its cost is
    // (split_block + 1) * k * 8 bytes of SLM per group, so at the k <= 32 ceiling 128
    // lanes need about 33 KB, which leaves a single resident group per Xe-core. Revisit
    // if the supported k ever grows.
    constexpr int split_block = 128;
    constexpr int max_splits  = 128;
    constexpr int min_cols    = 8192;

    int nsplit = 1;
    if (ncols >= min_cols) {
        // A partition is then always >= split_block = 128 columns, hence always more than
        // the k <= 32 ceiling, so no pass is ever padded with -FLT_MAX sentinels.
        const int64_t want = ncols / split_block;
        nsplit = (int) (want > max_splits ? max_splits : want);
    }

    if (nsplit > 1) {
        const int nchunk = (int) ((ncols + nsplit - 1) / nsplit);
        const size_t ncand = (size_t) nrows * nsplit * k;

        ggml_sycl_pool_alloc<float>   part_vals(ctx.pool(), ncand);
        ggml_sycl_pool_alloc<int32_t> part_idx(ctx.pool(), ncand);

        float *   pv = part_vals.get();
        int32_t * pi = part_idx.get();

        const sycl::range<1> block_dims(split_block);

        main_stream->submit([&](sycl::handler &cgh) {
            sycl::local_accessor<float, 1> shared_vals(sycl::range<1>((split_block + 1) * k), cgh);
            sycl::local_accessor<int, 1> shared_idx(sycl::range<1>((split_block + 1) * k), cgh);

            cgh.parallel_for(
                sycl::nd_range<1>(sycl::range<1>(nrows * nsplit) * block_dims, block_dims),
                [=](sycl::nd_item<1> item_ct1) {
                    const int grp = item_ct1.get_group(0);
                    const int row = grp / nsplit;
                    const int part = grp % nsplit;

                    const int begin = part * nchunk;
                    int end = begin + nchunk;
                    if (end > (int) ncols) {
                        end = (int) ncols;
                    }

                    top_k_scan_merge_f32(
                        src + (int64_t) row * ncols, nullptr, begin, end, k, split_block,
                        shared_vals.get_multi_ptr<sycl::access::decorated::no>().get(),
                        shared_idx.get_multi_ptr<sycl::access::decorated::no>().get(),
                        pv + (size_t) grp * k, pi + (size_t) grp * k, false, item_ct1);
                });
        });

        main_stream->submit([&](sycl::handler &cgh) {
            sycl::local_accessor<float, 1> shared_vals(sycl::range<1>((split_block + 1) * k), cgh);
            sycl::local_accessor<int, 1> shared_idx(sycl::range<1>((split_block + 1) * k), cgh);

            cgh.parallel_for(
                sycl::nd_range<1>(sycl::range<1>(nrows) * block_dims, block_dims),
                [=](sycl::nd_item<1> item_ct1) {
                    const int row = item_ct1.get_group(0);
                    const size_t off = (size_t) row * nsplit * k;

                    top_k_scan_merge_f32(
                        pv + off, pi + off, 0, nsplit * k, k, split_block,
                        shared_vals.get_multi_ptr<sycl::access::decorated::no>().get(),
                        shared_idx.get_multi_ptr<sycl::access::decorated::no>().get(),
                        nullptr, dst_indices + (int64_t) row * k, true, item_ct1);
                });
        });

        return;
    }

    const int block_size = 128;

    const sycl::range<1> block_dims(block_size);
    const sycl::range<1> grid_dims(nrows);

    main_stream->submit([&](sycl::handler &cgh) {
        sycl::local_accessor<float, 1> shared_vals(sycl::range<1>((block_size + 1) * k), cgh);
        sycl::local_accessor<int, 1> shared_idx(sycl::range<1>((block_size + 1) * k), cgh);

        cgh.parallel_for(
            sycl::nd_range<1>(grid_dims * block_dims, block_dims),
            [=](sycl::nd_item<1> item_ct1) {
                const int row = item_ct1.get_group(0);

                if (row >= nrows) return;

                top_k_scan_merge_f32(
                    src + (int64_t) row * ncols, nullptr, 0, (int) ncols, k, block_size,
                    shared_vals.get_multi_ptr<sycl::access::decorated::no>().get(),
                    shared_idx.get_multi_ptr<sycl::access::decorated::no>().get(),
                    nullptr, dst_indices + (int64_t) row * k, true, item_ct1);
            });
    });
}

static void argmax_f32_i32_sycl(const float *x, int *dst, const int ncols,
                               const int nrows, queue_ptr stream) {
    const sycl::range<3> block_dims(1, 1, SYCL_ARGMAX_BLOCK_SIZE);
    const sycl::range<3> block_nums(1, nrows, 1);
    const size_t shared_mem = 256 * sizeof(float);

    stream->submit([&](sycl::handler &cgh) {
        sycl::local_accessor<float, 1> shared_data(
            sycl::range<1>(shared_mem/sizeof(float)), cgh);
        sycl::local_accessor<int, 1> shared_indices(
            sycl::range<1>(shared_mem/sizeof(float)), cgh);

        cgh.parallel_for(
            sycl::nd_range<3>(block_nums * block_dims, block_dims),
            [=](sycl::nd_item<3> item_ct1) {
                const int tid = item_ct1.get_local_id(2);
                const int row = item_ct1.get_global_id(1);

                float max_val = -INFINITY;
                int max_idx = -1;

                for (int col = tid; col < ncols; col += 256) {
                    float val = x[row * ncols + col];
                    if (val > max_val) {
                        max_val = val;
                        max_idx = col;
                    }
                }

                shared_data[tid] = max_val;
                shared_indices[tid] = max_idx;
                item_ct1.barrier(sycl::access::fence_space::local_space);

                for (int stride = 256/2; stride > 0; stride >>= 1) {
                    if (tid < stride) {
                        float val1 = shared_data[tid];
                        float val2 = shared_data[tid + stride];
                        if (val2 > val1) {
                            shared_data[tid] = val2;
                            shared_indices[tid] = shared_indices[tid + stride];
                        }
                    }
                    item_ct1.barrier(sycl::access::fence_space::local_space);
                }


                if (tid == 0) {
                    dst[row] = shared_indices[0];
                }
            });
    });
}
static void diag_mask_inf_f32_sycl(const float *x, float *dst,
                                   const int ncols_x, const int nrows_x,
                                   const int rows_per_channel, const int n_past,
                                   queue_ptr stream) {
    const sycl::range<3> block_dims(1, SYCL_DIAG_MASK_INF_BLOCK_SIZE, 1);
    const int block_num_x = (ncols_x + SYCL_DIAG_MASK_INF_BLOCK_SIZE - 1) / SYCL_DIAG_MASK_INF_BLOCK_SIZE;
    const sycl::range<3> block_nums(1, block_num_x, nrows_x);
    stream->parallel_for(sycl::nd_range<3>(block_nums * block_dims, block_dims),
                         [=](sycl::nd_item<3> item_ct1) {
                             diag_mask_inf_f32(x, dst, ncols_x,
                                               rows_per_channel, n_past,
                                               item_ct1);
                         });
}

static dpct::err0 ggml_sycl_cpy_tensor_2d(void *dst,
                                          const struct ggml_tensor *src,
                                          int64_t i3, int64_t i2,
                                          int64_t i1_low, int64_t i1_high,
                                          queue_ptr stream) try {

    dpct::memcpy_direction kind;
    char * src_ptr;
    if (ggml_backend_buffer_is_host(src->buffer)) {
        kind = dpct::host_to_device;
        //GGML_SYCL_DEBUG("%s: Host buffer type src tensor\n", __func__);
        src_ptr = (char *) src->data;
        // GGML_SYCL_DEBUG("ggml_sycl_cpy_tensor_2d  GGML_BACKEND_TYPE_CPU src_ptr %p\n", src_ptr);
    } else if (ggml_backend_buffer_is_sycl(src->buffer)) {
        // If buffer is a SYCL buffer
        //GGML_SYCL_DEBUG("%s: SYCL buffer type src tensor\n", __func__);
        kind    = dpct::device_to_device;
        src_ptr = (char *) src->data;
    } else if (ggml_backend_buffer_is_sycl_split(src->buffer)) {
        /*
        If buffer is a SYCL split buffer
        */
        //GGML_SYCL_DEBUG("%s: Split buffer type src tensor\n", __func__);
        GGML_ASSERT(i1_low == 0 && i1_high == src->ne[1]);
        kind = dpct::device_to_device;
        ggml_tensor_extra_gpu * extra = (ggml_tensor_extra_gpu *) src->extra;
        int id;
        SYCL_CHECK(CHECK_TRY_ERROR(
            id = get_current_device_id()));
        // GGML_SYCL_DEBUG("current device index %d\n", id);
        src_ptr = (char *) extra->data_device[id];
    } else {
        // GGML_SYCL_DEBUG("GGML_ABORT("fatal error")\n");
        GGML_ABORT("fatal error");
    }
    char * dst_ptr = (char *) dst;

    GGML_TENSOR_LOCALS_1(int64_t, ne, src, ne);
    GGML_TENSOR_LOCALS(int64_t, nb, src, nb);
    const enum ggml_type type = src->type;
    const int64_t ts = ggml_type_size(type);
    const int64_t bs = ggml_blck_size(type);
    int64_t i1_diff = i1_high - i1_low;

    const char * x = src_ptr + i1_low*nb1 + i2*nb2 + i3*nb3;
    if (nb0 == ts && nb1 == ts*ne0/bs) {
        // GGML_SYCL_DEBUG("stream->memcpy: dst_ptr=%p, x=%p, size=%lu\n", dst_ptr, x, i1_diff * nb1);
        // return CHECK_TRY_ERROR(stream->memcpy(dst_ptr, x, i1_diff * nb1));
        return CHECK_TRY_ERROR(dpct::async_dpct_memcpy(dst_ptr, x, i1_diff * nb1,
                                    kind, *stream));

    } else if (nb0 == ts) {
        return CHECK_TRY_ERROR(
            dpct::async_dpct_memcpy(dst_ptr, ts * ne0 / bs, x, nb1,
                                    ts * ne0 / bs, i1_diff, kind, *stream));
    } else {
        for (int64_t i1 = 0; i1 < i1_diff; i1++) {
            const void * rx = (const void *) ((const char *) x + i1*nb1);
            void * rd = (void *) (dst_ptr + i1*ts*ne0/bs);
            // pretend the row is a matrix with cols=1
            dpct::err0 r = CHECK_TRY_ERROR(dpct::async_dpct_memcpy(
                rd, ts / bs, rx, nb0, ts / bs, ne0, kind, *stream));
            /*
            DPCT1001:85: The statement could not be removed.
            */
            /*
            DPCT1000:86: Error handling if-stmt was detected but could not be
            rewritten.
            */
            if (r != 0) return r;
        }
        return 0;
    }
}
catch (sycl::exception const &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
  std::exit(1);
}

inline void ggml_sycl_op_mul_mat_sycl(
    ggml_backend_sycl_context & ctx,
    const ggml_tensor *src0, const ggml_tensor *src1, ggml_tensor *dst,
    const char *src0_dd_i, const float *src1_ddf_i, const char *src1_ddq_i,
    float *dst_dd_i, const int64_t row_low, const int64_t row_high,
    const int64_t src1_ncols, const int64_t src1_padded_row_size,
    const queue_ptr &stream) try {

    GGML_ASSERT(src0_dd_i  != nullptr);
    GGML_ASSERT(src1_ddf_i != nullptr);
    GGML_ASSERT(dst_dd_i   != nullptr);

    const int64_t ne00 = src0->ne[0];
    const int64_t ne10 = src1->ne[0];
    GGML_ASSERT(ne00 == ne10);

    const int64_t row_diff = row_high - row_low;

    int id;
    SYCL_CHECK(
        CHECK_TRY_ERROR(id = get_current_device_id()));

    const int64_t ne0 = dst->ne[0]; // used by MKL only
    // the main device has a larger memory buffer to hold the results from all GPUs
    // ldc == nrows of the matrix that cuBLAS writes into
    int ldc = id == ctx.device ? ne0 : row_diff; // used by MKL only

#ifdef GGML_SYCL_F16
    bool use_fp16 = true;  // TODO(Yu) SYCL capability check
#else
    bool use_fp16 = false;
#endif

#if GGML_SYCL_DNNL && defined(GGML_SYCL_HAS_BF16)
    // Fast path for bf16 src0
    if (src0->type == GGML_TYPE_BF16 && g_ggml_sycl_enable_dnn &&
        ggml_sycl_dnnl_has_optimized_gemm(ggml_sycl_get_device()) && ggml_is_contiguous(src0) &&
        row_diff == src0->ne[1]) {
        using bf16_t = sycl::ext::oneapi::bfloat16;
        ggml_sycl_pool_alloc<bf16_t> src1_as_bf16(ctx.pool(), src1_ncols*ne10);
        if (src1->type != GGML_TYPE_BF16) {
            const to_bf16_sycl_t to_bf16_sycl = ggml_get_to_bf16_sycl(src1->type, dst);
            GGML_ASSERT(to_bf16_sycl != nullptr);
            to_bf16_sycl(src1_ddf_i, src1_as_bf16.get(), src1_ncols*ne10, stream);
        } else {
            stream->memcpy(src1_as_bf16.get(), src1_ddf_i, src1_ncols*ne10*sizeof(bf16_t));
        }
        DnnlGemmWrapper::row_gemm(ctx, row_diff, src1_ncols, ne10,
                                  src0_dd_i, DnnlGemmWrapper::to_dt<bf16_t>(),
                                  src1_as_bf16.get(), DnnlGemmWrapper::to_dt<bf16_t>(),
                                  dst_dd_i, DnnlGemmWrapper::to_dt<float>(), stream);
        GGML_UNUSED(dst);
        GGML_UNUSED(src1_ddq_i);
        GGML_UNUSED(src1_padded_row_size);
        return;
    }
#endif

    // dequantize inside the GEMM instead of writing the f16 weights out and reading them back; src1
    // goes in its own type, so there is no separate conversion pass. The A stage
    // decodes the weight bytes in place, so an installed persistent reorder must be
    // visible to it: pass the flag through; types without an SoA A-stage decline
    // inside and this falls through to the reorder-aware conversion route below.
    bool src0_reordered = false;
    if (const auto * src0_extra = static_cast<const ggml_tensor_extra_gpu *>(src0->extra)) {
        src0_reordered = src0_extra->optimized_feature.reorder;
    }
    if (ggml_is_quantized(src0->type) && ggml_is_contiguous(src0) && row_diff == src0->ne[1] &&
        ggml_sycl_fused_dequant_gemm(src0->type, src0_reordered, src0_dd_i, src1_ddf_i, src1->type, ggml_sycl_src1_prec(dst), dst_dd_i,
                                     row_diff, src1_ncols, ne10, ldc, ctx.pool(), stream)) {
        return;
    }

    // the f16 route converts src1 to f16 [TAG_GGML_PREC]
    use_fp16 = use_fp16 && ggml_sycl_src1_f16_ok(dst);
    if ((src0->type == GGML_TYPE_F16 || ggml_is_quantized(src0->type)) && use_fp16 && ggml_is_contiguous(src0) &&
        row_diff == src0->ne[1] && dst->op_params[0] == GGML_PREC_DEFAULT) {
        ggml_sycl_pool_alloc<sycl::half> src1_as_f16(ctx.pool());
        if (src1->type != GGML_TYPE_F16) {
            scope_op_debug_print scope_dbg_print(__func__, "/to_fp16_sycl", dst, /*num_src=*/2,
                                                 " : converting src1 to fp16");
            const to_fp16_sycl_t to_fp16_sycl = ggml_get_to_fp16_sycl(src1->type, dst);
            GGML_ASSERT(to_fp16_sycl != nullptr);
            size_t ne = src1_ncols*ne10;
            src1_as_f16.alloc(ne);
            to_fp16_sycl(src1_ddf_i, src1_as_f16.get(), ne, stream);
        }
        const sycl::half *src1_ptr = src1->type == GGML_TYPE_F16
                ? (const sycl::half *)src1_ddf_i
                : src1_as_f16.get();

        ggml_sycl_pool_alloc<sycl::half> src0_as_f16(ctx.pool());
        if (src0->type != GGML_TYPE_F16) {
            scope_op_debug_print scope_dbg_print(__func__, "/to_fp16_sycl", dst, /*num_src=*/2,
                                                 " : converting src0 to fp16");
            const to_fp16_sycl_t to_fp16_sycl = ggml_get_to_fp16_sycl(src0->type, dst);
            GGML_ASSERT(to_fp16_sycl != nullptr);
            size_t ne = row_diff*ne00;
            src0_as_f16.alloc(ne);
            to_fp16_sycl(src0_dd_i, src0_as_f16.get(), ne, stream);
        }
        const sycl::half *src0_ptr = src0->type == GGML_TYPE_F16
                                         ? (const sycl::half *)src0_dd_i
                                         : src0_as_f16.get();

#if GGML_SYCL_DNNL
        if (g_ggml_sycl_enable_dnn && ggml_sycl_dnnl_has_optimized_gemm(ggml_sycl_get_device())) {
                DnnlGemmWrapper::row_gemm(ctx,row_diff, src1_ncols , ne10, src0_ptr,
                                     DnnlGemmWrapper::to_dt<sycl::half>(), src1_ptr, DnnlGemmWrapper::to_dt<sycl::half>(),
                                      dst_dd_i, DnnlGemmWrapper::to_dt<float>(), stream);
        }
        else
#endif
        {
            const float alpha = 1.0f;
            const float beta  = 0.0f;
            SYCL_CHECK(CHECK_TRY_ERROR(dpct::gemm(
                *stream, oneapi::mkl::transpose::trans,
                oneapi::mkl::transpose::nontrans, row_diff, src1_ncols, ne10,
                &alpha, src0_ptr, dpct::library_data_t::real_half, ne00,
                src1_ptr, dpct::library_data_t::real_half, ne10, &beta,
                dst_dd_i, dpct::library_data_t::real_float, ldc,
                dpct::library_data_t::real_float)));
        }
    } else {
        ggml_sycl_pool_alloc<float> src0_ddq_as_f32(ctx.pool());
        ggml_sycl_pool_alloc<float> src1_ddq_as_f32(ctx.pool());
        if (src0->type != GGML_TYPE_F32) {
            scope_op_debug_print scope_dbg_print(__func__, "/to_fp32_sycl", dst, /*num_src=*/2,
                                                 " : converting src0 to fp32");
            const to_fp32_sycl_t to_fp32_sycl = ggml_get_to_fp32_sycl(src0->type, dst);
            GGML_ASSERT(to_fp32_sycl != nullptr);
            src0_ddq_as_f32.alloc(row_diff*ne00);
            to_fp32_sycl(src0_dd_i, src0_ddq_as_f32.get(), row_diff*ne00, stream);
        }
        if (src1->type != GGML_TYPE_F32) {
            scope_op_debug_print scope_dbg_print(__func__, "/to_fp32_sycl", dst, /*num_src=*/2,
                                                 " : converting src1 to fp32");
            const to_fp32_sycl_t to_fp32_sycl = ggml_get_to_fp32_sycl(src1->type, dst);
            GGML_ASSERT(to_fp32_sycl != nullptr);
            src1_ddq_as_f32.alloc(src1_ncols*ne10);
            to_fp32_sycl(src1_ddf_i, src1_ddq_as_f32.get(), src1_ncols*ne10, stream);
        }
        const float * src0_ddf_i = src0->type == GGML_TYPE_F32 ? (const float *) src0_dd_i : src0_ddq_as_f32.get();
        const float * src1_ddf1_i = src1->type == GGML_TYPE_F32 ? (const float *) src1_ddf_i : src1_ddq_as_f32.get();

        {
#if GGML_SYCL_DNNL
            const int64_t gemm_flops = (int64_t)row_diff * src1_ncols * ne10;
            const bool use_mkl_direct = gemm_flops < 256 * 256 * 256;
            if (g_ggml_sycl_enable_dnn && !use_mkl_direct && ggml_sycl_dnnl_has_optimized_gemm(ggml_sycl_get_device())) {
                DnnlGemmWrapper::row_gemm(ctx, row_diff, src1_ncols, ne10, src0_ddf_i,
                                          DnnlGemmWrapper::to_dt<float>(), src1_ddf1_i, DnnlGemmWrapper::to_dt<float>(),
                                          dst_dd_i, DnnlGemmWrapper::to_dt<float>(), stream);
            }
            else
#endif
            {
                const float alpha = 1.0f;
                const float beta  = 0.0f;
                SYCL_CHECK(CHECK_TRY_ERROR(oneapi::mkl::blas::column_major::gemm(
                    *stream, oneapi::mkl::transpose::trans, oneapi::mkl::transpose::nontrans, row_diff,
                    src1_ncols, ne10, dpct::get_value(&alpha, *stream), src0_ddf_i, ne00, src1_ddf1_i, ne10,
                    dpct::get_value(&beta, *stream), dst_dd_i, ldc)));
            }
        }
    }
    GGML_UNUSED(dst);
    GGML_UNUSED(src1_ddq_i);
    GGML_UNUSED(src1_padded_row_size);
}
catch (sycl::exception const &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
  std::exit(1);
}

inline void ggml_sycl_op_sum(ggml_backend_sycl_context & ctx, ggml_tensor *dst) {
    GGML_ASSERT(dst->src[0]->type == GGML_TYPE_F32);
    GGML_ASSERT( dst->type == GGML_TYPE_F32);
    dpct::queue_ptr main_stream = ctx.stream();
    SYCL_CHECK(ggml_sycl_set_device(ctx.device));
    const float * src0_dd = static_cast<const float *>(dst->src[0]->data);
    float *       dst_dd  = static_cast<float *>(dst->data);

    const int64_t ne = ggml_nelements(dst->src[0]);
    // the kernel indexes in 32-bit
    GGML_ASSERT(ne <= INT32_MAX);

    sum_rows_f32_sycl(src0_dd, dst_dd, ne, 1, main_stream);
}

inline void ggml_sycl_op_sum_rows(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    GGML_ASSERT(dst->src[0]->type == GGML_TYPE_F32);
    GGML_ASSERT( dst->type == GGML_TYPE_F32);
    dpct::queue_ptr main_stream = ctx.stream();
    SYCL_CHECK(ggml_sycl_set_device(ctx.device));
    const float * src0_dd = static_cast<const float *>(dst->src[0]->data);
    float *       dst_dd  = static_cast<float *>(dst->data);

    const int64_t ncols = dst->src[0]->ne[0];
    const int64_t nrows = ggml_nrows(dst->src[0]);
    // the kernel indexes in 32-bit
    GGML_ASSERT(ncols * nrows <= INT32_MAX);

    sum_rows_f32_sycl(src0_dd, dst_dd, ncols, nrows, main_stream);
}

inline void ggml_sycl_op_mean(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    GGML_ASSERT(dst->src[0]->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);

    dpct::queue_ptr main_stream = ctx.stream();
    SYCL_CHECK(ggml_sycl_set_device(ctx.device));

    const float * src0_dd = static_cast<const float *>(dst->src[0]->data);
    float *       dst_dd  = static_cast<float *>(dst->data);

    const int64_t ncols = dst->src[0]->ne[0];
    const int64_t nrows = ggml_nrows(dst->src[0]);
    // the kernel indexes in 32-bit
    GGML_ASSERT(ncols * nrows <= INT32_MAX);

    sum_rows_f32_sycl(src0_dd, dst_dd, ncols, nrows, main_stream);

    main_stream->parallel_for(
        sycl::range<1>(nrows),
        [=](sycl::id<1> row) {
            dst_dd[row] /= ncols;
        }
    );
}


inline void ggml_sycl_op_argsort(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    GGML_ASSERT(dst->src[0]->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_I32);
    dpct::queue_ptr main_stream = ctx.stream();
    SYCL_CHECK(ggml_sycl_set_device(ctx.device));
    const float * src0_dd = static_cast<const float *>(dst->src[0]->data);
    int32_t *       dst_dd  = static_cast<int32_t *>(dst->data);


    const int64_t ncols = dst->src[0]->ne[0];
    const int64_t nrows = ggml_nrows(dst->src[0]);

    enum ggml_sort_order order = (enum ggml_sort_order) dst->op_params[0];

    argsort_f32_i32_sycl(src0_dd, (int *)dst_dd, ncols, nrows, order,
                         main_stream, ctx.device, ctx.pool());
}

static void ggml_sycl_op_top_k(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];

    GGML_ASSERT(src0);
    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_I32);
    GGML_ASSERT(ggml_is_contiguous(src0));

    dpct::queue_ptr main_stream = ctx.stream();
    SYCL_CHECK(ggml_sycl_set_device(ctx.device));

    const float * src0_dd = static_cast<const float *>(src0->data);
    int32_t * dst_dd = static_cast<int32_t *>(dst->data);

    const int k = dst->ne[0];
    const int64_t ncols = src0->ne[0];
    const int64_t nrows = ggml_nrows(src0);

    GGML_ASSERT(k > 0);
    GGML_ASSERT(k <= ncols);

    if (k <= SYCL_TOP_K_SCAN_MERGE_MAX_K) {
        top_k_f32_sycl(ctx, src0_dd, dst_dd, ncols, nrows, k, main_stream);
    } else {
        ggml_sycl_top_k_radix(ctx, src0_dd, dst_dd, ncols, nrows, k, main_stream);
    }
}

inline void ggml_sycl_op_argmax(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    GGML_ASSERT(dst->src[0]->type == GGML_TYPE_F32);
    GGML_ASSERT( dst->type == GGML_TYPE_I32);

    dpct::queue_ptr main_stream = ctx.stream();
    SYCL_CHECK(ggml_sycl_set_device(ctx.device));
    const float * src0_dd = static_cast<const float *>(dst->src[0]->data);
    int32_t *       dst_dd  = static_cast<int32_t *>(dst->data);

    const int64_t ncols = dst->src[0]->ne[0];
    const int64_t nrows = ggml_nrows(dst->src[0]);
    // the kernel indexes in 32-bit
    GGML_ASSERT(ncols * nrows <= INT32_MAX);

    argmax_f32_i32_sycl(src0_dd, dst_dd, ncols, nrows, main_stream);
}

inline void ggml_sycl_op_diag_mask_inf(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    GGML_ASSERT(dst->src[0]->type == GGML_TYPE_F32);
    GGML_ASSERT( dst->type == GGML_TYPE_F32);
    dpct::queue_ptr main_stream = ctx.stream();
    SYCL_CHECK(ggml_sycl_set_device(ctx.device));
    const float * src0_dd = static_cast<const float *>(dst->src[0]->data);
    float *       dst_dd  = static_cast<float *>(dst->data);

    const int64_t ne00 = dst->src[0]->ne[0];
    const int64_t ne01 = dst->src[0]->ne[1];
    const int nrows0 = ggml_nrows(dst->src[0]);
    // the kernel indexes in 32-bit
    GGML_ASSERT(ne00 * nrows0 <= INT32_MAX);

    const int n_past = ((int32_t *) dst->op_params)[0];

    diag_mask_inf_f32_sycl(src0_dd, dst_dd, ne00, nrows0, ne01, n_past, main_stream);
}

static void tri_f32_sycl(
    const float * src,
    float * dst,
    const int64_t ne0,
    const int64_t ne1,
    const int64_t ne2,
    const int64_t ne3,
    const ggml_tri_type ttype,
    dpct::queue_ptr main_stream
) {
    const size_t total = (size_t) ne0 * (size_t) ne1 * (size_t) ne2 * (size_t) ne3;

    main_stream->parallel_for(sycl::range<1>(total), [=](sycl::id<1> tid) {
        const int64_t idx = (int64_t) tid[0];

        const int64_t i0 = idx % ne0;
        const int64_t t1 = idx / ne0;
        const int64_t i1 = t1 % ne1;

        bool keep = false;
        switch (ttype) {
            case GGML_TRI_TYPE_LOWER:      keep = (i0 <  i1); break;
            case GGML_TRI_TYPE_LOWER_DIAG: keep = (i0 <= i1); break;
            case GGML_TRI_TYPE_UPPER:      keep = (i0 >  i1); break;
            case GGML_TRI_TYPE_UPPER_DIAG: keep = (i0 >= i1); break;
            default: keep = false; break;
        }

        dst[idx] = keep ? src[idx] : 0.0f;
    });
}

static void ggml_sycl_op_tri(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];
    GGML_ASSERT(src0);

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type  == GGML_TYPE_F32);
    GGML_ASSERT(ggml_is_contiguous(src0));
    GGML_ASSERT(ggml_is_contiguous(dst));
    GGML_ASSERT(ggml_are_same_shape(src0, dst));

    dpct::queue_ptr main_stream = ctx.stream();
    SYCL_CHECK(ggml_sycl_set_device(ctx.device));

    const float * src0_dd = static_cast<const float *>(src0->data);
    float *       dst_dd  = static_cast<float *>(dst->data);

    const ggml_tri_type ttype = (ggml_tri_type) ggml_get_op_params_i32(dst, 0);

    const int64_t ne0 = src0->ne[0];
    const int64_t ne1 = src0->ne[1];
    const int64_t ne2 = src0->ne[2];
    const int64_t ne3 = src0->ne[3];

    tri_f32_sycl(src0_dd, dst_dd, ne0, ne1, ne2, ne3, ttype, main_stream);
}


inline void ggml_sycl_op_scale(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    GGML_ASSERT(dst->src[0]->type == GGML_TYPE_F32);
    GGML_ASSERT( dst->type == GGML_TYPE_F32);
    dpct::queue_ptr main_stream = ctx.stream();
    SYCL_CHECK(ggml_sycl_set_device(ctx.device));
    const float * src0_dd = static_cast<const float *>(dst->src[0]->data);
    float *       dst_dd  = static_cast<float *>(dst->data);

    float scale;
    float bias;
    memcpy(&scale, (float *) dst->op_params + 0, sizeof(float));
    memcpy(&bias,  (float *) dst->op_params + 1, sizeof(float));

    // the kernel indexes in 32-bit
    GGML_ASSERT(ggml_nelements(dst->src[0]) <= INT32_MAX);

    scale_f32_sycl(src0_dd, dst_dd, scale, bias, ggml_nelements(dst->src[0]), main_stream);
    /*
    DPCT1010:87: SYCL uses exceptions to report errors and does not use the
    error codes. The call was replaced with 0. You need to rewrite this code.
    */
    SYCL_CHECK(0);
}

static void ggml_sycl_set_peer_access(const int n_tokens, [[maybe_unused]] int main_device) {
    static bool peer_access_enabled = false;

    const bool enable_peer_access = n_tokens <= GGML_SYCL_PEER_MAX_BATCH_SIZE;

    if (peer_access_enabled == enable_peer_access) {
        return;
    }

#ifdef NDEBUG
    for (int i = 0; i < ggml_sycl_info().device_count; ++i) {
        SYCL_CHECK(ggml_sycl_set_device(i));
    }

    for (int i = 0; i < ggml_sycl_info().device_count; ++i) {
        SYCL_CHECK(ggml_sycl_set_device(i));

        for (int id_other = 0; id_other < ggml_sycl_info().device_count; ++id_other) {
            if (i == id_other) {
                continue;
            }
            if (i != main_device && id_other != main_device) {
                continue;
            }

            // int can_access_peer;
            // SYCL_CHECK(syclDeviceCanAccessPeer(&can_access_peer, id, id_other));
            // if (can_access_peer) {
            //     if (enable_peer_access) {
            //         SYCL_CHECK(syclDeviceEnablePeerAccess(id_other, 0));
            //     } else {
            //         SYCL_CHECK(syclDeviceDisablePeerAccess(id_other));
            //     }
            // }
        }
    }
#endif // NDEBUG

    peer_access_enabled = enable_peer_access;
}

template <template <int> typename quantize_f>
static void ggml_sycl_op_mul_mat(ggml_backend_sycl_context & ctx, const ggml_tensor *src0,
                                 const ggml_tensor *src1, ggml_tensor *dst,
                                 ggml_sycl_op_mul_mat_t op) try {

    GGML_TENSOR_LOCALS(int64_t, ne0, src0, ne);

    GGML_TENSOR_LOCALS(int64_t, ne1, src1, ne);
    const int64_t nrows1 = ggml_nrows(src1);

    GGML_ASSERT(ne03 == ne13);

    const int64_t ne0 = dst->ne[0];
    const int64_t ne1 = dst->ne[1];

    const int nb2 = dst->nb[2];
    const int nb3 = dst->nb[3];

    GGML_ASSERT(!ggml_backend_buffer_is_sycl_split(dst->buffer));
    GGML_ASSERT(!ggml_backend_buffer_is_sycl_split(src1->buffer));

    GGML_ASSERT(ne12 >= ne02 && ne12 % ne02 == 0);

    const int64_t i02_divisor = ne12 / ne02;

    const size_t src0_ts = ggml_type_size(src0->type);
    const size_t src0_bs = ggml_blck_size(src0->type);
    const size_t q8_1_ts = sizeof(block_q8_1);
    const size_t q8_1_bs = QK8_1;

    ggml_tensor_extra_gpu * src0_extra = (ggml_tensor_extra_gpu *) src0->extra;

    const bool src0_is_contiguous = ggml_is_contiguous(src0);
    const bool src1_is_contiguous = ggml_is_contiguous(src1);

    int64_t src1_padded_col_size = GGML_PAD(ne10, MATRIX_ROW_PADDING);

    const bool split = ggml_backend_buffer_is_sycl_split(src0->buffer);
    GGML_ASSERT(!(split && ne02 > 1));
    GGML_ASSERT(!(split && ne03 > 1));
    GGML_ASSERT(!(split && ne02 < ne12));

    std::array<float, GGML_SYCL_MAX_DEVICES> tensor_split;
    if (split) {
        // TODO: check that src0->buffer->buft is a split buffer type, replace GGML_BACKEND_TYPE_GPU_SPLIT check
        // GGML_ASSERT(src0->buffer != nullptr && src0->buffer->buft == ...);
        ggml_backend_sycl_split_buffer_type_context * buft_ctx = (ggml_backend_sycl_split_buffer_type_context *) src0->buffer->buft->context;
        tensor_split = buft_ctx->tensor_split;
    }

    struct dev_data {
        ggml_sycl_pool_alloc<char> src0_dd_alloc;
        ggml_sycl_pool_alloc<float> src1_ddf_alloc;
        ggml_sycl_pool_alloc<char> src1_ddq_alloc;
        ggml_sycl_pool_alloc<float> dst_dd_alloc;

        char *src0_dd = nullptr;
        float *src1_ddf = nullptr; // float
        char *src1_ddq = nullptr;  // q8_1
        float *dst_dd = nullptr;

        int64_t row_low;
        int64_t row_high;
    };

    dev_data dev[GGML_SYCL_MAX_DEVICES];

    int used_devices = 0;
    queue_ptr main_stream = ctx.stream();

    for (int i = 0; i < ggml_sycl_info().device_count; ++i) {
        // by default, use all rows
        dev[i].row_low  = 0;
        dev[i].row_high = ne01;

        // for multi GPU, get the row boundaries from tensor split
        // and round to mul_mat_q tile sizes
        if (split) {
            const int64_t rounding = get_row_rounding(src0->type, tensor_split);

            if (i != 0) {
                dev[i].row_low  = ne01*tensor_split[i];
                if (dev[i].row_low < ne01) {
                    dev[i].row_low -= dev[i].row_low % rounding;
                }
            }

            if (i != ggml_sycl_info().device_count - 1) {
                dev[i].row_high  = ne01*tensor_split[i + 1];
                if (dev[i].row_high < ne01) {
                    dev[i].row_high -= dev[i].row_high % rounding;
                }
            }
        }
    }

    constexpr bool quantize_enabled = !std::is_same_v<quantize_f<QK8_1 / WARP_SIZE>,
                                                      no_quantize_q8_1<QK8_1 / WARP_SIZE>>;
    for (int i = 0; i < ggml_sycl_info().device_count; ++i) {
        if ((!split && i != ctx.device) || dev[i].row_low == dev[i].row_high) {
            continue;
        }

        used_devices++;

        const bool src1_on_device = i == ctx.device;
        const bool  dst_on_device = i == ctx.device;

        ggml_sycl_set_device(i);
        queue_ptr stream = ctx.stream(i, 0);

        if (src0_is_contiguous) {
            // a split tensor keeps its rows in the per-device slices of the tensor extra
            dev[i].src0_dd = split ? (char *) src0_extra->data_device[i] : (char *) src0->data;
        } else {
            dev[i].src0_dd = dev[i].src0_dd_alloc.alloc(ctx.pool(i), ggml_nbytes(src0));
        }

        if (src1_on_device && src1_is_contiguous) {
            dev[i].src1_ddf = (float *) src1->data;
        } else {
            dev[i].src1_ddf = dev[i].src1_ddf_alloc.alloc(ctx.pool(i), ggml_nelements(src1));
        }

        if constexpr(quantize_enabled) {
            dev[i].src1_ddq = dev[i].src1_ddq_alloc.alloc(ctx.pool(i), nrows1*src1_padded_col_size*q8_1_ts/q8_1_bs);

            if (src1_on_device && src1_is_contiguous) {
                scope_op_debug_print scope_dbg_print(__func__, "/quantize_row_q8_1_sycl", dst,
                                                     /*num_src=*/2, " : converting src1 to Q8_1");
                try {
                    quantize_row_q8_1_sycl<quantize_f>(dev[i].src1_ddf, dev[i].src1_ddq, ne10, nrows1, src1_padded_col_size, stream);
                } catch (sycl::exception const &exc) {
                    std::cerr << "Quantize_row_q8_1_sycl error" << exc.what() << "Exception caught at file:" << __FILE__
                              << ", line:" << __LINE__ << std::endl;
                    std::exit(1);
                }
            }
        }

        if (dst_on_device) {
            dev[i].dst_dd = (float *) dst->data;
        } else {
            const size_t size_dst_ddf = split ? (dev[i].row_high - dev[i].row_low)*ne1 : ggml_nelements(dst);
            dev[i].dst_dd = dev[i].dst_dd_alloc.alloc(ctx.pool(i), size_dst_ddf);
        }
    }

    // if multiple devices are used they need to wait for the main device
    // here an event is recorded that signals that the main device has finished calculating the input data
    if (split && used_devices > 1) {
        ggml_sycl_set_device(ctx.device);
        SYCL_CHECK(CHECK_TRY_ERROR(
            *src0_extra->events[ctx.device][0] =
                ctx.stream()->ext_oneapi_submit_barrier()));
    }

    const int64_t src1_col_stride = split && used_devices > 1 ? MUL_MAT_SRC1_COL_STRIDE : ne11;
    for (int64_t src1_col_0 = 0; src1_col_0 < ne11; src1_col_0 += src1_col_stride) {
        const int64_t is = split ? (src1_col_0/src1_col_stride) % GGML_SYCL_MAX_STREAMS : 0;
        const int64_t src1_ncols = src1_col_0 + src1_col_stride > ne11 ? ne11 - src1_col_0 : src1_col_stride;
        for (int i = 0; i < ggml_sycl_info().device_count; ++i) {
            if ((!split && i != ctx.device) || dev[i].row_low == dev[i].row_high) {
                continue;
            }

            const bool src1_on_device = i == ctx.device;
            const bool  dst_on_device = i == ctx.device;
            const int64_t row_diff = dev[i].row_high - dev[i].row_low;

            ggml_sycl_set_device(i);
            queue_ptr stream = ctx.stream(i, is);

            // wait for main GPU data if necessary
            if (split && (i != ctx.device || is != 0)) {
                SYCL_CHECK(CHECK_TRY_ERROR(stream->ext_oneapi_submit_barrier(
                    {*src0_extra->events[ctx.device][0]})));
            }

            for (int64_t i0 = 0; i0 < ne13*ne12; ++i0) {
                const int64_t i03 = i0 / ne12;
                const int64_t i02 = i0 % ne12;

                const size_t src1_ddq_i_offset = (i0*ne11 + src1_col_0) * src1_padded_col_size*q8_1_ts/q8_1_bs;

                // for split tensors the data begins at i0 == i0_offset_low
                char  *  src0_dd_i =  dev[i].src0_dd + (i0/i02_divisor) * (ne01*ne00*src0_ts)/src0_bs;
                float * src1_ddf_i = (float *) ((char *) dev[i].src1_ddf +
                    (i0*ne11 + src1_col_0) * ne10 * ggml_type_size(src1->type));
                char  * src1_ddq_i = dev[i].src1_ddq +  src1_ddq_i_offset;
                float *   dst_dd_i =   dev[i].dst_dd + (i0*ne1  + src1_col_0) * (dst_on_device ? ne0 : row_diff);

                // the main device memory buffer can be on VRAM scratch, with space for all partial results
                // in that case an offset on dst_ddf_i is needed
                if (i == ctx.device) {
                    dst_dd_i += dev[i].row_low; // offset is 0 if no tensor split
                }

                // copy src0, src1 to device if necessary
                if (src1_is_contiguous) {
                    if (i != ctx.device) {
                        if constexpr (quantize_enabled) {
                            char * src1_ddq_i_source = dev[ctx.device].src1_ddq + src1_ddq_i_offset;
                            SYCL_CHECK(
                                CHECK_TRY_ERROR(stream
                                                    ->memcpy(src1_ddq_i, src1_ddq_i_source,
                                                             src1_ncols * src1_padded_col_size * q8_1_ts / q8_1_bs)
                                                    .wait()));
                        } else {
                            // src1 is not split, on the main device it lives at dev[ctx.device].src1_ddf
                            const char * src1_ddf_i_source = (const char *) dev[ctx.device].src1_ddf +
                                (i0 * ne11 + src1_col_0) * ne10 * ggml_type_size(src1->type);

                            SYCL_CHECK(
                                CHECK_TRY_ERROR(dev2dev_memcpy(i, *stream, ctx.device, *main_stream, src1_ddf_i, src1_ddf_i_source,
                                                               src1_ncols * ne10 * ggml_type_size(src1->type))));
                        }
                    }
                } else {
                    if (src1_on_device) {
                        SYCL_CHECK(ggml_sycl_cpy_tensor_2d(src1_ddf_i, src1, i03, i02, src1_col_0,
                                                           src1_col_0 + src1_ncols, stream));
                    } else {
                        GGML_ABORT("src1 is non-contiguous and not on device");
                    }

                    if constexpr (quantize_enabled) {
                        scope_op_debug_print scope_dbg_print(__func__, "/quantize_row_q8_1_sycl", dst,
                                                             /*num_src=*/2, " : converting src1 to Q8_1");
                        try {
                            quantize_row_q8_1_sycl<quantize_q8_1>(src1_ddf_i, src1_ddq_i, ne10, src1_ncols,
                                                                  src1_padded_col_size, stream);
                        } catch (const sycl::exception & exc) {
                            std::cerr << "Quantize_row_q8_1_sycl error" << exc.what()
                                      << "Exception caught at file:" << __FILE__ << ", line:" << __LINE__ << std::endl;
                            std::exit(1);
                        }
                    }
                }

                if (src1_col_0 == 0 && !src0_is_contiguous && i02 % i02_divisor == 0) {
                    SYCL_CHECK(ggml_sycl_cpy_tensor_2d(src0_dd_i, src0, i03, i02/i02_divisor, dev[i].row_low, dev[i].row_high, stream));
                }
                if (src1->type == GGML_TYPE_F16) {
                    src1_padded_col_size = (i0 * ne11 + src1_col_0) * ne10;
                }
                // do the computation
                SYCL_CHECK(CHECK_TRY_ERROR(op(ctx, src0, src1, dst, src0_dd_i, src1_ddf_i, src1_ddq_i, dst_dd_i,
                    dev[i].row_low, dev[i].row_high, src1_ncols, src1_padded_col_size, stream)));

                // copy dst to host or other device if necessary
                if (!dst_on_device) {
                    void * dst_off_device = dst->data;
                    if (split) {
                        // src0 = weight matrix is saved as a transposed matrix for better memory layout.
                        // dst is NOT transposed.
                        // The outputs of matrix matrix multiplications can therefore NOT simply be concatenated for >1 GPU.
                        // Instead they need to be copied to the correct slice in ne0 = dst row index.
                        // If dst is a vector with ne0 == 1 then you don't have to do this but it still produces correct results.
                        float * dhf_dst_i = (float *) ((char *) dst_off_device + i02*nb2 + i03*nb3);
                        GGML_ASSERT(dst->nb[1] == ne0*sizeof(float));
                        dhf_dst_i += src1_col_0*ne0 + dev[i].row_low;

                        SYCL_CHECK(CHECK_TRY_ERROR(dpct::async_dpct_memcpy(
                            dhf_dst_i, ne0 * sizeof(float), dst_dd_i,
                            row_diff * sizeof(float), row_diff * sizeof(float),
                            src1_ncols, dpct::device_to_device, *stream)));
                    } else {
                        float * dhf_dst_i = (float *) ((char *) dst_off_device + i02*nb2 + i03*nb3);
                        GGML_ASSERT(dst->nb[1] == ne0*sizeof(float));
                        dhf_dst_i += src1_col_0*ne0;
                        SYCL_CHECK(CHECK_TRY_ERROR(
                            stream->memcpy(dhf_dst_i, dst_dd_i,
                                           src1_ncols * ne0 * sizeof(float)).wait()));
                    }
                }

                // add event for the main device to wait on until other device is done
                if (split && (i != ctx.device || is != 0)) {
                    SYCL_CHECK(CHECK_TRY_ERROR(
                        *src0_extra->events[i][is] =
                            stream->ext_oneapi_submit_barrier()));
                }
            }
        }
    }

    // main device waits for all other devices to be finished
    if (split && ggml_sycl_info().device_count > 1) {
        int64_t is_max = (ne11 + MUL_MAT_SRC1_COL_STRIDE - 1) / MUL_MAT_SRC1_COL_STRIDE;
        is_max = is_max <= GGML_SYCL_MAX_STREAMS ? is_max : GGML_SYCL_MAX_STREAMS;

        ggml_sycl_set_device(ctx.device);
        for (int i = 0; i < ggml_sycl_info().device_count; ++i) {
            if (dev[i].row_low == dev[i].row_high) {
                continue;
            }
            for (int64_t is = 0; is < is_max; ++is) {
                SYCL_CHECK(CHECK_TRY_ERROR(
                    ctx.stream()->ext_oneapi_submit_barrier(
                        {*src0_extra->events[i][is]})));
            }
        }
    }
}
catch (sycl::exception const &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
  std::exit(1);
}

static void ggml_sycl_repeat_back(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/1);
    ggml_sycl_op_repeat_back(ctx, dst);
}

static void ggml_sycl_get_rows(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/2);
    ggml_sycl_op_get_rows(ctx, dst);
}

static void ggml_sycl_get_rows_back(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/2);
    ggml_sycl_op_get_rows_back(ctx, dst);
}

static void ggml_sycl_norm(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/1);
    ggml_sycl_op_norm(ctx, dst);
}

static void ggml_sycl_rms_norm(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/1);
    ggml_sycl_op_rms_norm(ctx, dst);
}

static void ggml_sycl_rms_norm_back(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/2);
    ggml_sycl_op_rms_norm_back(ctx, dst);
}

static void ggml_sycl_l2_norm(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/1);
    ggml_sycl_op_l2_norm(ctx, dst);
}

static void ggml_sycl_group_norm(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/1);
    ggml_sycl_op_group_norm(ctx, dst);
}

static void ggml_sycl_mul_mat_vec_p021(ggml_backend_sycl_context & ctx, const ggml_tensor *src0,
                                       const ggml_tensor *src1,
                                       ggml_tensor *dst) try {
    GGML_ASSERT(ggml_is_permuted(src0) && ggml_is_permuted(src1));
    GGML_ASSERT(!ggml_backend_buffer_is_sycl_split(src0->buffer));
    GGML_ASSERT(src0->nb[0] <= src0->nb[1] && src0->nb[2] <= src0->nb[3]); // 0213 permutation
    GGML_ASSERT(src1->nb[0] <= src1->nb[1] && src1->nb[2] <= src1->nb[3]); // 0213 permutation
    GGML_ASSERT(src0->type == GGML_TYPE_F16);
    GGML_ASSERT(src1->type == GGML_TYPE_F32);

    const int64_t ne00 = src0->ne[0];
    const int64_t ne01 = src0->ne[1];
    const int64_t ne02 = src0->ne[2];

    const int64_t ne12 = src1->ne[2];

    SYCL_CHECK(ggml_sycl_set_device(ctx.device));
    queue_ptr main_stream = ctx.stream();

    void  * src0_ddq = src0->data;
    float * src1_ddf = (float *) src1->data;
    float * dst_ddf  = (float *) dst->data;

    ggml_mul_mat_p021_f16_f32_sycl(src0_ddq, src1_ddf, dst_ddf, ne00, ne01, ne02, ne12, main_stream);
}
catch (sycl::exception const &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
  std::exit(1);
}

static void ggml_sycl_mul_mat_vec_nc(ggml_backend_sycl_context & ctx, const ggml_tensor *src0,
                                     const ggml_tensor *src1,
                                     ggml_tensor *dst) try {
    GGML_ASSERT(!ggml_is_transposed(src0));
    GGML_ASSERT(!ggml_is_transposed(src1));
    GGML_ASSERT(!ggml_is_permuted(src0));
    GGML_ASSERT(!ggml_backend_buffer_is_sycl_split(src0->buffer));
    GGML_ASSERT(src0->type == GGML_TYPE_F16);
    GGML_ASSERT(src1->type == GGML_TYPE_F32);
    GGML_ASSERT(src1->ne[1] == 1);
    GGML_ASSERT(src1->ne[3] == 1);

    const int64_t ne00 = src0->ne[0];
    const int64_t ne01 = src0->ne[1];
    const int64_t ne02 = src0->ne[2];

    const int64_t nb01 = src0->nb[1];
    const int64_t nb02 = src0->nb[2];

    const int64_t ne12 = src1->ne[2];
    const int64_t nb11 = src1->nb[1];

    SYCL_CHECK(ggml_sycl_set_device(ctx.device));
    queue_ptr main_stream = ctx.stream();

    void  * src0_ddq = src0->data;
    float * src1_ddf = (float *) src1->data;
    float * dst_ddf  = (float *) dst->data;

    const int64_t row_stride_x = nb01 / sizeof(sycl::half);
    const int64_t channel_stride_x = nb02 / sizeof(sycl::half);
    const int64_t channel_stride_y = nb11 / sizeof(float);

    ggml_mul_mat_vec_nc_f16_f32_sycl(src0_ddq, src1_ddf, dst_ddf, ne00, ne01, row_stride_x, ne02, ne12, channel_stride_x,channel_stride_y, main_stream);
}
catch (sycl::exception const &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
  std::exit(1);
}

static void k_compute_batched_ptrs(const sycl::half * src0_as_f16, const sycl::half * src1_as_f16, void * dst,
                                   const void ** ptrs_src, void ** ptrs_dst, int64_t ne12, int64_t ne13, int64_t ne23,
                                   size_t nb02, size_t nb03, size_t nb12, size_t nb13, size_t nbd2, size_t nbd3,
                                   int64_t r2, int64_t r3, const sycl::nd_item<3> & item_ct1) {
    const int64_t i13 = item_ct1.get_group(2) * item_ct1.get_local_range(2) + item_ct1.get_local_id(2);
    const int64_t i12 = item_ct1.get_group(1) * item_ct1.get_local_range(1) + item_ct1.get_local_id(1);

    if (i13 >= ne13 || i12 >= ne12) {
        return;
    }

    const int64_t i03 = i13 / r3;
    const int64_t i02 = i12 / r2;

    const uint8_t * src0_bytes = reinterpret_cast<const uint8_t *>(src0_as_f16);
    const uint8_t * src1_bytes = reinterpret_cast<const uint8_t *>(src1_as_f16);
    uint8_t *       dst_bytes  = static_cast<uint8_t *>(dst);

    ptrs_src[0 * ne23 + i12 + i13 * ne12] = src0_bytes + i02 * nb02 + i03 * nb03;
    ptrs_src[1 * ne23 + i12 + i13 * ne12] = src1_bytes + i12 * nb12 + i13 * nb13;
    ptrs_dst[0 * ne23 + i12 + i13 * ne12] = dst_bytes + i12 * nbd2 + i13 * nbd3;
}

static void ggml_sycl_mul_mat_batched_sycl(ggml_backend_sycl_context & ctx, const ggml_tensor * src0,
                                           const ggml_tensor * src1, ggml_tensor * dst) try {
    GGML_ASSERT(!ggml_is_transposed(src0));
    GGML_ASSERT(!ggml_is_transposed(src1));
    GGML_ASSERT(!ggml_backend_buffer_is_sycl_split(src0->buffer));
    GGML_ASSERT(src0->type == GGML_TYPE_F16);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);

    GGML_TENSOR_BINARY_OP_LOCALS

    // TODO: see https://github.com/ggml-org/llama.cpp/pull/13155
    // Batched mul_mat requires a rewrite to support both oneDNN and non-contiguous dst
    GGML_ASSERT(ggml_is_contiguous(dst));

    SYCL_CHECK(ggml_sycl_set_device(ctx.device));
    queue_ptr queue = ctx.stream();

    dpct::has_capability_or_fail(queue->get_device(), { sycl::aspect::fp16 });

    const sycl::half * src0_f16 = static_cast<const sycl::half *>(src0->data);
    float *            dst_ddf  = static_cast<float *>(dst->data);

    const sycl::half * src1_f16       = static_cast<const sycl::half *>(src1->data);
#if GGML_SYCL_DNNL
    const size_t       type_size_src0 = ggml_type_size(src0->type);
#endif
    const size_t       type_size_src1 = ggml_type_size(src1->type);

    bool is_src0_cont_2 = ggml_is_contiguous_2(src0);
    bool is_src1_cont_2 = ggml_is_contiguous_2(src1);

    // SRC1 strides
    int64_t                          s11 = nb11 / type_size_src1;
    int64_t                          s12 = nb12 / type_size_src1;
    int64_t                          s13 = nb13 / type_size_src1;
    ggml_sycl_pool_alloc<sycl::half> src1_f16_alloc(ctx.pool());

    // convert src1 to fp16
    if (src1->type != GGML_TYPE_F16) {
        scope_op_debug_print    scope_dbg_print(__func__, "/to_fp16_nc_sycl", dst, /*num_src=*/2,
                                                " : converting src1 to fp16");

#if GGML_SYCL_DNNL
        // iterate tensor dims and find the slowest moving dim and stride
        int last_dim=0;
        int last_str=0;
        size_t largest_str=0;
        for(int i = 0; i< 4; i++){
            // last stride is always the largest
            if(src1->nb[i] == largest_str){
                if(src1->ne[last_dim] == 1){
                    last_str = i;
                    last_dim = i;
                }
            }
            if(src1->nb[i] > largest_str){
                largest_str = src1->nb[i];
                last_str = i;
                last_dim = i;
            }

        }
        // oneDNN handles strided data and does not need overhead of ggml_get_to_fp16_nc_sycl
        const int64_t ne_src1 = src1->nb[last_str] * src1->ne[last_dim] / type_size_src1;
        src1_f16_alloc.alloc(ne_src1);
        const to_fp16_sycl_t to_fp16_sycl = ggml_get_to_fp16_sycl(src1->type, dst);
        GGML_ASSERT(to_fp16_sycl != nullptr);
        to_fp16_sycl(src1_f16, src1_f16_alloc.get(), ne_src1, queue);
# else
        const int64_t ne_src1 = ggml_nelements(src1);
        src1_f16_alloc.alloc(ne_src1);
        const to_fp16_nc_sycl_t to_fp16_nc_sycl = ggml_get_to_fp16_nc_sycl(src1->type);
        GGML_ASSERT(to_fp16_nc_sycl != nullptr);
        to_fp16_nc_sycl(src1_f16, src1_f16_alloc.get(), ne10, ne11, ne12, ne13, s11, s12, s13, queue);
#endif

        src1_f16 = src1_f16_alloc.get();
        s11      = ne10;
        s12      = ne11 * s11;
        s13      = ne12 * s12;

        is_src1_cont_2 = true;
    }

    ggml_sycl_pool_alloc<sycl::half> dst_f16(ctx.pool());

    dpct::library_data_t mkl_compute_type = dpct::library_data_t::real_float;
    dpct::library_data_t mkl_data_type    = dpct::library_data_t::real_float;

    // dst strides
    size_t nbd2 = dst->nb[2];
    size_t nbd3 = dst->nb[3];

    const float alpha_f32 = 1.0f;
    const float beta_f32  = 0.0f;

    const void * alpha = &alpha_f32;
    const void * beta  = &beta_f32;

    GGML_ASSERT(ne12 % ne02 == 0);
    GGML_ASSERT(ne13 % ne03 == 0);
    GGML_ASSERT(ne01 == static_cast<int64_t>(nb1/nb0));
    GGML_ASSERT(ne10 == ne00);

    // broadcast factors
    const int64_t r2 = ne12 / ne02;
    const int64_t r3 = ne13 / ne03;

#if GGML_SYCL_DNNL
    if (g_ggml_sycl_enable_dnn && ggml_sycl_dnnl_has_optimized_gemm(ggml_sycl_get_device())) {
            int64_t str_a0 = nb00 / type_size_src0;
            int64_t str_a1 = nb01 / type_size_src0;
            int64_t str_a2 = nb02 / type_size_src0;

            int64_t str_b0 = nb10 / type_size_src1;
            int64_t str_b1 = nb11 / type_size_src1;
            int64_t str_b2 = nb12 / type_size_src1;

            auto launch_gemm_for_batches = [&ctx, queue](const sycl::half *src0,
                                                const sycl::half *src1, float *dst,
                                                int64_t a0, int64_t a1, int64_t batcha,
                                                int64_t /*b0*/, int64_t b1, int64_t batchb,
                                                int64_t sa0, int64_t sa1, int64_t sa2,
                                                int64_t sb0, int64_t sb1, int64_t sb2,
                                                int64_t sd2) {
                bool supported_broadcast = batchb == batcha ? true
                        : batchb == 1 || batcha == 1        ? true
                                                            : false;
                if (supported_broadcast) {
                    DnnlGemmWrapper::gemm(ctx, a1, b1, a0, src0,
                            DnnlGemmWrapper::to_dt<sycl::half>(), sa0, sa1, sa2, src1,
                            DnnlGemmWrapper::to_dt<sycl::half>(), sb0, sb1, sb2, dst,
                            DnnlGemmWrapper::to_dt<float>(), queue, batcha, batchb);
                } else {
                    // iterate over batches from smaller set of matrices (matrix 0)
                    int64_t batches0 = batcha;
                    int64_t batches1 = batchb;

                    if (batches0 > batches1) {
                        int64_t num_mul_mats = batches1;
                        int64_t sub_batch = batches0 / num_mul_mats;
                        // src0 is batched and bigger, shift and multiply with src1
                        for (int64_t i0 = 0; i0 < num_mul_mats; i0++) {
                            const sycl::half *src0_shifted = src0 + (sa2 * i0 * sub_batch);
                            const sycl::half *src1_shifted = src1 + (sb2 * i0);
                            float *dst_shifted = dst + (sd2 * i0 * sub_batch);
                            DnnlGemmWrapper::gemm(ctx, a1, b1, a0, src0_shifted,
                                    DnnlGemmWrapper::to_dt<sycl::half>(), sa0, sa1, sa2,
                                    src1_shifted, DnnlGemmWrapper::to_dt<sycl::half>(), sb0,
                                    sb1, sb2, dst_shifted, DnnlGemmWrapper::to_dt<float>(),
                                    queue, sub_batch, 1);
                        }
                    } else {
                        int64_t num_mul_mats = batches0;
                        int64_t sub_batch = batches1 / num_mul_mats;
                        // src1 is batched and bigger, shift and multiply with src0
                        for (int64_t i1 = 0; i1 < num_mul_mats; i1++) {
                            const sycl::half *src0_shifted = src0 + (sa2 * i1);
                            const sycl::half *src1_shifted = src1 + (sb2 * i1 * sub_batch);
                            float *dst_shifted = dst + (sd2 * i1 * sub_batch);
                            DnnlGemmWrapper::gemm(ctx, a1, b1, a0, src0_shifted,
                                    DnnlGemmWrapper::to_dt<sycl::half>(), sa0, sa1, sa2,
                                    src1_shifted, DnnlGemmWrapper::to_dt<sycl::half>(), sb0,
                                    sb1, sb2, dst_shifted, DnnlGemmWrapper::to_dt<float>(),
                                    queue, 1, sub_batch);
                        }
                    }
                }
            };

            const bool cont_batches_dim2_a = nb02 * ne02 == nb03;
            const bool cont_batches_dim2_b = nb12 * ne12 == nb13;
            const bool cont_batches_dim3_a = ne02 == 1 && nb02 * ne01 == nb03;
            const bool cont_batches_dim3_b = ne12 == 1 && nb12 * ne11 == nb13;
            if (cont_batches_dim2_a && cont_batches_dim2_b) {
                // A batch is considered contiguous if the dimension 2 is not strided
                int64_t batches0 = ne02 * ne03;
                int64_t batches1 = ne12 * ne13;
                launch_gemm_for_batches(src0_f16, src1_f16, dst_ddf, ne00, ne01, batches0,
                        ne10, ne11, batches1, str_a0, str_a1, str_a2, str_b0, str_b1,
                        str_b2, nb2 / sizeof(float));
            } else if (cont_batches_dim3_a && cont_batches_dim3_b) {
                // This case is similar to the one above with the difference that only the batch in dimension 3 is used and the dimension 2 is of size 1.
                int64_t batches0 = ne02 * ne03;
                int64_t batches1 = ne12 * ne13;
                int64_t str_a3 = nb03 / type_size_src0;
                int64_t str_b3 = nb13 / type_size_src1;
                launch_gemm_for_batches(src0_f16, src1_f16, dst_ddf, ne00, ne01, batches0,
                        ne10, ne11, batches1, str_a0, str_a1, str_a3, str_b0, str_b1,
                        str_b3, nb2 / sizeof(float));
            } else {
                for (int64_t b_a = 0; b_a < ne03; b_a++) {
                    const sycl::half *src0_f16_shifted
                            = src0_f16 + (nb03 * b_a / type_size_src0);
                    const sycl::half *src1_f16_shifted
                            = src1_f16 + (nb13 * b_a / type_size_src1);
                    float *dst_shifted = dst_ddf + (nb3 * b_a / sizeof(float));
                    int64_t batches0 = ne02;
                    int64_t batches1 = ne12;
                    launch_gemm_for_batches(src0_f16_shifted, src1_f16_shifted, dst_shifted,
                            ne00, ne01, batches0, ne10, ne11, batches1, str_a0, str_a1,
                            str_a2, str_b0, str_b1, str_b2, nb2 / sizeof(float));
                }
            }

    }
    else
#endif
    {
        if (r2 == 1 && r3 == 1 && is_src0_cont_2 && is_src1_cont_2) {
            // with a [0, 2, 1, 3] perm. and ne02==1 the matrix strides need to be determined from dim 3:
            const int64_t sma = ne02 == 1 ? nb03/nb00 : nb02/nb00;
            const int64_t smb = ne12 == 1 ? s13       : s12;

            // there is no broadcast and src0, src1 are contiguous across dims 2, 3
            SYCL_CHECK(CHECK_TRY_ERROR(dpct::gemm_batch(*queue, oneapi::mkl::transpose::trans,
                                                        oneapi::mkl::transpose::nontrans, ne01, ne11, ne10, alpha,
                                                        src0_f16, dpct::library_data_t::real_half, nb01 / nb00, sma,
                                                        src1_f16, dpct::library_data_t::real_half, s11, smb, beta, dst_ddf,
                                                        mkl_data_type, ne0, ne1 * ne0, ne12 * ne13, mkl_compute_type)));
        } else {
            const int ne23 = ne12 * ne13;

            ggml_sycl_pool_alloc<const void *>         ptrs_src(ctx.pool(), 2 * ne23);
            ggml_sycl_pool_alloc<void *>               ptrs_dst(ctx.pool(), 1 * ne23);
            ggml_sycl_pool_alloc<matrix_info_t<float>> matrix_info(ctx.host_pool(), 1);

            sycl::range<3> block_dims(1, ne12, ne13);
            queue->submit([&](sycl::handler & cgh) {
                const void ** ptrs_src_get = ptrs_src.get();
                void **       ptrs_dst_get = ptrs_dst.get();
                size_t        nb12_scaled  = src1->type == GGML_TYPE_F16 ? nb12 : s12 * sizeof(sycl::half);
                size_t        nb13_scaled  = src1->type == GGML_TYPE_F16 ? nb13 : s13 * sizeof(sycl::half);
                cgh.parallel_for(sycl::nd_range<3>(block_dims, block_dims), [=](sycl::nd_item<3> item_ct1) {
                    k_compute_batched_ptrs(src0_f16, src1_f16, dst_ddf, ptrs_src_get, ptrs_dst_get, ne12, ne13, ne23, nb02,
                                           nb03, nb12_scaled, nb13_scaled, nbd2, nbd3, r2, r3, item_ct1);
                });
            });

            SYCL_CHECK(CHECK_TRY_ERROR(dpct::gemm_batch(
                *queue, oneapi::mkl::transpose::trans, oneapi::mkl::transpose::nontrans, ne01, ne11, ne10, alpha,
                (const void **) (ptrs_src.get() + 0 * ne23), dpct::library_data_t::real_half, nb01 / nb00,
                (const void **) (ptrs_src.get() + 1 * ne23), dpct::library_data_t::real_half, s11, beta,
                (void **) (ptrs_dst.get() + 0 * ne23), mkl_data_type, ne0, ne23, mkl_compute_type, matrix_info.get())));
        }
    }
} catch (const sycl::exception & exc) {
    std::cerr << exc.what() << "Exception caught at file:" << __FILE__ << ", line:" << __LINE__ << std::endl;
    std::exit(1);
}

enum class mul_mat_algo {
    DMMV         = 0,
    MMVQ         = 1,
    MUL_MAT_SYCL = 2,
};

inline bool ggml_sycl_supports_mmq(enum ggml_type type) {
    // TODO: accuracy issues in MMQ
    GGML_UNUSED(type);
    return false;
}

inline bool ggml_sycl_supports_reorder_mul_mat_sycl(enum ggml_type type) {
    switch (type) {
        case GGML_TYPE_Q1_0:
        case GGML_TYPE_Q4_0:
        case GGML_TYPE_Q8_0:
            return true;
        case GGML_TYPE_Q2_K:
        case GGML_TYPE_Q3_K:
        case GGML_TYPE_Q4_K:
        case GGML_TYPE_Q5_K:
        case GGML_TYPE_Q6_K:
            return !g_ggml_sycl_prioritize_dmmv;
        default:
            return false;
    }
}

inline bool ggml_sycl_supports_reorder_dmmv(enum ggml_type type) {
    switch (type) {
        case GGML_TYPE_Q1_0:
        case GGML_TYPE_Q4_0:
        case GGML_TYPE_Q8_0:
        case GGML_TYPE_Q2_K:
        case GGML_TYPE_Q3_K:
        case GGML_TYPE_Q4_K:
        case GGML_TYPE_Q5_K:
        case GGML_TYPE_Q6_K:
            return true;
        default:
            return false;
    }
}

inline bool ggml_sycl_supports_reorder_mmvq(enum ggml_type type) {
    switch (type) {
        case GGML_TYPE_Q1_0:
        case GGML_TYPE_Q4_0:
        case GGML_TYPE_Q8_0:
        case GGML_TYPE_Q2_K:
        case GGML_TYPE_Q3_K:
        case GGML_TYPE_Q4_K:
        case GGML_TYPE_Q5_K:
        case GGML_TYPE_Q6_K:
        case GGML_TYPE_IQ3_XXS:
        case GGML_TYPE_IQ3_S:
            return true;
        default:
            return false;
    }
}

static bool ggml_sycl_supports_reorder_esimd(enum ggml_type type) {
#ifdef GGML_SYCL_DMMV_HAS_ESIMD
    switch (type) {
        case GGML_TYPE_Q2_K:
        case GGML_TYPE_Q3_K:
        case GGML_TYPE_Q4_K:
        case GGML_TYPE_Q5_K:
        case GGML_TYPE_Q6_K:
        case GGML_TYPE_Q8_0:
            return true;
        default:
            return false;
    }
#else
    GGML_UNUSED(type);
    return false;
#endif
}

static bool ggml_sycl_supports_dmmv(enum ggml_type type) {
    switch (type) {
        case GGML_TYPE_Q1_0:
        case GGML_TYPE_Q4_0:
        case GGML_TYPE_Q4_1:
        case GGML_TYPE_Q5_0:
        case GGML_TYPE_Q5_1:
        case GGML_TYPE_Q8_0:
        case GGML_TYPE_Q2_K:
        case GGML_TYPE_Q3_K:
        case GGML_TYPE_Q4_K:
        case GGML_TYPE_Q5_K:
        case GGML_TYPE_Q6_K:
        case GGML_TYPE_F16:
        case GGML_TYPE_BF16:
            return true;
        default:
            return false;
    }
}

// Helper functions to unify device memory allocation for both async and sync paths
static inline void * sycl_ext_malloc_device(dpct::queue_ptr stream, size_t size) {
    bool use_async = g_ggml_sycl_use_async_mem_op;
#if defined(GGML_SYCL_GRAPH) && SYCL_EXT_ONEAPI_ASYNC_MEMORY_ALLOC
    if (use_async) {
        void * ptr = syclex::async_malloc(*stream, sycl::usm::alloc::device, size);
        ggml_sycl_memtrace_add(GGML_SYCL_MEM_ASYNC, ptr, size);
        return ptr;
    }
#else
    // If async allocation extension is not available, use_async should always be false.
    GGML_ASSERT(!use_async);
#endif
    return ggml_sycl_malloc_device(size, *stream);
}

static inline void sycl_ext_free(dpct::queue_ptr stream, void * ptr) {
    bool use_async = g_ggml_sycl_use_async_mem_op;
#if defined(GGML_SYCL_GRAPH) && SYCL_EXT_ONEAPI_ASYNC_MEMORY_ALLOC
    if (use_async) {
        ggml_sycl_memtrace_del(ptr);
        syclex::async_free(*stream, ptr);
        return;
    }
#else
    // If async allocation extension is not available, use_async should always be false.
    GGML_ASSERT(!use_async);
#endif
    ggml_sycl_free_device(ptr, *stream);
}

// RAII wrapper for temporary reorder buffers with optional host memory fallback.
// When device allocation fails and GGML_SYCL_HOST_MEM_FALLBACK is enabled,
// falls back to host memory so the reorder kernel can still run (over PCIe).
// Device access to host memory requires Linux kernel 6.8+ (Ubuntu 26.04+).
struct sycl_reorder_temp_buffer {
    void *          ptr  = nullptr;
    dpct::queue_ptr stream;

    sycl_reorder_temp_buffer(dpct::queue_ptr stream, size_t size) : stream(stream) {
        ptr = sycl_ext_malloc_device(stream, size);
#ifdef GGML_SYCL_HOST_MEM_FALLBACK
        if (!ptr) {
            ptr = sycl::malloc_host(size, *stream);
            if (ptr) {
                host_fallback = true;
                GGML_LOG_WARN("%s: device alloc of %zu bytes failed, using host memory fallback\n", __func__, size);
            }
        }
#endif
    }

    ~sycl_reorder_temp_buffer() {
        if (!ptr) {
            return;
        }
        if (host_fallback) {
            sycl::free(ptr, *stream);
        } else {
            sycl_ext_free(stream, ptr);
        }
    }

    explicit operator bool() const { return ptr != nullptr; }

    sycl_reorder_temp_buffer(const sycl_reorder_temp_buffer &)            = delete;
    sycl_reorder_temp_buffer & operator=(const sycl_reorder_temp_buffer &) = delete;

private:
    bool host_fallback = false;
};

static bool reorder_qw_q4_0(uint8_t * data_device, const int ncols, const int nrows, size_t size, size_t offset,
                            dpct::queue_ptr stream) {
    sycl_reorder_temp_buffer tmp(stream, size);
    if (!tmp) {
        GGML_LOG_WARN("%s: failed to allocate %zu bytes for reorder temp buffer, skipping reorder\n", __func__, size);
        return false;
    }
    uint8_t * tmp_buf = static_cast<uint8_t *>(tmp.ptr);

    sycl::event copy_event;
    SYCL_CHECK(CHECK_TRY_ERROR(copy_event = stream->memcpy(tmp_buf, data_device, size)));
    if (!g_ggml_sycl_use_async_mem_op) {
        { ggml_sycl_pipe_wait_scope ws_; copy_event.wait(); }
    }

    GGML_ASSERT((size % sizeof(block_q4_0) == 0));
    GGML_ASSERT((offset % sizeof(block_q4_0) == 0));
    int offset_blks = offset / sizeof(block_q4_0);
    auto qs_ptr      = data_device + offset_blks * QK4_0 / 2;
    auto d_ptr = (sycl::half*)(qs_ptr + ncols * nrows / 2) + offset_blks;

    auto reorder_event = stream->parallel_for(
        size / sizeof(block_q4_0),
            [=](auto i) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
            const block_q4_0* x = (const block_q4_0*)tmp_buf;
            const int ib = i;

            for (int j = 0; j < QK4_0/2; j ++)
            {
                *(qs_ptr + ib * QK4_0 / 2 + j) = x[ib].qs[j];
            }
            *(d_ptr + ib) = x[ib].d;
        });
    if (!g_ggml_sycl_use_async_mem_op) {
        { ggml_sycl_pipe_wait_scope ws_; reorder_event.wait_and_throw(); }
    }
    return true;
}

static bool reorder_qw_q8_0(uint8_t * data_device, const int ncols, const int nrows, size_t size, size_t offset,
                            dpct::queue_ptr stream) {
    sycl_reorder_temp_buffer tmp(stream, size);
    if (!tmp) {
        GGML_LOG_WARN("%s: failed to allocate %zu bytes for reorder temp buffer, skipping reorder\n", __func__, size);
        return false;
    }
    uint8_t * tmp_buf = static_cast<uint8_t *>(tmp.ptr);

    sycl::event copy_event;
    SYCL_CHECK(CHECK_TRY_ERROR(copy_event = stream->memcpy(tmp_buf, data_device, size)));
    if (!g_ggml_sycl_use_async_mem_op) {
        { ggml_sycl_pipe_wait_scope ws_; copy_event.wait(); }
    }

    GGML_ASSERT((size % sizeof(block_q8_0) == 0));
    GGML_ASSERT((offset % sizeof(block_q8_0) == 0));
    int offset_blks = offset / sizeof(block_q8_0);
    auto qs_ptr = data_device + offset_blks * QK8_0;
    auto d_ptr = (sycl::half*)(qs_ptr + ncols * nrows) + offset_blks;

    auto reorder_event = stream->parallel_for(
        size / sizeof(block_q8_0),
            [=](auto i) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
            const block_q8_0* x = (const block_q8_0*)tmp_buf;
            const int ib = i;

            for (int j = 0; j < QK8_0; j++)
            {
                *((int8_t*)qs_ptr + ib * QK8_0 + j) = x[ib].qs[j];
            }
            *(d_ptr + ib) = x[ib].d;
        });
    if (!g_ggml_sycl_use_async_mem_op) {
        { ggml_sycl_pipe_wait_scope ws_; reorder_event.wait_and_throw(); }
    }
    return true;
}

// Reorder each Q8_0 expert slice into a self-contained [qs][d] SoA layout.
static bool reorder_qw_q8_0_moe(uint8_t * data_device, size_t expert_bytes, int64_t n_expert, dpct::queue_ptr stream) {
    GGML_ASSERT(expert_bytes % sizeof(block_q8_0) == 0);
    const int    blocks_per_expert = (int) (expert_bytes / sizeof(block_q8_0));
    const size_t total_bytes       = expert_bytes * (size_t) n_expert;

    sycl_reorder_temp_buffer tmp(stream, total_bytes);
    if (!tmp) {
        GGML_LOG_WARN("%s: failed to allocate %zu bytes for reorder temp buffer, skipping reorder\n", __func__, total_bytes);
        return false;
    }
    uint8_t * tmp_buf = static_cast<uint8_t *>(tmp.ptr);

    sycl::event copy_event;
    SYCL_CHECK(CHECK_TRY_ERROR(copy_event = stream->memcpy(tmp_buf, data_device, total_bytes)));
    if (!g_ggml_sycl_use_async_mem_op) {
        copy_event.wait();
    }

    const int total_blocks = blocks_per_expert * (int) n_expert;
    auto reorder_event = stream->parallel_for(total_blocks, [=](auto gb_) {
        const int          gb   = gb_;
        const int          e    = gb / blocks_per_expert;
        const int          ib   = gb % blocks_per_expert;
        const block_q8_0 * x    = (const block_q8_0 *) (tmp_buf + (size_t) e * expert_bytes);
        uint8_t *          base = data_device + (size_t) e * expert_bytes;

        auto * qs_ptr = base;
        auto * d_ptr  = (sycl::half *) (qs_ptr + (size_t) blocks_per_expert * QK8_0);

        for (int j = 0; j < QK8_0; ++j) {
            *((int8_t *) qs_ptr + ib * QK8_0 + j) = x[ib].qs[j];
        }
        d_ptr[ib] = x[ib].d;
    });
    if (!g_ggml_sycl_use_async_mem_op) {
        reorder_event.wait_and_throw();
    }
    return true;
}

static bool reorder_qw_q4_k(uint8_t * data_device, size_t size, size_t offset, dpct::queue_ptr stream) {
    GGML_ASSERT(size % sizeof(block_q4_K) == 0);
    GGML_ASSERT(offset % sizeof(block_q4_K) == 0);

    const int nblocks = size / sizeof(block_q4_K);

    sycl_reorder_temp_buffer tmp(stream, size);
    if (!tmp) {
        GGML_LOG_WARN("%s: failed to allocate %zu bytes for reorder temp buffer, skipping reorder\n", __func__, size);
        return false;
    }
    uint8_t * tmp_buf = static_cast<uint8_t *>(tmp.ptr);

    sycl::event copy_event;
    SYCL_CHECK(CHECK_TRY_ERROR(copy_event = stream->memcpy(tmp_buf, data_device, size)));
    if (!g_ggml_sycl_use_async_mem_op) {
        { ggml_sycl_pipe_wait_scope ws_; copy_event.wait(); }
    }

    auto * qs_ptr     = data_device;
    auto * scales_ptr = qs_ptr + QK_K / 2 * nblocks;
    auto * dm_ptr     = (sycl::half2 *) (scales_ptr + K_SCALE_SIZE * nblocks);

    auto reorder_event = stream->parallel_for(nblocks, [=](auto i) {
        const block_q4_K * x  = (const block_q4_K *) tmp_buf;
        const int          ib = i;

        for (int j = 0; j < QK_K / 2; ++j) {
            qs_ptr[ib * (QK_K / 2) + j] = x[ib].qs[j];
        }

        for (int j = 0; j < K_SCALE_SIZE; ++j) {
            scales_ptr[ib * K_SCALE_SIZE + j] = x[ib].scales[j];
        }

        dm_ptr[ib] = x[ib].dm;
    });
    if (!g_ggml_sycl_use_async_mem_op) {
        { ggml_sycl_pipe_wait_scope ws_; reorder_event.wait_and_throw(); }
    }
    return true;
}

// Reorder each expert slice into a self-contained SoA layout.
static bool reorder_qw_q4_k_moe(uint8_t * data_device, size_t expert_bytes, int64_t n_expert, dpct::queue_ptr stream) {
    GGML_ASSERT(expert_bytes % sizeof(block_q4_K) == 0);
    const int    blocks_per_expert = (int) (expert_bytes / sizeof(block_q4_K));
    const size_t total_bytes       = expert_bytes * (size_t) n_expert;

    sycl_reorder_temp_buffer tmp(stream, total_bytes);
    if (!tmp) {
        GGML_LOG_WARN("%s: failed to allocate %zu bytes for reorder temp buffer, skipping reorder\n", __func__, total_bytes);
        return false;
    }
    uint8_t * tmp_buf = static_cast<uint8_t *>(tmp.ptr);

    sycl::event copy_event;
    SYCL_CHECK(CHECK_TRY_ERROR(copy_event = stream->memcpy(tmp_buf, data_device, total_bytes)));
    if (!g_ggml_sycl_use_async_mem_op) {
        { ggml_sycl_pipe_wait_scope ws_; copy_event.wait(); }
    }

    const int total_blocks = blocks_per_expert * (int) n_expert;
    auto reorder_event = stream->parallel_for(total_blocks, [=](auto gb_) {
        const int          gb   = gb_;
        const int          e    = gb / blocks_per_expert;
        const int          ib   = gb % blocks_per_expert;
        const block_q4_K * x    = (const block_q4_K *) (tmp_buf + (size_t) e * expert_bytes);
        uint8_t *          base = data_device + (size_t) e * expert_bytes;

        auto * qs_ptr     = base;
        auto * scales_ptr = qs_ptr + QK_K / 2 * blocks_per_expert;
        auto * dm_ptr     = (sycl::half2 *) (scales_ptr + K_SCALE_SIZE * blocks_per_expert);

        for (int j = 0; j < QK_K / 2; ++j) {
            qs_ptr[ib * (QK_K / 2) + j] = x[ib].qs[j];
        }
        for (int j = 0; j < K_SCALE_SIZE; ++j) {
            scales_ptr[ib * K_SCALE_SIZE + j] = x[ib].scales[j];
        }
        dm_ptr[ib] = x[ib].dm;
    });
    if (!g_ggml_sycl_use_async_mem_op) {
        { ggml_sycl_pipe_wait_scope ws_; reorder_event.wait_and_throw(); }
    }
    return true;
}

// Reorder each Q5_K expert slice into [qs][qh][scales][dm].
static bool reorder_qw_q5_k_moe(uint8_t * data_device, size_t expert_bytes, int64_t n_expert, dpct::queue_ptr stream) {
    GGML_ASSERT(expert_bytes % sizeof(block_q5_K) == 0);
    const int    blocks_per_expert = (int) (expert_bytes / sizeof(block_q5_K));
    const size_t total_bytes       = expert_bytes * (size_t) n_expert;

    sycl_reorder_temp_buffer tmp(stream, total_bytes);
    if (!tmp) {
        GGML_LOG_WARN("%s: failed to allocate %zu bytes for reorder temp buffer, skipping reorder\n", __func__, total_bytes);
        return false;
    }
    uint8_t * tmp_buf = static_cast<uint8_t *>(tmp.ptr);

    sycl::event copy_event;
    SYCL_CHECK(CHECK_TRY_ERROR(copy_event = stream->memcpy(tmp_buf, data_device, total_bytes)));
    if (!g_ggml_sycl_use_async_mem_op) {
        { ggml_sycl_pipe_wait_scope ws_; copy_event.wait(); }
    }

    const int total_blocks = blocks_per_expert * (int) n_expert;
    auto reorder_event = stream->parallel_for(total_blocks, [=](auto gb_) {
        const int          gb   = gb_;
        const int          e    = gb / blocks_per_expert;
        const int          ib   = gb % blocks_per_expert;
        const block_q5_K * x    = (const block_q5_K *) (tmp_buf + (size_t) e * expert_bytes);
        uint8_t *          base = data_device + (size_t) e * expert_bytes;

        auto * qs_ptr     = base;
        auto * qh_ptr     = qs_ptr + (QK_K / 2) * blocks_per_expert;
        auto * scales_ptr = qh_ptr + (QK_K / 8) * blocks_per_expert;
        auto * dm_ptr     = (sycl::half2 *) (scales_ptr + K_SCALE_SIZE * blocks_per_expert);

        for (int j = 0; j < QK_K / 2; ++j) {
            qs_ptr[ib * (QK_K / 2) + j] = x[ib].qs[j];
        }
        for (int j = 0; j < QK_K / 8; ++j) {
            qh_ptr[ib * (QK_K / 8) + j] = x[ib].qh[j];
        }
        for (int j = 0; j < K_SCALE_SIZE; ++j) {
            scales_ptr[ib * K_SCALE_SIZE + j] = x[ib].scales[j];
        }
        dm_ptr[ib] = x[ib].dm;
    });
    if (!g_ggml_sycl_use_async_mem_op) {
        { ggml_sycl_pipe_wait_scope ws_; reorder_event.wait_and_throw(); }
    }
    return true;
}

// Reorder each Q6_K expert slice into [ql][qh][scales][d].
static bool reorder_qw_q6_k_moe(uint8_t * data_device, size_t expert_bytes, int64_t n_expert, dpct::queue_ptr stream) {
    GGML_ASSERT(expert_bytes % sizeof(block_q6_K) == 0);
    const int    blocks_per_expert = (int) (expert_bytes / sizeof(block_q6_K));
    const size_t total_bytes       = expert_bytes * (size_t) n_expert;

    sycl_reorder_temp_buffer tmp(stream, total_bytes);
    if (!tmp) {
        GGML_LOG_WARN("%s: failed to allocate %zu bytes for reorder temp buffer, skipping reorder\n", __func__, total_bytes);
        return false;
    }
    uint8_t * tmp_buf = static_cast<uint8_t *>(tmp.ptr);

    sycl::event copy_event;
    SYCL_CHECK(CHECK_TRY_ERROR(copy_event = stream->memcpy(tmp_buf, data_device, total_bytes)));
    if (!g_ggml_sycl_use_async_mem_op) {
        { ggml_sycl_pipe_wait_scope ws_; copy_event.wait(); }
    }

    const int total_blocks = blocks_per_expert * (int) n_expert;
    auto reorder_event = stream->parallel_for(total_blocks, [=](auto gb_) {
        const int          gb   = gb_;
        const int          e    = gb / blocks_per_expert;
        const int          ib   = gb % blocks_per_expert;
        const block_q6_K * x    = (const block_q6_K *) (tmp_buf + (size_t) e * expert_bytes);
        uint8_t *          base = data_device + (size_t) e * expert_bytes;

        auto * ql_ptr     = base;
        auto * qh_ptr     = ql_ptr + (QK_K / 2) * blocks_per_expert;
        auto * scales_ptr = qh_ptr + (QK_K / 4) * blocks_per_expert;
        auto * d_ptr      = (sycl::half *) (scales_ptr + (QK_K / 16) * blocks_per_expert);

        for (int j = 0; j < QK_K / 2; ++j) {
            ql_ptr[ib * (QK_K / 2) + j] = x[ib].ql[j];
        }
        for (int j = 0; j < QK_K / 4; ++j) {
            qh_ptr[ib * (QK_K / 4) + j] = x[ib].qh[j];
        }
        for (int j = 0; j < QK_K / 16; ++j) {
            scales_ptr[ib * (QK_K / 16) + j] = x[ib].scales[j];
        }
        d_ptr[ib] = x[ib].d;
    });
    if (!g_ggml_sycl_use_async_mem_op) {
        { ggml_sycl_pipe_wait_scope ws_; reorder_event.wait_and_throw(); }
    }
    return true;
}

// Reorder each MXFP4 expert slice into [qs][e]: 16-byte nibble blocks, then one E8M0 byte per block.
// Experts are self-contained, so the tensor is reordered a few experts at a time through a small
// temporary: a whole-tensor temporary (hundreds of MB) can exceed the VRAM left on a nearly full card,
// and on Windows the driver then pages device memory out to host RAM instead of failing.
static bool reorder_qw_mxfp4_moe(uint8_t * data_device, size_t expert_bytes, int64_t n_expert, dpct::queue_ptr stream) {
    GGML_ASSERT(expert_bytes % sizeof(block_mxfp4) == 0);
    const int     blocks_per_expert = (int) (expert_bytes / sizeof(block_mxfp4));
    const size_t  max_chunk_bytes   = 32u << 20;
    const int64_t chunk_experts     = std::max<int64_t>(1, std::min<int64_t>(n_expert, (int64_t) (max_chunk_bytes / expert_bytes)));

    sycl_reorder_temp_buffer tmp(stream, (size_t) chunk_experts * expert_bytes);
    if (!tmp) {
        GGML_LOG_WARN("%s: failed to allocate %zu bytes for reorder temp buffer, skipping reorder\n", __func__,
                      (size_t) chunk_experts * expert_bytes);
        return false;
    }
    uint8_t * tmp_buf = static_cast<uint8_t *>(tmp.ptr);

    // the queue is in-order: each chunk's copy into tmp_buf waits for the previous chunk's kernel
    for (int64_t e0 = 0; e0 < n_expert; e0 += chunk_experts) {
        const int64_t n_chunk = std::min(chunk_experts, n_expert - e0);
        uint8_t *     chunk   = data_device + (size_t) e0 * expert_bytes;

        sycl::event copy_event;
        SYCL_CHECK(CHECK_TRY_ERROR(copy_event = stream->memcpy(tmp_buf, chunk, (size_t) n_chunk * expert_bytes)));
        if (!g_ggml_sycl_use_async_mem_op) {
            { ggml_sycl_pipe_wait_scope ws_; copy_event.wait(); }
        }

        const int total_blocks = blocks_per_expert * (int) n_chunk;
        auto reorder_event = stream->parallel_for(total_blocks, [=](auto gb_) {
            const int           gb   = gb_;
            const int           e    = gb / blocks_per_expert;
            const int           ib   = gb % blocks_per_expert;
            const block_mxfp4 * x    = (const block_mxfp4 *) (tmp_buf + (size_t) e * expert_bytes);
            uint8_t *           base = chunk + (size_t) e * expert_bytes;

            uint8_t * qs_ptr = base;
            uint8_t * e_ptr  = qs_ptr + (QK_MXFP4 / 2) * (size_t) blocks_per_expert;

            for (int j = 0; j < QK_MXFP4 / 2; ++j) {
                qs_ptr[(size_t) ib * (QK_MXFP4 / 2) + j] = x[ib].qs[j];
            }
            e_ptr[ib] = x[ib].e;
        });
        if (!g_ggml_sycl_use_async_mem_op) {
            { ggml_sycl_pipe_wait_scope ws_; reorder_event.wait_and_throw(); }
        }
    }
    return true;
}

static bool reorder_qw_q2_k(uint8_t * data_device, size_t size, size_t offset, dpct::queue_ptr stream) {
    GGML_ASSERT(size % sizeof(block_q2_K) == 0);
    GGML_ASSERT(offset % sizeof(block_q2_K) == 0);

    const int nblocks = size / sizeof(block_q2_K);

    sycl_reorder_temp_buffer tmp(stream, size);
    if (!tmp) {
        GGML_LOG_WARN("%s: failed to allocate %zu bytes for reorder temp buffer, skipping reorder\n", __func__, size);
        return false;
    }
    uint8_t * tmp_buf = static_cast<uint8_t *>(tmp.ptr);

    sycl::event copy_event;
    SYCL_CHECK(CHECK_TRY_ERROR(copy_event = stream->memcpy(tmp_buf, data_device, size)));
    if (!g_ggml_sycl_use_async_mem_op) {
        { ggml_sycl_pipe_wait_scope ws_; copy_event.wait(); }
    }

    auto *        qs_ptr     = data_device;
    auto *        scales_ptr = qs_ptr + (QK_K / 4) * nblocks;
    sycl::half2 * dm_ptr     = (sycl::half2 *) (scales_ptr + (QK_K / 16) * nblocks);

    auto reorder_event = stream->parallel_for(nblocks, [=](auto i) {
        const block_q2_K * x  = (const block_q2_K *) tmp_buf;
        const int          ib = i;

        for (int j = 0; j < QK_K / 4; ++j) {
            qs_ptr[ib * (QK_K / 4) + j] = x[ib].qs[j];
        }

        for (int j = 0; j < QK_K / 16; ++j) {
            scales_ptr[ib * (QK_K / 16) + j] = x[ib].scales[j];
        }

        dm_ptr[ib] = x[ib].dm;
    });
    if (!g_ggml_sycl_use_async_mem_op) {
        { ggml_sycl_pipe_wait_scope ws_; reorder_event.wait_and_throw(); }
    }
    return true;
}

static bool reorder_qw_q3_k(uint8_t * data_device, size_t size, size_t offset, dpct::queue_ptr stream) {
    GGML_ASSERT(size % sizeof(block_q3_K) == 0);
    GGML_ASSERT(offset % sizeof(block_q3_K) == 0);

    const int nblocks = size / sizeof(block_q3_K);

    sycl_reorder_temp_buffer tmp(stream, size);
    if (!tmp) {
        GGML_LOG_WARN("%s: failed to allocate %zu bytes for reorder temp buffer, skipping reorder\n", __func__, size);
        return false;
    }
    uint8_t * tmp_buf = static_cast<uint8_t *>(tmp.ptr);

    sycl::event copy_event;
    SYCL_CHECK(CHECK_TRY_ERROR(copy_event = stream->memcpy(tmp_buf, data_device, size)));
    if (!g_ggml_sycl_use_async_mem_op) {
        { ggml_sycl_pipe_wait_scope ws_; copy_event.wait(); }
    }

    auto *       qs_ptr     = data_device;
    auto *       hmask_ptr  = qs_ptr + (QK_K / 4) * nblocks;
    auto *       scales_ptr = hmask_ptr + (QK_K / 8) * nblocks;
    sycl::half * d_ptr      = (sycl::half *) (scales_ptr + 12 * nblocks);

    auto reorder_event = stream->parallel_for(nblocks, [=](auto i) {
        const block_q3_K * x  = (const block_q3_K *) tmp_buf;
        const int          ib = i;

        for (int j = 0; j < QK_K / 4; ++j) {
            qs_ptr[ib * (QK_K / 4) + j] = x[ib].qs[j];
        }

        for (int j = 0; j < QK_K / 8; ++j) {
            hmask_ptr[ib * (QK_K / 8) + j] = x[ib].hmask[j];
        }

        for (int j = 0; j < 12; ++j) {
            scales_ptr[ib * 12 + j] = x[ib].scales[j];
        }

        d_ptr[ib] = x[ib].d;
    });
    if (!g_ggml_sycl_use_async_mem_op) {
        { ggml_sycl_pipe_wait_scope ws_; reorder_event.wait_and_throw(); }
    }
    return true;
}

static bool reorder_qw_q5_k(uint8_t * data_device, size_t size, size_t offset, dpct::queue_ptr stream) {
    GGML_ASSERT(size % sizeof(block_q5_K) == 0);
    GGML_ASSERT(offset % sizeof(block_q5_K) == 0);

    const int nblocks = size / sizeof(block_q5_K);

    sycl_reorder_temp_buffer tmp(stream, size);
    if (!tmp) {
        GGML_LOG_WARN("%s: failed to allocate %zu bytes for reorder temp buffer, skipping reorder\n", __func__, size);
        return false;
    }
    uint8_t * tmp_buf = static_cast<uint8_t *>(tmp.ptr);

    sycl::event copy_event;
    SYCL_CHECK(CHECK_TRY_ERROR(copy_event = stream->memcpy(tmp_buf, data_device, size)));
    if (!g_ggml_sycl_use_async_mem_op) {
        { ggml_sycl_pipe_wait_scope ws_; copy_event.wait(); }
    }

    auto * qs_ptr     = data_device;
    auto * qh_ptr     = qs_ptr + (QK_K / 2) * nblocks;
    auto * scales_ptr = qh_ptr + (QK_K / 8) * nblocks;
    auto * dm_ptr     = (sycl::half2 *) (scales_ptr + K_SCALE_SIZE * nblocks);

    auto reorder_event = stream->parallel_for(nblocks, [=](auto i) {
        const block_q5_K * x  = (const block_q5_K *) tmp_buf;
        const int          ib = i;

        for (int j = 0; j < QK_K / 2; ++j) {
            qs_ptr[ib * (QK_K / 2) + j] = x[ib].qs[j];
        }

        for (int j = 0; j < QK_K / 8; ++j) {
            qh_ptr[ib * (QK_K / 8) + j] = x[ib].qh[j];
        }

        for (int j = 0; j < K_SCALE_SIZE; ++j) {
            scales_ptr[ib * K_SCALE_SIZE + j] = x[ib].scales[j];
        }

        dm_ptr[ib] = x[ib].dm;
    });
    if (!g_ggml_sycl_use_async_mem_op) {
        { ggml_sycl_pipe_wait_scope ws_; reorder_event.wait_and_throw(); }
    }
    return true;
}

static bool reorder_qw_q6_k(uint8_t * data_device, size_t size, size_t offset, dpct::queue_ptr stream) {
    GGML_ASSERT(size % sizeof(block_q6_K) == 0);
    GGML_ASSERT(offset % sizeof(block_q6_K) == 0);

    const int nblocks = size / sizeof(block_q6_K);

    sycl_reorder_temp_buffer tmp(stream, size);
    if (!tmp) {
        GGML_LOG_WARN("%s: failed to allocate %zu bytes for reorder temp buffer, skipping reorder\n", __func__, size);
        return false;
    }
    uint8_t * tmp_buf = static_cast<uint8_t *>(tmp.ptr);

    sycl::event copy_event;
    SYCL_CHECK(CHECK_TRY_ERROR(copy_event = stream->memcpy(tmp_buf, data_device, size)));
    if (!g_ggml_sycl_use_async_mem_op) {
        { ggml_sycl_pipe_wait_scope ws_; copy_event.wait(); }
    }

    auto *       ql_ptr     = data_device;
    auto *       qh_ptr     = ql_ptr + (QK_K / 2) * nblocks;
    auto *       scales_ptr = qh_ptr + (QK_K / 4) * nblocks;
    sycl::half * dm_ptr     = (sycl::half *) (scales_ptr + (QK_K / 16) * nblocks);

    auto reorder_event = stream->parallel_for(nblocks, [=](auto i) {
        const block_q6_K * x  = (const block_q6_K *) tmp_buf;
        const int          ib = i;

        const uint8_t * ql              = x[ib].ql;
        const uint8_t * qh              = x[ib].qh;
        uint8_t *       base_ql_ptr     = ql_ptr + (QK_K / 2) * ib;
        uint8_t *       base_qh_ptr     = qh_ptr + (QK_K / 4) * ib;
        uint8_t *       base_scales_ptr = scales_ptr + (QK_K / 16) * ib;

        for (int j = 0; j < QK_K / 2; ++j) {
            base_ql_ptr[j] = ql[j];
        }
        for (int j = 0; j < QK_K / 4; ++j) {
            base_qh_ptr[j] = qh[j];
        }

        for (int j = 0; j < QK_K / 16; ++j) {
            base_scales_ptr[j] = x[ib].scales[j];
        }

        dm_ptr[ib] = x[ib].d;
    });
    if (!g_ggml_sycl_use_async_mem_op) {
        { ggml_sycl_pipe_wait_scope ws_; reorder_event.wait_and_throw(); }
    }
    return true;
}

static bool reorder_qw_iq3_s(uint8_t * data_device, size_t size, dpct::queue_ptr stream) {
    GGML_ASSERT(size % sizeof(block_iq3_s) == 0);

    const int nblocks = size / sizeof(block_iq3_s);
    sycl_reorder_temp_buffer tmp(stream, size);
    if (!tmp) {
        GGML_LOG_WARN("%s: failed to allocate %zu bytes for reorder temp buffer, skipping reorder\n", __func__, size);
        return false;
    }
    uint8_t * tmp_buf = static_cast<uint8_t *>(tmp.ptr);

    sycl::event copy_event;
    SYCL_CHECK(CHECK_TRY_ERROR(copy_event = stream->memcpy(tmp_buf, data_device, size)));
    if (!g_ggml_sycl_use_async_mem_op) {
        copy_event.wait();
    }

    auto * qs_ptr       = data_device;
    auto * qh_ptr       = qs_ptr + (QK_K / 4) * nblocks;
    auto * signs_ptr    = qh_ptr + (QK_K / 32) * nblocks;
    auto * metadata_ptr = signs_ptr + (QK_K / 8) * nblocks;

    auto reorder_event = stream->parallel_for(nblocks, [=](auto i) {
        const block_iq3_s * x = (const block_iq3_s *) tmp_buf;
        const int ib = i;

        for (int j = 0; j < QK_K / 4; ++j) {
            qs_ptr[ib * (QK_K / 4) + j] = x[ib].qs[j];
        }
        for (int j = 0; j < QK_K / 32; ++j) {
            qh_ptr[ib * (QK_K / 32) + j] = x[ib].qh[j];
        }
        for (int j = 0; j < QK_K / 8; ++j) {
            signs_ptr[ib * (QK_K / 8) + j] = x[ib].signs[j];
        }

        uint8_t * metadata = metadata_ptr + ib * (sizeof(ggml_half) + IQ3S_N_SCALE);
        *reinterpret_cast<ggml_half *>(metadata) = x[ib].d;
        for (int j = 0; j < IQ3S_N_SCALE; ++j) {
            metadata[sizeof(ggml_half) + j] = x[ib].scales[j];
        }
    });
    if (!g_ggml_sycl_use_async_mem_op) {
        reorder_event.wait_and_throw();
    }
    return true;
}

static bool reorder_qw_iq3_xxs(uint8_t * data_device, size_t size, dpct::queue_ptr stream) {
    GGML_ASSERT(size % sizeof(block_iq3_xxs) == 0);

    const int nblocks = size / sizeof(block_iq3_xxs);
    sycl_reorder_temp_buffer tmp(stream, size);
    if (!tmp) {
        GGML_LOG_WARN("%s: failed to allocate %zu bytes for reorder temp buffer, skipping reorder\n", __func__, size);
        return false;
    }
    uint8_t * tmp_buf = static_cast<uint8_t *>(tmp.ptr);

    sycl::event copy_event;
    SYCL_CHECK(CHECK_TRY_ERROR(copy_event = stream->memcpy(tmp_buf, data_device, size)));
    if (!g_ggml_sycl_use_async_mem_op) {
        copy_event.wait();
    }

    auto * qs_ptr = data_device;
    auto * d_ptr  = reinterpret_cast<ggml_half *>(qs_ptr + (3 * QK_K / 8) * nblocks);

    auto reorder_event = stream->parallel_for(nblocks, [=](auto i) {
        const block_iq3_xxs * x = (const block_iq3_xxs *) tmp_buf;
        const int ib = i;

        for (int j = 0; j < 3 * QK_K / 8; ++j) {
            qs_ptr[ib * (3 * QK_K / 8) + j] = x[ib].qs[j];
        }
        d_ptr[ib] = x[ib].d;
    });
    if (!g_ggml_sycl_use_async_mem_op) {
        reorder_event.wait_and_throw();
    }
    return true;
}

static bool reorder_qw(const ggml_tensor * src0, dpct::queue_ptr stream) {
    uint8_t * data_device = (uint8_t *) src0->data;
    size_t ncols = src0->ne[0];
    size_t nrows = src0->ne[1];
    size_t size = ggml_nbytes(src0);

    // MoE expert weights are addressed per expert via nb[2], so each slice must
    // remain self-contained after reorder.
    if (src0->ne[2] > 1) {
        GGML_ASSERT((size_t) size == (size_t) src0->ne[2] * src0->nb[2]);
        switch (src0->type) {
            case GGML_TYPE_Q8_0:
                return reorder_qw_q8_0_moe(data_device, src0->nb[2], src0->ne[2], stream);
            case GGML_TYPE_Q4_K:
                return reorder_qw_q4_k_moe(data_device, src0->nb[2], src0->ne[2], stream);
            case GGML_TYPE_Q5_K:
                return reorder_qw_q5_k_moe(data_device, src0->nb[2], src0->ne[2], stream);
            case GGML_TYPE_Q6_K:
                return reorder_qw_q6_k_moe(data_device, src0->nb[2], src0->ne[2], stream);
            case GGML_TYPE_MXFP4:
                return reorder_qw_mxfp4_moe(data_device, src0->nb[2], src0->ne[2], stream);
            default:
                return false;
        }
    }

    switch (src0->type) {
        case GGML_TYPE_Q4_0:
            return reorder_qw_q4_0(data_device, ncols, nrows, size, 0, stream);
        case GGML_TYPE_Q8_0:
            return reorder_qw_q8_0(data_device, ncols, nrows, size, 0, stream);
        case GGML_TYPE_Q2_K:
            return reorder_qw_q2_k(data_device, size, 0, stream);
        case GGML_TYPE_Q3_K:
            return reorder_qw_q3_k(data_device, size, 0, stream);
        case GGML_TYPE_Q4_K:
            return reorder_qw_q4_k(data_device, size, 0, stream);
        case GGML_TYPE_Q5_K:
            return reorder_qw_q5_k(data_device, size, 0, stream);
        case GGML_TYPE_Q6_K:
            return reorder_qw_q6_k(data_device, size, 0, stream);
        case GGML_TYPE_IQ3_XXS:
            return reorder_qw_iq3_xxs(data_device, size, stream);
        case GGML_TYPE_IQ3_S:
            return reorder_qw_iq3_s(data_device, size, stream);
        default:
            return false;
    }
}

static bool should_reorder_tensor(ggml_backend_sycl_context& ctx, const ggml_tensor * dst) {
    return g_ggml_sycl_enable_optimize && //allow optimize, controlled by $GGML_SYCL_ENABLE_OPT
           ctx.opt_feature.reorder &&      //allow this device due to good perf, skip the devices with bad perf.
           dst->op == GGML_OP_MUL_MAT &&   //limit to some supported cases of Q4_0, to do for more cases.
           // ne[1] <= 8 so multi-column decode (spec / MTP verify) also bootstraps the reorder;
           // all reorderable types have a _switch_ncols kernel.
           dst->src[1]->ne[1] <= 8 && dst->src[1]->ne[2]==1 && dst->src[1]->ne[3]==1;
}

static void opt_for_reorder(ggml_backend_sycl_context * ctx, const ggml_tensor * src0, const ggml_tensor * /* src1 */,
                            ggml_tensor * dst, mul_mat_algo mm_algorithm) {
    if (!should_reorder_tensor(*ctx, dst)) {
        return;
    }

    ggml_tensor_extra_gpu * extra = static_cast<ggml_tensor_extra_gpu *>(src0->extra);
    if (!extra || extra->optimized_feature.reorder) {
        return;  // Skip permutations and already reordered tensors
    }

    switch (mm_algorithm) {
        case mul_mat_algo::DMMV:
            if (!ggml_sycl_supports_reorder_dmmv(src0->type)) {
                return;
            }
            break;
        case mul_mat_algo::MMVQ:
            if (!ggml_sycl_supports_reorder_mmvq(src0->type)) {
                return;
            }
            break;
        case mul_mat_algo::MUL_MAT_SYCL:
            if (!ggml_sycl_supports_reorder_mul_mat_sycl(src0->type)) {
                return;
            }
            break;
    }

    if (reorder_qw(src0, ctx->stream())) {
        extra->optimized_feature.reorder = true;  // Used to decode/dequan in next steps and avoid re-reordering
    }
}

// Lazily reorder supported MoE expert weights once their fused path is used.
// True when opt_for_reorder_id() would reorder src0 now. The reorder allocates a temp buffer and
// runs copies, so it must not happen while a SYCL graph is being recorded.
static bool ggml_sycl_mmid_reorder_pending(const ggml_backend_sycl_context * ctx, const ggml_tensor * src0) {
    if (!g_ggml_sycl_enable_optimize || !ctx->opt_feature.reorder) {
        return false;
    }
    if (src0->type != GGML_TYPE_Q8_0 && src0->type != GGML_TYPE_Q4_K && src0->type != GGML_TYPE_Q5_K &&
        src0->type != GGML_TYPE_Q6_K && src0->type != GGML_TYPE_MXFP4) {
        return false;
    }
    if (src0->type == GGML_TYPE_MXFP4 && (src0->nb[2] % 16 != 0 || (uintptr_t) src0->data % 16 != 0)) {
        return false;
    }
    const ggml_tensor_extra_gpu * extra = static_cast<const ggml_tensor_extra_gpu *>(src0->extra);
    return extra && !extra->optimized_feature.reorder;
}

static void opt_for_reorder_id(ggml_backend_sycl_context * ctx, const ggml_tensor * src0) {
    if (!ggml_sycl_mmid_reorder_pending(ctx, src0)) {
        return;
    }
    if (src0->type != GGML_TYPE_Q8_0 && src0->type != GGML_TYPE_Q4_K &&
        src0->type != GGML_TYPE_Q5_K && src0->type != GGML_TYPE_Q6_K && src0->type != GGML_TYPE_MXFP4) {
        return;
    }
    // The MXFP4 reorder kernels use 8-byte vector loads, so every expert slice must stay aligned.
    if (src0->type == GGML_TYPE_MXFP4 && (src0->nb[2] % 16 != 0 || (uintptr_t) src0->data % 16 != 0)) {
        return;
    }
    ggml_tensor_extra_gpu * extra = static_cast<ggml_tensor_extra_gpu *>(src0->extra);
    if (!extra || extra->optimized_feature.reorder) {
        return;
    }
    if (reorder_qw(src0, ctx->stream())) {
        extra->optimized_feature.reorder = true;
    }
}


static bool can_use_dequantize_mul_mat_vec(const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst) {
    // The F16/BF16 qk=1 kernel iterates with stride 2*DMMV_X, requiring ne[0] to be
    // a multiple of 2*DMMV_X. Quantized types use block-structured kernels that only
    // need ne[0] % DMMV_X == 0.
    const int64_t dmmv_x_required = (src0->type == GGML_TYPE_BF16 || src0->type == GGML_TYPE_F16) ?
                                    2*GGML_SYCL_DMMV_X : GGML_SYCL_DMMV_X;
    return ggml_sycl_supports_dmmv(src0->type) && src1->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32 &&
           src0->ne[0] % dmmv_x_required == 0 && src1->ne[1] == 1;
}

// reordered weights of the types in ggml_sycl_xmx_supports_type() take the XMX kernel for wider batches
static bool can_use_xmx_batch(int device, const ggml_tensor * src0, const ggml_tensor * src1) {
#ifdef GGML_SYCL_MMVQ_HAS_XMX
    const auto * extra = static_cast<const ggml_tensor_extra_gpu *>(src0->extra);
    return ggml_sycl_xmx_enabled(device) && ggml_sycl_xmx_supports_type(src0->type) && extra &&
           extra->optimized_feature.reorder && src1->ne[1] <= g_ggml_sycl_xmx_max_cols && src1->ne[2] == 1 &&
           src1->ne[3] == 1;
#else
    GGML_UNUSED(device);
    GGML_UNUSED(src0);
    GGML_UNUSED(src1);
    return false;
#endif // GGML_SYCL_MMVQ_HAS_XMX
}

static bool can_use_mul_mat_vec_q(int device, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst) {
    return ggml_is_quantized(src0->type) && src1->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32 &&
           (src1->ne[1] <= MMVQ_MAX_BATCH_SIZE || can_use_xmx_batch(device, src0, src1));
}

// F32 weight x F32 activations, 1..8 activation columns. One sub-group per weight row; each lane sums a strided
// slice of the row (float4 when the row and the activation columns are 16-byte aligned), then a sub-group
// reduction. fp32 throughout, so results match the GEMM path up to summation order.
#define GGML_SYCL_F32_MMV_MAX_COLS 8
static bool ggml_sycl_can_use_f32_mmv(const ggml_tensor * src0, const ggml_tensor * src1, const ggml_tensor * dst) {
    static const bool disabled = ggml_sycl_get_env("GGML_SYCL_DISABLE_F32_MMV", 0) != 0;
    if (disabled) {
        return false;
    }
    return ggml_get_op_params_i32(dst, 1) != GGML_HINT_SRC0_IS_HADAMARD &&
           src0->type == GGML_TYPE_F32 && src1->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32 &&
           src1->ne[1] >= 1 && src1->ne[1] <= GGML_SYCL_F32_MMV_MAX_COLS &&
           src0->ne[2] == 1 && src0->ne[3] == 1 && src1->ne[2] == 1 && src1->ne[3] == 1 &&
           src0->nb[0] == sizeof(float) && src1->nb[0] == sizeof(float) && dst->nb[0] == sizeof(float) &&
           src0->ne[0] == src1->ne[0] && !ggml_backend_buffer_is_sycl_split(src0->buffer);
}

template <int ncols_y>
static void mul_mat_vec_f32_kernel(const float * __restrict__ w, const float * __restrict__ y, float * __restrict__ dst,
                                   const int ncols, const int nrows, const int64_t w_stride, const int64_t y_stride,
                                   const int64_t dst_stride, const bool vec4, const sycl::nd_item<3> & item) {
    const int row = item.get_group(2) * item.get_local_range(1) + item.get_local_id(1);
    if (row >= nrows) {
        return;
    }
    const int lane = item.get_local_id(2);
    const float * wr = w + (int64_t) row * w_stride;
    float sum[ncols_y];
#pragma unroll
    for (int j = 0; j < ncols_y; ++j) {
        sum[j] = 0.0f;
    }
    if (vec4) {
        const sycl::float4 * w4 = reinterpret_cast<const sycl::float4 *>(wr);
        for (int c = lane; c < ncols / 4; c += WARP_SIZE) {
            const sycl::float4 wv = w4[c];
#pragma unroll
            for (int j = 0; j < ncols_y; ++j) {
                const sycl::float4 yv = reinterpret_cast<const sycl::float4 *>(y + j * y_stride)[c];
                sum[j] += wv.x() * yv.x() + wv.y() * yv.y() + wv.z() * yv.z() + wv.w() * yv.w();
            }
        }
    } else {
        for (int c = lane; c < ncols; c += WARP_SIZE) {
            const float wv = wr[c];
#pragma unroll
            for (int j = 0; j < ncols_y; ++j) {
                sum[j] += wv * y[j * y_stride + c];
            }
        }
    }
    const auto sg = item.get_sub_group();
#pragma unroll
    for (int j = 0; j < ncols_y; ++j) {
        const float total = sycl::reduce_over_group(sg, sum[j], sycl::plus<float>());
        if (lane == 0) {
            dst[j * dst_stride + row] = total;
        }
    }
}

template <int ncols_y>
static void launch_mul_mat_vec_f32(const float * w, const float * y, float * dst, int ncols, int nrows, int64_t w_stride,
                                   int64_t y_stride, int64_t dst_stride, bool vec4, dpct::queue_ptr stream) {
    constexpr int rows_per_wg = 4;
    const sycl::range<3> block_dims(1, rows_per_wg, WARP_SIZE);
    const sycl::range<3> block_nums(1, 1, (nrows + rows_per_wg - 1) / rows_per_wg);
    stream->parallel_for(sycl::nd_range<3>(block_nums * block_dims, block_dims),
                         [=](sycl::nd_item<3> item) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                             mul_mat_vec_f32_kernel<ncols_y>(w, y, dst, ncols, nrows, w_stride, y_stride, dst_stride,
                                                             vec4, item);
                         });
}

// Wide rows (e.g. DeepSeek-V4 hyper-connection fn weights, 16384 x 24): one sub-group per row leaves every lane a
// long serial chain and most of the GPU idle. Split each row over a whole work-group and reduce over the group.
#define GGML_SYCL_F32_MMV_WIDE_SG 16
static void launch_mul_mat_vec_f32_wide(const float * w, const float * y, float * dst, int ncols, int nrows,
                                        int64_t w_stride, dpct::queue_ptr stream) {
    constexpr int lanes = GGML_SYCL_F32_MMV_WIDE_SG * WARP_SIZE;
    const sycl::range<3> block_dims(1, GGML_SYCL_F32_MMV_WIDE_SG, WARP_SIZE);
    const sycl::range<3> block_nums(1, 1, nrows);
    const int n4 = ncols / 4;
    stream->parallel_for(sycl::nd_range<3>(block_nums * block_dims, block_dims),
                         [=](sycl::nd_item<3> item) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                             const int row  = item.get_group(2);
                             const int lane = item.get_local_id(1) * WARP_SIZE + item.get_local_id(2);
                             const sycl::float4 * w4 = reinterpret_cast<const sycl::float4 *>(w + (int64_t) row * w_stride);
                             const sycl::float4 * y4 = reinterpret_cast<const sycl::float4 *>(y);
                             float sum = 0.0f;
                             for (int c = lane; c < n4; c += lanes) {
                                 const sycl::float4 wv = w4[c];
                                 const sycl::float4 yv = y4[c];
                                 sum += wv.x() * yv.x() + wv.y() * yv.y() + wv.z() * yv.z() + wv.w() * yv.w();
                             }
                             const float total = sycl::reduce_over_group(item.get_group(), sum, sycl::plus<float>());
                             if (lane == 0) {
                                 dst[row] = total;
                             }
                         });
}

// Multi-column variant of the wide kernel (verify batches: 2..8 activation columns). One work-group per row, the row
// is split over the whole group and every column is reduced separately; the weight row is read once for all columns.
template <int ncols_y>
static void launch_mul_mat_vec_f32_wide_n(const float * w, const float * y, float * dst, int ncols, int nrows,
                                          int64_t w_stride, int64_t y_stride, int64_t dst_stride,
                                          dpct::queue_ptr stream) {
    constexpr int lanes = GGML_SYCL_F32_MMV_WIDE_SG * WARP_SIZE;
    const sycl::range<3> block_dims(1, GGML_SYCL_F32_MMV_WIDE_SG, WARP_SIZE);
    const sycl::range<3> block_nums(1, 1, nrows);
    const int n4 = ncols / 4;
    stream->parallel_for(sycl::nd_range<3>(block_nums * block_dims, block_dims),
                         [=](sycl::nd_item<3> item) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                             const int row  = item.get_group(2);
                             const int lane = item.get_local_id(1) * WARP_SIZE + item.get_local_id(2);
                             const sycl::float4 * w4 = reinterpret_cast<const sycl::float4 *>(w + (int64_t) row * w_stride);
                             float sum[ncols_y];
#pragma unroll
                             for (int j = 0; j < ncols_y; ++j) {
                                 sum[j] = 0.0f;
                             }
                             for (int c = lane; c < n4; c += lanes) {
                                 const sycl::float4 wv = w4[c];
#pragma unroll
                                 for (int j = 0; j < ncols_y; ++j) {
                                     const sycl::float4 yv = reinterpret_cast<const sycl::float4 *>(y + j * y_stride)[c];
                                     sum[j] += wv.x() * yv.x() + wv.y() * yv.y() + wv.z() * yv.z() + wv.w() * yv.w();
                                 }
                             }
#pragma unroll
                             for (int j = 0; j < ncols_y; ++j) {
                                 const float total = sycl::reduce_over_group(item.get_group(), sum[j], sycl::plus<float>());
                                 if (lane == 0) {
                                     dst[row + j * dst_stride] = total;
                                 }
                             }
                         });
}

static void ggml_sycl_mul_mat_vec_f32(ggml_backend_sycl_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1,
                                      ggml_tensor * dst) {
    const float * w = (const float *) src0->data;
    const float * y = (const float *) src1->data;
    float *       d = (float *) dst->data;
    const int     ncols      = (int) src0->ne[0];
    const int     nrows      = (int) src0->ne[1];
    const int64_t w_stride   = src0->nb[1] / sizeof(float);
    const int64_t y_stride   = src1->nb[1] / sizeof(float);
    const int64_t dst_stride = dst->nb[1] / sizeof(float);
    const bool vec4 = ncols % 4 == 0 && w_stride % 4 == 0 && y_stride % 4 == 0 &&
                      ((uintptr_t) w % 16) == 0 && ((uintptr_t) y % 16) == 0;
    dpct::queue_ptr stream = ctx.stream();
    // GGML_SYCL_F32_MMV_WIDE_MIN_COLS=0 disables the wide-row kernel
    static const int wide_min_cols = ggml_sycl_get_env("GGML_SYCL_F32_MMV_WIDE_MIN_COLS", 4096);
    if (wide_min_cols > 0 && src1->ne[1] == 1 && vec4 && ncols >= wide_min_cols) {
        launch_mul_mat_vec_f32_wide(w, y, d, ncols, nrows, w_stride, stream);
        return;
    }
    // Multi-column activations (MTP / draft verify batches): split each row over a work-group as well. Few rows with a
    // long reduction (hc inject 4 x 10240, ssm alpha/beta 48 x 2560, shared-expert gate 1 x 2560) otherwise run on a
    // handful of sub-groups. GGML_SYCL_F32_MMV_WIDE_N=0 disables; _WIDE_SMALL_ROWS / _SMALL_MIN_COLS tune the small case.
    static const bool wide_n_enabled = ggml_sycl_get_env("GGML_SYCL_F32_MMV_WIDE_N", 1) != 0;
    static const int  wide_small_rows = ggml_sycl_get_env("GGML_SYCL_F32_MMV_WIDE_SMALL_ROWS", 64);
    static const int  wide_small_cols = ggml_sycl_get_env("GGML_SYCL_F32_MMV_WIDE_SMALL_MIN_COLS", 2048);
    const bool wide_shape = (wide_min_cols > 0 && ncols >= wide_min_cols) ||
                            (wide_small_rows > 0 && nrows <= wide_small_rows && ncols >= wide_small_cols);
    if (wide_n_enabled && vec4 && wide_shape && src1->ne[1] >= 1) {
        switch (src1->ne[1]) {
            case 1: launch_mul_mat_vec_f32_wide_n<1>(w, y, d, ncols, nrows, w_stride, y_stride, dst_stride, stream); return;
            case 2: launch_mul_mat_vec_f32_wide_n<2>(w, y, d, ncols, nrows, w_stride, y_stride, dst_stride, stream); return;
            case 3: launch_mul_mat_vec_f32_wide_n<3>(w, y, d, ncols, nrows, w_stride, y_stride, dst_stride, stream); return;
            case 4: launch_mul_mat_vec_f32_wide_n<4>(w, y, d, ncols, nrows, w_stride, y_stride, dst_stride, stream); return;
            case 5: launch_mul_mat_vec_f32_wide_n<5>(w, y, d, ncols, nrows, w_stride, y_stride, dst_stride, stream); return;
            case 6: launch_mul_mat_vec_f32_wide_n<6>(w, y, d, ncols, nrows, w_stride, y_stride, dst_stride, stream); return;
            case 7: launch_mul_mat_vec_f32_wide_n<7>(w, y, d, ncols, nrows, w_stride, y_stride, dst_stride, stream); return;
            case 8: launch_mul_mat_vec_f32_wide_n<8>(w, y, d, ncols, nrows, w_stride, y_stride, dst_stride, stream); return;
            default: break;
        }
    }
    switch (src1->ne[1]) {
        case 1: launch_mul_mat_vec_f32<1>(w, y, d, ncols, nrows, w_stride, y_stride, dst_stride, vec4, stream); break;
        case 2: launch_mul_mat_vec_f32<2>(w, y, d, ncols, nrows, w_stride, y_stride, dst_stride, vec4, stream); break;
        case 3: launch_mul_mat_vec_f32<3>(w, y, d, ncols, nrows, w_stride, y_stride, dst_stride, vec4, stream); break;
        case 4: launch_mul_mat_vec_f32<4>(w, y, d, ncols, nrows, w_stride, y_stride, dst_stride, vec4, stream); break;
        case 5: launch_mul_mat_vec_f32<5>(w, y, d, ncols, nrows, w_stride, y_stride, dst_stride, vec4, stream); break;
        case 6: launch_mul_mat_vec_f32<6>(w, y, d, ncols, nrows, w_stride, y_stride, dst_stride, vec4, stream); break;
        case 7: launch_mul_mat_vec_f32<7>(w, y, d, ncols, nrows, w_stride, y_stride, dst_stride, vec4, stream); break;
        case 8: launch_mul_mat_vec_f32<8>(w, y, d, ncols, nrows, w_stride, y_stride, dst_stride, vec4, stream); break;
        default: GGML_ABORT("unsupported ncols_y for F32 mat-vec");
    }
}

static bool ggml_sycl_can_use_bf16_mmv(const ggml_tensor * w, const ggml_tensor * x, const ggml_tensor * dst) {
    static const bool enabled = ggml_sycl_get_env("GGML_SYCL_BF16_MMV", 0) != 0;
    return enabled && w->type == GGML_TYPE_BF16 && x->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32 &&
           x->ne[1] == 1 && w->ne[1] <= 1024 && w->ne[2] == 1 && w->ne[3] == 1 &&
           x->ne[2] == 1 && x->ne[3] == 1 && w->nb[0] == sizeof(uint16_t) && x->nb[0] == sizeof(float) &&
           dst->nb[0] == sizeof(float) && w->ne[0] == x->ne[0] && !ggml_backend_buffer_is_sycl_split(w->buffer);
}

static void ggml_sycl_mul_mat_vec_bf16(ggml_backend_sycl_context & ctx, const ggml_tensor * src0,
                                      const ggml_tensor * src1, ggml_tensor * dst) {
    const uint16_t * w = (const uint16_t *) src0->data;
    const float * x = (const float *) src1->data;
    float * out = (float *) dst->data;
    const int k = (int) src0->ne[0];
    const int m = (int) src0->ne[1];
    const int64_t stride = src0->nb[1] / sizeof(uint16_t);
    const bool vec4 = k % 4 == 0 && stride % 4 == 0 && (uintptr_t) w % 8 == 0 && (uintptr_t) x % 16 == 0;
    constexpr int rows = 4;
    const sycl::range<3> local(1, rows, WARP_SIZE);
    const sycl::range<3> groups(1, 1, (m + rows - 1) / rows);
    ctx.stream()->parallel_for(sycl::nd_range<3>(groups * local, local),
        [=](sycl::nd_item<3> item) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
            const int row = item.get_group(2) * rows + item.get_local_id(1);
            if (row >= m) { return; }
            const int lane = item.get_local_id(2);
            const uint16_t * wr = w + row * stride;
            float sum = 0.0f;
            if (vec4) {
                for (int i = lane; i < k / 4; i += WARP_SIZE) {
                    const sycl::ushort4 v = reinterpret_cast<const sycl::ushort4 *>(wr)[i];
                    const sycl::float4 a = reinterpret_cast<const sycl::float4 *>(x)[i];
                    sum += sycl::bit_cast<float>((uint32_t) v.x() << 16) * a.x() +
                           sycl::bit_cast<float>((uint32_t) v.y() << 16) * a.y() +
                           sycl::bit_cast<float>((uint32_t) v.z() << 16) * a.z() +
                           sycl::bit_cast<float>((uint32_t) v.w() << 16) * a.w();
                }
            } else {
                for (int i = lane; i < k; i += WARP_SIZE) {
                    sum += sycl::bit_cast<float>((uint32_t) wr[i] << 16) * x[i];
                }
            }
            const float total = sycl::reduce_over_group(item.get_sub_group(), sum, sycl::plus<float>());
            if (lane == 0) { out[row] = total; }
        });
}

// Q8_0 weights with ne[2] batches, each applied to its own single activation column (DeepSeek-V4's grouped
// attention output projection, [4096, 1024, 8] x [4096, 1, 8]). The generic path launches one mat-vec per
// batch; this quantizes all columns once and runs a single launch over all batches. Plain layout only.
static bool ggml_sycl_can_use_q8_0_grouped_mmv(const ggml_tensor * src0, const ggml_tensor * src1, const ggml_tensor * dst) {
    static const bool disabled = ggml_sycl_get_env("GGML_SYCL_DISABLE_Q8_GROUPED_MMV", 0) != 0;
    if (disabled) {
        return false;
    }
    const ggml_tensor_extra_gpu * extra = static_cast<const ggml_tensor_extra_gpu *>(src0->extra);
    return src0->type == GGML_TYPE_Q8_0 && src1->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32 &&
           src0->ne[2] > 1 && src0->ne[3] == 1 && src1->ne[1] == 1 && src1->ne[2] == src0->ne[2] && src1->ne[3] == 1 &&
           src0->ne[0] == src1->ne[0] && src0->ne[0] % QK8_0 == 0 &&
           (int64_t) src0->ne[1] * src0->ne[2] * (src0->ne[0] / QK8_0) < INT_MAX &&
           ggml_is_contiguous(src0) && ggml_is_contiguous(dst) &&
           // src1 may be a permuted view; with one column per batch it only needs dense, adjacent columns
           src1->nb[0] == sizeof(float) && src1->nb[2] == (size_t) src1->ne[0] * sizeof(float) &&
           src0->nb[2] == (size_t) src0->ne[1] * (src0->ne[0] / QK8_0) * sizeof(block_q8_0) &&
           !ggml_backend_buffer_is_sycl_split(src0->buffer);
}

static void ggml_sycl_mul_mat_vec_q8_0_grouped_op(ggml_backend_sycl_context & ctx, const ggml_tensor * src0,
                                                  const ggml_tensor * src1, ggml_tensor * dst) {
    const int ne00    = (int) src0->ne[0];
    const int ngroups = (int) src0->ne[2];
    const queue_ptr stream = ctx.stream();
    const int src1_padded_cols = GGML_PAD(ne00, MATRIX_ROW_PADDING);

    // Move the weights to the reorder layout once, slice by slice: each ne[2] slice becomes a self-contained SoA,
    // which is what the reorder-aware per-batch paths (multi-token calls) expect from src0_dd_i.
    static const bool reorder_enabled = ggml_sycl_get_env("GGML_SYCL_Q8_GROUPED_REORDER", 1) != 0;
    ggml_tensor_extra_gpu * extra = static_cast<ggml_tensor_extra_gpu *>(src0->extra);
    if (reorder_enabled && g_ggml_sycl_enable_optimize && ctx.opt_feature.reorder && extra &&
        !extra->optimized_feature.reorder) {
        for (int g = 0; g < ngroups; ++g) {
            const bool ok = reorder_qw_q8_0((uint8_t *) src0->data + (size_t) g * src0->nb[2], ne00, (int) src0->ne[1],
                                            src0->nb[2], 0, stream);
            if (!ok) {
                GGML_ASSERT(g == 0 && "grouped Q8_0 reorder failed part-way");
                break;
            }
            extra->optimized_feature.reorder = true;
        }
    }
    const bool   use_reorder    = extra && extra->optimized_feature.reorder;
    const size_t bytes_per_qrow = (size_t) src1_padded_cols * sizeof(block_q8_1) / QK8_1;

    ggml_sycl_pool_alloc<char> src1_q8_alloc(ctx.pool(), (size_t) ngroups * bytes_per_qrow);
    char * src1_ddq = src1_q8_alloc.get();

    if (use_reorder) {
        quantize_row_q8_1_sycl<quantize_and_reorder_q8_1_soa>((const float *) src1->data, src1_ddq, ne00, ngroups,
                                                              src1_padded_cols, stream);
        ggml_sycl_mul_mat_vec_q8_0_grouped_reorder(src0->data, src1_ddq, (float *) dst->data, ne00, (int) src0->ne[1],
                                                   ngroups, src0->nb[2], bytes_per_qrow, stream);
        return;
    }
    quantize_row_q8_1_sycl<quantize_q8_1>((const float *) src1->data, src1_ddq, ne00, ngroups, src1_padded_cols, stream);
    ggml_sycl_mul_mat_vec_q8_0_grouped(src0->data, src1_ddq, (float *) dst->data, ne00, (int) src0->ne[1], ngroups,
                                       src1_padded_cols / QK8_1, stream);
}

static void ggml_sycl_mul_mat(ggml_backend_sycl_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst);

// A batched Q8_0 weight whose ne[2] slices were reordered by the grouped mat-vec path above.
static bool ggml_sycl_is_slice_reordered_q8_0(const ggml_tensor * src0) {
    const ggml_tensor_extra_gpu * extra = static_cast<const ggml_tensor_extra_gpu *>(src0->extra);
    return src0->type == GGML_TYPE_Q8_0 && src0->ne[2] > 1 && src0->ne[3] == 1 && extra && extra->optimized_feature.reorder;
}

// Each slice is a self-contained reordered matrix: hand them to the 2-D paths one at a time.
static void ggml_sycl_mul_mat_q8_0_grouped_slices(ggml_backend_sycl_context & ctx, const ggml_tensor * src0,
                                                  const ggml_tensor * src1, ggml_tensor * dst) {
    GGML_ASSERT(src1->ne[2] == src0->ne[2] && src1->ne[3] == 1 && dst->ne[2] == src0->ne[2]);
    ggml_tensor src0_row = *src0;
    ggml_tensor src1_row = *src1;
    ggml_tensor dst_row  = *dst;
    src0_row.ne[2] = 1;
    src0_row.nb[3] = src0->nb[2];
    src1_row.ne[2] = 1;
    src1_row.nb[3] = src1->nb[2];
    dst_row.ne[2]  = 1;
    dst_row.nb[3]  = dst->nb[2];
    dst_row.src[0] = &src0_row;
    dst_row.src[1] = &src1_row;

    // The generic path quantizes non-contiguous activations with the plain q8_1 layout even for reordered weights,
    // so always hand it dense columns (DSV4 passes a permuted view here).
    GGML_ASSERT(src1->type == GGML_TYPE_F32 && src1->nb[0] == sizeof(float));
    const size_t col_bytes = (size_t) src1->ne[0] * sizeof(float);
    const bool   dense     = src1->nb[1] == col_bytes;
    ggml_sycl_pool_alloc<char> src1_dense(ctx.pool());
    if (!dense) {
        src1_dense.alloc(col_bytes * (size_t) src1->ne[1]);
        src1_row.nb[1] = col_bytes;
        src1_row.nb[2] = col_bytes * (size_t) src1->ne[1];
        src1_row.nb[3] = src1_row.nb[2];
    }
    const queue_ptr stream = ctx.stream();
    for (int64_t g = 0; g < src0->ne[2]; ++g) {
        src0_row.data = (char *) src0->data + g * src0->nb[2];
        if (dense) {
            src1_row.data = (char *) src1->data + g * src1->nb[2];
        } else {
            const char * from = (const char *) src1->data + g * src1->nb[2];
            float *      to   = (float *) src1_dense.get();
            const size_t nb1  = src1->nb[1];
            const size_t ne0  = (size_t) src1->ne[0];
            stream->parallel_for(sycl::range<2>((size_t) src1->ne[1], ne0), [=](sycl::id<2> id) {
                to[id[0] * ne0 + id[1]] = *(const float *) (from + id[0] * nb1 + id[1] * sizeof(float));
            });
            src1_row.data = src1_dense.get();
        }
        dst_row.data = (char *) dst->data + g * dst->nb[2];
        ggml_sycl_mul_mat(ctx, &src0_row, &src1_row, &dst_row);
    }
}

// Batched (3-D) Q4_K/Q5_K/Q6_K weight times 1..8 columns per slice, as GLM-5's MLA attn_k_b / attn_v_b: one quantize and one
// mat-vec launch instead of a 2-D call per slice. src1 may be a permuted view; it is gathered into dense columns first.
static bool ggml_sycl_mul_mat_vec_q_grouped_op(ggml_backend_sycl_context & ctx, const ggml_tensor * src0,
                                               const ggml_tensor * src1, ggml_tensor * dst) {
    static const bool disabled = ggml_sycl_get_env("GGML_SYCL_DISABLE_QK_GROUPED_MMV", 0) != 0;
    if (disabled || dst->op != GGML_OP_MUL_MAT) {
        return false;
    }
    if (!(src0->type == GGML_TYPE_Q6_K || src0->type == GGML_TYPE_Q5_K || src0->type == GGML_TYPE_Q4_K)) {
        return false;
    }
    const ggml_tensor_extra_gpu * extra = static_cast<const ggml_tensor_extra_gpu *>(src0->extra);
    if (src1->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32 || src0->ne[2] <= 1 || src0->ne[3] != 1 ||
        src1->ne[2] != src0->ne[2] || src1->ne[3] != 1 || src1->ne[1] < 1 || src1->ne[1] > 8 ||
        src0->ne[0] != src1->ne[0] || src0->ne[0] % QK_K != 0 || !ggml_is_contiguous(src0) ||
        !ggml_is_contiguous(dst) || src1->nb[0] != sizeof(float) || ggml_backend_buffer_is_sycl_split(src0->buffer) ||
        (extra && extra->optimized_feature.reorder) ||
        (int64_t) src0->ne[1] * src0->ne[2] >= INT_MAX) {
        return false;
    }
    const int ne00 = (int) src0->ne[0], ne01 = (int) src0->ne[1], ngroups = (int) src0->ne[2];
    const int ncols_dst = (int) src1->ne[1];
    const queue_ptr stream = ctx.stream();

    const size_t col_bytes = (size_t) ne00 * sizeof(float);
    const bool   dense     = src1->nb[1] == col_bytes && src1->nb[2] == col_bytes * (size_t) ncols_dst;
    ggml_sycl_pool_alloc<float> src1_dense(ctx.pool());
    const float * src1_ptr = (const float *) src1->data;
    if (!dense) {
        src1_dense.alloc((size_t) ne00 * ncols_dst * ngroups);
        float *      to   = src1_dense.get();
        const char * from = (const char *) src1->data;
        const size_t nb1 = src1->nb[1], nb2 = src1->nb[2];
        const int    n1  = ncols_dst;
        stream->parallel_for(sycl::range<3>((size_t) ngroups, (size_t) n1, (size_t) ne00), [=](sycl::id<3> id) {
            to[((size_t) id[0] * n1 + id[1]) * ne00 + id[2]] =
                *(const float *) (from + id[0] * nb2 + id[1] * nb1 + id[2] * sizeof(float));
        });
        src1_ptr = to;
    }
    const int padded_cols = GGML_PAD(ne00, MATRIX_ROW_PADDING);
    ggml_sycl_pool_alloc<char> src1_q8(ctx.pool(), (size_t) ngroups * ncols_dst * padded_cols * sizeof(block_q8_1) / QK8_1);
    quantize_row_q8_1_sycl<quantize_q8_1>(src1_ptr, src1_q8.get(), ne00, ncols_dst * ngroups, padded_cols, stream);
    return ggml_sycl_mul_mat_vec_q_grouped(src0->type, src0->data, src1_q8.get(), (float *) dst->data, ne00, ne01, ngroups,
                                           ncols_dst, padded_cols / QK8_1, src0->nb[2], stream);
}

static void ggml_sycl_mul_mat(ggml_backend_sycl_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/2);

    if (dst->op == GGML_OP_MUL_MAT && ggml_sycl_can_use_f32_mmv(src0, src1, dst)) {
        ggml_sycl_mul_mat_vec_f32(ctx, src0, src1, dst);
        return;
    }

    if (dst->op == GGML_OP_MUL_MAT && ggml_sycl_can_use_bf16_mmv(src0, src1, dst)) {
        ggml_sycl_mul_mat_vec_bf16(ctx, src0, src1, dst);
        return;
    }

    if (ggml_sycl_mul_mat_vec_q_grouped_op(ctx, src0, src1, dst)) {
        return;
    }

    if (dst->op == GGML_OP_MUL_MAT && ggml_sycl_can_use_q8_0_grouped_mmv(src0, src1, dst)) {
        ggml_sycl_mul_mat_vec_q8_0_grouped_op(ctx, src0, src1, dst);
        return;
    }

    if (dst->op == GGML_OP_MUL_MAT && ggml_sycl_is_slice_reordered_q8_0(src0)) {
        ggml_sycl_mul_mat_q8_0_grouped_slices(ctx, src0, src1, dst);
        return;
    }

    // Handle HADAMARAD hint given from further up the pipeline and pass it to the correct
    // kernel.
    //
    // The op check is not redundant: this backend also routes MUL_MAT_ID through here with a
    // stack copy of dst, which carries MUL_MAT_ID's own op_params. ggml_mul_mat_set_hint()
    // asserts GGML_OP_MUL_MAT for the same reason.
    if (dst->op == GGML_OP_MUL_MAT && ggml_get_op_params_i32(dst, 1) == GGML_HINT_SRC0_IS_HADAMARD &&
        ggml_sycl_op_fwht(ctx, src1, dst)) {
        return;
    }

    const bool split = ggml_backend_buffer_is_sycl_split(src0->buffer);
    int64_t min_compute_capability = INT_MAX;

    if (split) {
        ggml_backend_sycl_split_buffer_type_context * buft_ctx =
            (ggml_backend_sycl_split_buffer_type_context *) src0->buffer->buft->context;
        auto & tensor_split = buft_ctx->tensor_split;
        for (int id = 0; id < ggml_sycl_info().device_count; ++id) {
            // skip devices that are not going to do any work:
            if (tensor_split[id] >= (id + 1 < ggml_sycl_info().device_count ? tensor_split[id + 1] : 1.0f)) {
                continue;
            }

            if (min_compute_capability > ggml_sycl_info().devices[id].cc) {
                min_compute_capability = ggml_sycl_info().devices[id].cc;
            }
        }
    } else {
        min_compute_capability = ggml_sycl_info().devices[ctx.device].cc;
    }

    // check data types and tensor shapes for custom matrix multiplication kernels:
    bool use_dequantize_mul_mat_vec = can_use_dequantize_mul_mat_vec(src0, src1, dst);
    if (src1->ne[1] == 1 && ggml_sycl_xmx_min_cols(src0->type) == 1 && can_use_xmx_batch(ctx.device, src0, src1)) {
        use_dequantize_mul_mat_vec = false;
    }
#ifdef GGML_SYCL_F16
    // dmmv may convert src1 to f16 in this build [TAG_GGML_PREC]
    use_dequantize_mul_mat_vec = use_dequantize_mul_mat_vec && ggml_sycl_src1_f16_ok(dst);
#endif

    bool use_mul_mat_vec_q = can_use_mul_mat_vec_q(ctx.device, src0, src1, dst);

    bool use_mul_mat_q =  ggml_sycl_supports_mmq(src0->type)
        && src1->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32;


    // mmvq and mmq need the __dp4a instruction which is available for gen12+
    // Workaround in https://github.com/ggml-org/llama.cpp/commit/95f84d5ce8b449a9b16009434aca800df504a02e
    use_mul_mat_q = use_mul_mat_q && (src0->type != GGML_TYPE_IQ2_XXS);
#ifdef SYCL_USE_XMX
    use_mul_mat_q = use_mul_mat_q && (src1->ne[1] <= MMQ_MAX_BATCH_SIZE);
#endif // SYCL_USE_XMX

    // When reorder is enabled, both ESIMD, MMVQ and DMMV kernels may be used. For
    // best performance use ESIMD when supported, followed by MMVQ, and finally DMMV.
    // But the reordered ESIMD path cannot be used without reordered MMVQ. A later
    // multi-token call (ne[1] in 2..8) will take the MMVQ path and it would read the
    // reordered bytes as if they were still the unreordered layout.

    if (!g_ggml_sycl_prioritize_dmmv && ((should_reorder_tensor(ctx, dst) &&
                                          ggml_sycl_supports_reorder_mmvq(src0->type)))) {
        bool use = g_ggml_sycl_enable_esimd && ggml_sycl_supports_reorder_esimd(src0->type);
        // Arc770 get benefit with Q4_0 by skipping MMVQ path
        if (!(ggml_sycl_info().devices[ctx.device].hw_info.arch ==
                    gpu_arch::intel_gpu_acm_g10 &&
                src0->type == GGML_TYPE_Q4_0)) {
            use = use || !use_mul_mat_vec_q;
        }
        use_dequantize_mul_mat_vec = use_dequantize_mul_mat_vec && use;
    }

    static const bool small_q8_dmmv = ggml_sycl_get_env("GGML_SYCL_SMALL_Q8_DMMV", 0) != 0;
    if (small_q8_dmmv && !split && src0->type == GGML_TYPE_Q8_0 && src0->ne[1] <= 1024 &&
        src0->ne[0] <= 16384 && src1->ne[1] == 1 && use_dequantize_mul_mat_vec == false &&
        can_use_dequantize_mul_mat_vec(src0, src1, dst)) {
        use_dequantize_mul_mat_vec = true;
    }

    if (!split && src0->type == GGML_TYPE_F16 && ggml_is_permuted(src0) && ggml_is_permuted(src1) && src1->ne[1] == 1) {
        // TODO: Refactor and cleanup of mul mat dispatching.
        if (src0->ne[3] == 1 && src1->ne[3] == 1) {
            // KQ single-batch
            // mmv p021 was specific for these dimensions
            ggml_sycl_mul_mat_vec_p021(ctx, src0, src1, dst);
        } else {
            // The kernel from the if path is faster for that specific case, but does not support all mul mats.
            ggml_sycl_mul_mat_batched_sycl(ctx, src0, src1, dst);
        }
    } else if (!split && src0->type == GGML_TYPE_F16 && !ggml_is_contiguous(src0) && !ggml_is_transposed(src1) && src1->ne[1] == 1 && src1->ne[3] == 1) {
        // KQV single-batch
        ggml_sycl_mul_mat_vec_nc(ctx, src0, src1, dst);
    } else if (!split && src0->type == GGML_TYPE_F16 && !ggml_is_transposed(src0) && !ggml_is_transposed(src1) && src1->ne[2] * src1->ne[3] > 1) {
        // KQ + KQV multi-batch
        ggml_sycl_mul_mat_batched_sycl(ctx, src0, src1, dst);
    } else if (use_dequantize_mul_mat_vec) {
        opt_for_reorder(&ctx, src0, src1, dst, mul_mat_algo::DMMV);
        ggml_sycl_op_mul_mat<no_quantize_q8_1>(ctx, src0, src1, dst, ggml_sycl_op_dequantize_mul_mat_vec);
    } else if (use_mul_mat_vec_q) {
        opt_for_reorder(&ctx, src0, src1, dst, mul_mat_algo::MMVQ);
        ggml_tensor_extra_gpu * extra = static_cast<ggml_tensor_extra_gpu *>(src0->extra);
        if (extra && extra->optimized_feature.reorder) {
            ggml_sycl_op_mul_mat<quantize_and_reorder_q8_1_soa>(ctx, src0, src1, dst, ggml_sycl_op_mul_mat_vec_q);
        } else {
            ggml_sycl_op_mul_mat<quantize_q8_1>(ctx, src0, src1, dst, ggml_sycl_op_mul_mat_vec_q);
        }
    } else if (use_mul_mat_q) {
        ggml_sycl_op_mul_mat<quantize_q8_1>(ctx, src0, src1, dst, ggml_sycl_op_mul_mat_q);
    } else {
        ggml_sycl_op_mul_mat<no_quantize_q8_1>(ctx, src0, src1, dst, ggml_sycl_op_mul_mat_sycl);
    }
}

// {mul_mat(gate), mul_mat(up), GLU} over the standard (non-reorder) weight layout,
// for quant pairs the reorder kernel does not cover (mixed gate/up types, e.g. UD-Q4_K_XL's
// iq4_xs gate + q5_K up). Two launches replace five: one shared q8_1 quantization and one
// dual-GEMV+GLU.
static bool ggml_sycl_mul_mat_glu_mmvq_plain(ggml_backend_sycl_context & ctx, ggml_tensor * glu,
                                             ggml_tensor * gate, ggml_tensor * up, const ggml_tensor * wu,
                                             const ggml_tensor * wg, const ggml_tensor * act) {
    // weights already migrated to the reorder layout would be misread by the plain kernel
    const auto * extra_u = static_cast<const ggml_tensor_extra_gpu *>(wu->extra);
    const auto * extra_g = static_cast<const ggml_tensor_extra_gpu *>(wg->extra);
    if ((extra_u && extra_u->optimized_feature.reorder) || (extra_g && extra_g->optimized_feature.reorder)) {
        return false;
    }

    // the plain kernel only covers q5_K / iq4_xs pairs: bail before quantizing the activation
    const auto plain_type = [](ggml_type t) { return t == GGML_TYPE_Q5_K || t == GGML_TYPE_IQ4_XS; };
    if (!plain_type(wu->type) || !plain_type(wg->type)) {
        return false;
    }

    // log the up mat-mul: glu's own srcs are the two intermediates the fusion never materialises
    scope_op_debug_print scope_dbg_print(__func__, up, /*num_src=*/2, " : fused with gate + GLU (plain layout)");

    const int64_t ne00 = wu->ne[0];
    const int64_t ne11 = act->ne[1];

    const queue_ptr stream = ctx.stream();
    const int src1_padded_cols = GGML_PAD((int) ne00, MATRIX_ROW_PADDING);

    ggml_sycl_pool_alloc<char> src1_q8_alloc(ctx.pool(),
        (size_t) ne11 * src1_padded_cols * sizeof(block_q8_1) / QK8_1);
    char * src1_ddq = src1_q8_alloc.get();

    quantize_row_q8_1_sycl<quantize_q8_1>((const float *) act->data, src1_ddq, (int) ne00, (int) ne11,
                                          src1_padded_cols, stream);

    return ggml_sycl_mul_mat_vec_q_glu_plain(wg->type, wu->type, ggml_get_glu_op(glu), wg->data, wu->data,
                                             src1_ddq, (float *) glu->data, (int) ne00, (int) wu->ne[1],
                                             (int) ne11,
                                             /*stride_col_y=*/src1_padded_cols / QK8_1,
                                             /*stride_col_dst=*/(int) glu->ne[0], stream);
    GGML_UNUSED(gate);
}

// Gate and up weights of one type on the XMX engines: one launch reads the activations once and applies the GLU.
// Both weights must already be in the reorder layout. Returns false if the kernel declined.
static bool ggml_sycl_mul_mat_glu_xmx(ggml_backend_sycl_context & ctx,
                                      ggml_tensor *               glu,
                                      ggml_tensor *               up,
                                      const ggml_tensor *         wu,
                                      const ggml_tensor *         wg,
                                      const ggml_tensor *         act) {
    const int64_t ne00 = wu->ne[0];
    const int64_t ne11 = act->ne[1];

    const queue_ptr stream           = ctx.stream();
    const int       src1_padded_cols = GGML_PAD((int) ne00, MATRIX_ROW_PADDING);
    const int       stride_y_bytes   = src1_padded_cols * (int) sizeof(block_q8_1) / QK8_1;

    // log the up mat-mul: glu's own srcs are the two intermediates the fusion never materialises
    scope_op_debug_print scope_dbg_print(__func__, up, /*num_src=*/2, " : fused with gate + GLU (XMX)");

    ggml_sycl_pool_alloc<char> src1_q8_alloc(ctx.pool(), (size_t) ne11 * stride_y_bytes);
    char *                     src1_ddq = src1_q8_alloc.get();

    quantize_row_q8_1_sycl<quantize_and_reorder_q8_1_soa>((const float *) act->data, src1_ddq, (int) ne00, (int) ne11,
                                                          src1_padded_cols, stream);

    return ggml_sycl_mul_mat_vec_q_glu_xmx(ctx.device, wu->type, ggml_get_glu_op(glu), wu->data, wg->data, src1_ddq,
                                           (float *) glu->data, (int) ne00, (int) wu->ne[1], (int) ne11, stride_y_bytes,
                                           /*stride_col_dst=*/(int) glu->ne[0], stream);
}

#ifdef GGML_SYCL_DMMV_HAS_ESIMD
// {mul_mat(gate), mul_mat(up), GLU} on the reorder layout via the ESIMD DMMV kernel, for
// same-type K-quant or Q8_0 pairs and a single activation column. It reads the F32 activation
// directly, so one launch replaces three and no q8_1 quantization is needed.
static bool ggml_sycl_mul_mat_glu_dmmv_esimd(ggml_backend_sycl_context & ctx, ggml_tensor * glu,
                                             ggml_tensor * gate, ggml_tensor * up, const ggml_tensor * wu,
                                             const ggml_tensor * wg, const ggml_tensor * act) {
    // the kernel reads the activation while writing the output; glu may reuse the dead activation's memory
    const char * act_begin = (const char *) act->data;
    const char * glu_begin = (const char *) glu->data;
    if (act_begin < glu_begin + ggml_nbytes(glu) && glu_begin < act_begin + ggml_nbytes(act)) {
        return false;
    }

    // install the reorder (SoA) layout the kernel needs, as the unfused DMMV path would
    opt_for_reorder(&ctx, wu, act, up, mul_mat_algo::DMMV);
    opt_for_reorder(&ctx, wg, act, gate, mul_mat_algo::DMMV);

    const auto * extra_u = static_cast<const ggml_tensor_extra_gpu *>(wu->extra);
    const auto * extra_g = static_cast<const ggml_tensor_extra_gpu *>(wg->extra);
    if (!extra_u || !extra_g || !extra_u->optimized_feature.reorder || !extra_g->optimized_feature.reorder) {
        return false;
    }

    // log the up mat-mul: glu's own srcs are the two intermediates the fusion never materialises
    scope_op_debug_print scope_dbg_print(__func__, up, /*num_src=*/2, " : fused with gate + GLU (ESIMD)");

    return ggml_sycl_dequantize_mul_mat_vec_glu_reorder_esimd(wu->type, ggml_get_glu_op(glu), wu->data, wg->data,
                                                              (const float *) act->data, (float *) glu->data,
                                                              (int) wu->ne[0], (int) wu->ne[1], ctx.stream());
}
#endif

// Fused dense-FFN mat-vec for the {mul_mat(gate), mul_mat(up), GLU} subgraph at node_idx.
// Returns false if it declined, in which case the caller runs the three nodes normally.
static bool ggml_sycl_mul_mat_glu_fused(ggml_backend_sycl_context & ctx, ggml_cgraph * cgraph, int node_idx) {
    if (!ggml_sycl_can_fuse(cgraph, node_idx, { GGML_OP_MUL_MAT, GGML_OP_MUL_MAT, GGML_OP_GLU }, {})) {
        return false;
    }

    ggml_tensor *       glu  = cgraph->nodes[node_idx + 2];
    ggml_tensor *       gate = glu->src[0];
    ggml_tensor *       up   = glu->src[1];
    const ggml_tensor * wu   = up->src[0];
    const ggml_tensor * wg   = gate->src[0];
    const ggml_tensor * act  = up->src[1];

    // this writes glu->data directly rather than the per-device row slices that
    // ggml_sycl_op_mul_mat() stitches back together, so it cannot serve split weights
    if (ggml_backend_buffer_is_sycl_split(wu->buffer) || ggml_backend_buffer_is_sycl_split(wg->buffer)) {
        return false;
    }

#ifdef GGML_SYCL_DMMV_HAS_ESIMD
    // single-column same-type K-quant or Q8_0 pairs take the ESIMD kernel, as the unfused mul-mats
    // would via DMMV; if the reorder layout could not be installed, fall through to mmvq
    if (g_ggml_sycl_enable_esimd && wu->type == wg->type && act->ne[1] == 1 &&
        ggml_sycl_supports_reorder_esimd(wu->type) &&
        ggml_sycl_mul_mat_glu_dmmv_esimd(ctx, glu, gate, up, wu, wg, act)) {
        return true;
    }
#endif

    // with DMMV prioritised the unfused path would not have gone through mmvq at all
    if (g_ggml_sycl_prioritize_dmmv) {
        return false;
    }

    // gate and up of one type the XMX kernel handles, it needs the reorder layout like the unfused mmvq path
    if (wg->type == wu->type && gate->src[1] == act && act->ne[1] >= ggml_sycl_xmx_min_cols(wu->type) &&
        act->ne[1] <= g_ggml_sycl_xmx_glu_max_cols) {
        opt_for_reorder(&ctx, wu, act, up, mul_mat_algo::MMVQ);
        opt_for_reorder(&ctx, wg, act, gate, mul_mat_algo::MMVQ);
        if (can_use_xmx_batch(ctx.device, wu, act) && can_use_xmx_batch(ctx.device, wg, act) &&
            ggml_sycl_mul_mat_glu_xmx(ctx, glu, up, wu, wg, act)) {
            return true;
        }
    }

    // the kernels below take up to MMVQ_MAX_BATCH_SIZE columns, a wider batch got here for the XMX kernel
    if (act->ne[1] > MMVQ_MAX_BATCH_SIZE) {
        return false;
    }

    // quant pairs the reorder kernel does not serve (mixed gate/up types, q5_K off BMG) take the
    // standard-layout fused path instead; same-type q4_K / q5_K / q8_0 / q6_K keep the reorder path below
    static const bool q6_glu_reorder = ggml_sycl_get_env("GGML_SYCL_Q6_K_GLU_REORDER", 1) != 0;
    const bool reorder_pair = wg->type == wu->type &&
        (wu->type == GGML_TYPE_Q4_K || wu->type == GGML_TYPE_Q8_0 || (q6_glu_reorder && wu->type == GGML_TYPE_Q6_K) ||
         (wu->type == GGML_TYPE_Q5_K && ggml_sycl_q5_k_mmvq_reuse(ctx.device)));
    if (!reorder_pair) {
        const auto is_plain_type = [](ggml_type type) {
            return type == GGML_TYPE_Q5_K || type == GGML_TYPE_IQ4_XS;
        };
        if (!is_plain_type(wu->type) || !is_plain_type(wg->type)) {
            return false;
        }
        return ggml_sycl_mul_mat_glu_mmvq_plain(ctx, glu, gate, up, wu, wg, act);
    }

    // past 5 columns the two unfused q5_K GEMVs are faster than the fused kernel
    if (wu->type == GGML_TYPE_Q5_K && act->ne[1] > 5) {
        return false;
    }

    // install the reorder (SoA) layout the fused kernel needs, as the unfused mmvq path would;
    // a no-op once done. after the bail checks so a declined op does not pay for it.
    opt_for_reorder(&ctx, wu, act, up, mul_mat_algo::MMVQ);
    opt_for_reorder(&ctx, wg, act, gate, mul_mat_algo::MMVQ);

    const auto * extra_u = static_cast<const ggml_tensor_extra_gpu *>(wu->extra);
    const auto * extra_g = static_cast<const ggml_tensor_extra_gpu *>(wg->extra);
    if (!extra_u || !extra_g || !extra_u->optimized_feature.reorder || !extra_g->optimized_feature.reorder) {
        return false;
    }

    // log the up mat-mul: glu's own srcs are the two intermediates the fusion never materialises
    scope_op_debug_print scope_dbg_print(__func__, up, /*num_src=*/2, " : fused with gate + GLU");

    const int64_t ne00 = wu->ne[0];
    const int64_t ne11 = act->ne[1];

    const queue_ptr stream           = ctx.stream();
    const int       src1_padded_cols = GGML_PAD((int) ne00, MATRIX_ROW_PADDING);

    // one activation, quantized once and fully consumed into src1_ddq before the GEMV on this
    // in-order queue, so glu->data aliasing the dead activation needs no memory-range check
    ggml_sycl_pool_alloc<char> src1_q8_alloc(ctx.pool(),
                                             (size_t) ne11 * src1_padded_cols * sizeof(block_q8_1) / QK8_1);
    char *                     src1_ddq = src1_q8_alloc.get();

    quantize_row_q8_1_sycl<quantize_and_reorder_q8_1_soa>((const float *) act->data, src1_ddq, (int) ne00, (int) ne11,
                                                          src1_padded_cols, stream);

    return ggml_sycl_mul_mat_vec_q_glu_reorder(wu->type, ggml_get_glu_op(glu), wu->data, wg->data, src1_ddq,
                                               (float *) glu->data, (int) ne00, (int) wu->ne[1], (int) ne11,
                                               /*stride_col_y_bytes=*/src1_padded_cols * (int) sizeof(block_q8_1) /
                                                   QK8_1,
                                               /*stride_col_dst=*/(int) glu->ne[0], stream);
}

// Batch the run of consecutive L2_NORM siblings starting at node_idx into one launch.
// Returns the number of extra graph nodes consumed, or 0 if the run is shorter than two
// (the caller then runs the norm through the per-tensor kernel).
static int ggml_sycl_l2_norm_batch_fused(ggml_backend_sycl_context & ctx, ggml_cgraph * cgraph, int node_idx) {
    const ggml_tensor * node = cgraph->nodes[node_idx];
    if (ggml_sycl_info().device_count != 1 || node->type != GGML_TYPE_F32 ||
        node->src[0]->type != GGML_TYPE_F32 || node->src[0]->ne[0] >= 1024) {
        return 0;
    }

    ggml_tensor * batch[GGML_SYCL_L2_BATCH_MAX];
    int           count = 0;
    int           last  = node_idx;
    float         eps0;
    memcpy(&eps0, node->op_params, sizeof(float));

    // Conservative aliasing test: the batched norms run concurrently in one kernel,
    // so none may read what another writes, and none may write where another writes.
    auto overlaps = [](const ggml_tensor * a, const ggml_tensor * b) {
        const char * ab = (const char *) a->data;
        const char * bb = (const char *) b->data;
        return ab < bb + ggml_nbytes(b) && bb < ab + ggml_nbytes(a);
    };

    for (int j = node_idx; j < cgraph->n_nodes && count < GGML_SYCL_L2_BATCH_MAX; ++j) {
        ggml_tensor * nj = cgraph->nodes[j];
        if (ggml_is_empty(nj) || nj->op == GGML_OP_RESHAPE || nj->op == GGML_OP_TRANSPOSE ||
            nj->op == GGML_OP_VIEW || nj->op == GGML_OP_PERMUTE || nj->op == GGML_OP_NONE ||
            (nj->flags & GGML_TENSOR_FLAG_COMPUTE) == 0) {
            continue;  // not a launch; cannot break a run of adjacent norms
        }
        if (nj->op != GGML_OP_L2_NORM || nj->type != GGML_TYPE_F32 ||
            nj->src[0]->type != GGML_TYPE_F32 || !ggml_are_same_shape(nj, node) ||
            !ggml_are_same_shape(nj->src[0], node->src[0])) {
            break;  // any other launch ends the run
        }
        bool same_nb = true;
        for (int d = 0; d < GGML_MAX_DIMS; ++d) {
            if (nj->nb[d] != node->nb[d] || nj->src[0]->nb[d] != node->src[0]->nb[d]) {
                same_nb = false;
                break;
            }
        }
        if (!same_nb) {
            break;  // one nb[] stride set is shared by the whole batch
        }
        float epsj;
        memcpy(&epsj, nj->op_params, sizeof(float));
        if (epsj != eps0) {
            break;  // eps mismatch ends the run
        }
        bool indep = true;
        for (int k = 0; k < count; ++k) {
            if (overlaps(nj->src[0], batch[k]) || overlaps(nj, batch[k])) {
                indep = false;
                break;
            }
        }
        if (!indep) {
            break;  // an overlapping tensor would race inside one launch
        }
        batch[count++] = nj;
        last           = j;
    }
    if (count < 2) {
        return 0;  // a lone norm falls through to the per-tensor kernel
    }
    ggml_sycl_l2_norm_batch(ctx, batch, count);
    return last - node_idx;
}


__dpct_inline__ static void k_copy_src1_to_contiguous(
    const char *__restrict__ src1_original, char *__restrict__ src1_contiguous,
    const mmid_row_mapping *__restrict__ row_mapping,
    int64_t ne11, int64_t ne10, size_t nb11, size_t nb12,
    const sycl::nd_item<3> &item_ct1) {
    const int32_t src1_row = item_ct1.get_group(2);

    const int32_t iid1 = row_mapping[src1_row].i2;
    const int32_t id   = row_mapping[src1_row].i1;

    float * src1_row_contiguous = (float *)(src1_contiguous + src1_row*nb11);
    if (id < 0) {
        // padding row of a tile-aligned expert block
        for (int i = item_ct1.get_local_id(2); i < ne10; i += item_ct1.get_local_range(2)) {
            src1_row_contiguous[i] = 0.0f;
        }
        return;
    }

    const int64_t i11 = id % ne11;
    const int64_t i12 = iid1;

    const float * src1_row_original = (const float *)(src1_original + i11*nb11 + i12*nb12);

#pragma unroll
    for (int i = item_ct1.get_local_id(2); i < ne10;
         i += item_ct1.get_local_range(2)) {
        src1_row_contiguous[i] = src1_row_original[i];
    }
}

__dpct_inline__ static void k_copy_dst_from_contiguous(
    char *__restrict__ dst_original, const char *__restrict__ dst_contiguous,
    const mmid_row_mapping *__restrict__ row_mapping, int64_t ne0, size_t nb1,
    size_t nb2, const sycl::nd_item<3> &item_ct1) {
    int32_t i = item_ct1.get_group(2);

    const int32_t i1 = row_mapping[i].i1;
    const int32_t i2 = row_mapping[i].i2;
    if (i1 < 0) {
        return; // padding row
    }

    const float * dst_row_contiguous = (const float *)(dst_contiguous + i*nb1);
    float * dst_row_original = (float *)(dst_original + i1*nb1 + i2*nb2);

#pragma unroll
    for (int j = item_ct1.get_local_id(2); j < ne0;
         j += item_ct1.get_local_range(2)) {
        dst_row_original[j] = dst_row_contiguous[j];
    }
}

// Fused MoE TG fast path. Returns false to fall back to the per-expert loop below.
// Shape/type preconditions of the fused single-token MoE GEMV (no host sync on this path).
static bool ggml_sycl_mmid_fused_applicable(const ggml_tensor * src0, const ggml_tensor * src1,
                                            const ggml_tensor * ids, const ggml_tensor * dst) {
    const int64_t ne10 = src1->ne[0];
    const int64_t ne11 = src1->ne[1];
    // one token, or a small batch: every (token, slot) pair is one kernel entry, laid out token-major exactly like
    // ids / dst. Large batches stay on the per-expert GEMM loop, which shares each expert's dequantized weights.
    static const int max_tokens = ggml_sycl_get_env("GGML_SYCL_MMID_FUSED_MAX_TOKENS", 64);
    const int64_t n_tokens = src1->ne[2];
    int why = 0;
    if (n_tokens != 1 && (n_tokens > max_tokens || src1->ne[3] != 1)) why = 1;
    else if (src1->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32) why = 2;
    else if (ne10 != src0->ne[0] || ne10 % QK8_1 != 0) why = 3;
    else if (!ggml_is_contiguous(src1)) why = 4;
    else if (ids->ne[1] != n_tokens) why = 5;
    else if (ne11 != 1 && ne11 != ids->ne[0]) why = 6;
    else if (n_tokens > 1) {
        // ids may be a strided view (qwen4exp: rows of the full top-k result); they are compacted on the device
        if (ids->type != GGML_TYPE_I32 || ids->nb[0] != sizeof(int32_t) || ids->nb[1] < (size_t) ids->ne[0] * sizeof(int32_t) ||
            ids->nb[1] % sizeof(int32_t) != 0) why = 7;
        else if (!ggml_is_contiguous(dst) || dst->ne[1] != ids->ne[0] || dst->ne[2] != n_tokens) why = 8;
        else if (!ggml_sycl_mul_mat_vec_q_id_supports_type(src0->type)) why = 9;
    }
    static const bool fused_debug = ggml_sycl_get_env("GGML_SYCL_MMID_FUSED_DEBUG", 0) != 0;
    static std::atomic<int> fused_debug_count{0};
    if (fused_debug && n_tokens > 1 && n_tokens <= 64 && fused_debug_count.fetch_add(1) < 24) {
        fprintf(stderr, "[SYCL] MMID fused-check why=%d max_tokens=%d w=%s src1 ne=[%lld,%lld,%lld,%lld] nb=[%zu,%zu,%zu,%zu] ids type=%s ne=[%lld,%lld] nb=[%zu,%zu] dst ne=[%lld,%lld,%lld] nb=[%zu,%zu,%zu] op1=%s\n",
                why, max_tokens, ggml_type_name(src0->type),
                (long long) src1->ne[0], (long long) src1->ne[1], (long long) src1->ne[2], (long long) src1->ne[3],
                src1->nb[0], src1->nb[1], src1->nb[2], src1->nb[3],
                ggml_type_name(ids->type), (long long) ids->ne[0], (long long) ids->ne[1], ids->nb[0], ids->nb[1],
                (long long) dst->ne[0], (long long) dst->ne[1], (long long) dst->ne[2], dst->nb[0], dst->nb[1], dst->nb[2],
                ggml_op_name(src1->op));
    }
    return why == 0;
}

static bool ggml_sycl_mul_mat_id_mmvq_fused(
    ggml_backend_sycl_context & ctx, const ggml_tensor * src0,
    const ggml_tensor * src1, const ggml_tensor * ids, ggml_tensor * dst)
{
    if (!ggml_sycl_mmid_fused_applicable(src0, src1, ids, dst)) return false;
    const int64_t ne10 = src1->ne[0];
    const int64_t ne11 = src1->ne[1];
    const int64_t n_ids_per_group = ids->ne[0];
    const int64_t n_tokens        = src1->ne[2];

    const queue_ptr stream           = ctx.stream();
    const int       src1_padded_cols = GGML_PAD((int) ne10, MATRIX_ROW_PADDING);
    const int       n_experts_used   = (int) (n_ids_per_group * n_tokens);   // kernel entries
    const int       nrows            = (int) src0->ne[1];

    // Lazily reorder supported expert weights into a per-expert SoA layout, then run the reorder
    // GEMV. Placed after the bail checks so a non-dispatchable op does not pay the reorder cost.
    opt_for_reorder_id(&ctx, src0);
    const ggml_tensor_extra_gpu * src0_extra =
        static_cast<const ggml_tensor_extra_gpu *>(src0->extra);
    const bool use_reorder = src0_extra && src0_extra->optimized_feature.reorder;

    const int n_src1_rows = (int) (ne11 * n_tokens);
    ggml_sycl_pool_alloc<char> src1_q8_alloc(ctx.pool(),
        (size_t) n_src1_rows * src1_padded_cols * sizeof(block_q8_1) / QK8_1);
    char * src1_ddq = src1_q8_alloc.get();
    if (use_reorder) {
        quantize_row_q8_1_sycl<quantize_and_reorder_q8_1_soa>(
            (const float *) src1->data, src1_ddq, (int) ne10, n_src1_rows,
            src1_padded_cols, stream);
    } else {
        quantize_row_q8_1_sycl<quantize_q8_1>(
            (const float *) src1->data, src1_ddq, (int) ne10, n_src1_rows,
            src1_padded_cols, stream);
    }

    // strided ids rows: gather them into a contiguous [n_ids, n_tokens] buffer for the kernels
    ggml_sycl_pool_alloc<int32_t> ids_compact(ctx.pool());
    const int32_t * ids_dev = (const int32_t *) ids->data;
    if (n_tokens > 1 && ids->nb[1] != (size_t) n_ids_per_group * sizeof(int32_t)) {
        ids_compact.alloc((size_t) n_experts_used);
        int32_t * ids_out             = ids_compact.get();
        const int32_t * ids_in        = (const int32_t *) ids->data;
        const int64_t   ids_row_elems = (int64_t) (ids->nb[1] / sizeof(int32_t));
        const int64_t   n_per_row     = n_ids_per_group;
        stream->parallel_for(sycl::range<1>((size_t) n_experts_used), [=](sycl::id<1> idx) {
            const int64_t e = (int64_t) idx[0];
            ids_out[e] = ids_in[(e / n_per_row) * ids_row_elems + e % n_per_row];
        });
        ids_dev = ids_out;
    }

    const size_t bytes_per_qrow = (size_t) src1_padded_cols * sizeof(block_q8_1) / QK8_1;
    // one activation row for everything (single token, shared), one per token (shared, batch), or one per entry
    const size_t src1_row_stride = n_src1_rows == 1 ? 0 : bytes_per_qrow;
    const int    src1_row_div    = ne11 == 1 ? (int) n_ids_per_group : 1;

    if (use_reorder) {
        return ggml_sycl_mul_mat_vec_q_id_reorder(
            src0->type, src0->data, src1_ddq, ids_dev,
            (float *) dst->data, (int) ne10, nrows, n_experts_used,
            /*expert_weight_stride=*/ src0->nb[2],
            /*dst_row_stride=*/ dst->nb[1],
            src1_row_stride, src1_row_div, stream);
    }
    return ggml_sycl_mul_mat_vec_q_id(
        src0->type, src0->data, src1_ddq, ids_dev,
        (float *) dst->data, (int) ne10, nrows, n_experts_used,
        /*expert_weight_stride=*/ src0->nb[2],
        /*dst_row_stride=*/ dst->nb[1],
        src1_row_stride, src1_row_div, stream);
}

// convert.cpp
void ggml_sycl_dequantize_q4_K_reorder_slices(const void * vx, sycl::half * y, const int64_t k, const size_t slice_bytes,
                                              const int64_t n_slices, dpct::queue_ptr stream);
void ggml_sycl_dequantize_q4_K_reorder_indexed_slices(const void * vx, sycl::half * y,
    const int64_t k, const size_t slice_bytes, const int64_t * slice_ids, const int64_t n_slices,
    const int64_t output_first, dpct::queue_ptr stream);

// Bumped at the start of every graph compute; lets per-graph caches detect a new graph.
static std::atomic<uint64_t> g_ggml_sycl_graph_serial[GGML_SYCL_MAX_DEVICES];   // per device: cards may compute concurrently

struct mmid_compact_tile {
    int64_t expert_id;
    int64_t token_tile;
    int64_t row_count;
    int64_t row_offset;
};

// Routing of one ids tensor, reusable by the MUL_MAT_ID nodes that share it within a graph compute
// (gate / up / down projections of one MoE layer).
struct ggml_sycl_mmid_route {
    const ggml_tensor * ids      = nullptr;
    const void *        ids_data = nullptr;
    uint64_t            serial   = 0;
    int64_t             n_as     = 0;
    std::vector<char>             ids_host;
    std::vector<int64_t>          expert_row_counts;
    std::vector<int64_t>          expert_row_offsets;
    std::vector<mmid_row_mapping> routed_row_src;
    mmid_row_mapping *  dev_mapping     = nullptr;   // device copy of routed_row_src
    size_t              dev_mapping_cap = 0;
    sycl::queue *       dev_queue       = nullptr;
    bool                sorted          = false;
    std::vector<mmid_compact_tile> compact_tiles;
    std::vector<int64_t> compact_expert_tile_offsets;
    mmid_compact_tile * dev_compact_tiles = nullptr;
    size_t compact_tiles_cap = 0;
    sycl::queue * compact_tiles_queue = nullptr;
    std::vector<int64_t> selective_expert_ids;
    int64_t * dev_selective_expert_ids = nullptr;
    size_t selective_expert_ids_cap = 0;
    sycl::queue * selective_expert_ids_queue = nullptr;
    bool selective_expert_ids_ready = false;
    // small-first layout: the rows of experts with few routed rows come first (one fused MMVQ launch)
    std::vector<int32_t> small_row_ids;        // expert id of each small row
    int32_t * dev_small_row_ids = nullptr;     // device copy, owned by dev_queue
    size_t    small_row_ids_cap = 0;
    int64_t   n_small_rows  = 0;
    int64_t   n_layout_rows = 0;               // routed rows + padding rows
};
static ggml_sycl_mmid_route g_ggml_sycl_mmid_routes[GGML_SYCL_MAX_DEVICES];

// counting sort of the routed rows by expert id (row_id_i, as chosen by the router):
// builds a projection of a memory layout where each expert's slice is contiguous
static void mmid_counting_sort_rows(
        const ggml_tensor * ids, const char * ids_host,
        int64_t n_ids, int64_t n_as, int64_t n_routed_rows,
        int64_t small_max, bool pad_tiles,
        std::vector<int64_t> & expert_counts,
        std::vector<int64_t> & expert_row_offsets,
        std::vector<mmid_row_mapping> & routed_row_src,
        int64_t & n_small_rows, int64_t & n_layout_rows) {

    // frequencies: how many routed rows each expert "owns"
    expert_counts.assign(n_as, 0);
    for (int64_t iid1 = 0; iid1 < ids->ne[1]; iid1++) {
        for (int64_t id = 0; id < n_ids; id++) {
            const int32_t row_id_i = *(const int32_t *) (ids_host + iid1*ids->nb[1] + id*ids->nb[0]);
            GGML_ASSERT(row_id_i >= 0 && row_id_i < n_as);
            expert_counts[row_id_i]++;
        }
    }

    // where each expert's slice starts (row indices)
    expert_row_offsets.assign(n_as + 1, 0);
    n_small_rows = 0;
    if (small_max < 0) {
        for (int64_t i02 = 0; i02 < n_as; i02++) {
            expert_row_offsets[i02 + 1] = expert_row_offsets[i02] + expert_counts[i02];
        }
        n_layout_rows = n_routed_rows;
    } else {
        // small experts first, then the large ones (optionally each padded to full 8-row tiles)
        int64_t next = 0;
        for (int64_t i02 = 0; i02 < n_as; i02++) {
            if (expert_counts[i02] <= small_max) {
                expert_row_offsets[i02] = next;
                next += expert_counts[i02];
            }
        }
        n_small_rows = next;
        for (int64_t i02 = 0; i02 < n_as; i02++) {
            if (expert_counts[i02] > small_max) {
                expert_row_offsets[i02] = next;
                next += pad_tiles ? (expert_counts[i02] + 7) / 8 * 8 : expert_counts[i02];
            }
        }
        expert_row_offsets[n_as] = next;
        n_layout_rows = next;
    }

    std::vector<int64_t> expert_row_next = expert_row_offsets;
    routed_row_src.assign(n_layout_rows, mmid_row_mapping{-1, -1});
    for (int64_t iid1 = 0; iid1 < ids->ne[1]; iid1++) {
        for (int64_t id = 0; id < n_ids; id++) {
            const int32_t row_id_i = *(const int32_t *) (ids_host + iid1*ids->nb[1] + id*ids->nb[0]);
            GGML_ASSERT(row_id_i >= 0 && row_id_i < n_as);

            // find and validate the next free row for a given expert (row_id_i)
            const int64_t routed_row = expert_row_next[row_id_i]++;
            GGML_ASSERT(routed_row >= expert_row_offsets[row_id_i]);
            GGML_ASSERT(routed_row < expert_row_offsets[row_id_i] + expert_counts[row_id_i]);
            routed_row_src[routed_row] = {(int32_t) id, (int32_t) iid1};
        }
    }
}

static void ggml_sycl_mul_mat_id(ggml_backend_sycl_context & ctx,
                                 ggml_tensor *dst) try {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/3);
    const ggml_tensor *src0 = dst->src[0];
    const ggml_tensor *src1 = dst->src[1];
    GGML_ASSERT(!ggml_backend_buffer_is_sycl_split(src0->buffer) && "mul_mat_id does not support split buffers");

    const ggml_tensor *ids = dst->src[2];
    GGML_TENSOR_BINARY_OP_LOCALS

    const queue_ptr stream = ctx.stream();

    const int64_t n_as = ne02;
    const int64_t n_ids = ids->ne[0];

    if (ggml_sycl_mul_mat_id_mmvq_fused(ctx, src0, src1, ids, dst)) {
        return;
    }

    static const bool test_reorder_enabled = ggml_sycl_get_env("GGML_SYCL_MMID_TEST_REORDER", 0) != 0;
    if (test_reorder_enabled && src0->type == GGML_TYPE_Q4_K && ids->ne[1] > MMVQ_MAX_BATCH_SIZE &&
        ggml_is_contiguous(src0) && src1->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32 &&
        ne01 == ne0 && ne01 % 16 == 0 && ne00 == ne10 && ne00 % 16 == 0 &&
        nb11 == sizeof(float) * ne10 && nb1 == sizeof(float) * ne0 &&
        dst->op_params[0] == GGML_PREC_DEFAULT &&
        ggml_sycl_mmid_reorder_pending(&ctx, src0)) {
        opt_for_reorder_id(&ctx, src0);
    }
    // The per-expert loop below reads the experts in whatever layout they have: reorder MXFP4 here as well, so prompt processing does not depend on a single-token decode having run first.
    if (src0->type == GGML_TYPE_MXFP4) {
        opt_for_reorder_id(&ctx, src0);
    }


    static const bool route_cache_enabled = ggml_sycl_get_env("GGML_SYCL_MMID_ROUTE_CACHE", 1) != 0;
    ggml_sycl_mmid_route & route = g_ggml_sycl_mmid_routes[ctx.device];
    static const bool selective_dequant_enabled =
        ggml_sycl_get_env("GGML_SYCL_MMID_SELECTIVE_DEQUANT", 0) != 0;
    static const bool small_fused_enabled = ggml_sycl_get_env("GGML_SYCL_MMID_SMALL_FUSED", 0) != 0;
    static const bool pad_tiles_enabled   = ggml_sycl_get_env("GGML_SYCL_MMID_PAD_TILES", 0) != 0;
    // experts with more routed rows than this take the GEMM / routed paths
    static const int64_t small_max = std::max(0, std::min((int) MMVQ_MAX_BATCH_SIZE,
        ggml_sycl_get_env("GGML_SYCL_MMID_SMALL_MAX_ROWS", (int) MMVQ_MAX_BATCH_SIZE)));
    const int64_t small_rows_max = small_max; // kernel-capturable copy
    const bool    tiles_padded   = pad_tiles_enabled;
    const uint64_t graph_serial = g_ggml_sycl_graph_serial[ctx.device].load();
    const bool route_hit = route_cache_enabled && route.ids == ids && route.ids_data == ids->data &&
                           route.serial == graph_serial && route.n_as == n_as &&
                           route.ids_host.size() == ggml_nbytes(ids) &&
                           (route.dev_queue == nullptr || route.dev_queue == stream) &&
                           (route.selective_expert_ids_queue == nullptr || route.selective_expert_ids_queue == stream) &&
                           (!selective_dequant_enabled || route.selective_expert_ids_ready);
    if (!route_hit) {
        // the queue must be drained before the host vectors of the previous route are overwritten: their upload
        // to the device may still be pending
        if (route.selective_expert_ids_queue != nullptr && route.selective_expert_ids_queue != stream) {
            { ggml_sycl_pipe_wait_scope ws_; SYCL_CHECK(CHECK_TRY_ERROR(route.selective_expert_ids_queue->wait())); }
            if (route.dev_selective_expert_ids != nullptr) {
                sycl::free(route.dev_selective_expert_ids, *route.selective_expert_ids_queue);
                route.dev_selective_expert_ids = nullptr;
                route.selective_expert_ids_cap = 0;
            }
            route.selective_expert_ids_queue = nullptr;
        }
        if (route.dev_queue != nullptr && route.dev_queue != stream) {
            { ggml_sycl_pipe_wait_scope ws_; SYCL_CHECK(CHECK_TRY_ERROR(route.dev_queue->wait())); }
        }
        { ggml_sycl_pipe_wait_scope ws_; SYCL_CHECK(CHECK_TRY_ERROR(stream->wait())); }
        route.ids      = ids;
        route.ids_data = ids->data;
        route.serial   = graph_serial;
        route.n_as     = n_as;
        route.sorted   = false;
        route.selective_expert_ids_ready = false;
        route.ids_host.resize(ggml_nbytes(ids));
        SYCL_CHECK(CHECK_TRY_ERROR(
            stream->memcpy(route.ids_host.data(), ids->data, ggml_nbytes(ids))));
        { ggml_sycl_pipe_wait_scope ws_; SYCL_CHECK(CHECK_TRY_ERROR(stream->wait())); }
    }
    std::vector<char> & ids_host = route.ids_host;

    ggml_tensor src0_row = *src0;
    ggml_tensor src1_row = *src1;
    ggml_tensor dst_row = *dst;

    char *src0_original = (char *)src0->data;
    char *src1_original = (char *)src1->data;
    char *dst_original = (char *)dst->data;

    src0_row.ne[2] = 1;
    src0_row.ne[3] = 1;
    src0_row.nb[3] = nb02;

    src1_row.ne[1] = 1;
    src1_row.ne[2] = 1;
    src1_row.ne[3] = 1;
    src1_row.nb[2] = nb11;
    src1_row.nb[3] = nb11;

    dst_row.ne[1] = 1;
    dst_row.ne[2] = 1;
    dst_row.ne[3] = 1;
    dst_row.nb[2] = nb1;
    dst_row.nb[3] = nb1;
    if (ne12 == 1) {
        for (int64_t iid1 = 0; iid1 < ids->ne[1]; iid1++) {
            for (int64_t id = 0; id < n_ids; id++) {
                const int32_t i02 = *(const int32_t *) (ids_host.data() + iid1*ids->nb[1] + id*ids->nb[0]);
                GGML_ASSERT(i02 >= 0 && i02 < n_as);

                const int64_t i11 = id % ne11;
                const int64_t i12 = iid1;

                const int64_t i1 = id;
                const int64_t i2 = i12;

            src0_row.data = src0_original + i02*nb02;
            src1_row.data = src1_original + i11*nb11 + i12*nb12;
            dst_row.data = dst_original + i1*nb1 + i2*nb2;

            ggml_sycl_mul_mat(ctx, &src0_row, &src1_row, &dst_row);
            }
        }
    } else {
        const int64_t n_routed_rows = ids->ne[1] * n_ids;
        ggml_sycl_pool_alloc<char> src1_contiguous(ctx.pool());
        ggml_sycl_pool_alloc<char>  dst_contiguous(ctx.pool());

        // how many "owned" routed rows to pass to each expert, where each expert's slice starts, and the sources
        // (slot/token pairs) of the contiguous rows; computed once per ids tensor and graph (see the route cache)
        std::vector<int64_t> &          expert_row_counts  = route.expert_row_counts;
        std::vector<int64_t> &          expert_row_offsets = route.expert_row_offsets;
        std::vector<mmid_row_mapping> & routed_row_src     = route.routed_row_src;

        if (!route.sorted) {
            mmid_counting_sort_rows(ids, ids_host.data(), n_ids, n_as, n_routed_rows,
                                    ((small_fused_enabled || pad_tiles_enabled) &&
                                     ids->ne[1] < ggml_sycl_get_env("GGML_SYCL_MMID_GROUPED_MIN_TOKENS", 32)) ? small_max : -1,
                                    pad_tiles_enabled,
                                    expert_row_counts, expert_row_offsets, routed_row_src,
                                    route.n_small_rows, route.n_layout_rows);
            route.small_row_ids.clear();
            if (small_fused_enabled && route.n_small_rows > 0) {
                route.small_row_ids.resize((size_t) route.n_small_rows);
                for (int64_t expert = 0; expert < n_as; ++expert) {
                    if (expert_row_counts[expert] <= small_max) {
                        for (int64_t r = 0; r < expert_row_counts[expert]; ++r) {
                            route.small_row_ids[(size_t) (expert_row_offsets[expert] + r)] = (int32_t) expert;
                        }
                    }
                }
                if (route.dev_small_row_ids == nullptr || route.small_row_ids_cap < route.small_row_ids.size() ||
                    route.dev_queue != stream) {
                    if (route.dev_small_row_ids != nullptr) {
                        sycl::free(route.dev_small_row_ids, *route.dev_queue);
                    }
                    route.small_row_ids_cap = route.small_row_ids.size();
                    route.dev_small_row_ids = sycl::malloc_device<int32_t>(route.small_row_ids_cap, *stream);
                    GGML_ASSERT(route.dev_small_row_ids != nullptr);
                }
                SYCL_CHECK(CHECK_TRY_ERROR(stream->memcpy(route.dev_small_row_ids, route.small_row_ids.data(),
                    route.small_row_ids.size() * sizeof(int32_t))));
            } else if (route.dev_small_row_ids != nullptr && route.dev_queue != stream) {
                sycl::free(route.dev_small_row_ids, *route.dev_queue);
                route.dev_small_row_ids = nullptr;
                route.small_row_ids_cap = 0;
            }
            if (selective_dequant_enabled) {
                route.selective_expert_ids.clear();
                for (int64_t expert = 0; expert < n_as; ++expert) {
                    if (expert_row_counts[expert] > small_max) {
                        route.selective_expert_ids.push_back(expert);
                    }
                }
                const size_t required = route.selective_expert_ids.size();
                if (required > 0) {
                    if (route.dev_selective_expert_ids == nullptr || route.selective_expert_ids_cap < required) {
                        if (route.dev_selective_expert_ids != nullptr) {
                            { ggml_sycl_pipe_wait_scope ws_; SYCL_CHECK(CHECK_TRY_ERROR(stream->wait())); }
                            sycl::free(route.dev_selective_expert_ids, *route.selective_expert_ids_queue);
                        }
                        route.dev_selective_expert_ids = sycl::malloc_device<int64_t>(required, *stream);
                        GGML_ASSERT(route.dev_selective_expert_ids != nullptr);
                        route.selective_expert_ids_cap = required;
                        route.selective_expert_ids_queue = stream;
                    }
                    GGML_ASSERT(route.selective_expert_ids_queue == stream);
                    SYCL_CHECK(CHECK_TRY_ERROR(stream->memcpy(route.dev_selective_expert_ids,
                        route.selective_expert_ids.data(), required * sizeof(int64_t))));
                }
                route.selective_expert_ids_ready = true;
            }
            if (route.dev_mapping == nullptr || route.dev_mapping_cap < (size_t) route.n_layout_rows || route.dev_queue != stream) {
                if (route.dev_mapping != nullptr) {
                    sycl::free(route.dev_mapping, *route.dev_queue);
                }
                route.dev_mapping_cap = (size_t) route.n_layout_rows;
                route.dev_queue       = stream;
                route.dev_mapping     = sycl::malloc_device<mmid_row_mapping>(route.dev_mapping_cap, *stream);
                GGML_ASSERT(route.dev_mapping != nullptr);
            }
            SYCL_CHECK(CHECK_TRY_ERROR(
                    stream->memcpy(route.dev_mapping, routed_row_src.data(), route.n_layout_rows*sizeof(mmid_row_mapping))));
            static const bool compact_metadata_enabled =
                ggml_sycl_get_env("GGML_SYCL_MMID_ROUTED_COMPACT", 0) != 0 ||
                ggml_sycl_get_env("GGML_SYCL_MMID_ROUTED_TILE_STATS", 0) != 0;
            if (compact_metadata_enabled) {
                if (route.compact_tiles_queue != nullptr && route.compact_tiles_queue != stream) {
                    { ggml_sycl_pipe_wait_scope ws_; SYCL_CHECK(CHECK_TRY_ERROR(route.compact_tiles_queue->wait())); }
                }
                route.compact_tiles.clear();
                route.compact_expert_tile_offsets.assign((size_t)n_as + 1, 0);
                for (int64_t expert = 0; expert < n_as; ++expert) {
                    const int64_t rows = expert_row_counts[expert];
                    const int64_t tiles = rows > small_max ? (rows + 7) / 8 : 0;
                    route.compact_expert_tile_offsets[expert + 1] =
                        route.compact_expert_tile_offsets[expert] + tiles;
                    for (int64_t tile = 0; tile < tiles; ++tile) {
                        route.compact_tiles.push_back({expert, tile, pad_tiles_enabled ? tiles * 8 : rows, expert_row_offsets[expert]});
                    }
                }
                if (!route.compact_tiles.empty()) {
                    const size_t required = route.compact_tiles.size();
                    if (route.dev_compact_tiles == nullptr || route.compact_tiles_cap < required ||
                        route.compact_tiles_queue != stream) {
                        if (route.dev_compact_tiles != nullptr) {
                            if (route.compact_tiles_queue != stream) {
                                { ggml_sycl_pipe_wait_scope ws_; SYCL_CHECK(CHECK_TRY_ERROR(route.compact_tiles_queue->wait())); }
                            }
                            sycl::free(route.dev_compact_tiles, *route.compact_tiles_queue);
                        }
                        route.dev_compact_tiles = sycl::malloc_device<mmid_compact_tile>(required, *stream);
                        GGML_ASSERT(route.dev_compact_tiles != nullptr);
                        route.compact_tiles_cap = required;
                        route.compact_tiles_queue = stream;
                    }
                    SYCL_CHECK(CHECK_TRY_ERROR(stream->memcpy(route.dev_compact_tiles, route.compact_tiles.data(),
                        required * sizeof(mmid_compact_tile))));
                }
            }
            route.sorted = true;
        }
        struct { mmid_row_mapping * p; mmid_row_mapping * get() const { return p; } } dev_row_mapping = { route.dev_mapping };

        const int64_t n_layout_rows = route.n_layout_rows;
        GGML_ASSERT(n_layout_rows >= n_routed_rows);
        src1_contiguous.alloc(sizeof(float)*n_layout_rows*ne10);
        dst_contiguous.alloc(sizeof(float)*n_layout_rows*ne0);
        src1_row.data = src1_contiguous.get();
        dst_row.data  =  dst_contiguous.get();

        const unsigned int max_work_group_size = ggml_sycl_info().max_work_group_sizes[ctx.device];
        assert(max_work_group_size % (WARP_SIZE * WARP_SIZE) == 0);

        {
            sycl::range<3> block_dims(1, 1, std::min((unsigned int)ne10, max_work_group_size));
            sycl::range<3> grid_dims(1, 1, n_layout_rows);
            stream->submit([&](sycl::handler &cgh) {
                char *__restrict src1_contiguous_get =
                    src1_contiguous.get();
                mmid_row_mapping *__restrict dev_row_mapping_get =
                    dev_row_mapping.get();

                cgh.parallel_for(
                    sycl::nd_range<3>(grid_dims * block_dims, block_dims),
                    [=](sycl::nd_item<3> item_ct1) {
                        k_copy_src1_to_contiguous(
                            src1_original, src1_contiguous_get,
                            dev_row_mapping_get,
                            ne11, ne10, nb11, nb12,
                            item_ct1);
                    });
            });
        }

        // grouped XMX GEMM over every routed expert (upstream #29245); reads the quantized experts directly.
        // Large batches only: decode-sized batches keep the fork's small-expert and direct-GEMM paths.
        bool grouped = false;
        static const bool grouped_enabled = ggml_sycl_get_env("GGML_SYCL_MMID_GROUPED", 1) != 0;
        static const int  grouped_min_tokens = ggml_sycl_get_env("GGML_SYCL_MMID_GROUPED_MIN_TOKENS", 32);
        if (grouped_enabled && ids->ne[1] >= grouped_min_tokens && n_layout_rows == n_routed_rows &&
            ggml_is_contiguous(src0) && src1->type == GGML_TYPE_F32 &&
            dst->type == GGML_TYPE_F32 && nb11 == sizeof(float)*ne10 && nb1 == sizeof(float)*ne0 &&
            g_ggml_sycl_dynamic_precision != GGML_SYCL_DYNAMIC_PRECISION_F32 &&
            ggml_sycl_xmx_gather_type_enabled(src0->type)) {
            const bool src0_reordered =
                src0->extra && ((const ggml_tensor_extra_gpu *) src0->extra)->optimized_feature.reorder;
            grouped = ggml_sycl_grouped_dequant_gemm(src0->type, src0_reordered, src0_original, nb02,
                                                     (const float *) src1_contiguous.get(), ggml_sycl_src1_prec(dst),
                                                     (float *) dst_contiguous.get(),
                                                     expert_row_offsets.data(), n_as, ne01, ne10, n_routed_rows,
                                                     ctx.mmid_tile_schedule_host, ctx.pool(), stream);
        }

        // Small experts: one q8_1 quantization of their rows and one fused MMVQ launch over (row, expert) entries.
        bool small_fused = false;
        ggml_sycl_pool_alloc<char> small_q8(ctx.pool());
        if (!grouped && small_fused_enabled && route.n_small_rows > 0 && route.dev_small_row_ids != nullptr &&
            src1->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32 &&
            ne00 == ne10 && ne10 % QK8_1 == 0 && nb11 == sizeof(float)*ne10 && nb1 == sizeof(float)*ne0 &&
            ggml_sycl_mul_mat_vec_q_id_supports_type(src0->type)) {
            const ggml_tensor_extra_gpu * extra_small = static_cast<const ggml_tensor_extra_gpu *>(src0->extra);
            const bool reordered_small = extra_small && extra_small->optimized_feature.reorder;
            const bool reorder_kernel  = src0->type == GGML_TYPE_Q4_K || src0->type == GGML_TYPE_Q5_K ||
                                         src0->type == GGML_TYPE_Q6_K || src0->type == GGML_TYPE_MXFP4 || src0->type == GGML_TYPE_Q8_0;
            if (!reordered_small || reorder_kernel) {
                const int    padded_cols    = GGML_PAD((int) ne10, MATRIX_ROW_PADDING);
                const size_t bytes_per_qrow = (size_t) padded_cols * sizeof(block_q8_1) / QK8_1;
                const int    n_small        = (int) route.n_small_rows;
                small_q8.alloc((size_t) n_small * bytes_per_qrow);
                if (reordered_small) {
                    quantize_row_q8_1_sycl<quantize_and_reorder_q8_1_soa>(
                        (const float *) src1_contiguous.get(), small_q8.get(), (int) ne10, n_small, padded_cols, stream);
                    small_fused = ggml_sycl_mul_mat_vec_q_id_reorder(
                        src0->type, src0->data, small_q8.get(), route.dev_small_row_ids,
                        (float *) dst_contiguous.get(), (int) ne10, (int) ne01, n_small,
                        nb02, nb1, bytes_per_qrow, 1, stream);
                } else {
                    quantize_row_q8_1_sycl<quantize_q8_1>(
                        (const float *) src1_contiguous.get(), small_q8.get(), (int) ne10, n_small, padded_cols, stream);
                    small_fused = ggml_sycl_mul_mat_vec_q_id(
                        src0->type, src0->data, small_q8.get(), route.dev_small_row_ids,
                        (float *) dst_contiguous.get(), (int) ne10, (int) ne01, n_small,
                        nb02, nb1, bytes_per_qrow, 1, stream);
                }
                GGML_ASSERT(small_fused);
            }
        }
        if (ggml_sycl_get_env("GGML_SYCL_MMID_ROUTED_STATS", 0) != 0 && (small_fused_enabled || pad_tiles_enabled)) {
            fprintf(stderr,
                "[SYCL] MMID small-fused device=%d tokens=%lld routed_rows=%lld layout_rows=%lld small_rows=%lld small_max=%lld small_fused=%d pad_tiles=%d\n",
                ctx.device, (long long)ids->ne[1], (long long)n_routed_rows, (long long)n_layout_rows,
                (long long)route.n_small_rows, (long long)small_max, small_fused ? 1 : 0, tiles_padded ? 1 : 0);
        }

        // Direct per-expert GEMM: one fp16 conversion of all routed activations, then dequantize + gemm per expert.
        bool direct_gemm = false;
        bool routed_gemm = false;
        ggml_sycl_pool_alloc<int64_t> routed_dev_counts(ctx.pool());
        ggml_sycl_pool_alloc<int64_t> routed_dev_offsets(ctx.pool());
        ggml_sycl_pool_alloc<sycl::half> src1_f16(ctx.pool());
        ggml_sycl_pool_alloc<sycl::half> expert_f16(ctx.pool());
        // experts dequantized per launch (their fp16 copies sit side by side in expert_f16)
        int64_t chunk_max = 1, chunk_first = -1, chunk_count = 0;
        bool    chunk_q4k_reorder = false;
        bool    per_expert_convert = false;
#if defined(GGML_SYCL_F16)
        static const bool direct_gemm_enabled = ggml_sycl_get_env("GGML_SYCL_MMID_DIRECT_GEMM", 1) != 0;
        const to_fp16_sycl_t expert_to_fp16 =
            (!grouped && direct_gemm_enabled && !g_ggml_sycl_enable_dnn && ggml_is_quantized(src0->type) &&
             src1->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32 && ggml_is_contiguous(src0) &&
             dst->op_params[0] == GGML_PREC_DEFAULT && nb11 == sizeof(float)*ne10 && nb1 == sizeof(float)*ne0)
                ? ggml_get_to_fp16_sycl(src0->type, dst) : nullptr;
        if (expert_to_fp16 != nullptr) {
            bool any_big = false;
            for (int64_t i02 = 0; i02 < n_as && !any_big; i02++) {
                any_big = expert_row_counts[i02] > small_max;
            }
            if (any_big) {
                const to_fp16_sycl_t f32_to_fp16 = ggml_get_to_fp16_sycl(GGML_TYPE_F32, dst);
                if (f32_to_fp16 != nullptr) {
                    static const int chunk_mb = ggml_sycl_get_env("GGML_SYCL_MMID_CHUNK_MB", 256);
                    const ggml_tensor_extra_gpu * extra0 = static_cast<const ggml_tensor_extra_gpu *>(src0->extra);
                    const bool   reordered    = extra0 && extra0->optimized_feature.reorder;
                    const size_t expert_bytes = (size_t) ne01 * ne00 * ggml_type_size(src0->type) / ggml_blck_size(src0->type);
                    chunk_q4k_reorder = reordered && src0->type == GGML_TYPE_Q4_K;
                    static const bool routed_gemm_enabled =
                        ggml_sycl_get_env("GGML_SYCL_MMID_ROUTED_GEMM", 0) != 0;
                    static const int routed_min_tokens = std::max(0, std::min(4096,
                        ggml_sycl_get_env("GGML_SYCL_MMID_ROUTED_MIN_TOKENS", 0)));
                    static const bool routed_any_enabled =
                        ggml_sycl_get_env("GGML_SYCL_MMID_ROUTED_ANY", 0) != 0;
                    const bool routed_candidate = (src0->type == GGML_TYPE_Q4_K || routed_any_enabled) &&
                        ne01 == ne0 && ne01 % 16 == 0 && ne00 == ne10 && ne00 % 16 == 0;
                    const bool short_routed_batch = routed_gemm_enabled && routed_candidate &&
                        routed_min_tokens > 0 && ids->ne[1] < routed_min_tokens;
                    // each large expert is converted by its own launch into its chunk slot; valid for every layout
                    per_expert_convert = routed_any_enabled && routed_gemm_enabled && routed_candidate &&
                        !short_routed_batch && !chunk_q4k_reorder;
                    if (chunk_mb > 0 && nb02 == expert_bytes && (!reordered || chunk_q4k_reorder || per_expert_convert)) {
                        const int64_t fit = ((int64_t) chunk_mb * 1024 * 1024) / ((int64_t) ne01 * ne00 * (int64_t) sizeof(sycl::half));
                        chunk_max = std::max<int64_t>(1, std::min<int64_t>(64, fit));
                    }
                    if (short_routed_batch) {
                        chunk_max = 1;
                    }
                    src1_f16.alloc((size_t) n_layout_rows * ne10);
                    expert_f16.alloc((size_t) chunk_max * ne01 * ne00);
                    f32_to_fp16(src1_contiguous.get(), src1_f16.get(), n_layout_rows * ne10, stream);
                    direct_gemm = true;
                    routed_gemm = routed_gemm_enabled && routed_candidate && !short_routed_batch;
                    if (ggml_sycl_get_env("GGML_SYCL_MMID_ROUTED_STATS", 0) != 0) {
                        fprintf(stderr,
                            "[SYCL] MMID routed-threshold device=%d tokens=%lld min_tokens=%d routed_gemm=%d chunk_max=%lld chunking=%d\n",
                            ctx.device, (long long)ids->ne[1], routed_min_tokens, routed_gemm ? 1 : 0,
                            (long long)chunk_max, chunk_max > 1 ? 1 : 0);
                    }
                }
            }
        }
#endif
        const bool selective_dequant = selective_dequant_enabled && routed_gemm && chunk_q4k_reorder;

#if defined(GGML_SYCL_F16)
        const bool compact_dispatch = routed_gemm &&
            ggml_sycl_get_env("GGML_SYCL_MMID_ROUTED_COMPACT", 0) != 0;
        if (routed_gemm) {
            if (!compact_dispatch) {
                routed_dev_counts.alloc((size_t)n_as);
                routed_dev_offsets.alloc((size_t)n_as + 1);
                SYCL_CHECK(CHECK_TRY_ERROR(stream->memcpy(routed_dev_counts.get(), expert_row_counts.data(),
                                                           (size_t)n_as * sizeof(int64_t))));
                SYCL_CHECK(CHECK_TRY_ERROR(stream->memcpy(routed_dev_offsets.get(), expert_row_offsets.data(),
                                                           ((size_t)n_as + 1) * sizeof(int64_t))));
            }

            const int64_t output_tiles = ne01 / 16;
            uint64_t useful_tile_pairs = 0;
            uint64_t dense_tile_pairs = 0;
            uint64_t scheduled_groups = 0;
            int64_t total_experts = 0;
            int64_t total_chunks = 0;
            uint64_t selective_converted_experts = 0;
            uint64_t selective_staged_dense_experts = 0;
            static const bool routed_stats_enabled =
                ggml_sycl_get_env("GGML_SYCL_MMID_ROUTED_STATS", 0) != 0;
            static const bool cost_stats_enabled =
                ggml_sycl_get_env("GGML_SYCL_MMID_COST_STATS", 0) != 0;
            const bool collect_cost_stats = cost_stats_enabled && ids->ne[1] >= 58;
            static const bool cost_timing_enabled =
                ggml_sycl_get_env("GGML_SYCL_MMID_COST_TIMING", 0) != 0;
            static std::atomic<uint32_t> cost_timing_samples[GGML_SYCL_MAX_DEVICES]{};
            int cost_timing_op = 0;
            if (cost_timing_enabled && ids->ne[1] >= 128 && ctx.device >= 0 && ctx.device < GGML_SYCL_MAX_DEVICES) {
                const uint32_t sample = cost_timing_samples[ctx.device].fetch_add(1, std::memory_order_relaxed);
                if (sample < 4) {
                    cost_timing_op = (int)sample + 1;
                }
            }
            uint64_t cost_staged_experts = 0;
            uint64_t cost_xmx_experts = 0;
            uint64_t cost_small_nonempty = 0;
            uint64_t cost_zero_experts = 0;
            uint64_t cost_large_runs = 0;
            uint64_t cost_full_tiles = 0;
            uint64_t cost_tail_tiles = 0;
            uint64_t cost_staged_bytes = 0;
            uint64_t cost_xmx_used_bytes = 0;
            int64_t cost_dequant_us = 0;
            int64_t cost_xmx_us = 0;
            static const bool routed_shadow_enabled =
                ggml_sycl_get_env("GGML_SYCL_MMID_ROUTED_SHADOW", 0) != 0;
            static std::atomic<bool> routed_shadow_done[GGML_SYCL_MAX_DEVICES]{};
            int64_t shadow_expert = -1;
            int64_t shadow_chunk_first = -1;

            for (int64_t first = 0; first < n_as; first += chunk_max) {
                const int64_t count = std::min<int64_t>(chunk_max, n_as - first);
                int64_t max_token_tiles = 0;
                int64_t chunk_experts = 0;
                uint64_t chunk_small_nonempty = 0;
                uint64_t chunk_zero_experts = 0;
                uint64_t chunk_large_runs = 0;
                uint64_t chunk_full_tiles = 0;
                uint64_t chunk_tail_tiles = 0;
                bool in_large_run = false;
                int64_t selective_active_begin = 0;
                int64_t selective_active_count = 0;
                if (selective_dequant) {
                    const auto active_begin = std::lower_bound(route.selective_expert_ids.begin(),
                        route.selective_expert_ids.end(), first);
                    const auto active_end = std::lower_bound(active_begin, route.selective_expert_ids.end(), first + count);
                    selective_active_begin = active_begin - route.selective_expert_ids.begin();
                    selective_active_count = active_end - active_begin;
                }
                for (int64_t j = 0; j < count; ++j) {
                    const int64_t rows = expert_row_counts[first + j];
                    if (rows > small_max) {
                        const int64_t tiles = (rows + 7) / 8;
                        max_token_tiles = std::max<int64_t>(max_token_tiles, tiles);
                        useful_tile_pairs += (uint64_t)tiles;
                        ++chunk_experts;
                        if (collect_cost_stats) {
                            if (!in_large_run) {
                                ++chunk_large_runs;
                            }
                            in_large_run = true;
                            chunk_full_tiles += (uint64_t)(rows / 8);
                            chunk_tail_tiles += (rows % 8) != 0 ? 1 : 0;
                        }
                    } else {
                        if (collect_cost_stats) {
                            if (rows == 0) {
                                ++chunk_zero_experts;
                            } else {
                                ++chunk_small_nonempty;
                            }
                            in_large_run = false;
                        }
                    }
                }
                if (max_token_tiles == 0) {
                    continue;
                }
                if (selective_dequant) {
                    GGML_ASSERT(selective_active_count == chunk_experts);
                }
                if (routed_stats_enabled) {
                    selective_staged_dense_experts += (uint64_t)count;
                    selective_converted_experts += selective_dequant ? (uint64_t)chunk_experts : (uint64_t)count;
                }
                if (collect_cost_stats) {
                    const uint64_t expert_bytes = (uint64_t)ne01 * (uint64_t)ne00 * sizeof(sycl::half);
                    cost_staged_experts += selective_dequant ? (uint64_t)chunk_experts : (uint64_t)count;
                    cost_xmx_experts += (uint64_t)chunk_experts;
                    cost_small_nonempty += selective_dequant ? 0 : chunk_small_nonempty;
                    cost_zero_experts += selective_dequant ? 0 : chunk_zero_experts;
                    cost_large_runs += chunk_large_runs;
                    cost_full_tiles += chunk_full_tiles;
                    cost_tail_tiles += chunk_tail_tiles;
                    cost_staged_bytes += (selective_dequant ? (uint64_t)chunk_experts : (uint64_t)count) * expert_bytes;
                    cost_xmx_used_bytes += (uint64_t)chunk_experts * expert_bytes;
                }
                dense_tile_pairs += (uint64_t)count * (uint64_t)max_token_tiles;

                chunk_first = first;
                chunk_count = count;
                if (cost_timing_op > 0) {
                    { ggml_sycl_pipe_wait_scope ws_; SYCL_CHECK(CHECK_TRY_ERROR(stream->wait())); }
                }
                const int64_t dequant_start_us = cost_timing_op > 0 ? ggml_time_us() : 0;
                if (selective_dequant) {
                    if (selective_active_count > 0) {
                        ggml_sycl_dequantize_q4_K_reorder_indexed_slices(src0_original, expert_f16.get(), ne01 * ne00,
                            nb02, route.dev_selective_expert_ids + selective_active_begin,
                            selective_active_count, first, stream);
                        static const bool selective_shadow_enabled =
                            ggml_sycl_get_env("GGML_SYCL_MMID_SELECTIVE_DEQUANT_SHADOW", 0) != 0;
                        static std::atomic<bool> selective_shadow_done[GGML_SYCL_MAX_DEVICES]{};
                        if (selective_shadow_enabled && ids->ne[1] >= 128 &&
                            ctx.device >= 0 && ctx.device < GGML_SYCL_MAX_DEVICES) {
                            bool expected = false;
                            if (selective_shadow_done[ctx.device].compare_exchange_strong(expected, true)) {
                                const uint64_t elements_per_expert = (uint64_t)ne01 * (uint64_t)ne00;
                                const uint64_t max_selected_elements = 128ull * 1024 * 1024;
                                const uint64_t max_dense_elements = 128ull * 1024 * 1024;
                                const bool within_bounds = elements_per_expert > 0 &&
                                    (uint64_t)selective_active_count <= max_selected_elements / elements_per_expert &&
                                    (uint64_t)count <= max_dense_elements / elements_per_expert;
                                if (!within_bounds) {
                                    fprintf(stderr,
                                        "[SYCL] MMID selective-shadow device=%d tokens=%lld skipped=1 selected_experts=%lld dense_experts=%lld selected_limit_halves=%llu dense_limit_halves=%llu\n",
                                        ctx.device, (long long)ids->ne[1], (long long)selective_active_count,
                                        (long long)count, (unsigned long long)max_selected_elements,
                                        (unsigned long long)max_dense_elements);
                                } else {
                                    const size_t selected_elements = (size_t)selective_active_count * (size_t)elements_per_expert;
                                    const size_t dense_elements = (size_t)count * (size_t)elements_per_expert;
                                    ggml_sycl_pool_alloc<sycl::half> dense_reference(ctx.pool(), dense_elements);
                                    ggml_sycl_dequantize_q4_K_reorder_slices(src0_original + first * nb02,
                                        dense_reference.get(), ne01 * ne00, nb02, count, stream);
                                    std::vector<sycl::half> actual(selected_elements);
                                    std::vector<sycl::half> reference(selected_elements);
                                    for (int64_t active = 0; active < selective_active_count; ++active) {
                                        const int64_t expert = route.selective_expert_ids[selective_active_begin + active];
                                        GGML_ASSERT(expert >= first && expert < first + count);
                                        const size_t dense_slot = (size_t)(expert - first) * (size_t)elements_per_expert;
                                        const size_t selected_slot = (size_t)active * (size_t)elements_per_expert;
                                        const size_t bytes = (size_t)elements_per_expert * sizeof(sycl::half);
                                        SYCL_CHECK(CHECK_TRY_ERROR(stream->memcpy(actual.data() + selected_slot,
                                            expert_f16.get() + dense_slot, bytes)));
                                        SYCL_CHECK(CHECK_TRY_ERROR(stream->memcpy(reference.data() + selected_slot,
                                            dense_reference.get() + dense_slot, bytes)));
                                    }
                                    { ggml_sycl_pipe_wait_scope ws_; SYCL_CHECK(CHECK_TRY_ERROR(stream->wait())); }
                                    uint64_t bit_mismatches = 0;
                                    uint64_t nonfinite = 0;
                                    float max_abs = 0.0f;
                                    double error_sq = 0.0;
                                    double reference_sq = 0.0;
                                    for (size_t i = 0; i < selected_elements; ++i) {
                                        uint16_t actual_bits, reference_bits;
                                        std::memcpy(&actual_bits, &actual[i], sizeof(actual_bits));
                                        std::memcpy(&reference_bits, &reference[i], sizeof(reference_bits));
                                        if (actual_bits != reference_bits) {
                                            ++bit_mismatches;
                                        }
                                        const float actual_value = (float)actual[i];
                                        const float reference_value = (float)reference[i];
                                        if (!std::isfinite(actual_value) || !std::isfinite(reference_value)) {
                                            ++nonfinite;
                                            continue;
                                        }
                                        const float error = std::fabs(actual_value - reference_value);
                                        max_abs = std::max(max_abs, error);
                                        error_sq += (double)error * error;
                                        reference_sq += (double)reference_value * reference_value;
                                    }
                                    const double nmse = error_sq / std::max(reference_sq, 1.0e-30);
                                    fprintf(stderr,
                                        "[SYCL] MMID selective-shadow device=%d tokens=%lld selected_experts=%lld dense_experts=%lld elements=%llu bit_mismatches=%llu max_abs=%.8g nmse=%.8g nonfinite=%llu\n",
                                        ctx.device, (long long)ids->ne[1], (long long)selective_active_count,
                                        (long long)count, (unsigned long long)selected_elements,
                                        (unsigned long long)bit_mismatches, max_abs, nmse,
                                        (unsigned long long)nonfinite);
                                }
                            }
                        }
                    }
                } else if (chunk_q4k_reorder) {
                    ggml_sycl_dequantize_q4_K_reorder_slices(src0_original + first * nb02, expert_f16.get(), ne01 * ne00,
                                                             nb02, count, stream);
                } else if (per_expert_convert) {
                    for (int64_t j = 0; j < count; ++j) {
                        if (expert_row_counts[first + j] > small_max) {
                            expert_to_fp16(src0_original + (first + j) * nb02, expert_f16.get() + j * ne01 * ne00,
                                           ne01 * ne00, stream);
                        }
                    }
                } else {
                    expert_to_fp16(src0_original + first * nb02, expert_f16.get(), count * ne01 * ne00, stream);
                }
                if (cost_timing_op > 0) {
                    { ggml_sycl_pipe_wait_scope ws_; SYCL_CHECK(CHECK_TRY_ERROR(stream->wait())); }
                    cost_dequant_us += ggml_time_us() - dequant_start_us;
                }

                const int64_t pair_begin = compact_dispatch ? route.compact_expert_tile_offsets[first] : 0;
                const int64_t pair_end = compact_dispatch ? route.compact_expert_tile_offsets[first + count] : 0;
                const int64_t pair_count = pair_end - pair_begin;
                const size_t groups = compact_dispatch
                    ? (size_t)pair_count * (size_t)output_tiles
                    : (size_t)count * (size_t)max_token_tiles * (size_t)output_tiles;
                GGML_ASSERT(groups > 0 && groups <= SIZE_MAX / 16);
                scheduled_groups += (uint64_t)groups;
                const mmid_compact_tile * compact_tiles = route.dev_compact_tiles == nullptr
                    ? nullptr : route.dev_compact_tiles + pair_begin;
                const sycl::half * weights = expert_f16.get();
                const sycl::half * activations = src1_f16.get();
                const int64_t * counts = routed_dev_counts.get();
                const int64_t * offsets = routed_dev_offsets.get();
                float * output = (float *)dst_contiguous.get();
                const int64_t k_size = ne00;
                const int64_t m_size = ne01;
                const int64_t output_stride = ne0;
                const int64_t xmx_start_us = cost_timing_op > 0 ? ggml_time_us() : 0;
                stream->submit([&](sycl::handler &cgh) {
                    sycl::local_accessor<sycl::half, 1> tail_a(sycl::range<1>(8 * 16), cgh);
                    sycl::local_accessor<float, 1> tail_c(sycl::range<1>(8 * 16), cgh);
                    cgh.parallel_for(
                        sycl::nd_range<1>(sycl::range<1>(groups * 16), sycl::range<1>(16)),
                        [=](sycl::nd_item<1> item) [[sycl::reqd_sub_group_size(16)]] {
                            const size_t group_id = item.get_group(0);
                            const size_t channel_tile = group_id % (size_t)output_tiles;
                            size_t token_tile;
                            size_t expert_local;
                            int64_t expert_id;
                            int64_t row_count;
                            int64_t row_offset;
                            if (compact_dispatch) {
                                const size_t pair = group_id / (size_t)output_tiles;
                                const mmid_compact_tile descriptor = compact_tiles[pair];
                                expert_id = descriptor.expert_id;
                                token_tile = (size_t)descriptor.token_tile;
                                row_count = descriptor.row_count;
                                row_offset = descriptor.row_offset;
                                expert_local = (size_t)(expert_id - first);
                            } else {
                                token_tile = (group_id / (size_t)output_tiles) % (size_t)max_token_tiles;
                                expert_local = group_id / ((size_t)output_tiles * (size_t)max_token_tiles);
                                expert_id = first + (int64_t)expert_local;
                                row_count = counts[expert_id];
                                row_offset = offsets[expert_id];
                                if (row_count <= small_rows_max || (int64_t)token_tile >= (row_count + 7) / 8) {
                                    return;
                                }
                            }

                            const int64_t token_base = row_offset + (int64_t)token_tile * 8;
                            const sycl::half * a_base = activations + token_base * k_size;
                            const sycl::half * b_base = weights + (int64_t)expert_local * m_size * k_size + (int64_t)channel_tile * 16 * k_size;
                            auto a_ptr = sycl::address_space_cast<sycl::access::address_space::global_space,
                                sycl::access::decorated::no>(a_base);
                            auto b_ptr = sycl::address_space_cast<sycl::access::address_space::global_space,
                                sycl::access::decorated::no>(b_base);
                            auto c_ptr = sycl::address_space_cast<sycl::access::address_space::global_space,
                                sycl::access::decorated::no>(output + token_base * output_stride + (int64_t)channel_tile * 16);

                            sycl_mx::joint_matrix<sycl::sub_group, sycl::half, sycl_mx::use::a, 8, 16, sycl_mx::layout::row_major> a;
                            sycl_mx::joint_matrix<sycl::sub_group, sycl::half, sycl_mx::use::b, 16, 16, sycl_mx::layout::col_major> b;
                            sycl_mx::joint_matrix<sycl::sub_group, float, sycl_mx::use::accumulator, 8, 16> c;
                            sycl_mx::joint_matrix_fill(item.get_sub_group(), c, 0.0f);
                            const int64_t remaining_rows = row_count - (int64_t)token_tile * 8;
                            const int64_t valid_rows = (tiles_padded || remaining_rows >= 8) ? 8 : remaining_rows;
                            const bool token_tail = valid_rows < 8;
                            auto tail_a_ptr = tail_a.get_multi_ptr<sycl::access::decorated::no>();
                            for (int64_t k = 0; k < k_size; k += 16) {
                                if (token_tail) {
                                    for (int idx = (int)item.get_local_id(0); idx < 8 * 16; idx += 16) {
                                        const int row = idx / 16;
                                        const int col = idx % 16;
                                        tail_a[idx] = row < valid_rows ? a_base[row * k_size + k + col] : sycl::half(0.0f);
                                    }
                                    item.barrier(sycl::access::fence_space::local_space);
                                    sycl_mx::joint_matrix_load(item.get_sub_group(), a, tail_a_ptr, 16);
                                    item.barrier(sycl::access::fence_space::local_space);
                                } else {
                                    sycl_mx::joint_matrix_load(item.get_sub_group(), a, a_ptr + k, (size_t)k_size);
                                }
                                sycl_mx::joint_matrix_load(item.get_sub_group(), b, b_ptr + k, (size_t)k_size);
                                sycl_mx::joint_matrix_mad(item.get_sub_group(), c, a, b, c);
                            }
                            if (token_tail) {
                                sycl_mx::joint_matrix_store(item.get_sub_group(), c, tail_c.get_multi_ptr<sycl::access::decorated::no>(), 16, sycl_mx::layout::row_major);
                                item.barrier(sycl::access::fence_space::local_space);
                                for (int idx = (int)item.get_local_id(0); idx < 8 * 16; idx += 16) {
                                    const int row = idx / 16;
                                    const int col = idx % 16;
                                    if (row < valid_rows) {
                                        output[(token_base + row) * output_stride + (int64_t)channel_tile * 16 + col] = tail_c[idx];
                                    }
                                }
                            } else {
                                sycl_mx::joint_matrix_store(item.get_sub_group(), c, c_ptr, (size_t)output_stride, sycl_mx::layout::row_major);
                            }
                        });
                });
                if (cost_timing_op > 0) {
                    { ggml_sycl_pipe_wait_scope ws_; SYCL_CHECK(CHECK_TRY_ERROR(stream->wait())); }
                    cost_xmx_us += ggml_time_us() - xmx_start_us;
                }
                ++total_chunks;
                total_experts += chunk_experts;

                // Compare the first tile of one real expert against the same-N oneMKL path.
                if (routed_shadow_enabled && shadow_expert < 0 && ctx.device >= 0 &&
                    ctx.device < GGML_SYCL_MAX_DEVICES) {
                    for (int64_t j = 0; j < count; ++j) {
                        const int64_t candidate = first + j;
                        if (expert_row_counts[candidate] > small_max) {
                            const uint64_t elements = (uint64_t)expert_row_counts[candidate] * (uint64_t)ne0;
                            if (elements <= 16ull * 1024 * 1024) {
                                shadow_expert = candidate;
                                shadow_chunk_first = first;
                                break;
                            }
                        }
                    }
                    if (shadow_expert >= 0) {
                        bool expected = false;
                        if (routed_shadow_done[ctx.device].compare_exchange_strong(expected, true)) {
                            const int64_t shadow_rows = std::min<int64_t>(expert_row_counts[shadow_expert], 8);
                            const int64_t shadow_n = expert_row_counts[shadow_expert];
                            const int64_t shadow_offset = expert_row_offsets[shadow_expert];
                            ggml_sycl_pool_alloc<float> shadow_output(ctx.pool(), (size_t)shadow_n * (size_t)ne0);
                            const sycl::half * shadow_weights = expert_f16.get() +
                                (shadow_expert - shadow_chunk_first) * ne01 * ne00;
                            const float alpha = 1.0f, beta = 0.0f;
                            SYCL_CHECK(CHECK_TRY_ERROR(dpct::gemm(
                                *stream, oneapi::mkl::transpose::trans, oneapi::mkl::transpose::nontrans,
                                ne01, shadow_n, ne10, &alpha,
                                shadow_weights, dpct::library_data_t::real_half, ne00,
                                src1_f16.get() + shadow_offset * ne10, dpct::library_data_t::real_half, ne10,
                                &beta, shadow_output.get(), dpct::library_data_t::real_float, ne0,
                                dpct::library_data_t::real_float)));
                            std::vector<float> actual((size_t)shadow_rows * (size_t)ne0);
                            std::vector<float> reference((size_t)shadow_rows * (size_t)ne0);
                            SYCL_CHECK(CHECK_TRY_ERROR(stream->memcpy(actual.data(),
                                (float *)dst_contiguous.get() + shadow_offset * ne0, actual.size() * sizeof(float))));
                            SYCL_CHECK(CHECK_TRY_ERROR(stream->memcpy(reference.data(), shadow_output.get(),
                                reference.size() * sizeof(float))));
                            { ggml_sycl_pipe_wait_scope ws_; stream->wait(); }
                            double error_sq = 0.0, reference_sq = 0.0;
                            float max_abs = 0.0f, max_rel = 0.0f;
                            size_t nonfinite = 0;
                            for (size_t x = 0; x < actual.size(); ++x) {
                                if (!std::isfinite(actual[x]) || !std::isfinite(reference[x])) {
                                    ++nonfinite;
                                    continue;
                                }
                                const float error = std::fabs(actual[x] - reference[x]);
                                max_abs = std::max(max_abs, error);
                                max_rel = std::max(max_rel, error / std::max(std::fabs(reference[x]), 1.0e-6f));
                                error_sq += (double)error * error;
                                reference_sq += (double)reference[x] * reference[x];
                            }
                            const double nmse = error_sq / std::max(reference_sq, 1.0e-30);
                            fprintf(stderr,
                                "[SYCL] MMID routed shadow device=%d expert=%lld rows=%lld width=%lld k=%lld max_abs=%.8g max_rel=%.8g nmse=%.8g nonfinite=%zu\n",
                                ctx.device, (long long)shadow_expert, (long long)shadow_rows,
                                (long long)ne0, (long long)ne00, max_abs, max_rel, nmse, nonfinite);
                        }
                    }
                    shadow_expert = -1;
                }
            }
            if (collect_cost_stats) {
                fprintf(stderr,
                    "[SYCL] MMID cost-stats device=%d tokens=%lld routed_rows=%lld width=%lld k=%lld selective_dequant=%d staged_experts=%llu xmx_experts=%llu small_nonempty=%llu zero=%llu large_runs=%llu full8_tiles=%llu tail_tiles=%llu staged_fp16_bytes=%llu xmx_used_fp16_bytes=%llu chunks=%lld\n",
                    ctx.device, (long long)ids->ne[1], (long long)n_routed_rows, (long long)ne0, (long long)ne00,
                    (int)selective_dequant, (unsigned long long)cost_staged_experts,
                    (unsigned long long)cost_xmx_experts,
                    (unsigned long long)cost_small_nonempty, (unsigned long long)cost_zero_experts,
                    (unsigned long long)cost_large_runs, (unsigned long long)cost_full_tiles,
                    (unsigned long long)cost_tail_tiles, (unsigned long long)cost_staged_bytes,
                    (unsigned long long)cost_xmx_used_bytes, (long long)total_chunks);
            }
            if (cost_timing_op > 0) {
                fprintf(stderr,
                    "[SYCL] MMID cost-time opnum=%d device=%d tokens=%lld chunks=%lld dequant_us=%lld xmx_us=%lld timed=1 op_name=\"%s\" weights=\"%s\"\n",
                    cost_timing_op, ctx.device, (long long)ids->ne[1], (long long)total_chunks,
                    (long long)cost_dequant_us, (long long)cost_xmx_us, dst->name, src0->name);
            }
            if (ggml_sycl_get_env("GGML_SYCL_MMID_ROUTED_TILE_STATS", 0) != 0 && routed_gemm && ids->ne[1] >= 128) {
                const uint64_t useful_groups = useful_tile_pairs * (uint64_t)output_tiles;
                const uint64_t dense_groups = dense_tile_pairs * (uint64_t)output_tiles;
                const double fill = dense_groups == 0 ? 1.0 : (double)useful_groups / (double)dense_groups;
                static std::atomic<uint64_t> tile_stats_sequence{0};
                const uint64_t sequence = tile_stats_sequence.fetch_add(1, std::memory_order_relaxed) + 1;
                fprintf(stderr,
                    "[SYCL] MMID routed tile-stats op=%llu device=%d tokens=%lld routed_rows=%lld shape=%lldx%lld useful_wg=%llu dense_wg=%llu scheduled_wg=%llu fill=%.5f experts=%lld chunks=%lld compact=%d\n",
                    (unsigned long long)sequence, ctx.device, (long long)ids->ne[1], (long long)n_routed_rows,
                    (long long)ne0, (long long)ne00, (unsigned long long)useful_groups,
                    (unsigned long long)dense_groups, (unsigned long long)scheduled_groups,
                    fill, (long long)total_experts, (long long)total_chunks, compact_dispatch ? 1 : 0);
            }
            if (routed_stats_enabled && total_chunks > 0) {
                fprintf(stderr,
                    "[SYCL] MMID selective-stage device=%d tokens=%lld selective_dequant=%d converted=%llu staged_dense=%llu chunks=%lld\n",
                    ctx.device, (long long)ids->ne[1], (int)selective_dequant,
                    (unsigned long long)selective_converted_experts,
                    (unsigned long long)selective_staged_dense_experts, (long long)total_chunks);
            }
            static std::atomic<bool> routed_stats_logged[GGML_SYCL_MAX_DEVICES]{};
            if (routed_stats_enabled && total_experts > 0 && ctx.device >= 0 && ctx.device < GGML_SYCL_MAX_DEVICES) {
                bool expected = false;
                if (routed_stats_logged[ctx.device].compare_exchange_strong(expected, true)) {
                    fprintf(stderr,
                        "[SYCL] MMID routed GEMM device=%d experts=%lld joint_matrix_launches=%lld tile=8x16x16\n",
                        ctx.device, (long long)total_experts, (long long)total_chunks);
                }
            }
        }
#endif

        for (int64_t i02 = 0; i02 < n_as && !grouped; i02++) {
            const int64_t num_src1_rows = expert_row_counts[i02];

            if (num_src1_rows == 0) {
                continue;
            }

            const int64_t expert_row_offset = expert_row_offsets[i02];

            src0_row.data = src0_original + i02*nb02;

            GGML_ASSERT(nb11 == sizeof(float)*ne10);
            GGML_ASSERT(nb1 == sizeof(float)*ne0);
            src1_row.data = src1_contiguous.get() + expert_row_offset*nb11;
            src1_row.ne[1] = num_src1_rows;

            src1_row.nb[1] = nb11;
            src1_row.nb[2] = num_src1_rows*nb11;
            src1_row.nb[3] = num_src1_rows*nb11;

            dst_row.data = dst_contiguous.get() + expert_row_offset*nb1;
            dst_row.ne[1] = num_src1_rows;
            dst_row.nb[1] = nb1;
            dst_row.nb[2] = num_src1_rows*nb1;
            dst_row.nb[3] = num_src1_rows*nb1;

            if (small_fused && num_src1_rows <= small_max) {
                // done by the fused small-expert launch
                continue;
            }
#if defined(GGML_SYCL_F16)
            if (routed_gemm && num_src1_rows > small_max) {
                // The routed joint_matrix launch above handled every row, including a zero-padded tail tile.
                continue;
            }
            if (direct_gemm && num_src1_rows > small_max) {
                // same fp16 gemm the generic path runs for this expert, minus its per-call activation conversion
                if (i02 < chunk_first || i02 >= chunk_first + chunk_count) {
                    chunk_first = i02;
                    chunk_count = std::min<int64_t>(chunk_max, n_as - i02);
                    if (chunk_q4k_reorder) {
                        ggml_sycl_dequantize_q4_K_reorder_slices(src0_original + i02*nb02, expert_f16.get(), ne01 * ne00,
                                                                 nb02, chunk_count, stream);
                    } else {
                        // plain layouts are contiguous across experts (checked above); chunk_count is 1 otherwise
                        expert_to_fp16(src0_original + i02*nb02, expert_f16.get(), chunk_count * ne01 * ne00, stream);
                    }
                }
                const sycl::half * expert_w = expert_f16.get() + (i02 - chunk_first) * ne01 * ne00;
                const float alpha = 1.0f;
                const float beta  = 0.0f;
                SYCL_CHECK(CHECK_TRY_ERROR(dpct::gemm(
                    *stream, oneapi::mkl::transpose::trans, oneapi::mkl::transpose::nontrans,
                    ne01, num_src1_rows, ne10,
                    &alpha, expert_w, dpct::library_data_t::real_half, ne00,
                    src1_f16.get() + expert_row_offset * ne10, dpct::library_data_t::real_half, ne10,
                    &beta, (float *) dst_row.data, dpct::library_data_t::real_float, ne0,
                    dpct::library_data_t::real_float)));
                continue;
            }
#endif

            ggml_sycl_mul_mat(ctx, &src0_row, &src1_row, &dst_row);
        }

        {
            sycl::range<3> block_dims(1, 1, std::min((unsigned int)ne0, max_work_group_size));
            sycl::range<3> grid_dims(1, 1, n_layout_rows);
            stream->submit([&](sycl::handler &cgh) {
                const char *__restrict dst_contiguous_get =
                    dst_contiguous.get();
                const mmid_row_mapping *__restrict dev_row_mapping_get =
                    dev_row_mapping.get();

                cgh.parallel_for(
                    sycl::nd_range<3>(grid_dims * block_dims, block_dims),
                    [=](sycl::nd_item<3> item_ct1) {
                        k_copy_dst_from_contiguous(dst_original,
                                                   dst_contiguous_get,
                                                   dev_row_mapping_get,
                                                   ne0, nb1, nb2, item_ct1);
                    });
            });
        }
    }
}

catch (sycl::exception const &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
  std::exit(1);
}

static void ggml_sycl_scale(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/1);
    ggml_sycl_op_scale(ctx, dst);
}

static void ggml_sycl_diag_mask_inf(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/1);
    ggml_sycl_op_diag_mask_inf(ctx, dst);
}

static void ggml_sycl_pool2d(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/1);
    ggml_sycl_op_pool2d(ctx, dst);
}

static void ggml_sycl_pool1d(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/1);
    ggml_sycl_op_pool1d(ctx, dst);
}

static void ggml_sycl_im2col(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/2);
    ggml_sycl_op_im2col(ctx, dst);
}

static void ggml_sycl_im2col_3d(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/2);
    ggml_sycl_op_im2col_3d(ctx, dst);
}

static void ggml_sycl_col2im_1d(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/1);
    ggml_sycl_op_col2im_1d(ctx, dst);
}

static void ggml_sycl_conv_3d(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/2);
    ggml_sycl_op_conv_3d(ctx, dst);
}

static void ggml_sycl_sum(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/1);
    GGML_ASSERT(ggml_is_contiguous(dst->src[0]));
    ggml_sycl_op_sum(ctx, dst);
}

static void ggml_sycl_sum_rows(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/1);
    GGML_ASSERT(ggml_is_contiguous(dst->src[0]));
    ggml_sycl_op_sum_rows(ctx, dst);
}

static void ggml_sycl_mean(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/1);
    GGML_ASSERT(ggml_is_contiguous(dst->src[0]));
    ggml_sycl_op_mean(ctx, dst);
}

static void ggml_sycl_argsort(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/1);
    GGML_ASSERT(ggml_is_contiguous(dst->src[0]));
    ggml_sycl_op_argsort(ctx, dst);
}

static void ggml_sycl_argmax(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/1);
    GGML_ASSERT(ggml_is_contiguous(dst->src[0]));
    ggml_sycl_op_argmax(ctx, dst);
}


static void ggml_sycl_set_main_device(const int main_device) try {
    if (dpct::get_current_device_id() == static_cast<unsigned int> (main_device)) {
        return;
    }
    check_allow_gpu_index(main_device);
    dpct::select_device(main_device);

    if (g_ggml_sycl_debug) {
        dpct::device_info prop;
        SYCL_CHECK(CHECK_TRY_ERROR(dpct::get_device_info(
            prop, dpct::dev_mgr::instance().get_device(main_device))));
        GGML_LOG_INFO("Using device %d (%s) as main device\n",
                main_device, prop.get_name());
    }
}
catch (sycl::exception const &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
  std::exit(1);
}

static bool ggml_sycl_compute_forward(ggml_backend_sycl_context & ctx, struct ggml_tensor * dst) try {
    GGML_SYCL_DEBUG("[SYCL] ggml_sycl_compute_forward: dst=%s, op=%s\n", dst->name, ggml_op_name(dst->op));
    if (!g_sycl_loaded) return false;

    if (dst->src[0] != nullptr && ggml_backend_buffer_is_sycl_split(dst->src[0]->buffer)) {
        ggml_sycl_set_peer_access(dst->src[1]->ne[1], ctx.device);
    }

    switch (dst->op) {
        case GGML_OP_ARGMAX:
            ggml_sycl_argmax(ctx, dst);
            break;
        case GGML_OP_CONV_2D:
            ggml_sycl_op_conv2d(ctx, dst);
            break;
        case GGML_OP_CONV_2D_DW:
            ggml_sycl_op_conv2d_dw(ctx, dst);
            break;
        case GGML_OP_CONV_3D:
            ggml_sycl_conv_3d(ctx, dst);
            break;
        case GGML_OP_CONV_TRANSPOSE_1D:
            ggml_sycl_op_conv_transpose_1d(ctx, dst);
            break;
        case GGML_OP_CONV_TRANSPOSE_2D:
            ggml_sycl_op_conv2d_transpose(ctx, dst);
            break;
        case GGML_OP_REPEAT:
            ggml_sycl_repeat(ctx, dst);
            break;
        case GGML_OP_REPEAT_BACK:
            ggml_sycl_repeat_back(ctx, dst);
            break;
        case GGML_OP_GET_ROWS:
            ggml_sycl_get_rows(ctx, dst);
            break;
        case GGML_OP_GET_ROWS_BACK:
            ggml_sycl_get_rows_back(ctx, dst);
            break;
        case GGML_OP_SET:
            ggml_sycl_op_set(ctx, dst);
            break;
        case GGML_OP_SET_ROWS:
            ggml_sycl_op_set_rows(ctx, dst);
            break;
        case GGML_OP_DSV4_HC_PRE:
            ggml_sycl_op_dsv4_hc_pre(ctx, dst);
            break;
        case GGML_OP_DSV4_HC_COMB:
            ggml_sycl_op_dsv4_hc_comb(ctx, dst);
            break;
        case GGML_OP_DSV4_HC_POST:
            ggml_sycl_op_dsv4_hc_post(ctx, dst);
            break;
        case GGML_OP_LIGHTNING_INDEXER:
            ggml_sycl_op_lightning_indexer(ctx, dst);
            break;
        case GGML_OP_DUP:
            ggml_sycl_dup(ctx, dst);
            break;
        case GGML_OP_ADD:
        case GGML_OP_ADD1: // TODO: more efficient implementation
            ggml_sycl_add(ctx, dst);
            break;
        case GGML_OP_ADD_ID:
            ggml_sycl_add_id(ctx, dst);
            break;
        case GGML_OP_SUB:
            ggml_sycl_sub(ctx, dst);
            break;
        case GGML_OP_COUNT_EQUAL:
            ggml_sycl_count_equal(ctx, dst);
            break;
        case GGML_OP_ACC:
            ggml_sycl_acc(ctx, dst);
            break;
        case GGML_OP_MUL:
            ggml_sycl_mul(ctx, dst);
            break;
        case GGML_OP_LOG:
            ggml_sycl_log(ctx, dst);
            break;
        case GGML_OP_DIV:
            ggml_sycl_div(ctx, dst);
            break;
        case GGML_OP_UNARY:
            switch (ggml_get_unary_op(dst)) {
                case GGML_UNARY_OP_NEG:
                    ggml_sycl_neg(ctx, dst);
                    break;
                case GGML_UNARY_OP_STEP:
                    ggml_sycl_step(ctx, dst);
                    break;
                case GGML_UNARY_OP_GELU:
                    ggml_sycl_gelu(ctx, dst);
                    break;
                case GGML_UNARY_OP_SILU:
                    ggml_sycl_silu(ctx, dst);
                    break;
                case GGML_UNARY_OP_GELU_QUICK:
                    ggml_sycl_gelu_quick(ctx, dst);
                    break;
                case GGML_UNARY_OP_GELU_ERF:
                    ggml_sycl_gelu_erf(ctx, dst);
                    break;
                case GGML_UNARY_OP_TANH:
                    ggml_sycl_tanh(ctx, dst);
                    break;
                case GGML_UNARY_OP_RELU:
                    ggml_sycl_relu(ctx, dst);
                    break;
                case GGML_UNARY_OP_SIGMOID:
                    ggml_sycl_sigmoid(ctx, dst);
                    break;
                case GGML_UNARY_OP_HARDSIGMOID:
                    ggml_sycl_hardsigmoid(ctx, dst);
                    break;
                case GGML_UNARY_OP_HARDSWISH:
                    ggml_sycl_hardswish(ctx, dst);
                    break;
                case GGML_UNARY_OP_EXP:
                    ggml_sycl_exp(ctx, dst);
                    break;
                case GGML_UNARY_OP_EXPM1:
                    ggml_sycl_expm1(ctx, dst);
                    break;
                case GGML_UNARY_OP_SOFTPLUS:
                    ggml_sycl_softplus(ctx, dst);
                    break;
                case GGML_UNARY_OP_SGN:
                    ggml_sycl_sgn(ctx, dst);
                    break;
                case GGML_UNARY_OP_ABS:
                    ggml_sycl_abs(ctx, dst);
                    break;
                case GGML_UNARY_OP_ELU:
                    ggml_sycl_elu(ctx, dst);
                    break;
                case GGML_UNARY_OP_XIELU:
                    ggml_sycl_xielu(ctx, dst);
                    break;
                case GGML_UNARY_OP_FLOOR:
                    ggml_sycl_floor(ctx, dst);
                    break;
                case GGML_UNARY_OP_CEIL:
                    ggml_sycl_ceil(ctx, dst);
                    break;
                case GGML_UNARY_OP_ROUND:
                    ggml_sycl_round(ctx, dst);
                    break;
                case GGML_UNARY_OP_TRUNC:
                    ggml_sycl_trunc(ctx, dst);
                    break;
                default:
                    return false;
            }
            break;
        case GGML_OP_GLU:
            switch (ggml_get_glu_op(dst)) {
                case GGML_GLU_OP_REGLU:
                    ggml_sycl_reglu(ctx, dst);
                    break;
                case GGML_GLU_OP_GEGLU:
                    ggml_sycl_geglu(ctx, dst);
                    break;
                case GGML_GLU_OP_SWIGLU:
                    ggml_sycl_swiglu(ctx, dst);
                    break;
                case GGML_GLU_OP_SWIGLU_OAI:
                    ggml_sycl_swiglu_oai(ctx, dst);
                    break;
                case GGML_GLU_OP_SWIGLU_CLAMP:
                    ggml_sycl_swiglu_clamp(ctx, dst);
                    break;
                case GGML_GLU_OP_GEGLU_ERF:
                    ggml_sycl_geglu_erf(ctx, dst);
                    break;
                case GGML_GLU_OP_GEGLU_QUICK:
                    ggml_sycl_geglu_quick(ctx, dst);
                    break;
                default:
                    return false;
            }
            break;
        case GGML_OP_NORM:
            ggml_sycl_norm(ctx, dst);
            break;
        case GGML_OP_GROUP_NORM:
            ggml_sycl_group_norm(ctx, dst);
            break;
        case GGML_OP_CONCAT:
            ggml_sycl_op_concat(ctx, dst);
            break;
        case GGML_OP_PAD_REFLECT_1D:
            ggml_sycl_op_pad_reflect_1d(ctx,dst);
            break;
        case GGML_OP_UPSCALE:
            ggml_sycl_upscale(ctx, dst);
            break;
        case GGML_OP_PAD:
            ggml_sycl_pad(ctx, dst);
            break;
        case GGML_OP_LEAKY_RELU:
            ggml_sycl_leaky_relu(ctx, dst);
            break;
        case GGML_OP_RMS_NORM_BACK:
            ggml_sycl_rms_norm_back(ctx, dst);
            break;
        case GGML_OP_RMS_NORM:
            ggml_sycl_rms_norm(ctx, dst);
            break;
        case GGML_OP_L2_NORM:
            ggml_sycl_l2_norm(ctx, dst);
            break;
        case GGML_OP_MUL_MAT:
            if (dst->src[0]->ne[3] != dst->src[1]->ne[3]) {
                return false;
            }
            /* ggml_sycl_mul_mat_id is dependent on ggml_sycl_mul_mat */
            ggml_sycl_mul_mat(ctx, dst->src[0], dst->src[1], dst);
            break;
        case GGML_OP_MUL_MAT_ID:
            if (dst->src[0]->ne[3] != dst->src[1]->ne[3]) {
                return false;
            }
            ggml_sycl_mul_mat_id(ctx, dst);
            break;
        case GGML_OP_OUT_PROD:
            ggml_sycl_op_out_prod(ctx, dst);
            break;
        case GGML_OP_SCALE:
            ggml_sycl_scale(ctx, dst);
            break;
        case GGML_OP_SQR:
            ggml_sycl_sqr(ctx, dst);
            break;
        case GGML_OP_SQRT:
            ggml_sycl_sqrt(ctx, dst);
            break;
        case GGML_OP_SIN:
            ggml_sycl_sin(ctx, dst);
            break;
        case GGML_OP_COS:
            ggml_sycl_cos(ctx, dst);
            break;
        case GGML_OP_CLAMP:
            ggml_sycl_clamp(ctx, dst);
            break;
        case GGML_OP_CPY:
            ggml_sycl_cpy(ctx, dst->src[0], dst->src[1]);
            break;
        case GGML_OP_CONT:
            ggml_sycl_dup(ctx, dst);
            break;
        case GGML_OP_NONE:
        case GGML_OP_RESHAPE:
        case GGML_OP_VIEW:
        case GGML_OP_PERMUTE:
        case GGML_OP_TRANSPOSE:
            GGML_SYCL_DEBUG("%s: Tensor NO-OP\n", __func__);
            break;
        case GGML_OP_TRI:
            ggml_sycl_op_tri(ctx, dst);
            break;
        case GGML_OP_DIAG_MASK_INF:
            ggml_sycl_diag_mask_inf(ctx, dst);
            break;
        case GGML_OP_SOFT_MAX:
            ggml_sycl_op_soft_max(ctx, dst);
            break;
        case GGML_OP_SOFT_MAX_BACK:
            ggml_sycl_op_soft_max_back(ctx, dst);
            break;
        case GGML_OP_CROSS_ENTROPY_LOSS:
            ggml_sycl_cross_entropy_loss(ctx, dst);
            break;
        case GGML_OP_CROSS_ENTROPY_LOSS_BACK:
            ggml_sycl_cross_entropy_loss_back(ctx, dst);
            break;
        case GGML_OP_ROPE:
            ggml_sycl_rope(ctx, dst);
            break;
        case GGML_OP_ROPE_BACK:
            ggml_sycl_rope_back(ctx, dst);
            break;
        case GGML_OP_IM2COL:
            ggml_sycl_im2col(ctx, dst);
            break;
        case GGML_OP_IM2COL_3D:
            ggml_sycl_im2col_3d(ctx, dst);
            break;
        case GGML_OP_COL2IM_1D:
            ggml_sycl_col2im_1d(ctx, dst);
            break;
        case GGML_OP_POOL_2D:
            ggml_sycl_pool2d(ctx, dst);
            break;
        case GGML_OP_POOL_1D:
            ggml_sycl_pool1d(ctx, dst);
            break;
        case GGML_OP_SUM:
            ggml_sycl_sum(ctx, dst);
            break;
        case GGML_OP_SUM_ROWS:
            ggml_sycl_sum_rows(ctx, dst);
            break;
        case GGML_OP_MEAN:
            ggml_sycl_mean(ctx, dst);
            break;
        case GGML_OP_ARGSORT:
            ggml_sycl_argsort(ctx, dst);
            break;
        case GGML_OP_TOP_K:
            ggml_sycl_op_top_k(ctx, dst);
            break;
        case GGML_OP_TIMESTEP_EMBEDDING:
            ggml_sycl_op_timestep_embedding(ctx, dst);
            break;
        case GGML_OP_RWKV_WKV6:
            ggml_sycl_op_rwkv_wkv6(ctx, dst);
            break;
        case GGML_OP_RWKV_WKV7:
            ggml_sycl_op_rwkv_wkv7(ctx, dst);
            break;
        case GGML_OP_GATED_LINEAR_ATTN:
            ggml_sycl_op_gated_linear_attn(ctx, dst);
            break;
        case GGML_OP_GATED_DELTA_NET:
            ggml_sycl_gated_delta_net(ctx, dst);
            break;
        case GGML_OP_OPT_STEP_ADAMW:
            ggml_sycl_opt_step_adamw(ctx, dst);
            break;
        case GGML_OP_OPT_STEP_SGD:
            ggml_sycl_opt_step_sgd(ctx, dst);
            break;
        case GGML_OP_SSM_CONV:
            ggml_sycl_ssm_conv(ctx, dst);
            break;
        case GGML_OP_SSM_SCAN:
            ggml_sycl_ssm_scan(ctx, dst);
            break;
        case GGML_OP_FILL:
            ggml_sycl_fill(ctx, dst);
            break;
        case GGML_OP_CUMSUM:
            ggml_sycl_cumsum(ctx, dst);
            break;
        case GGML_OP_DIAG:
            ggml_sycl_diag(ctx, dst);
            break;
        case GGML_OP_SOLVE_TRI:
            ggml_sycl_solve_tri(ctx, dst);
            break;
        case GGML_OP_ROLL:
            ggml_sycl_roll(ctx, dst);
            break;
        case GGML_OP_ARANGE:
            ggml_sycl_arange(ctx, dst);
            break;
        case GGML_OP_FLASH_ATTN_EXT:
            ggml_sycl_flash_attn_ext(ctx, dst);
            break;
        default:
            return false;
    }

    return true;
} catch (sycl::exception & e) {
    std::cerr << e.what() << "Exception caught at file:" << __FILE__ << ", line:" << __LINE__ << std::endl;
    std::cerr << "Error OP "<<ggml_op_name(dst->op)<< std::endl;
    std::exit(1);
}

GGML_API void ggml_backend_sycl_get_device_description(int device, char *description,
                                      size_t description_size) try {
    GGML_SYCL_DEBUG("[SYCL] call ggml_backend_sycl_get_device_description\n");
    dpct::device_info prop;
    SYCL_CHECK(CHECK_TRY_ERROR(dpct::get_device_info(
        prop, dpct::dev_mgr::instance().get_device(device))));
    snprintf(description, description_size, "%s", prop.get_name());
}
catch (sycl::exception const &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
  std::exit(1);
}

static bool sycl_get_mem_info(int device, size_t * free, size_t * total) {
    GGML_SYCL_DEBUG("[SYCL] [%s] g_ggml_sycl_get_mem_api=%d\n",
        __func__, g_ggml_sycl_get_mem_api);

    MemoryAPIType mem_api_type = MemoryAPIType::MEMORY_API_TYPE_SYCL;

#ifdef GGML_SYCL_SUPPORT_LEVEL_ZERO_API
    mem_api_type = get_zes_init_res() == ZE_RESULT_SUCCESS ?
        (MemoryAPIType) g_ggml_sycl_get_mem_api : MemoryAPIType::MEMORY_API_TYPE_SYCL;
#else
    mem_api_type = MemoryAPIType::MEMORY_API_TYPE_SYCL;
#endif
    bool res = get_memory_size(dpct::dev_mgr::instance().get_device(device),
        *free, *total, mem_api_type);
    GGML_SYCL_DEBUG("[SYCL] [%s] total = %zu free = %zu\n", __func__, *total, *free);
    return res;
}

void ggml_backend_sycl_get_device_memory(int device, size_t * free, size_t * total) try {
    GGML_SYCL_DEBUG("[SYCL] call ggml_backend_sycl_get_device_memory\n");
    if (!sycl_get_mem_info(device, free, total)) {
        GGML_ABORT("[%s] failed to get device memory size", __func__);
    }
    ggml_sycl_memtrace_report_device("device memory query", device, *free, *total);
} catch (const sycl::exception & exc) {
    std::cerr << exc.what() << "Exception caught at file:" << __FILE__ << ", line:" << __LINE__ << std::endl;
    std::exit(1);
}

////////////////////////////////////////////////////////////////////////////////

// backend

static const char * ggml_backend_sycl_get_name(ggml_backend_t backend) {

    ggml_backend_sycl_context * sycl_ctx = (ggml_backend_sycl_context *)backend->context;

    return sycl_ctx->name.c_str();
}

static void ggml_backend_sycl_free(ggml_backend_t backend) {
    ggml_backend_sycl_context * sycl_ctx = (ggml_backend_sycl_context *)backend->context;
    sycl_pipe::drain(sycl_ctx->device);

    delete sycl_ctx;
    delete backend;
}

static void ggml_backend_sycl_set_tensor_async(ggml_backend_t backend,
                                               ggml_tensor *tensor,
                                               const void *data, size_t offset,
                                               size_t size) try {
    sycl_pipe::submit_scope pipe_ss_;
    GGML_SYCL_DEBUG("[SYCL] call %s", __func__);
    GGML_SYCL_DEBUG("%s", debug_get_tensor_str(": tensor", tensor).c_str());
    GGML_SYCL_DEBUG(" size=%zu offset=%zu\n", size, offset);
    ggml_backend_sycl_context * sycl_ctx = (ggml_backend_sycl_context *)backend->context;
    ggml_backend_buffer_t buf = tensor->view_src ? tensor->view_src->buffer : tensor->buffer;

    GGML_ASSERT(buf->buft == ggml_backend_sycl_buffer_type(sycl_ctx->device) && "unsupported buffer type");
    const queue_ptr stream = sycl_ctx->stream(sycl_ctx->device, 0);
    if (sycl_pipe::pending(sycl_ctx->device)) {
        auto   held   = std::make_shared<std::vector<char>>((const char *) data, (const char *) data + size);
        char * target = (char *) tensor->data + offset;
        sycl_pipe::held_add(sycl_ctx->device, (const char *) tensor->data, offset, held);
        sycl_pipe::enqueue(sycl_ctx->device, [held, target, stream] {
            {
                sycl_pipe::submit_scope ss_;
                { ggml_sycl_pipe_wait_scope ws_; SYCL_CHECK(CHECK_TRY_ERROR(stream->wait())); }
                sycl::event ev;
                SYCL_CHECK(CHECK_TRY_ERROR(ev = stream->memcpy(target, held->data(), held->size())));
                { ggml_sycl_pipe_wait_scope ws_; SYCL_CHECK(CHECK_TRY_ERROR(ev.wait())); }
            }
            sycl_pipe::held_done(target, held);
        }, "set");
        return;
    }
    SYCL_CHECK(CHECK_TRY_ERROR(
        (stream)->memcpy((char *)tensor->data + offset, data, size)));
}
catch (sycl::exception const &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
  std::exit(1);
}

static void ggml_backend_sycl_get_tensor_async(ggml_backend_t backend,
                                               const ggml_tensor *tensor,
                                               void *data, size_t offset,
                                               size_t size) try {
    sycl_pipe::submit_scope pipe_ss_;
    GGML_SYCL_DEBUG("[SYCL] call %s", __func__);
    GGML_SYCL_DEBUG("%s", debug_get_tensor_str(": tensor", tensor).c_str());
    GGML_SYCL_DEBUG(" size=%zu offset=%zu\n", size, offset);
    ggml_backend_sycl_context * sycl_ctx = (ggml_backend_sycl_context *)backend->context;
    ggml_backend_buffer_t buf = tensor->view_src ? tensor->view_src->buffer : tensor->buffer;

    GGML_ASSERT(buf->buft == ggml_backend_sycl_buffer_type(sycl_ctx->device) && "unsupported buffer type");
    const queue_ptr stream = sycl_ctx->stream(sycl_ctx->device, 0);
    // Drain the queue before reading (see ggml_backend_sycl_buffer_get_tensor): this is the path that returns logits
    // and the backend-sampled draft tokens, and a USM copy is not reliably ordered against the kernels queued before it.
    static const bool wait_first = ggml_sycl_get_env("GGML_SYCL_GET_TENSOR_WAIT", 1) != 0;
    if (sycl_pipe::pending(sycl_ctx->device)) {
        // asynchronous read: the caller keeps `data` valid until it synchronizes, so the copy can follow the queued work
        const char * source = (const char *) tensor->data + offset;
        sycl_pipe::enqueue(sycl_ctx->device, [data, source, size, stream] {
            sycl_pipe::submit_scope ss_;
            { ggml_sycl_pipe_wait_scope ws_; SYCL_CHECK(CHECK_TRY_ERROR(stream->wait())); }
            sycl::event ev;
            SYCL_CHECK(CHECK_TRY_ERROR(ev = stream->memcpy(data, source, size)));
            { ggml_sycl_pipe_wait_scope ws_; SYCL_CHECK(CHECK_TRY_ERROR(ev.wait())); }
        }, "get");
        return;
    }
    if (wait_first) {
        { ggml_sycl_pipe_wait_scope ws_; SYCL_CHECK(CHECK_TRY_ERROR((stream)->wait())); }
    }
    SYCL_CHECK(CHECK_TRY_ERROR((stream)->memcpy(
        data, (const char *)tensor->data + offset, size)));
}
catch (sycl::exception const &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
  std::exit(1);
}

static bool ggml_backend_sycl_cpy_tensor_async(ggml_backend_t backend,
                                               const ggml_tensor *src,
                                               ggml_tensor *dst) try {
    ggml_backend_sycl_context * sycl_ctx = (ggml_backend_sycl_context *)backend->context;
    bool is_cpy_supported                = dst->buffer->buft == ggml_backend_sycl_buffer_type(sycl_ctx->device) &&
                            ggml_backend_buffer_is_sycl(src->buffer);
    GGML_SYCL_DEBUG("[SYCL] call %s", __func__);
    GGML_SYCL_DEBUG("%s", debug_get_tensor_str(": dst", dst).c_str());
    GGML_SYCL_DEBUG("%s", debug_get_tensor_str(" src", src).c_str());
    GGML_SYCL_DEBUG(" is_cpy_supported=%d\n", is_cpy_supported);
    if (is_cpy_supported) {
        /*
        DPCT1009:215: SYCL uses exceptions to report errors and does not use the
        error codes. The original code was commented out and a warning string
        was inserted. You need to rewrite this code.
        */
        const queue_ptr stream = sycl_ctx->stream(sycl_ctx->device, 0);
        SYCL_CHECK(CHECK_TRY_ERROR((stream)->memcpy(
            dst->data, src->data, ggml_nbytes(dst))));
        return true;
    }

    return false;
}
catch (sycl::exception const &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
  std::exit(1);
}

// asynchronous copy between two cards: only with the pipeline running (otherwise the scheduler's blocking path is used)
static bool ggml_backend_sycl_cpy_tensor_async_pipe(ggml_backend_t backend_src, ggml_backend_t backend_dst,
                                                    const ggml_tensor * src, ggml_tensor * dst) {
    if (!ggml_backend_is_sycl(backend_src) || !ggml_backend_is_sycl(backend_dst)) {
        return false;
    }
    ggml_backend_sycl_context * ctx_src = (ggml_backend_sycl_context *) backend_src->context;
    ggml_backend_sycl_context * ctx_dst = (ggml_backend_sycl_context *) backend_dst->context;
    if (!sycl_pipe::active(ctx_dst->device) || sycl_pipe::t_worker_device >= 0) {
        return false;
    }
    // nothing in flight (decode): the scheduler's direct card-to-card copy is faster than going through host memory
    if (!sycl_pipe::pending(ctx_src->device) && !sycl_pipe::pending(ctx_dst->device)) {
        return false;
    }
    ggml_backend_buffer_t buf_src = src->view_src ? src->view_src->buffer : src->buffer;
    ggml_backend_buffer_t buf_dst = dst->view_src ? dst->view_src->buffer : dst->buffer;
    if (buf_src->buft != ggml_backend_sycl_buffer_type(ctx_src->device) || buf_dst->buft != ggml_backend_sycl_buffer_type(ctx_dst->device)) {
        return false;
    }
    sycl_pipe::held_report("async", src, ctx_src->device, ctx_dst->device);
    sycl_pipe::copy_async(ctx_src->device, (const char *) src->data, ctx_dst->device, (char *) dst->data, ggml_nbytes(src));
    return true;
}

static void ggml_backend_sycl_synchronize(ggml_backend_t backend) try {
    sycl_pipe::submit_scope pipe_ss_;
    GGML_SYCL_DEBUG("[SYCL] call %s\n", __func__);
    ggml_backend_sycl_context * sycl_ctx = (ggml_backend_sycl_context *)backend->context;
    sycl_pipe::drain(sycl_ctx->device);
    const queue_ptr stream = sycl_ctx->stream(sycl_ctx->device, 0);
    { ggml_sycl_pipe_wait_scope ws_; SYCL_CHECK(CHECK_TRY_ERROR((stream)->wait())); }

    GGML_UNUSED(backend);
}
catch (sycl::exception const &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
  std::exit(1);
}

static bool ggml_sycl_is_view_or_noop(const ggml_tensor * t) {
    return ggml_is_empty(t) || t->op == GGML_OP_RESHAPE || t->op == GGML_OP_TRANSPOSE ||
           t->op == GGML_OP_VIEW || t->op == GGML_OP_PERMUTE || t->op == GGML_OP_NONE;
}

// match gated_delta_net + the strided cpy that scatters its state snapshots into the cache
// (slot i -> rollback group i, slot 0 newest), so the kernel can write them and skip the cpy.
// returns the number of following nodes to skip (0 = no fusion)
// ported from ggml_cuda_try_gdn_cache_fusion - pure graph inspection, backend-agnostic
static int ggml_sycl_try_gdn_cache_fusion(const ggml_cgraph * cgraph, int node_idx,
                                          ggml_sycl_gated_delta_net_fused_cache & fused_state_cpy) {
    if (!g_ggml_sycl_enable_fusion) {
        return 0;
    }

    const ggml_tensor * gdn = cgraph->nodes[node_idx];
    // the kernel skips the snapshot tail, so the gdn output must not be a graph output, and the cpy
    // found below is taken to be its only reader, as it is in every graph that builds this op
    if (gdn->op != GGML_OP_GATED_DELTA_NET || gdn->type != GGML_TYPE_F32 ||
        (gdn->flags & GGML_TENSOR_FLAG_OUTPUT)) {
        return 0;
    }

    const ggml_tensor * src_v     = gdn->src[2];
    const int64_t       S_v       = src_v->ne[0];
    const int64_t       H         = src_v->ne[1];
    const int64_t       n_tokens  = src_v->ne[2];
    const int64_t       n_seqs    = src_v->ne[3];
    const int64_t       D         = S_v * S_v * H;
    const int64_t       K         = ggml_get_op_params_i32(gdn, 0); // snapshot slot count
    const int64_t       n_written = std::min<int64_t>(n_tokens, K); // newest n_written slots are written

    // snapshot tail starts right after the attention scores
    const size_t tail_off = ggml_row_size(GGML_TYPE_F32, S_v * H * n_tokens * n_seqs);

    // the cpy must be the first node the compute loop below runs, so nothing can read the cache first.
    // skip exactly what that loop skips: views, no-ops, and nodes the graph does not compute.
    const ggml_tensor * cpy  = nullptr;
    int                 skip = 0;
    for (int j = node_idx + 1; j < cgraph->n_nodes && cpy == nullptr; ++j) {
        const ggml_tensor * n = cgraph->nodes[j];
        if (ggml_sycl_is_view_or_noop(n) || (n->flags & GGML_TENSOR_FLAG_COMPUTE) == 0) {
            continue;
        }
        if (n->op != GGML_OP_CPY || (n->flags & GGML_TENSOR_FLAG_OUTPUT)) {
            return 0;
        }
        cpy  = n;
        skip = j - node_idx;
    }
    if (cpy == nullptr) {
        return 0;
    }

    const ggml_tensor * src = cpy->src[0]; // view of the gdn snapshot tail
    const ggml_tensor * dst = cpy->src[1]; // cache view the kernel writes to

    // src must be this gdn's snapshot tail (contiguous, at the tail offset)
    if (src->op != GGML_OP_VIEW || src->view_src != gdn || src->view_offs != tail_off ||
        !ggml_is_contiguous(src)) {
        return 0;
    }

    // dst is the [D, n_seqs, n_written] cache view, with the per-seq stride D that the kernel assumes.
    // ggml_cpy pins src to the same element count, so src needs no shape check of its own.
    const std::array<int64_t, GGML_MAX_DIMS> expected_ne = { D, n_seqs, n_written, 1 };
    if (dst->op != GGML_OP_VIEW || dst->type != GGML_TYPE_F32 || dst->data == nullptr ||
        !std::equal(expected_ne.begin(), expected_ne.end(), dst->ne) ||
        dst->nb[0] != ggml_type_size(GGML_TYPE_F32) ||
        dst->nb[1] != (size_t) ggml_row_size(GGML_TYPE_F32, D)) {
        return 0;
    }

    fused_state_cpy.data        = (float *) dst->data; // rollback group 0 (newest)
    fused_state_cpy.slot_stride = K > 1 ? (int64_t) (dst->nb[2] / sizeof(float)) : 0;
    return skip;
}

// {ADD, RMS_NORM, MUL} where the ADD result stays live as the next residual. That gives the group
// two outputs (ADD and MUL), which the single-output chain check ggml_can_fuse cannot express.
static bool ggml_sycl_can_fuse_add_rms_norm_mul(const ggml_cgraph * cgraph, int i) {
    if (!g_ggml_sycl_enable_fusion || i + 2 >= cgraph->n_nodes) {
        return false;
    }
    const ggml_tensor * add = cgraph->nodes[i];
    const ggml_tensor * rms = cgraph->nodes[i + 1];
    const ggml_tensor * mul = cgraph->nodes[i + 2];
    if (add->op != GGML_OP_ADD || rms->op != GGML_OP_RMS_NORM || mul->op != GGML_OP_MUL) {
        return false;
    }
    if (rms->src[0] != add || (mul->src[0] != rms && mul->src[1] != rms)) {
        return false;
    }
    if (!ggml_can_fuse_subgraph(cgraph, i, { GGML_OP_ADD, GGML_OP_RMS_NORM, GGML_OP_MUL }, { i, i + 2 })) {
        return false;
    }
    const ggml_tensor * a = add->src[0];
    const ggml_tensor * b = add->src[1];
    const ggml_tensor * w = mul->src[0] == rms ? mul->src[1] : mul->src[0];
    for (const ggml_tensor * t : { a, b, add, rms, mul, w }) {
        if (t->type != GGML_TYPE_F32) {
            return false;
        }
    }
    // no broadcast of the residual operands; one contiguous layout shared by a, b and the sum
    if (!ggml_are_same_shape(a, add) || !ggml_are_same_shape(b, add) || !ggml_are_same_shape(mul, add)) {
        return false;
    }
    if (!ggml_is_contiguous(a) || !ggml_is_contiguous(b) || !ggml_is_contiguous(add) || !ggml_is_contiguous(mul)) {
        return false;
    }
    // the kernel indexes the weight as w[col] and broadcasts it like the unfused MUL
    if (w->ne[0] != add->ne[0] || w->nb[0] != sizeof(float) || !ggml_is_contiguous_rows(w)) {
        return false;
    }
    for (int d = 1; d < 4; ++d) {
        if (add->ne[d] % w->ne[d] != 0) {
            return false;
        }
    }
    return true;
}

// Opt-in contiguous F32 SCALE -> SiLU. Preserve all intermediate consumers.
static bool ggml_sycl_can_fuse_scale_silu(const ggml_cgraph * graph, int i) {
    static const bool enabled = ggml_sycl_get_env("GGML_SYCL_SCALE_SILU", 0) != 0;
    if (!enabled || !g_ggml_sycl_enable_fusion ||
        !ggml_can_fuse_subgraph(graph, i, { GGML_OP_SCALE, GGML_OP_UNARY }, { i + 1 })) {
        return false;
    }
    const ggml_tensor * scale = graph->nodes[i];
    const ggml_tensor * out = graph->nodes[i + 1];
    const ggml_tensor * input = scale->src[0];
    if (out->src[0] != scale || scale->view_src || out->view_src || ggml_get_unary_op(out) != GGML_UNARY_OP_SILU ||
        ggml_get_op_params_f32(scale, 1) != 0.0f ||
        input->type != GGML_TYPE_F32 || out->type != GGML_TYPE_F32 ||
        !ggml_is_contiguous(input) || !ggml_is_contiguous(out) || !ggml_are_same_shape(input, out)) {
        return false;
    }
    const uintptr_t src = (uintptr_t) input->data;
    const uintptr_t dst = (uintptr_t) out->data;
    return src == dst || src + ggml_nbytes(input) <= dst || dst + ggml_nbytes(out) <= src;
}

static void ggml_sycl_scale_silu(ggml_backend_sycl_context & ctx, const ggml_tensor * scale, ggml_tensor * out) {
    const float * input = (const float *) scale->src[0]->data;
    float * output = (float *) out->data;
    const int64_t n = ggml_nelements(out);
    const float factor = ggml_get_op_params_f32(scale, 0);
    constexpr size_t block = 256;
    ctx.stream()->parallel_for(
        sycl::nd_range<1>(sycl::range<1>(((n + block - 1) / block) * block), sycl::range<1>(block)),
        [=](sycl::nd_item<1> item) {
            const int64_t j = item.get_global_id(0);
            if (j < n) {
                const float x = input[j] * factor;
                output[j] = x / (1.0f + sycl::exp(-x));
            }
        });
}


// ---- Flash-Next / Qwen-style decode small-op fusions (r60) ----------------------------------------------------------
// Each replaces a short chain of tiny launches with one kernel that computes the same fp32 expressions in the same
// order, so results are bit-identical to the unfused chain.

static bool ggml_sycl_fusion_out_disjoint_or_same(const ggml_tensor * out, const ggml_tensor * src, bool allow_same) {
    const uintptr_t ob = (uintptr_t) out->data, oe = ob + ggml_nbytes(out);
    const uintptr_t sb = (uintptr_t) src->data, se = sb + ggml_nbytes(src);
    if (allow_same && ob == sb && oe == se) { return true; }
    return oe <= sb || se <= ob;
}

// (1) MoE expert sum: ADD(v0, v1), ADD(.., v2) ... over the n_expert_used views of ffn_moe_weighted -> one kernel
#define GGML_SYCL_MOE_SUM_MAX 16
struct ggml_sycl_moe_sum_match {
    int n_adds = 0;
    int n_terms = 0;
    const float * p[GGML_SYCL_MOE_SUM_MAX];
    int64_t ne0 = 0, ntok = 0, src_s1 = 0;
    ggml_tensor * out = nullptr;
};

static bool ggml_sycl_match_moe_sum(const ggml_cgraph * g, int i, ggml_sycl_moe_sum_match & m) {
    static const bool disabled = ggml_sycl_get_env("GGML_SYCL_DISABLE_MOE_SUM_FUSION", 0) != 0;
    if (disabled || !g_ggml_sycl_enable_fusion || i + 2 >= g->n_nodes) { return false; }
    const ggml_tensor * a = g->nodes[i];
    if (a->op != GGML_OP_ADD || a->type != GGML_TYPE_F32 || !a->src[0] || !a->src[1]) { return false; }
    const ggml_tensor * v0 = a->src[0];
    const ggml_tensor * v1 = a->src[1];
    if (v0->op != GGML_OP_VIEW || v1->op != GGML_OP_VIEW || !v0->view_src || v0->view_src != v1->view_src) { return false; }
    const ggml_tensor * base = v0->view_src;
    if (base->type != GGML_TYPE_F32 || base->ne[3] != 1 || base->nb[0] != sizeof(float)) { return false; }
    if (v0->type != GGML_TYPE_F32 || v0->ne[2] != 1 || v0->ne[3] != 1 || v0->nb[0] != sizeof(float)) { return false; }
    const int64_t ne0 = v0->ne[0], ntok = v0->ne[1];
    if (v1->ne[0] != ne0 || v1->ne[1] != ntok || v0->nb[1] != v1->nb[1]) { return false; }
    const size_t stride = v1->view_offs - v0->view_offs;
    if (v1->view_offs <= v0->view_offs || stride != base->nb[1] || v0->view_offs != 0) { return false; }
    if (base->ne[0] != ne0 || base->ne[2] != ntok) { return false; }
    m.p[0] = (const float *) v0->data;
    m.p[1] = (const float *) v1->data;
    int terms = 2;
    int L = 1;
    while (i + L < g->n_nodes && terms < GGML_SYCL_MOE_SUM_MAX) {
        const ggml_tensor * n = g->nodes[i + L];
        if (n->op != GGML_OP_ADD || n->src[0] != g->nodes[i + L - 1] || !n->src[1]) { break; }
        const ggml_tensor * vk = n->src[1];
        if (vk->op != GGML_OP_VIEW || vk->view_src != base || vk->type != GGML_TYPE_F32 ||
            vk->ne[0] != ne0 || vk->ne[1] != ntok || vk->nb[1] != v0->nb[1] ||
            vk->view_offs != (size_t) terms * stride) { break; }
        m.p[terms++] = (const float *) vk->data;
        L++;
    }
    if (terms != base->ne[1] || terms < 3) { return false; }
    std::vector<ggml_op> ops(L, GGML_OP_ADD);
    int outs[1] = { i + L - 1 };
    if (!ggml_can_fuse_subgraph(g, i, (int) ops.size(), ops.data(), outs, 1)) { return false; }
    ggml_tensor * out = g->nodes[i + L - 1];
    if (out->type != GGML_TYPE_F32 || !ggml_is_contiguous(out) || out->ne[0] != ne0 || out->ne[1] != ntok ||
        out->ne[2] != 1 || out->ne[3] != 1) { return false; }
    if (!ggml_sycl_fusion_out_disjoint_or_same(out, base, false)) { return false; }
    m.n_adds = L; m.n_terms = terms; m.ne0 = ne0; m.ntok = ntok; m.src_s1 = (int64_t) (v0->nb[1] / sizeof(float)); m.out = out;
    return true;
}

struct ggml_sycl_moe_sum_ptrs { const float * p[GGML_SYCL_MOE_SUM_MAX]; };

static void ggml_sycl_moe_sum(ggml_backend_sycl_context & ctx, const ggml_sycl_moe_sum_match & m) {
    float * dst = (float *) m.out->data;
    const int64_t n = m.ne0 * m.ntok;
    const int64_t ne0 = m.ne0, s1 = m.src_s1;
    const int nt = m.n_terms;
    ggml_sycl_moe_sum_ptrs pp;
    for (int k = 0; k < GGML_SYCL_MOE_SUM_MAX; ++k) { pp.p[k] = k < nt ? m.p[k] : m.p[0]; }
    constexpr size_t block = 256;
    ctx.stream()->parallel_for(
        sycl::nd_range<1>(sycl::range<1>(((n + block - 1) / block) * block), sycl::range<1>(block)),
        [=](sycl::nd_item<1> item) {
            const int64_t j = item.get_global_id(0);
            if (j >= n) { return; }
            const int64_t d = j % ne0, t = j / ne0;
            const int64_t off = d + t * s1;
            float acc = pp.p[0][off] + pp.p[1][off];
            for (int k = 2; k < nt; ++k) { acc = acc + pp.p[k][off]; }
            dst[j] = acc;
        });
}

// (2) shared-expert gate tail: SIGMOID(g) -> MUL(shexp, sig) -> ADD(moe_out, gated)
static bool ggml_sycl_match_shexp_gate(const ggml_cgraph * g, int i) {
    static const bool disabled = ggml_sycl_get_env("GGML_SYCL_DISABLE_SHEXP_GATE_FUSION", 0) != 0;
    if (disabled || !g_ggml_sycl_enable_fusion || i + 2 >= g->n_nodes) { return false; }
    const ggml_tensor * sg = g->nodes[i];
    const ggml_tensor * mu = g->nodes[i + 1];
    const ggml_tensor * ad = g->nodes[i + 2];
    if (sg->op != GGML_OP_UNARY || ggml_get_unary_op(sg) != GGML_UNARY_OP_SIGMOID || mu->op != GGML_OP_MUL || ad->op != GGML_OP_ADD) { return false; }
    if (mu->src[1] != sg || ad->src[1] != mu || !mu->src[0] || !ad->src[0] || !sg->src[0]) { return false; }
    const ggml_tensor * gin = sg->src[0];
    const ggml_tensor * shx = mu->src[0];
    const ggml_tensor * moe = ad->src[0];
    const ggml_tensor * all[] = { gin, shx, moe, sg, mu, ad };
    for (const ggml_tensor * t : all) {
        if (t->type != GGML_TYPE_F32 || t->ne[2] != 1 || t->ne[3] != 1) { return false; }
    }
    const int64_t D = shx->ne[0], T = shx->ne[1];
    if (sg->ne[0] != 1 || sg->ne[1] != T || gin->ne[0] != 1 || gin->ne[1] != T) { return false; }
    if (mu->ne[0] != D || mu->ne[1] != T || ad->ne[0] != D || ad->ne[1] != T || moe->ne[0] != D || moe->ne[1] != T) { return false; }
    if (!ggml_is_contiguous(gin) || !ggml_is_contiguous(shx) || !ggml_is_contiguous(moe) || !ggml_is_contiguous(ad) || !ggml_is_contiguous(sg)) { return false; }
    if (!ggml_can_fuse_subgraph(g, i, { GGML_OP_UNARY, GGML_OP_MUL, GGML_OP_ADD }, { i + 2 })) { return false; }
    return ggml_sycl_fusion_out_disjoint_or_same(ad, moe, true) && ggml_sycl_fusion_out_disjoint_or_same(ad, shx, true) &&
           ggml_sycl_fusion_out_disjoint_or_same(ad, gin, false);
}

static void ggml_sycl_shexp_gate(ggml_backend_sycl_context & ctx, const ggml_cgraph * g, int i) {
    const ggml_tensor * gin = g->nodes[i]->src[0];
    const ggml_tensor * shx = g->nodes[i + 1]->src[0];
    const ggml_tensor * moe = g->nodes[i + 2]->src[0];
    float * dst = (float *) g->nodes[i + 2]->data;
    const float * pg = (const float *) gin->data;
    const float * ps = (const float *) shx->data;
    const float * pm = (const float *) moe->data;
    const int64_t D = shx->ne[0];
    const int64_t n = D * shx->ne[1];
    constexpr size_t block = 256;
    ctx.stream()->parallel_for(
        sycl::nd_range<1>(sycl::range<1>(((n + block - 1) / block) * block), sycl::range<1>(block)),
        [=](sycl::nd_item<1> item) {
            #pragma clang fp contract(off)
            const int64_t j = item.get_global_id(0);
            if (j >= n) { return; }
            const int64_t t = j / D;
            const float x = pg[t];
            const float sig = 1.0f / (1.0f + sycl::exp(-x));
            const float gated = ps[j] * sig;
            dst[j] = pm[j] + gated;
        });
}

// (3) GDN decay gate: ADD(alpha, dt_bias) -> SOFTPLUS -> MUL(.., ssm_a)
static bool ggml_sycl_match_ssm_gate(const ggml_cgraph * g, int i) {
    static const bool disabled = ggml_sycl_get_env("GGML_SYCL_DISABLE_SSM_GATE_FUSION", 0) != 0;
    if (disabled || !g_ggml_sycl_enable_fusion || i + 2 >= g->n_nodes) { return false; }
    const ggml_tensor * ad = g->nodes[i];
    const ggml_tensor * sp = g->nodes[i + 1];
    const ggml_tensor * mu = g->nodes[i + 2];
    if (ad->op != GGML_OP_ADD || sp->op != GGML_OP_UNARY || ggml_get_unary_op(sp) != GGML_UNARY_OP_SOFTPLUS || mu->op != GGML_OP_MUL) { return false; }
    if (sp->src[0] != ad || mu->src[0] != sp || !ad->src[0] || !ad->src[1] || !mu->src[1]) { return false; }
    const ggml_tensor * al = ad->src[0];
    const ggml_tensor * bi = ad->src[1];
    const ggml_tensor * sa = mu->src[1];
    const ggml_tensor * all[] = { al, bi, sa, ad, sp, mu };
    for (const ggml_tensor * t : all) {
        if (t->type != GGML_TYPE_F32 || t->ne[2] != 1 || t->ne[3] != 1) { return false; }
    }
    const int64_t H = al->ne[0], T = al->ne[1];
    if (bi->ne[0] != H || bi->ne[1] != 1 || sa->ne[0] != H || sa->ne[1] != 1) { return false; }
    if (ad->ne[0] != H || ad->ne[1] != T || mu->ne[0] != H || mu->ne[1] != T) { return false; }
    if (!ggml_is_contiguous(al) || !ggml_is_contiguous(bi) || !ggml_is_contiguous(sa) || !ggml_is_contiguous(mu)) { return false; }
    if (!ggml_can_fuse_subgraph(g, i, { GGML_OP_ADD, GGML_OP_UNARY, GGML_OP_MUL }, { i + 2 })) { return false; }
    return ggml_sycl_fusion_out_disjoint_or_same(mu, al, true) && ggml_sycl_fusion_out_disjoint_or_same(mu, bi, false) &&
           ggml_sycl_fusion_out_disjoint_or_same(mu, sa, false);
}

static void ggml_sycl_ssm_gate(ggml_backend_sycl_context & ctx, const ggml_cgraph * g, int i) {
    const ggml_tensor * al = g->nodes[i]->src[0];
    const ggml_tensor * bi = g->nodes[i]->src[1];
    const ggml_tensor * sa = g->nodes[i + 2]->src[1];
    float * dst = (float *) g->nodes[i + 2]->data;
    const float * pa = (const float *) al->data;
    const float * pb = (const float *) bi->data;
    const float * ps = (const float *) sa->data;
    const int64_t H = al->ne[0];
    const int64_t n = H * al->ne[1];
    constexpr size_t block = 128;
    ctx.stream()->parallel_for(
        sycl::nd_range<1>(sycl::range<1>(((n + block - 1) / block) * block), sycl::range<1>(block)),
        [=](sycl::nd_item<1> item) {
            #pragma clang fp contract(off)
            const int64_t j = item.get_global_id(0);
            if (j >= n) { return; }
            const int64_t h = j % H;
            const float x  = pa[j] + pb[h];
            const float ax = sycl::fabs(x);
            const float m  = sycl::fmax(x, 0.0f);
            const float sp = m + sycl::log1p(sycl::exp(-ax));
            dst[j] = sp * ps[h];
        });
}

static int ggml_sycl_next_compute_node(const ggml_cgraph * graph, int i);

// (4) GDN conv-state rollback snapshots: after CONCAT(conv_state, qkv) the graph copies K sliding 3-column windows of
// the concat result into K slots of the recurrent cache with K x (CONT, CPY). One kernel writes all of them straight
// from the concat output (the concat itself still runs: SSM_CONV reads it).
#define GGML_SYCL_CONV_SNAP_MAX 8
struct ggml_sycl_conv_snap_match {
    int k = 0;
    int last = 0;
    const float * src[GGML_SYCL_CONV_SNAP_MAX];
    float * dst[GGML_SYCL_CONV_SNAP_MAX];
    int64_t s1[GGML_SYCL_CONV_SNAP_MAX];
    int64_t w = 0, rows = 0;
};

static bool ggml_sycl_match_conv_snap(const ggml_cgraph * g, int i, ggml_sycl_conv_snap_match & m) {
    static const bool disabled = ggml_sycl_get_env("GGML_SYCL_DISABLE_CONV_SNAP_FUSION", 0) != 0;
    if (disabled || !g_ggml_sycl_enable_fusion) { return false; }
    const ggml_tensor * c = g->nodes[i];
    if (c->op != GGML_OP_CONCAT || c->type != GGML_TYPE_F32 || c->ne[2] != 1 || c->ne[3] != 1 ||
        ggml_get_op_params_i32(c, 0) != 0) { return false; }
    std::vector<int> idxs;
    std::vector<ggml_op> ops;
    std::vector<int> outs;
    int prev = i;
    int k = 0;
    while (k < GGML_SYCL_CONV_SNAP_MAX) {
        const int ia = ggml_sycl_next_compute_node(g, prev);
        if (ia < 0) { break; }
        const int ib = ggml_sycl_next_compute_node(g, ia);
        if (ib < 0) { break; }
        const ggml_tensor * a = g->nodes[ia];
        const ggml_tensor * b = g->nodes[ib];
        if (a->op != GGML_OP_CONT || b->op != GGML_OP_CPY || b->src[0] != a || !b->src[1]) { break; }
        const ggml_tensor * v = a->src[0];
        if (!v || v->op != GGML_OP_VIEW || v->view_src != c || v->type != GGML_TYPE_F32 || a->type != GGML_TYPE_F32) { break; }
        const ggml_tensor * d = b->src[1];
        if (d->type != GGML_TYPE_F32 || b->type != GGML_TYPE_F32 || !ggml_is_contiguous(d) || !ggml_is_contiguous(a) ||
            ggml_nelements(d) != ggml_nelements(a)) { break; }
        if (a->ne[2] != 1 || a->ne[3] != 1 || v->ne[0] != a->ne[0] || v->ne[1] != a->ne[1] ||
            v->nb[0] != sizeof(float) || v->nb[1] % sizeof(float) != 0) { break; }
        if (k > 0 && (a->ne[0] != m.w || a->ne[1] != m.rows)) { break; }
        m.w = a->ne[0];
        m.rows = a->ne[1];
        m.src[k] = (const float *) v->data;
        m.dst[k] = (float *) d->data;
        m.s1[k] = (int64_t) (v->nb[1] / sizeof(float));
        idxs.push_back(ia); ops.push_back(GGML_OP_CONT);
        idxs.push_back(ib); ops.push_back(GGML_OP_CPY);
        outs.push_back(ib);
        m.last = ib;
        prev = ib;
        ++k;
    }
    if (k < 2) { return false; }
    m.k = k;
    if (!ggml_can_fuse_subgraph_ext(g, idxs.data(), (int) idxs.size(), ops.data(), outs.data(), (int) outs.size())) { return false; }
    // destinations must not alias the concat result or each other
    const uintptr_t cb = (uintptr_t) c->data, ce = cb + ggml_nbytes(c);
    const size_t nbytes = (size_t) (m.w * m.rows) * sizeof(float);
    for (int a = 0; a < k; ++a) {
        const uintptr_t da = (uintptr_t) m.dst[a];
        if (da < ce && cb < da + nbytes) { return false; }
        for (int b = a + 1; b < k; ++b) {
            const uintptr_t db = (uintptr_t) m.dst[b];
            if (da < db + nbytes && db < da + nbytes) { return false; }
        }
    }
    return true;
}

struct ggml_sycl_conv_snap_args {
    const float * src[GGML_SYCL_CONV_SNAP_MAX];
    float * dst[GGML_SYCL_CONV_SNAP_MAX];
    int64_t s1[GGML_SYCL_CONV_SNAP_MAX];
};

static void ggml_sycl_conv_snap(ggml_backend_sycl_context & ctx, const ggml_sycl_conv_snap_match & m) {
    ggml_sycl_conv_snap_args a;
    for (int k = 0; k < GGML_SYCL_CONV_SNAP_MAX; ++k) {
        const int kk = k < m.k ? k : 0;
        a.src[k] = m.src[kk]; a.dst[k] = m.dst[kk]; a.s1[k] = m.s1[kk];
    }
    const int64_t w = m.w, per = m.w * m.rows;
    const int64_t n = per * m.k;
    constexpr size_t block = 256;
    ctx.stream()->parallel_for(
        sycl::nd_range<1>(sycl::range<1>(((n + block - 1) / block) * block), sycl::range<1>(block)),
        [=](sycl::nd_item<1> item) {
            const int64_t j = item.get_global_id(0);
            if (j >= n) { return; }
            const int k = (int) (j / per);
            const int64_t r = j - (int64_t) k * per;
            const int64_t row = r / w, col = r - row * w;
            a.dst[k][r] = a.src[k][row * a.s1[k] + col];
        });
}

static bool ggml_sycl_can_fuse_hc_post_gate(const ggml_cgraph * graph, int i) {
    static const bool disabled = ggml_sycl_get_env("GGML_SYCL_DISABLE_HC_POST_GATE", 0) != 0;
    if (disabled || !g_ggml_sycl_enable_fusion ||
        !ggml_can_fuse_subgraph(graph, i, { GGML_OP_SCALE, GGML_OP_UNARY, GGML_OP_SCALE, GGML_OP_DSV4_HC_POST }, { i + 3 })) {
        return false;
    }
    const ggml_tensor * a = graph->nodes[i];
    const ggml_tensor * b = graph->nodes[i + 1];
    const ggml_tensor * c = graph->nodes[i + 2];
    const ggml_tensor * d = graph->nodes[i + 3];
    const ggml_tensor * input = a->src[0];
    if (b->src[0] != a || c->src[0] != b || d->src[2] != c ||
        ggml_get_unary_op(b) != GGML_UNARY_OP_SIGMOID ||
        ggml_get_op_params_f32(a, 1) != 0.0f || ggml_get_op_params_f32(c, 1) != 0.0f ||
        input->type != GGML_TYPE_F32 || !ggml_are_same_shape(input, c)) {
        return false;
    }
    const uintptr_t out_begin = (uintptr_t) d->data;
    const uintptr_t out_end = out_begin + ggml_nbytes(d);
    const ggml_tensor * sources[] = { input, d->src[0], d->src[1], d->src[3] };
    for (const ggml_tensor * src : sources) {
        if (!src) { continue; }
        const uintptr_t begin = (uintptr_t) src->data;
        if (begin < out_end && out_begin < begin + ggml_nbytes(src)) {
            return false;
        }
    }
    return true;
}

// ---- DeepSeek-V4 small-op fusions ---------------------------------------------------------------------------
// Decode runs thousands of launches on tensors of a few elements; each costs a fixed host round trip.

static int ggml_sycl_next_compute_node(const ggml_cgraph * graph, int i) {
    for (int j = i + 1; j < graph->n_nodes; ++j) {
        if (!ggml_sycl_is_view_or_noop(graph->nodes[j])) {
            return (graph->nodes[j]->flags & GGML_TENSOR_FLAG_COMPUTE) ? j : -1;
        }
    }
    return -1;
}

static int ggml_sycl_tensor_use_count(const ggml_cgraph * graph, const ggml_tensor * t) {
    const size_t pos = ggml_hash_find(&graph->visited_hash_set, t);
    if (!ggml_bitset_get(graph->visited_hash_set.used, pos)) {
        return 0;
    }
    return graph->use_counts[pos];
}

static bool ggml_sycl_ranges_disjoint(const void * a, size_t na, const void * b, size_t nb) {
    const uintptr_t pa = (uintptr_t) a;
    const uintptr_t pb = (uintptr_t) b;
    return pa + na <= pb || pb + nb <= pa;
}

// dense single-column F32 vector (views of a wider tensor keep the parent's nb[1], so ggml_is_contiguous is too strict)
static bool ggml_sycl_is_f32_column(const ggml_tensor * t) {
    return t->type == GGML_TYPE_F32 && t->nb[0] == sizeof(float) && ggml_nelements(t) == t->ne[0];
}

struct ggml_sycl_hc_gate_match {
    const ggml_tensor * x = nullptr;
    const ggml_tensor * s = nullptr;
    const ggml_tensor * b = nullptr;
    ggml_tensor *       out = nullptr;
    int                 last = -1;
};

// out = factor * sigmoid(x * s + b) + bias, the DSV4 hyper-connection pre/post gate
static bool ggml_sycl_match_hc_gate(const ggml_cgraph * graph, int i, ggml_sycl_hc_gate_match & m) {
    static const bool disabled = ggml_sycl_get_env("GGML_SYCL_DISABLE_HC_GATE_FUSION", 0) != 0;
    if (disabled || !g_ggml_sycl_enable_fusion) {
        return false;
    }
    const ggml_tensor * mul = graph->nodes[i];
    const int ia = ggml_sycl_next_compute_node(graph, i);
    const int iu = ia < 0 ? -1 : ggml_sycl_next_compute_node(graph, ia);
    const int is = iu < 0 ? -1 : ggml_sycl_next_compute_node(graph, iu);
    if (mul->op != GGML_OP_MUL || is < 0) {
        return false;
    }
    const ggml_tensor * add   = graph->nodes[ia];
    const ggml_tensor * unary = graph->nodes[iu];
    ggml_tensor *       scale = graph->nodes[is];
    if (add->op != GGML_OP_ADD || unary->op != GGML_OP_UNARY || scale->op != GGML_OP_SCALE ||
        ggml_get_unary_op(unary) != GGML_UNARY_OP_SIGMOID ||
        add->src[0] != mul || unary->src[0] != add || scale->src[0] != unary ||
        !ggml_node_has_n_uses(graph, i, 1) || !ggml_node_has_n_uses(graph, ia, 1) || !ggml_node_has_n_uses(graph, iu, 1)) {
        return false;
    }
    const ggml_tensor * x = mul->src[0];
    const ggml_tensor * s = mul->src[1];
    const ggml_tensor * b = add->src[1];
    const int64_t n = ggml_nelements(scale);
    if (!ggml_sycl_is_f32_column(x) || !ggml_sycl_is_f32_column(b) || !ggml_sycl_is_f32_column(scale) ||
        s->type != GGML_TYPE_F32 || ggml_nelements(s) != 1 || n < 1 || n > 4096 ||
        ggml_nelements(x) != n || ggml_nelements(b) != n || ggml_nelements(mul) != n) {
        return false;
    }
    // element j only depends on element j, so writing in place over x is fine; a partial overlap is not
    const size_t nbytes = (size_t) n * sizeof(float);
    if (scale->data != x->data && !ggml_sycl_ranges_disjoint(scale->data, nbytes, x->data, nbytes)) {
        return false;
    }
    if (!ggml_sycl_ranges_disjoint(scale->data, nbytes, b->data, nbytes) ||
        !ggml_sycl_ranges_disjoint(scale->data, nbytes, s->data, sizeof(float))) {
        return false;
    }
    m.x = x; m.s = s; m.b = b; m.out = scale; m.last = is;
    return true;
}

static void ggml_sycl_hc_gate(ggml_backend_sycl_context & ctx, const ggml_sycl_hc_gate_match & m) {
    const float * x   = (const float *) m.x->data;
    const float * s   = (const float *) m.s->data;
    const float * b   = (const float *) m.b->data;
    float *       out = (float *) m.out->data;
    const size_t  n      = (size_t) ggml_nelements(m.out);
    const float   factor = ggml_get_op_params_f32(m.out, 0);
    const float   bias   = ggml_get_op_params_f32(m.out, 1);
    ctx.stream()->parallel_for(sycl::range<1>(n), [=](sycl::id<1> id) {
        const size_t j = id[0];
        const float  v = x[j] * s[0] + b[j];
        out[j] = factor * (1.0f / (1.0f + sycl::exp(-v))) + bias;
    });
}

struct ggml_sycl_moe_norm_match {
    const ggml_tensor * w = nullptr;
    const ggml_tensor * clamp = nullptr;
    ggml_tensor *       out = nullptr;
    int                 last = -1;
};

// out = factor * w / clamp(sum(w), lo, hi) + bias over the handful of selected-expert weights
static bool ggml_sycl_match_moe_norm(const ggml_cgraph * graph, int i, ggml_sycl_moe_norm_match & m) {
    static const bool disabled = ggml_sycl_get_env("GGML_SYCL_DISABLE_MOE_NORM_FUSION", 0) != 0;
    if (disabled || !g_ggml_sycl_enable_fusion) {
        return false;
    }
    const ggml_tensor * sum = graph->nodes[i];
    const int ic = ggml_sycl_next_compute_node(graph, i);
    const int id = ic < 0 ? -1 : ggml_sycl_next_compute_node(graph, ic);
    const int is = id < 0 ? -1 : ggml_sycl_next_compute_node(graph, id);
    if (sum->op != GGML_OP_SUM_ROWS || is < 0) {
        return false;
    }
    const ggml_tensor * clamp = graph->nodes[ic];
    const ggml_tensor * div   = graph->nodes[id];
    ggml_tensor *       scale = graph->nodes[is];
    if (clamp->op != GGML_OP_CLAMP || div->op != GGML_OP_DIV || scale->op != GGML_OP_SCALE ||
        clamp->src[0] != sum || div->src[1] != clamp ||
        !ggml_node_has_n_uses(graph, i, 1) || !ggml_node_has_n_uses(graph, ic, 1) || !ggml_node_has_n_uses(graph, id, 1)) {
        return false;
    }
    // SCALE reads DIV directly or through one reshape/view of it
    const ggml_tensor * scaled = scale->src[0];
    if (scaled != div && !(scaled->view_src == div && scaled->data == div->data &&
                           ggml_sycl_tensor_use_count(graph, scaled) == 1)) {
        return false;
    }
    const ggml_tensor * w  = sum->src[0];
    const ggml_tensor * w2 = div->src[0];
    const int64_t n = ggml_nelements(w);
    if (!ggml_sycl_is_f32_column(w) || w2->type != GGML_TYPE_F32 || w2->data != w->data || ggml_nelements(w2) != n ||
        w2->nb[0] != sizeof(float) || ggml_nelements(sum) != 1 || ggml_nelements(div) != n ||
        scale->type != GGML_TYPE_F32 || ggml_nelements(scale) != n || !ggml_is_contiguous(scale) || n < 1 || n > 64) {
        return false;
    }
    // every work item re-reads all of w, so the output must not overlap it
    if (!ggml_sycl_ranges_disjoint(scale->data, (size_t) n * sizeof(float), w->data, (size_t) n * sizeof(float))) {
        return false;
    }
    m.w = w; m.clamp = clamp; m.out = scale; m.last = is;
    return true;
}

static void ggml_sycl_moe_norm(ggml_backend_sycl_context & ctx, const ggml_sycl_moe_norm_match & m) {
    const float * w   = (const float *) m.w->data;
    float *       out = (float *) m.out->data;
    const int     n      = (int) ggml_nelements(m.out);
    const float   lo     = ggml_get_op_params_f32(m.clamp, 0);
    const float   hi     = ggml_get_op_params_f32(m.clamp, 1);
    const float   factor = ggml_get_op_params_f32(m.out, 0);
    const float   bias   = ggml_get_op_params_f32(m.out, 1);
    ctx.stream()->parallel_for(sycl::range<1>((size_t) n), [=](sycl::id<1> id) {
        float sum = 0.0f;
        for (int k = 0; k < n; ++k) {
            sum += w[k];
        }
        sum = sycl::fmin(sycl::fmax(sum, lo), hi);
        out[id[0]] = factor * (w[id[0]] / sum) + bias;
    });
}

// ---- qwen4exp QSA indexer: gather + mean-pool of the cached block member keys in one launch ----------------
struct ggml_sycl_qsa_pool_match {
    const ggml_tensor * keys = nullptr;   // [d, n_kv, 1] F16/F32, rows nb[1] apart
    const ggml_tensor * idx  = nullptr;   // I32 [r * n_blocks]
    ggml_tensor *       out  = nullptr;   // SCALE node, F32 [d, n_blocks]
    int                 r    = 0;
    int                 last = -1;
};

static bool ggml_sycl_match_qsa_pool(const ggml_cgraph * graph, int i, ggml_sycl_qsa_pool_match & m) {
    static const bool disabled = ggml_sycl_get_env("GGML_SYCL_DISABLE_QSA_POOL_FUSION", 0) != 0;
    if (disabled || !g_ggml_sycl_enable_fusion) {
        return false;
    }
    const ggml_tensor * g = graph->nodes[i];
    if (g->op != GGML_OP_GET_ROWS || g->type != GGML_TYPE_F32 || g->view_src != nullptr ||
        (g->flags & GGML_TENSOR_FLAG_OUTPUT) || ggml_node_get_use_count(graph, i) != 1) {
        return false;
    }
    const ggml_tensor * keys = g->src[0];
    const ggml_tensor * idx  = g->src[1];
    const int64_t d     = g->ne[0];
    const int64_t n_idx = g->ne[1];
    if ((keys->type != GGML_TYPE_F16 && keys->type != GGML_TYPE_F32) || keys->nb[0] != ggml_type_size(keys->type) ||
        keys->ne[0] != d || keys->ne[2] != 1 || keys->ne[3] != 1 ||
        idx->type != GGML_TYPE_I32 || idx->nb[0] != sizeof(int32_t) || ggml_nelements(idx) != n_idx ||
        g->ne[2] != 1 || g->ne[3] != 1) {
        return false;
    }

    // CONT(slice 0), then {CONT(slice k), ADD}*, then SCALE; views in between are skipped
    const ggml_tensor * members = nullptr;   // the reshape of g that the slices view
    const ggml_tensor * acc     = nullptr;
    int64_t n_blocks = 0;
    int     r        = 0;
    int     j        = i;
    for (;;) {
        j = ggml_sycl_next_compute_node(graph, j);
        if (j < 0) {
            return false;
        }
        const ggml_tensor * node = graph->nodes[j];
        if (node->op == GGML_OP_SCALE) {
            if (r < 2 || node->src[0] != acc || ggml_get_op_params_f32(node, 1) != 0.0f) {
                return false;
            }
            break;
        }
        if (node->op != GGML_OP_CONT || r >= 16) {
            return false;
        }
        const ggml_tensor * slice = node->src[0];
        if (slice->view_src != g || slice->src[0] == nullptr || slice->type != GGML_TYPE_F32 ||
            !ggml_node_has_n_uses(graph, j, 1)) {
            return false;
        }
        if (r == 0) {
            members  = slice->src[0];
            n_blocks = slice->ne[1];
            if (n_blocks < 1 || n_idx % n_blocks != 0) {
                return false;
            }
        }
        const int64_t r_total = n_idx / n_blocks;
        // slice k = member k of every block: offset k*d floats, blocks r_total*d floats apart
        if (slice->src[0] != members || slice->ne[0] != d || slice->ne[1] != n_blocks || slice->ne[2] != 1 ||
            slice->ne[3] != 1 || slice->nb[0] != sizeof(float) || slice->nb[1] != (size_t) (r_total * d) * sizeof(float) ||
            slice->view_offs != (size_t) ((int64_t) r * d) * sizeof(float) || node->ne[0] != d || node->ne[1] != n_blocks) {
            return false;
        }
        if (r == 0) {
            acc = node;
        } else {
            const int ja = ggml_sycl_next_compute_node(graph, j);
            if (ja < 0) {
                return false;
            }
            const ggml_tensor * add = graph->nodes[ja];
            if (add->op != GGML_OP_ADD || add->src[0] != acc || add->src[1] != node || !ggml_node_has_n_uses(graph, ja, 1) ||
                add->ne[0] != d || add->ne[1] != n_blocks) {
                return false;
            }
            acc = add;
            j   = ja;
        }
        r++;
    }
    ggml_tensor * scale = graph->nodes[j];
    if (members == nullptr || (int64_t) r * n_blocks != n_idx || ggml_sycl_tensor_use_count(graph, members) != r ||
        scale->type != GGML_TYPE_F32 || scale->ne[0] != d || scale->ne[1] != n_blocks || !ggml_is_contiguous(scale)) {
        return false;
    }
    const size_t keys_bytes = (size_t) keys->nb[1] * (size_t) keys->ne[1];
    if (!ggml_sycl_ranges_disjoint(scale->data, ggml_nbytes(scale), keys->data, keys_bytes) ||
        !ggml_sycl_ranges_disjoint(scale->data, ggml_nbytes(scale), idx->data, ggml_nbytes(idx))) {
        return false;
    }
    m.keys = keys; m.idx = idx; m.out = scale; m.r = r; m.last = j;
    return true;
}

static void ggml_sycl_qsa_pool(ggml_backend_sycl_context & ctx, const ggml_sycl_qsa_pool_match & m) {
    const char *    keys       = (const char *) m.keys->data;
    const int32_t * idx        = (const int32_t *) m.idx->data;
    float *         out        = (float *) m.out->data;
    const size_t    row_stride = m.keys->nb[1];
    const size_t    d          = (size_t) m.out->ne[0];
    const size_t    n_blocks   = (size_t) m.out->ne[1];
    const int       r          = m.r;
    const float     factor     = ggml_get_op_params_f32(m.out, 0);
    if (m.keys->type == GGML_TYPE_F16) {
        ctx.stream()->parallel_for(sycl::range<2>(n_blocks, d), [=](sycl::id<2> id) {
            const size_t b = id[0];
            const size_t c = id[1];
            float sum = 0.0f;
            for (int k = 0; k < r; ++k) {
                const sycl::half * row = (const sycl::half *) (keys + (size_t) idx[b * r + k] * row_stride);
                sum += (float) row[c];
            }
            out[b * d + c] = sum * factor;
        });
    } else {
        ctx.stream()->parallel_for(sycl::range<2>(n_blocks, d), [=](sycl::id<2> id) {
            const size_t b = id[0];
            const size_t c = id[1];
            float sum = 0.0f;
            for (int k = 0; k < r; ++k) {
                const float * row = (const float *) (keys + (size_t) idx[b * r + k] * row_stride);
                sum += row[c];
            }
            out[b * d + c] = sum * factor;
        });
    }
}

// Diagnostic per-op timer (GGML_SYCL_PROFILE_OPS=1). It waits on the queue after every op, which
// changes scheduling and must never be used for throughput comparisons.
struct ggml_sycl_profile_node {
    ggml_backend_sycl_context * ctx;
    ggml_op op;
    std::map<int, std::pair<int64_t, int>> * totals;
    std::map<std::string, std::pair<int64_t, int>> * mm_totals = nullptr;
    std::string mm_key;
    int64_t start_us;
    ggml_sycl_profile_node(ggml_backend_sycl_context * ctx, const ggml_tensor * node, std::map<int, std::pair<int64_t, int>> * totals,
                           std::map<std::string, std::pair<int64_t, int>> * mm)
        : ctx(ctx), op(node->op), totals(totals), start_us(totals ? ggml_time_us() : 0) {
        if (totals && mm && (node->op == GGML_OP_MUL_MAT || node->op == GGML_OP_MUL_MAT_ID)) {
            // weight family: tensor name with every digit run collapsed to N (blk.12.attn_q.weight -> blk.N.attn_q.weight)
            const ggml_tensor * w = node->src[0];
            const ggml_tensor * x = node->src[1];
            std::string name;
            bool in_digits = false;
            for (const char * c = w->name; *c; c++) {
                if (*c >= '0' && *c <= '9') { if (!in_digits) { name += 'N'; } in_digits = true; }
                else { name += *c; in_digits = false; }
            }
            char buf[320];
            snprintf(buf, sizeof(buf), "%s %s w=%s[%lld,%lld,%lld] x=%s[%lld,%lld,%lld]%s", ggml_op_name(node->op), name.c_str(),
                     ggml_type_name(w->type), (long long) w->ne[0], (long long) w->ne[1], (long long) w->ne[2],
                     ggml_type_name(x->type), (long long) x->ne[0], (long long) x->ne[1], (long long) x->ne[2],
                     ggml_is_contiguous(x) ? "" : " x-noncontig");
            mm_key = buf;
            mm_totals = mm;
        }
    }
    ~ggml_sycl_profile_node() {
        if (!totals) {
            return;
        }
        { ggml_sycl_pipe_wait_scope ws_; ctx->stream()->wait(); }
        const int64_t dt = ggml_time_us() - start_us;
        auto & entry = (*totals)[(int) op];
        entry.first += dt;
        entry.second++;
        if (mm_totals) {
            auto & e = (*mm_totals)[mm_key];
            e.first += dt;
            e.second++;
        }
    }
};

#include <chrono>
#include <mutex>
#include <thread>

// ---- idle heartbeat (GGML_SYCL_IDLE_HEARTBEAT_SEC, default off) ---------------------------------------------
// Keeps the driver from treating a loaded-but-idle process as inactive and demoting its VRAM to system RAM.
static std::atomic<int64_t> g_sycl_heartbeat_last_us{0};
static std::mutex           g_sycl_heartbeat_mutex;
static std::vector<std::pair<sycl::queue *, uint32_t *>> g_sycl_heartbeat_queues;

static void ggml_sycl_idle_heartbeat_note(sycl::queue * q) {
    static const int period_s = ggml_sycl_get_env("GGML_SYCL_IDLE_HEARTBEAT_SEC", 0);
    if (period_s <= 0) {
        return;
    }
    g_sycl_heartbeat_last_us.store(ggml_time_us());
    {
        std::lock_guard<std::mutex> lock(g_sycl_heartbeat_mutex);
        bool known = false;
        for (const auto & e : g_sycl_heartbeat_queues) {
            known = known || e.first == q;
        }
        if (!known) {
            uint32_t * scratch = sycl::malloc_device<uint32_t>(64, *q);
            if (scratch != nullptr) {
                g_sycl_heartbeat_queues.emplace_back(q, scratch);
            }
        }
    }
    static std::once_flag once;
    std::call_once(once, [] {
        std::thread([] {
            const int64_t period_us = (int64_t) period_s * 1000000;
            for (;;) {
                std::this_thread::sleep_for(std::chrono::seconds(period_s));
                if (ggml_time_us() - g_sycl_heartbeat_last_us.load() < period_us) {
                    continue;
                }
                std::vector<std::pair<sycl::queue *, uint32_t *>> queues;
                {
                    std::lock_guard<std::mutex> lock(g_sycl_heartbeat_mutex);
                    queues = g_sycl_heartbeat_queues;
                }
                for (const auto & e : queues) {
                    try {
                        sycl_pipe::submit_scope ss_;
                        uint32_t * scratch = e.second;
                        e.first->parallel_for(sycl::range<1>(64), [=](sycl::id<1> i) { scratch[i[0]] += 1u; }).wait();
                    } catch (const sycl::exception &) {
                        // best effort
                    }
                }
            }
        }).detach();
    });
}

static void ggml_backend_sycl_graph_compute_impl(ggml_backend_sycl_context * sycl_ctx, ggml_cgraph * cgraph) {
    ggml_sycl_set_main_device(sycl_ctx->device);
    ggml_sycl_idle_heartbeat_note(sycl_ctx->stream());
    g_ggml_sycl_graph_serial[sycl_ctx->device].fetch_add(1);

    static const bool profile_nodes = ggml_sycl_get_env("GGML_SYCL_PROFILE_OPS", 0) != 0;
    static std::atomic<uint64_t> profile_graph_sequence{0};
    const uint64_t graph_sequence = profile_nodes ? profile_graph_sequence++ : 0;
    std::map<int, std::pair<int64_t, int>> profile_totals;
    std::map<std::string, std::pair<int64_t, int>> profile_mm;
    const int64_t profile_t0 = profile_nodes ? ggml_time_us() : 0;

    // GGML_SYCL_DUMP_GRAPH=N: print the node sequence of the first N decode-sized graphs (<= 8 tokens) with the indices
    // that actually started a kernel launch (everything else was absorbed by a fusion).
    static const int dump_graphs_max = ggml_sycl_get_env("GGML_SYCL_DUMP_GRAPH", 0);
    static std::atomic<int> dump_graphs_done{0};
    bool dump_this = false;
    int dump_gid = -1;
    std::string dump_buf;
    std::vector<int> dump_exec;
    if (dump_graphs_max > 0 && cgraph->n_nodes >= 100 && dump_graphs_done.load() < dump_graphs_max) {
        int64_t ntok = -1;
        for (int i = 0; i < cgraph->n_nodes && ntok < 0; i++) {
            const ggml_tensor * t = cgraph->nodes[i];
            if (t->op == GGML_OP_MUL_MAT_ID && t->src[1]) { ntok = t->src[1]->ne[2]; }
        }
        if (ntok >= 1 && ntok <= 8) {
            dump_this = true;
            dump_gid = dump_graphs_done.fetch_add(1);
            char hb[256];
            snprintf(hb, sizeof hb, "[GD] %d graph device %d nodes %d ntok %lld\n", dump_gid, sycl_ctx->device, cgraph->n_nodes, (long long) ntok);
            dump_buf += hb;
            for (int i = 0; i < cgraph->n_nodes; i++) {
                const ggml_tensor * t = cgraph->nodes[i];
                if (ggml_sycl_is_view_or_noop(t) || (t->flags & GGML_TENSOR_FLAG_COMPUTE) == 0) { continue; }
                char nb[512];
                snprintf(nb, sizeof nb, "[GN] %d %d %s %s %s %lld,%lld,%lld,%lld t=%s s0=%s s1=%s\n", dump_gid, i, ggml_op_name(t->op),
                        t->op == GGML_OP_UNARY ? ggml_unary_op_name(ggml_get_unary_op(t)) : "-", t->name,
                        (long long) t->ne[0], (long long) t->ne[1], (long long) t->ne[2], (long long) t->ne[3],
                        ggml_type_name(t->type), t->src[0] ? t->src[0]->name : "-", t->src[1] ? t->src[1]->name : "-");
                dump_buf += nb;
            }
        }
    }

    for (int i = 0; i < cgraph->n_nodes; i++) {
        ggml_tensor * node = cgraph->nodes[i];
        if (ggml_sycl_is_view_or_noop(node)) {
            continue;
        }
        if (dump_this && (node->flags & GGML_TENSOR_FLAG_COMPUTE) != 0) { dump_exec.push_back(i); }
        if ((node->flags & GGML_TENSOR_FLAG_COMPUTE) == 0) {
            continue;
        }
        sycl_pipe::g_cur_node[sycl_ctx->device].store(i, std::memory_order_relaxed);
        sycl_pipe::g_cur_op[sycl_ctx->device].store((int) node->op, std::memory_order_relaxed);
        ggml_sycl_profile_node profile_node(sycl_ctx, node, profile_nodes ? &profile_totals : nullptr, &profile_mm);

        // debug: what the routed-expert matmuls consume (ids, weights) for single-token graphs
        static const int route_dump_n = ggml_sycl_get_env("GGML_SYCL_DSV4_ROUTE_DUMP", 0);
        static const bool route_dump = route_dump_n > 0;
        static std::atomic<int> route_dump_count{0};
        if (route_dump && route_dump_count.load() < route_dump_n && node->src[1] != nullptr) {
            const bool is_gate = node->op == GGML_OP_MUL_MAT_ID && strncmp(node->name, "ffn_moe_gate-", 13) == 0 &&
                                 node->src[2] != nullptr && node->src[2]->ne[1] == 1;
            const bool is_wgt  = node->op == GGML_OP_MUL && strncmp(node->name, "ffn_moe_weighted-", 17) == 0 &&
                                 node->src[1]->ne[2] == 1;
            if (is_gate || is_wgt) {
                const ggml_tensor * t = is_gate ? node->src[2] : node->src[1];
                const int n = (int) std::min<int64_t>(6, is_gate ? t->ne[0] : t->ne[1]);
                std::vector<char> host(64);
                { ggml_sycl_pipe_wait_scope ws_; sycl_ctx->stream()->wait(); }
                if (is_gate) {
                    sycl_ctx->stream()->memcpy(host.data(), t->data, n * sizeof(int32_t)).wait();
                    const int32_t * v = (const int32_t *) host.data();
                    fprintf(stderr, "[DUMP] %s ids:", node->name);
                    for (int k = 0; k < n; ++k) { fprintf(stderr, " %d", v[k]); }
                } else {
                    sycl_ctx->stream()->memcpy(host.data(), t->data, n * sizeof(float)).wait();
                    const float * v = (const float *) host.data();
                    fprintf(stderr, "[DUMP] %s weights:", node->name);
                    for (int k = 0; k < n; ++k) { fprintf(stderr, " %.7f", v[k]); }
                }
                fprintf(stderr, "\n");
                route_dump_count++;
            }
        }

        if (node->op == GGML_OP_SCALE && ggml_sycl_can_fuse_scale_silu(cgraph, i)) {
            ggml_sycl_scale_silu(*sycl_ctx, node, cgraph->nodes[i + 1]);
            i += 1;
            continue;
        }

        if (node->op == GGML_OP_SCALE && ggml_sycl_can_fuse_hc_post_gate(cgraph, i)) {
            ggml_sycl_op_dsv4_hc_post(*sycl_ctx, cgraph->nodes[i + 3], node->src[0],
                                    ggml_get_op_params_f32(node, 0), ggml_get_op_params_f32(cgraph->nodes[i + 2], 0));
            i += 3;
            continue;
        }

        if (node->op == GGML_OP_GET_ROWS) {
            ggml_sycl_qsa_pool_match qsa_pool;
            if (ggml_sycl_match_qsa_pool(cgraph, i, qsa_pool)) {
                ggml_sycl_qsa_pool(*sycl_ctx, qsa_pool);
                i = qsa_pool.last;
                continue;
            }
        }

        if (node->op == GGML_OP_MUL) {
            ggml_sycl_hc_gate_match hc_gate;
            if (ggml_sycl_match_hc_gate(cgraph, i, hc_gate)) {
                ggml_sycl_hc_gate(*sycl_ctx, hc_gate);
                i = hc_gate.last;
                continue;
            }
        }

        if (node->op == GGML_OP_SUM_ROWS) {
            ggml_sycl_moe_norm_match moe_norm;
            if (ggml_sycl_match_moe_norm(cgraph, i, moe_norm)) {
                ggml_sycl_moe_norm(*sycl_ctx, moe_norm);
                i = moe_norm.last;
                continue;
            }
        }

        if (node->op == GGML_OP_CONCAT) {
            ggml_sycl_conv_snap_match snap;
            if (ggml_sycl_match_conv_snap(cgraph, i, snap)) {
                if (!ggml_sycl_compute_forward(*sycl_ctx, node)) {
                    GGML_ABORT("concat failed before conv snapshot fusion");
                }
                ggml_sycl_conv_snap(*sycl_ctx, snap);
                i = snap.last;
                continue;
            }
        }
        if (node->op == GGML_OP_ADD) {
            if (ggml_sycl_match_ssm_gate(cgraph, i)) {
                ggml_sycl_ssm_gate(*sycl_ctx, cgraph, i);
                i += 2;
                continue;
            }
            ggml_sycl_moe_sum_match moe_sum;
            if (ggml_sycl_match_moe_sum(cgraph, i, moe_sum)) {
                ggml_sycl_moe_sum(*sycl_ctx, moe_sum);
                i += moe_sum.n_adds - 1;
                continue;
            }
        }
        if (node->op == GGML_OP_UNARY && ggml_sycl_match_shexp_gate(cgraph, i)) {
            ggml_sycl_shexp_gate(*sycl_ctx, cgraph, i);
            i += 2;
            continue;
        }
        const int nodes_to_skip = ggml_sycl_fuse(*sycl_ctx, cgraph, i);
        if (nodes_to_skip != 0) {
            i += nodes_to_skip;
            continue;
        }
#ifndef NDEBUG
        assert(node->buffer->buft == ggml_backend_sycl_buffer_type(sycl_ctx->device));
        for (int j = 0; j < GGML_MAX_SRC; j++) {
            if (node->src[j] != nullptr) {
                assert(node->src[j]->buffer->buft == ggml_backend_sycl_buffer_type(sycl_ctx->device));
            }
        }
#endif
        // gated_delta_net -> cpy: scatter recurrent-state snapshots into the cache
        if (node->op == GGML_OP_GATED_DELTA_NET) {
            ggml_sycl_gated_delta_net_fused_cache fused_state_cpy;
            const int gdn_nodes_to_skip = ggml_sycl_try_gdn_cache_fusion(cgraph, i, fused_state_cpy);
            if (gdn_nodes_to_skip > 0) {
                ggml_sycl_op_gated_delta_net_fused_cache(*sycl_ctx, node, fused_state_cpy);
                i += gdn_nodes_to_skip;
                continue;
            }
        }
        if (node->op == GGML_OP_ADD && ggml_sycl_can_fuse_add_rms_norm_mul(cgraph, i)) {
            ggml_sycl_op_add_rms_norm_mul_fused(*sycl_ctx, node, cgraph->nodes[i + 1], cgraph->nodes[i + 2]);
            i += 2;
            continue;
        }
        if (node->op == GGML_OP_RMS_NORM &&
            ggml_sycl_can_fuse(cgraph, i, { GGML_OP_RMS_NORM, GGML_OP_MUL, GGML_OP_ADD }, {})) {
            ggml_sycl_op_rms_norm_fused_add(*sycl_ctx, node, cgraph->nodes[i + 1], cgraph->nodes[i + 2]);
            i += 2;
            continue;
        }
        if (node->op == GGML_OP_RMS_NORM &&
            ggml_sycl_can_fuse(cgraph, i, { GGML_OP_RMS_NORM, GGML_OP_MUL }, {})) {
            ggml_sycl_op_rms_norm_fused(*sycl_ctx, node, cgraph->nodes[i + 1]);
            i++;
            continue;
        }
        // qwen35 GDN l2 norms are emitted as rms_norm + scalar scale (models.h
        // build_gdn_l2_norm), which the rms_norm+mul fusion above cannot match
        if (node->op == GGML_OP_RMS_NORM &&
            ggml_sycl_can_fuse(cgraph, i, { GGML_OP_RMS_NORM, GGML_OP_SCALE }, {})) {
            ggml_sycl_op_rms_norm_scale_fused(*sycl_ctx, node, cgraph->nodes[i + 1]);
            i++;
            continue;
        }
        if (node->op == GGML_OP_ADD &&
            ggml_sycl_can_fuse(cgraph, i, { GGML_OP_ADD, GGML_OP_ADD }, {})) {
            ggml_sycl_op_add_add_fused(*sycl_ctx, node, cgraph->nodes[i + 1]);
            i++;
            continue;
        }
        if (node->op == GGML_OP_UNARY &&
            ggml_sycl_can_fuse(cgraph, i, { GGML_OP_UNARY, GGML_OP_MUL }, { ggml_get_unary_op(node) })) {
            ggml_sycl_op_unary_mul_fused(*sycl_ctx, node, cgraph->nodes[i + 1]);
            i++;
            continue;
        }
        // ADD(bias) + UNARY + MUL(scale) with both broadcast over dim 0, the form the branch
        // above cannot take; ggml_get_unary_op() asserts, so check the op first.
        if (node->op == GGML_OP_ADD && i + 2 < cgraph->n_nodes &&
            cgraph->nodes[i + 1]->op == GGML_OP_UNARY &&
            ggml_sycl_can_fuse(cgraph, i, { GGML_OP_ADD, GGML_OP_UNARY, GGML_OP_MUL },
                               { ggml_get_unary_op(cgraph->nodes[i + 1]) })) {
            ggml_sycl_op_add_unary_mul_fused(*sycl_ctx, node, cgraph->nodes[i + 1], cgraph->nodes[i + 2]);
            i += 2;
            continue;
        }

        // Batch consecutive independent same-shape F32 L2_NORM siblings (the GDN q/k
        // norms) into one launch; sources are strided views of the fused qkv buffer, so
        // the scan skips the interleaved view nodes instead of breaking on them.
        if (node->op == GGML_OP_L2_NORM) {
            const int l2_batch_skip = ggml_sycl_l2_norm_batch_fused(*sycl_ctx, cgraph, i);
            if (l2_batch_skip > 0) {
                i += l2_batch_skip;
                continue;
            }
        }

        if (node->op == GGML_OP_SSM_CONV &&
            ggml_sycl_can_fuse(cgraph, i, { GGML_OP_SSM_CONV, GGML_OP_ADD, GGML_OP_UNARY }, { GGML_UNARY_OP_SILU })) {
            ggml_sycl_ssm_conv_fused(*sycl_ctx, node, cgraph->nodes[i + 1], cgraph->nodes[i + 2]);
            i += 2;
            continue;
        }

        if (node->op == GGML_OP_SSM_CONV &&
            ggml_sycl_can_fuse(cgraph, i, { GGML_OP_SSM_CONV, GGML_OP_UNARY }, { GGML_UNARY_OP_SILU })) {
            ggml_sycl_ssm_conv_fused(*sycl_ctx, node, nullptr, cgraph->nodes[i + 1]);
            i++;
            continue;
        }

        if (node->op == GGML_OP_MUL_MAT && ggml_sycl_mul_mat_glu_fused(*sycl_ctx, cgraph, i)) {
            i += 2;
            continue;
        }

        bool ok = ggml_sycl_compute_forward(*sycl_ctx, node);
        if (!ok) {
            GGML_LOG_ERROR("%s: error: op not supported %s (%s)\n", __func__, node->name, ggml_op_name(node->op));
        }
        GGML_ASSERT(ok);
    }
    if (dump_this) {
        dump_buf += "[GE] " + std::to_string(dump_gid);
        for (int e : dump_exec) { dump_buf += " " + std::to_string(e); }
        dump_buf += "\n";
        fputs(dump_buf.c_str(), stderr);
        fflush(stderr);
    }
    if (profile_nodes) {
        int64_t sum = 0; int n = 0;
        for (const auto & [op, value] : profile_totals) {
            fprintf(stderr, "[SYCL-OP-PROFILE] graph=%llu device=%d op=%s count=%d elapsed_us=%lld\n",
                    (unsigned long long) graph_sequence, sycl_ctx->device, ggml_op_name((ggml_op) op), value.second,
                    (long long) value.first);
            sum += value.first; n += value.second;
        }
        for (const auto & [key, value] : profile_mm) {
            fprintf(stderr, "[SYCL-MM-PROFILE] graph=%llu device=%d count=%d elapsed_us=%lld key=%s\n",
                    (unsigned long long) graph_sequence, sycl_ctx->device, value.second, (long long) value.first, key.c_str());
        }
        fprintf(stderr, "[SYCL-OP-PROFILE] graph=%llu device=%d TOTAL ops=%d op_us=%lld wall_us=%lld\n",
                (unsigned long long) graph_sequence, sycl_ctx->device, n, (long long) sum,
                (long long) (ggml_time_us() - profile_t0));
    }
}

#ifdef GGML_SYCL_GRAPH
static bool ggml_sycl_graph_uses_split_buffer(const ggml_cgraph * cgraph) {
    for (int i = 0; i < cgraph->n_nodes; i++) {
        const ggml_tensor * node = cgraph->nodes[i];
        for (int j = 0; j < GGML_MAX_SRC; j++) {
            const ggml_tensor * src = node->src[j];
            if (src && src->buffer && ggml_backend_buffer_is_sycl_split(src->buffer)) {
                return true;
            }
        }
    }
    return false;
}

static bool check_graph_compatibility(const ggml_backend_sycl_context * sycl_ctx, ggml_cgraph * cgraph) {
    // A sycl_ex::command_graph object can only be created for a single device. Each backend gets the
    // part of the graph that runs on its own device, so only row-split buffers (one op spanning
    // devices) rule graphs out.
    if (ggml_sycl_info().device_count > 1 && ggml_sycl_graph_uses_split_buffer(cgraph)) {
        GGML_LOG_INFO("%s: disabling SYCL graphs due to row-split buffers across devices\n", __func__);
        return false;
    }

    for (int i = 0; i < cgraph->n_nodes; i++) {
        const ggml_op node_op = cgraph->nodes[i]->op;
        switch (node_op) {
            default:
                break;
            case GGML_OP_CONCAT: {
                // only the contiguous dim==3 path does a blocking memcpy+wait; the other paths are kernels
                const ggml_tensor * node = cgraph->nodes[i];
                const int32_t dim = ((const int32_t *) node->op_params)[0];
                if (dim == 3 && ggml_is_contiguous(node->src[0]) && ggml_is_contiguous(node->src[1])) {
                    GGML_LOG_INFO("%s: disabling SYCL graphs due to unsupported node type %s\n", __func__,
                                  ggml_op_name(node_op));
                    return false;
                }
                break;
            }
            case GGML_OP_MUL_MAT_ID: {
                // The generic path copies the expert ids to the host and waits. The fused single-token GEMV
                // runs entirely on the device, so it is graph-safe when it is certain to be taken: shapes
                // qualify, the type has a kernel, the src1 pool allocation can be async, and no lazy weight
                // reorder (temp alloc + copies) is still pending. The first decode after load therefore runs
                // without a graph and performs the reorder.
                const ggml_tensor * node = cgraph->nodes[i];
                const ggml_tensor * src0 = node->src[0];
                const bool ok = g_ggml_sycl_use_async_mem_op &&
                                !ggml_backend_buffer_is_sycl_split(src0->buffer) &&
                                ggml_sycl_mmid_fused_applicable(src0, node->src[1], node->src[2], node) &&
                                ggml_sycl_mul_mat_vec_q_id_supports_type(src0->type) &&
                                !ggml_sycl_mmid_reorder_pending(sycl_ctx, src0);
                if (!ok) {
                    GGML_LOG_INFO("%s: disabling SYCL graphs due to unsupported node type %s\n", __func__,
                                  ggml_op_name(node_op));
                    return false;
                }
                break;
            }
            case GGML_OP_MUL_MAT:
                // We cannot use graphs with ggml_sycl_mul_mat() when SYCL async memory allocation extensions are not available,
                // as SYCL malloc / free and host wait calls are not supported when recording to a graph which are all present
                // in reordering.
                if (!g_ggml_sycl_use_async_mem_op) {
                    GGML_LOG_INFO(
                        "%s: disabling SYCL graphs due to unsupported node type when using a compiler without the "
                        "oneAPI async memory allocation extension "
                        "%s\n",
                        __func__, ggml_op_name(node_op));
                    return false;
                }
        }
    }
    return true;
}
#endif

static ggml_status ggml_backend_sycl_graph_compute(ggml_backend_t backend, ggml_cgraph * cgraph) {
    sycl_pipe::submit_scope pipe_ss_;
    auto * sycl_ctx = static_cast<ggml_backend_sycl_context *>(backend->context);

    if (sycl_pipe::enabled() && sycl_pipe::t_worker_device < 0) {
        if (sycl_pipe::trace() && cgraph->n_nodes <= 16 && sycl_pipe::pending(sycl_ctx->device)) {
            for (int i = 0; i < cgraph->n_nodes; ++i) {
                const ggml_tensor * n = cgraph->nodes[i];
                fprintf(stderr, "[SYCL-PIPE] small graph device=%d node %d/%d %s '%s' [%lld,%lld,%lld,%lld] src0='%s' src1='%s'" "\n", sycl_ctx->device, i, cgraph->n_nodes,
                        ggml_op_name(n->op), n->name, (long long) n->ne[0], (long long) n->ne[1], (long long) n->ne[2], (long long) n->ne[3],
                        n->src[0] ? n->src[0]->name : "", n->src[1] ? n->src[1]->name : "");
            }
        }
        sycl_pipe::held_forget_device(sycl_ctx->device);
        static const bool follow = ggml_sycl_get_env("GGML_SYCL_PIPELINE_FOLLOW", 1) != 0;
        if (sycl_pipe::is_batch_graph(cgraph) || (follow && sycl_pipe::pending(sycl_ctx->device))) {
            sycl_pipe::ensure_started();
            const bool counted = sycl_pipe::is_batch_graph(cgraph);
            if (counted) {
                sycl_pipe::graph_admit(sycl_ctx->device);
            }
            auto copy = std::make_shared<sycl_pipe::graph_copy>(cgraph);
            if (sycl_pipe::trace()) {
                fprintf(stderr, "[SYCL-PIPE] queue graph device=%d nodes=%d t=%lld\n", sycl_ctx->device, cgraph->n_nodes, (long long) ggml_time_us());
            }
            sycl_pipe::enqueue(sycl_ctx->device, [sycl_ctx, copy, counted] {
                const int64_t tq = ggml_time_us();
                sycl_pipe::t_lock_wait_us = 0; sycl_pipe::t_unlocked_us = 0; sycl_pipe::t_releases = 0;
                sycl_pipe::submit_scope ss_;
                const int64_t t0 = ggml_time_us();
                ggml_backend_sycl_graph_compute_impl(sycl_ctx, &copy->g);
                { ggml_sycl_pipe_wait_scope ws_; SYCL_CHECK(CHECK_TRY_ERROR(sycl_ctx->stream()->wait())); }
                if (sycl_pipe::trace()) {
                    fprintf(stderr, "[SYCL-PIPE] done  graph device=%d start=%lld end=%lld lockwait_ms=%lld unlocked_ms=%lld releases=%lld\n", sycl_ctx->device, (long long) tq, (long long) ggml_time_us(),
                            (long long) (sycl_pipe::t_lock_wait_us / 1000), (long long) (sycl_pipe::t_unlocked_us / 1000), (long long) sycl_pipe::t_releases);
                }
                if (counted) {
                    sycl_pipe::graph_finished(sycl_ctx->device);
                }
            }, "graph");
            return GGML_STATUS_SUCCESS;
        }
        // a decode-sized graph runs inline, after whatever is still queued for this card
        sycl_pipe::drain(sycl_ctx->device);
    }

#ifdef GGML_SYCL_GRAPH
    bool use_sycl_graph = false;
    if (g_ggml_sycl_enable_graph) {
        use_sycl_graph = check_graph_compatibility(sycl_ctx, cgraph);
    }
    if (use_sycl_graph) {
        const bool graph_support = dpct::get_device(sycl_ctx->device).has(sycl::aspect::ext_oneapi_limited_graph);
        if (!graph_support) {
            GGML_SYCL_DEBUG("[SYCL-GRAPH] can not use graphs on device:%d\n", sycl_ctx->device);
            ggml_backend_sycl_graph_compute_impl(sycl_ctx, cgraph);
            return GGML_STATUS_SUCCESS;
        }

        sycl_ex::command_graph model_sycl_graph(*(sycl_ctx->stream()), {sycl_ex::property::graph::assume_buffer_outlives_graph{}});

        model_sycl_graph.begin_recording(*(sycl_ctx->stream()));
        ggml_backend_sycl_graph_compute_impl(sycl_ctx, cgraph);
        model_sycl_graph.end_recording();

        const bool graph_update_support = dpct::get_device(sycl_ctx->device).has(sycl::aspect::ext_oneapi_graph);
        if (!sycl_ctx->exec_graph || !graph_update_support) {
            auto exec_graph = graph_update_support ? model_sycl_graph.finalize(sycl_ex::property::graph::updatable{}) :
                                                     model_sycl_graph.finalize();
            sycl_ctx->exec_graph = std::make_unique<
                sycl_ex::command_graph<sycl_ex::graph_state::executable>>(exec_graph);
        } else {
            try {
                sycl_ctx->exec_graph->update(model_sycl_graph);
                GGML_SYCL_DEBUG("[SYCL-GRAPH] update success\n");
            } catch (sycl::exception const & e) {
                GGML_SYCL_DEBUG("[SYCL-GRAPH] Exception when updating graph, %s\n", e.what());
                auto exec_graph = model_sycl_graph.finalize({sycl_ex::property::graph::updatable{}});
                sycl_ctx->exec_graph = std::make_unique<
                    sycl_ex::command_graph<sycl_ex::graph_state::executable>>(exec_graph);
            }
        }

        sycl_ctx->stream()->ext_oneapi_graph(*(sycl_ctx->exec_graph));
    } else
#endif
    {
        ggml_backend_sycl_graph_compute_impl(sycl_ctx, cgraph);
    }
    return GGML_STATUS_SUCCESS;
}

static void ggml_backend_sycl_event_record(ggml_backend_t backend, ggml_backend_event_t event)
try
{
    sycl_pipe::submit_scope pipe_ss_;
    ggml_backend_sycl_context *sycl_ctx =
        (ggml_backend_sycl_context *)backend->context;

    sycl::event *sycl_event = static_cast<sycl::event *>(event->context);

    const queue_ptr &stream = sycl_ctx->stream(sycl_ctx->device, 0);
    // Record the current state of the queue
    SYCL_CHECK(CHECK_TRY_ERROR(*sycl_event = stream->ext_oneapi_submit_barrier()));
    sycl_pipe::event_record(sycl_ctx->device, event);
}
catch (sycl::exception const &exc)
{
    std::cerr << exc.what() << "Exception caught at file:" << __FILE__
              << ", line:" << __LINE__ << std::endl;
    std::exit(1);
}

static void ggml_backend_sycl_event_wait(ggml_backend_t backend, ggml_backend_event_t event) try {
    sycl_pipe::submit_scope pipe_ss_;
    GGML_SYCL_DEBUG("[SYCL] call %s\n", __func__);
    sycl::event* sycl_event = static_cast<sycl::event*>(event->context);

    if (ggml_backend_is_sycl(backend)) {
        sycl_pipe::event_wait(event);
        { ggml_sycl_pipe_wait_scope ws_; SYCL_CHECK(CHECK_TRY_ERROR(sycl_event->wait())); }
    } else
        GGML_ABORT("fatal error");
} catch (sycl::exception const& exc) {
    std::cerr << exc.what() << "Exception caught at file:" << __FILE__
              << ", line:" << __LINE__ << std::endl;
    std::exit(1);
}

static ggml_backend_i ggml_backend_sycl_interface = {
    /* .get_name                = */ ggml_backend_sycl_get_name,
    /* .free                    = */ ggml_backend_sycl_free,
    /* .set_tensor_async        = */ ggml_backend_sycl_set_tensor_async,
    /* .get_tensor_async        = */ ggml_backend_sycl_get_tensor_async,
    /* .set_tensor_2d_async     = */ NULL,
    /* .get_tensor_2d_async     = */ NULL,
    /* .cpy_tensor_async        = */ ggml_backend_sycl_cpy_tensor_async_pipe, // (old: ggml_backend_sycl_cpy_tensor_async)
                                           // // TODO: update for the new
                                           // interface
    /* .synchronize             = */ ggml_backend_sycl_synchronize,
    /* .graph_plan_create       = */ NULL,
    /* .graph_plan_free         = */ NULL,
    /* .graph_plan_update       = */ NULL,
    /* .graph_plan_compute      = */ NULL,
    /* .graph_compute           = */ ggml_backend_sycl_graph_compute,
    /* .event_record            = */ ggml_backend_sycl_event_record,
    /* .event_wait              = */ ggml_backend_sycl_event_wait,
    /* .graph_optimize          = */ NULL,
};

static ggml_guid_t ggml_backend_sycl_guid() {
    static ggml_guid guid = { 0x58, 0x05, 0x13, 0x8f, 0xcd, 0x3a, 0x61, 0x9d, 0xe7, 0xcd, 0x98, 0xa9, 0x03, 0xfd, 0x7c, 0x53 };
    return &guid;
}

bool ggml_backend_is_sycl(ggml_backend_t backend) {
    return backend != NULL && ggml_guid_matches(backend->guid, ggml_backend_sycl_guid());
}

int ggml_backend_sycl_get_device_count(void) {
    return ggml_sycl_info().device_count;
}


// backend device

static const char * ggml_backend_sycl_device_get_name(ggml_backend_dev_t dev) {
    ggml_backend_sycl_device_context * ctx = (ggml_backend_sycl_device_context *)dev->context;
    return ctx->name.c_str();
}

static const char * ggml_backend_sycl_device_get_description(ggml_backend_dev_t dev) {
    ggml_backend_sycl_device_context * ctx = (ggml_backend_sycl_device_context *)dev->context;
    return ctx->description.c_str();
}

static void ggml_backend_sycl_device_get_memory(ggml_backend_dev_t dev, size_t * free, size_t * total) {
    GGML_SYCL_DEBUG("[SYCL] call %s\n", __func__);
    ggml_backend_sycl_device_context * ctx = (ggml_backend_sycl_device_context *) dev->context;
    if (!sycl_get_mem_info(ctx->device, free, total)) {
        GGML_ABORT("[%s] failed to get device memory size", __func__);
    }
    GGML_SYCL_DEBUG("[SYCL] call %s total %zu free %zu\n", __func__, *total, *free);
    ggml_sycl_memtrace_report_device("device memory query (dev)", ctx->device, *free, *total);
}

static enum ggml_backend_dev_type ggml_backend_sycl_device_get_type(ggml_backend_dev_t dev) {
    ggml_backend_sycl_device_context * ctx = (ggml_backend_sycl_device_context *)dev->context;
    const sycl_device_info & info = ggml_sycl_info().devices[ctx->device];
    if (info.l0_device_type_valid && !info.l0_discrete_gpu) {
        return GGML_BACKEND_DEVICE_TYPE_IGPU;
    }
    return GGML_BACKEND_DEVICE_TYPE_GPU;
}

static void ggml_backend_sycl_device_get_props(ggml_backend_dev_t dev, ggml_backend_dev_props * props) {
    props->name        = ggml_backend_sycl_device_get_name(dev);
    props->description = ggml_backend_sycl_device_get_description(dev);
    props->type        = ggml_backend_sycl_device_get_type(dev);
    ggml_backend_sycl_device_get_memory(dev, &props->memory_free, &props->memory_total);

    bool host_buffer = getenv("GGML_SYCL_NO_PINNED") == nullptr;
#ifdef GGML_SYCL_NO_PEER_COPY
    bool events = false;
#else
    bool events = true;
#endif

    props->caps = {
        /* .async                 = */ true,
        /* .host_buffer           = */ host_buffer,
        /* .buffer_from_host_ptr  = */ false,
        /* .events                = */ events,
        /* .mmap_support          = */ true,
    };
}

static ggml_backend_t ggml_backend_sycl_device_init(ggml_backend_dev_t dev, const char * params) {
    GGML_UNUSED(params);
    ggml_backend_sycl_device_context * ctx = (ggml_backend_sycl_device_context *)dev->context;
    return ggml_backend_sycl_init(ctx->device);
}

static ggml_backend_buffer_type_t ggml_backend_sycl_device_get_buffer_type(ggml_backend_dev_t dev) {
    ggml_backend_sycl_device_context * ctx = (ggml_backend_sycl_device_context *)dev->context;
    return ggml_backend_sycl_buffer_type(ctx->device);
}

static ggml_backend_buffer_type_t ggml_backend_sycl_device_get_host_buffer_type(ggml_backend_dev_t dev) {
    ggml_backend_sycl_device_context * ctx = (ggml_backend_sycl_device_context *) dev->context;
    return ggml_backend_sycl_host_buffer_type_for_device(ctx->device);
}

static ggml_backend_buffer_t ggml_backend_sycl_device_buffer_from_host_ptr(ggml_backend_dev_t dev, void * ptr, size_t size, size_t max_tensor_size) {
    GGML_UNUSED(dev);
    GGML_UNUSED(ptr);
    GGML_UNUSED(size);
    GGML_UNUSED(max_tensor_size);
    return nullptr;
}

static bool do_ggml_backend_sycl_device_supports_op(ggml_backend_dev_t dev, const ggml_tensor * op) {
    ggml_backend_sycl_device_context *sycl_ctx =
        (ggml_backend_sycl_device_context *)dev->context;
    int device = sycl_ctx->device;
    switch (op->op) {
        case GGML_OP_CONV_TRANSPOSE_1D:
            {
                ggml_type src0_type = op->src[0]->type;
                ggml_type src1_type = op->src[1]->type;
                if (src0_type == GGML_TYPE_F32 && src1_type == GGML_TYPE_F32) {
                    return true;
                }
                return false;
            }
        case GGML_OP_CONV_2D:
            // WHCN: contiguous [KW,KH,IC,OC] kernel and [W,H,C,N] input
            // CWHN: channel-fastest input (C fastest) and kernel with OC fastest
            return (ggml_is_contiguous(op->src[0]) && ggml_is_contiguous(op->src[1])) ||
                   (op->src[1]->nb[2] == ggml_type_size(op->src[1]->type) &&
                    op->src[0]->nb[3] == ggml_type_size(op->src[0]->type));
        case GGML_OP_CONV_2D_DW:
        case GGML_OP_CONV_TRANSPOSE_2D:
            return true;
        case GGML_OP_UNARY:
            switch (ggml_get_unary_op(op)) {
                case GGML_UNARY_OP_SGN:
                case GGML_UNARY_OP_ABS:
                case GGML_UNARY_OP_NEG:
                case GGML_UNARY_OP_STEP:
                case GGML_UNARY_OP_RELU:
                case GGML_UNARY_OP_HARDSIGMOID:
                case GGML_UNARY_OP_TANH:
                case GGML_UNARY_OP_GELU:
                case GGML_UNARY_OP_SILU:
                case GGML_UNARY_OP_SIGMOID:
                case GGML_UNARY_OP_HARDSWISH:
                case GGML_UNARY_OP_GELU_QUICK:
                case GGML_UNARY_OP_GELU_ERF:
                case GGML_UNARY_OP_EXP:
                case GGML_UNARY_OP_EXPM1:
                case GGML_UNARY_OP_SOFTPLUS:
                case GGML_UNARY_OP_ELU:
                case GGML_UNARY_OP_XIELU:
                case GGML_UNARY_OP_CEIL:
                    return true;
                case GGML_UNARY_OP_FLOOR:
                case GGML_UNARY_OP_ROUND:
                case GGML_UNARY_OP_TRUNC:
                    return true;
                default:
                    return false;
            }
        case GGML_OP_GLU:
            switch (ggml_get_glu_op(op)) {
                case GGML_GLU_OP_REGLU:
                case GGML_GLU_OP_GEGLU:
                case GGML_GLU_OP_SWIGLU:
                case GGML_GLU_OP_GEGLU_ERF:
                case GGML_GLU_OP_GEGLU_QUICK:
                case GGML_GLU_OP_SWIGLU_CLAMP:
                    // the fused GLU kernels only have f32/f16 instantiations
                    return ggml_is_contiguous_1(op->src[0]) &&
                           (op->type == GGML_TYPE_F32 || op->type == GGML_TYPE_F16);
                case GGML_GLU_OP_SWIGLU_OAI:
                    // ggml_sycl_op_swiglu_oai asserts f32 (an f16 request aborted test-backend-ops)
                    return ggml_is_contiguous_1(op->src[0]) && op->type == GGML_TYPE_F32 &&
                           op->src[0]->type == GGML_TYPE_F32;
                default:
                    return false;
            }
        case GGML_OP_MUL_MAT:
        case GGML_OP_MUL_MAT_ID:
            {
                struct ggml_tensor * a = op->src[0];
                struct ggml_tensor * b = op->src[1];

                if (a->ne[3] != b->ne[3]) {
                    return false;
                }

                ggml_type src0_type = op->src[0]->type;

                // TODO: The configuration below needs more work to be supported with oneDNN
                if (ggml_is_permuted(a) && !ggml_is_contiguous(a) &&
                    a->ne[2] > 1 && a->ne[3] > 1 && src0_type == GGML_TYPE_F16) {
                  return false;
                }

                // TODO: This specific configuration can fail with oneDNN and needs more debugging
                if (!ggml_is_permuted(a) && ggml_is_permuted(b) && b->ne[2] > 1 && b->ne[3] > 1 &&
                    a->ne[0] > 128 && a->ne[2] == 1 && src0_type == GGML_TYPE_F16) {
                    return false;
                }

                if (src0_type == GGML_TYPE_TQ2_0 || src0_type == GGML_TYPE_TQ1_0) {
                    return false;
                }

                return true;
            }
        case GGML_OP_OUT_PROD:
            return op->type == GGML_TYPE_F32 &&
                   (op->src[0]->type == GGML_TYPE_F32 ||
                    (op->src[0]->type == GGML_TYPE_Q1_0 && op->src[0]->ne[2] == op->src[1]->ne[2] &&
                     op->src[0]->ne[3] == op->src[1]->ne[3])) &&
                   op->src[1]->type == GGML_TYPE_F32;
        case GGML_OP_GET_ROWS:
            {
                switch (op->src[0]->type) {
                    case GGML_TYPE_I32:
                    case GGML_TYPE_F16:
                    case GGML_TYPE_BF16:
                    case GGML_TYPE_F32:
                    case GGML_TYPE_Q1_0:
                    case GGML_TYPE_MXFP4:
                    case GGML_TYPE_NVFP4:
                    case GGML_TYPE_IQ2_XXS:
                    case GGML_TYPE_IQ2_XS:
                    case GGML_TYPE_IQ2_S:
                    case GGML_TYPE_IQ3_XXS:
                    case GGML_TYPE_IQ1_S:
                    case GGML_TYPE_IQ1_M:
                    case GGML_TYPE_IQ3_S:
                    case GGML_TYPE_IQ4_NL:
                    case GGML_TYPE_IQ4_XS:
                    case GGML_TYPE_Q2_K:
                    case GGML_TYPE_Q3_K:
                    case GGML_TYPE_Q4_0:
                    case GGML_TYPE_Q4_1:
                    case GGML_TYPE_Q4_K:
                    case GGML_TYPE_Q5_0:
                    case GGML_TYPE_Q5_1:
                    case GGML_TYPE_Q5_K:
                    case GGML_TYPE_Q6_K:
                    case GGML_TYPE_Q8_0:
                        return true;
                    default:
                        return false;
                }
            }
        case GGML_OP_GET_ROWS_BACK:
            // return true;
             return op->type == GGML_TYPE_F32 &&
                 (op->src[0]->type == GGML_TYPE_F32 || op->src[0]->type == GGML_TYPE_F16) &&
                 op->src[1]->type == GGML_TYPE_I32 &&
                 op->ne[2] == 1 && op->ne[3] == 1;
         case GGML_OP_SET:
               return (op->type == GGML_TYPE_F32) &&
                      (op->src[0] && op->src[1]) &&
                      (op->src[0]->type == GGML_TYPE_F32) &&
                      (op->src[1]->type == GGML_TYPE_F32);

        case GGML_OP_SET_ROWS:
            {
                if (op->type == GGML_TYPE_TQ2_0 || op->type == GGML_TYPE_TQ1_0) {
                    return false;
                }
                auto res = (op->src[0]->type == GGML_TYPE_F32 || op->src[0]->type == GGML_TYPE_F16 ||
                            op->src[0]->type == GGML_TYPE_BF16) &&
                           (op->src[1]->type == GGML_TYPE_I64 || op->src[1]->type == GGML_TYPE_I32);
                return res;
            }
        case GGML_OP_DSV4_HC_PRE:
            return op->src[0]->type == GGML_TYPE_F32 && op->src[1]->type == GGML_TYPE_F32 &&
                op->type == GGML_TYPE_F32;
        case GGML_OP_DSV4_HC_COMB:
            return op->src[0]->type == GGML_TYPE_F32 && op->src[1]->type == GGML_TYPE_F32 &&
                op->src[2]->type == GGML_TYPE_F32 && op->type == GGML_TYPE_F32;
        case GGML_OP_DSV4_HC_POST:
            return op->src[0]->type == GGML_TYPE_F32 && op->src[1]->type == GGML_TYPE_F32 &&
                op->src[2]->type == GGML_TYPE_F32 &&
                (op->src[3] == nullptr || op->src[3]->type == GGML_TYPE_F32) &&
                op->type == GGML_TYPE_F32;
        case GGML_OP_LIGHTNING_INDEXER:
            return op->src[0]->type == GGML_TYPE_F32 &&
                (op->src[1]->type == GGML_TYPE_F16 || op->src[1]->type == GGML_TYPE_F32 ||
                 op->src[1]->type == GGML_TYPE_BF16 || op->src[1]->type == GGML_TYPE_Q8_0 ||
                 op->src[1]->type == GGML_TYPE_Q5_1 || op->src[1]->type == GGML_TYPE_Q5_0 ||
                 op->src[1]->type == GGML_TYPE_Q4_1 || op->src[1]->type == GGML_TYPE_Q4_0 ||
                 op->src[1]->type == GGML_TYPE_IQ4_NL) &&
                op->src[2]->type == GGML_TYPE_F32 &&
                op->src[3]->type == GGML_TYPE_F16 &&
                op->type == GGML_TYPE_F32 &&
                op->src[0]->ne[0] == WARP_SIZE * 8;
        case GGML_OP_CPY:
            {
                ggml_type src0_type = op->src[0]->type;
                ggml_type src1_type = op->src[1]->type;

                if (src0_type == GGML_TYPE_F16) {
                    if (src1_type == GGML_TYPE_Q2_K ||
                        src1_type == GGML_TYPE_Q3_K ||
                        src1_type == GGML_TYPE_Q4_K ||
                        src1_type == GGML_TYPE_Q5_K ||
                        src1_type == GGML_TYPE_Q6_K ||
                        src1_type == GGML_TYPE_IQ2_XXS ||
                        src1_type == GGML_TYPE_IQ2_XS ||
                        src1_type == GGML_TYPE_IQ2_S ||
                        src1_type == GGML_TYPE_IQ3_XXS ||
                        src1_type == GGML_TYPE_IQ1_S ||
                        src1_type == GGML_TYPE_IQ1_M ||
                        src1_type == GGML_TYPE_IQ3_S ||
                        src1_type == GGML_TYPE_IQ4_XS) {
                        return false;
                    }
                }

                if (src0_type == GGML_TYPE_BF16) {
                    if (src1_type == GGML_TYPE_Q4_0 || //big error in ut
                        src1_type == GGML_TYPE_Q4_1 || //big error in ut
                        src1_type == GGML_TYPE_Q8_0 || //big error in ut
                        src1_type == GGML_TYPE_Q2_K ||
                        src1_type == GGML_TYPE_Q3_K ||
                        src1_type == GGML_TYPE_Q4_K ||
                        src1_type == GGML_TYPE_Q5_K ||
                        src1_type == GGML_TYPE_Q6_K ||
                        src1_type == GGML_TYPE_IQ2_XXS ||
                        src1_type == GGML_TYPE_IQ2_XS ||
                        src1_type == GGML_TYPE_IQ2_S ||
                        src1_type == GGML_TYPE_IQ3_XXS ||
                        src1_type == GGML_TYPE_IQ1_S ||
                        src1_type == GGML_TYPE_IQ1_M ||
                        src1_type == GGML_TYPE_IQ3_S ||
                        src1_type == GGML_TYPE_IQ4_XS) {
                        return false;
                    }
                }

                if (src0_type == GGML_TYPE_F32) {
                    if (src1_type == GGML_TYPE_Q2_K ||
                        src1_type == GGML_TYPE_Q3_K ||
                        src1_type == GGML_TYPE_Q4_K ||
                        src1_type == GGML_TYPE_Q5_K ||
                        src1_type == GGML_TYPE_Q6_K ||
                        src1_type == GGML_TYPE_IQ2_XXS ||
                        src1_type == GGML_TYPE_IQ2_XS ||
                        src1_type == GGML_TYPE_IQ2_S ||
                        src1_type == GGML_TYPE_IQ3_XXS ||
                        src1_type == GGML_TYPE_IQ1_S ||
                        src1_type == GGML_TYPE_IQ1_M ||
                        src1_type == GGML_TYPE_IQ3_S ||
                        src1_type == GGML_TYPE_IQ4_XS) {
                        return false;
                    }
                }

                if (src1_type == GGML_TYPE_F32) {
                    if (src0_type == GGML_TYPE_Q1_0 ||
                        src0_type == GGML_TYPE_NVFP4 ||
                        src0_type == GGML_TYPE_Q2_K ||
                        src0_type == GGML_TYPE_Q3_K ||
                        src0_type == GGML_TYPE_Q4_K ||
                        src0_type == GGML_TYPE_Q5_K ||
                        src0_type == GGML_TYPE_Q6_K ||
                        src0_type == GGML_TYPE_IQ2_XXS ||
                        src0_type == GGML_TYPE_IQ2_XS ||
                        src0_type == GGML_TYPE_IQ2_S ||
                        src0_type == GGML_TYPE_IQ3_XXS ||
                        src0_type == GGML_TYPE_IQ1_S ||
                        src0_type == GGML_TYPE_IQ1_M ||
                        src0_type == GGML_TYPE_IQ3_S ||
                        src0_type == GGML_TYPE_IQ4_NL ||
                        src0_type == GGML_TYPE_IQ4_XS
                    ) {
                        return false;
                    }
                }

                if (src0_type == src1_type) {
                    if (src1_type == GGML_TYPE_IQ2_XXS ||
                        src1_type == GGML_TYPE_IQ2_XS ||
                        src1_type == GGML_TYPE_IQ2_S ||
                        src1_type == GGML_TYPE_IQ3_XXS ||
                        src1_type == GGML_TYPE_IQ3_S ||
                        src1_type == GGML_TYPE_IQ1_S ||
                        src1_type == GGML_TYPE_IQ1_M ||
                        src1_type == GGML_TYPE_TQ2_0 ||
                        src1_type == GGML_TYPE_TQ1_0) {
                        return false;
                    }
                }

                if (src0_type == GGML_TYPE_TQ2_0 || src1_type == GGML_TYPE_TQ2_0 ||
                    src0_type == GGML_TYPE_TQ1_0 || src1_type == GGML_TYPE_TQ1_0) {
                    return false;
                }

                return true;
            }
        case GGML_OP_REPEAT_BACK:
            {
                ggml_type src0_type = op->src[0]->type;
                return src0_type == GGML_TYPE_F32;
            }
        case GGML_OP_CONCAT:
        case GGML_OP_DUP:
        case GGML_OP_ARGMAX:
        case GGML_OP_NONE:
        case GGML_OP_RESHAPE:
        case GGML_OP_VIEW:
        case GGML_OP_PERMUTE:
        case GGML_OP_TRANSPOSE:
        case GGML_OP_ADD:
        case GGML_OP_ADD1:
        case GGML_OP_ADD_ID:
        case GGML_OP_SUB:
        case GGML_OP_COUNT_EQUAL:
        case GGML_OP_MUL:
        case GGML_OP_DIV:
        case GGML_OP_REPEAT:
            return true;
        case GGML_OP_PAD_REFLECT_1D:
            return ggml_is_contiguous(op->src[0]) && op-> type == GGML_TYPE_F32 && op->src[0]->type == GGML_TYPE_F32;
        case GGML_OP_SQR:
        case GGML_OP_SQRT:
        case GGML_OP_SIN:
        case GGML_OP_COS:
        case GGML_OP_CLAMP:
        case GGML_OP_LOG:
        case GGML_OP_NORM:
        case GGML_OP_L2_NORM:
        case GGML_OP_GROUP_NORM:
        case GGML_OP_RMS_NORM:
            return true;
        case GGML_OP_RMS_NORM_BACK:
            return ggml_is_contiguous(op->src[0]);
        case GGML_OP_SCALE:
            // the scale kernel is f32 only
            return op->type == GGML_TYPE_F32 && op->src[0]->type == GGML_TYPE_F32;
        case GGML_OP_CONT:
            return true;
        case GGML_OP_TRI:
            {
                const ggml_tensor * src0 = op->src[0];
                return src0 &&
                       op->type == GGML_TYPE_F32 &&
                       ggml_is_contiguous(src0);
            }
        case GGML_OP_DIAG_MASK_INF:
            return true;
        case GGML_OP_SOFT_MAX:
            return true;
        case GGML_OP_SOFT_MAX_BACK: {
            float max_bias = 0.0f;
            memcpy(&max_bias, (const float *) op->op_params + 1, sizeof(float));
            return max_bias == 0.0f;
        }
        case GGML_OP_ROPE:
        case GGML_OP_ROPE_BACK:
        case GGML_OP_IM2COL:
        case GGML_OP_IM2COL_3D:
        case GGML_OP_UPSCALE:
            return true;
        case GGML_OP_COL2IM_1D:
            return ggml_is_contiguous(op->src[0]) &&
                   (op->type == GGML_TYPE_F32 || op->type == GGML_TYPE_F16
#ifdef GGML_SYCL_HAS_BF16
                    || op->type == GGML_TYPE_BF16
#endif
                   ) &&
                   op->src[0]->type == op->type;
        case GGML_OP_CONV_3D: {
            const int32_t * opts = (const int32_t *) op->op_params;
            // zero kernel volume, channels or output makes the im2col GEMM degenerate (k == 0 or m == 0)
            // and MKL BLAS rejects it; let those fall back to the CPU backend
            const int64_t knl_n_total = op->src[0]->ne[0] * op->src[0]->ne[1] * op->src[0]->ne[2] * opts[9];
            const int64_t patch_total = op->ne[0] * op->ne[1] * op->ne[2] * opts[10];
            return op->type == GGML_TYPE_F32 &&
                   (op->src[0]->type == GGML_TYPE_F32 || op->src[0]->type == GGML_TYPE_F16) &&
                   op->src[1]->type == GGML_TYPE_F32 &&
                   ggml_is_contiguous(op->src[0]) &&
                   ggml_is_contiguous(op->src[1]) &&
                   knl_n_total > 0 && patch_total > 0;
        }
        case GGML_OP_SUM:
        case GGML_OP_SUM_ROWS:
        case GGML_OP_MEAN:
            return ggml_is_contiguous(op->src[0]);
        case GGML_OP_ARGSORT:
            return true;
        case GGML_OP_TOP_K: {
            const ggml_tensor * src0 = op->src[0];
            const int k = op->ne[0];
            return src0 &&
                op->type == GGML_TYPE_I32 &&
                src0->type == GGML_TYPE_F32 &&
                ggml_is_contiguous(src0) &&
                k > 0 && k <= src0->ne[0];
        }
        case GGML_OP_POOL_2D:
        case GGML_OP_POOL_1D:
        case GGML_OP_ACC:
            return true;
        case GGML_OP_PAD:
            if (ggml_get_op_params_i32(op, 8) != 0) {
                return false;
            }
            return true;
        case GGML_OP_LEAKY_RELU:
        case GGML_OP_TIMESTEP_EMBEDDING:
        case GGML_OP_RWKV_WKV6:
        case GGML_OP_RWKV_WKV7:
        case GGML_OP_GATED_LINEAR_ATTN:
        case GGML_OP_GATED_DELTA_NET:
        case GGML_OP_OPT_STEP_ADAMW:
        case GGML_OP_OPT_STEP_SGD:
            return true;
        case GGML_OP_SSM_CONV:
            return op->type == GGML_TYPE_F32 &&
                   op->src[0]->type == GGML_TYPE_F32 &&
                   op->src[1]->type == GGML_TYPE_F32;
        case GGML_OP_ROLL:
            return op->type == GGML_TYPE_F32;
        case GGML_OP_ARANGE:
            return op->type == GGML_TYPE_F32;
        case GGML_OP_SSM_SCAN:
            if (op->src[3]->ne[0] == 1) {
                // Mamba2
                // (kernel only supports (d_state == 128 || d_state == 256) && d_head % WARP_SIZE == 0)
                return (op->src[0]->ne[0] == 128 || op->src[0]->ne[0] == 256) && op->src[0]->ne[1] % WARP_SIZE == 0;
            } else {
                // TODO Mamba-1 not yet ported to SYCL
                return false;
            }
        case GGML_OP_FILL:
        case GGML_OP_CUMSUM:
        case GGML_OP_DIAG:
        case GGML_OP_CROSS_ENTROPY_LOSS:
        case GGML_OP_CROSS_ENTROPY_LOSS_BACK:
            return true;
        case GGML_OP_SOLVE_TRI:
            return op->src[0]->ne[0] <= SYCL_SOLVE_TRI_MAX_N && op->src[1]->ne[0] <= SYCL_SOLVE_TRI_MAX_K;
        case GGML_OP_FLASH_ATTN_EXT:
            return ggml_sycl_flash_attn_ext_supported(device, op);
        default:
            return false;
    }

    GGML_UNUSED(dev);
}

static bool ggml_backend_sycl_device_supports_op(ggml_backend_dev_t dev, const ggml_tensor * op) {
    bool res = do_ggml_backend_sycl_device_supports_op(dev, op);
    GGML_SYCL_DEBUG("[SYCL] call %s op->op=%s op->type=%s -> %s\n", __func__, ggml_op_name(op->op),
                    ggml_type_name(op->type), res ? "true" : "false");
    return res;
}

static bool ggml_backend_sycl_device_supports_buft(ggml_backend_dev_t dev, ggml_backend_buffer_type_t buft) {
    if (buft->iface.get_name != ggml_backend_sycl_buffer_type_get_name) {
        return false;
    }
    ggml_backend_sycl_buffer_type_context * buft_ctx = (ggml_backend_sycl_buffer_type_context *)buft->context;
    ggml_backend_sycl_device_context * sycl_ctx = (ggml_backend_sycl_device_context *)dev->context;
    return buft_ctx->device == sycl_ctx->device;
}

static int64_t get_op_batch_size(const ggml_tensor * op) {
    switch (op->op) {
        case GGML_OP_GET_ROWS:
            return 0;
        case GGML_OP_MUL_MAT:
            return op->ne[1];
        case GGML_OP_MUL_MAT_ID:
        case GGML_OP_ROPE:
            return op->ne[2];
        default:
            return ggml_nrows(op);
    }
}

static bool ggml_backend_sycl_device_offload_op(ggml_backend_dev_t dev, const ggml_tensor * op) {
    ggml_backend_sycl_device_context * sycl_ctx = (ggml_backend_sycl_device_context *)dev->context;
    return get_op_batch_size(op) >= sycl_ctx->op_offload_min_batch_size;
}

static ggml_backend_event_t
ggml_backend_sycl_device_event_new(ggml_backend_dev_t dev) {

#ifdef GGML_SYCL_NO_PEER_COPY
    return nullptr;
#else
  sycl::event *event_ptr = new sycl::event();

  return new ggml_backend_event{
      /* .device = */ dev,
      /* .context = */ event_ptr,
  };
#endif
}

static void ggml_backend_sycl_device_event_free(ggml_backend_dev_t dev, ggml_backend_event_t event) try {
  GGML_UNUSED(dev);
  if (event == nullptr) {
    return;
  }

  sycl_pipe::event_forget(event);
  if (event->context != nullptr) {
    sycl::event *sycl_event = static_cast<sycl::event *>(event->context);
    delete sycl_event;
    event->context = nullptr;
  }

  delete event;
} catch (sycl::exception const &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
  std::exit(1);
}


static void ggml_backend_sycl_device_event_synchronize(ggml_backend_dev_t dev, ggml_backend_event_t event) try {
  GGML_UNUSED(dev);
  GGML_SYCL_DEBUG("[SYCL] call %s\n", __func__);

  sycl::event *sycl_event = static_cast<sycl::event *>(event->context);
  { ggml_sycl_pipe_wait_scope ws_; SYCL_CHECK(CHECK_TRY_ERROR(sycl_event->wait())); }
} catch (sycl::exception const &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
  std::exit(1);
}

static const ggml_backend_device_i ggml_backend_sycl_device_interface = {
    /* .get_name                = */ ggml_backend_sycl_device_get_name,
    /* .get_description         = */ ggml_backend_sycl_device_get_description,
    /* .get_memory              = */ ggml_backend_sycl_device_get_memory,
    /* .get_type                = */ ggml_backend_sycl_device_get_type,
    /* .get_props               = */ ggml_backend_sycl_device_get_props,
    /* .init_backend            = */ ggml_backend_sycl_device_init,
    /* .get_buffer_type         = */ ggml_backend_sycl_device_get_buffer_type,
    /* .get_host_buffer_type    = */ ggml_backend_sycl_device_get_host_buffer_type,
    /* .buffer_from_host_ptr    = */ ggml_backend_sycl_device_buffer_from_host_ptr,
    /* .supports_op             = */ ggml_backend_sycl_device_supports_op,
    /* .supports_buft           = */ ggml_backend_sycl_device_supports_buft,
    /* .offload_op              = */ ggml_backend_sycl_device_offload_op,
    /* .event_new               = */ ggml_backend_sycl_device_event_new,
    /* .event_free              = */ ggml_backend_sycl_device_event_free,
    /* .event_synchronize       = */ ggml_backend_sycl_device_event_synchronize,
};

// backend reg

struct ggml_backend_sycl_reg_context {
    std::vector<ggml_backend_dev_t> devices;
};

static const char * ggml_backend_sycl_reg_get_name(ggml_backend_reg_t reg) {
    GGML_UNUSED(reg);
    return GGML_SYCL_NAME;
}

static size_t ggml_backend_sycl_reg_get_device_count(ggml_backend_reg_t reg) {
    ggml_backend_sycl_reg_context * ctx = (ggml_backend_sycl_reg_context *)reg->context;
    return ctx->devices.size();
}

static ggml_backend_dev_t ggml_backend_sycl_reg_get_device(ggml_backend_reg_t reg, size_t index) {
    ggml_backend_sycl_reg_context * ctx = (ggml_backend_sycl_reg_context *)reg->context;
    GGML_ASSERT(index < ctx->devices.size());
    return ctx->devices[index];
}

// ==========================================================================
// Tensor parallelism (--split-mode tensor) for the SYCL backend.
//
// The meta-backend invokes these three entry points via get_proc_address:
//   * ggml_backend_sycl_comm_init             - one-time per-graph setup
//   * ggml_backend_sycl_comm_allreduce_tensor - per-allreduce step
//   * ggml_backend_sycl_comm_free             - tear-down
//
// For N=2 (dual-GPU), this is a degenerate ring allreduce with dual paths
// chosen by tensor size:
//
//   * Small (nelem < 32K): FP32 direct memcpy + per-device ADD
//     kernel. The kernel depends_on() its corresponding memcpy event
//     so it doesn't read partial data. Both devices run in parallel.
//
//   * Large (nelem >= 32K): BF16-compressed. Each device compresses
//     its FP32 partial to BF16 locally, cross-device memcpys
//     to the peer (half the PCI bandwidth), where it is decompressed
//     and added into the local FP32 partial. 6 SYCL submissions per
//     allreduce (2 compress + 2 memcpy + 2 decompress-add) vs the
//     4 for the small path, but the bandwidth saving > 6 GB/s PCIe x 2
//     dominates for larger tensors.
//
// Storage: A persistent uint8_t buffer per device, sized to
// 4 * nelem bytes. Both paths reinterpret the same bytes (small path
// as nelem floats; large path as outbox + inbox = 2*nelem uint16_t
// each, using the full 4*nelem byte budget either way). Single
// alloc+free per device keeps the SYCL pool's strict-LIFO invariant
// trivial.
//
// For N=3..GGML_SYCL_MAX_DEVICES, ggml_sycl_comm_allreduce_nway stages every
// partial through pinned host memory (see the comment there).
//
// For other cases (non-F32/F16, non-contiguous, GGML_SYCL_COMM_NWAY=0 with
// N>2), comm_init or comm_allreduce_tensor returns null/false, causing the
// meta-backend to use its generic butterfly all-reduce fallback.
// ==========================================================================

struct ggml_backend_sycl_comm_context {
    std::vector<ggml_backend_t> backends;
    // ONE persistent per-device byte buffer, 4*(N-1)*nelem bytes (4*nelem for N=2).  Both the
    // FP32 small-tensor path and the BF16 large-tensor path share it
    // by reinterpreting.
    std::vector<std::unique_ptr<ggml_sycl_pool_alloc<uint8_t>>> bufs;
    int64_t buf_nelem = 0;

    // pinned host staging per device context, reused across calls (queues are in-order)
    std::vector<uint8_t *> host_out;
    size_t    host_bytes  = 0;

    void free_host() {
        for (size_t i = 0; i < host_out.size(); ++i) {
            const sycl::context sctx = ((ggml_backend_sycl_context *) backends[i]->context)->stream()->get_context();
            if (host_out[i]) { sycl::free(host_out[i], sctx); host_out[i] = nullptr; }
        }
        host_bytes = 0;
    }
};

// dst0 <- src1 and dst1 <- src0 through pinned host memory (PR #30956: each queue uploads straight from the
// other device's staging buffer, no CPU memcpy; blocking on the uploads so host_out can be reused next call)
static void ggml_sycl_comm_exchange(ggml_backend_sycl_comm_context * comm, queue_ptr q0, queue_ptr q1,
                                    void * dst0, const void * src1, void * dst1, const void * src0,
                                    size_t nbytes) {
    if (comm->host_bytes < nbytes) {
        { ggml_sycl_pipe_wait_scope ws_; q0->wait(); }
        { ggml_sycl_pipe_wait_scope ws_; q1->wait(); }
        comm->free_host();
        comm->host_out[0] = sycl::malloc_host<uint8_t>(nbytes, q0->get_context());
        comm->host_out[1] = sycl::malloc_host<uint8_t>(nbytes, q1->get_context());
        GGML_ASSERT(comm->host_out[0] && comm->host_out[1]);
        comm->host_bytes = nbytes;
    }

    sycl::event e0 = q0->memcpy(comm->host_out[0], src0, nbytes);
    sycl::event e1 = q1->memcpy(comm->host_out[1], src1, nbytes);
    { ggml_sycl_pipe_wait_scope ws_; e0.wait(); }
    { ggml_sycl_pipe_wait_scope ws_; e1.wait(); }
    sycl::event u0 = q0->memcpy(dst0, comm->host_out[1], nbytes);
    sycl::event u1 = q1->memcpy(dst1, comm->host_out[0], nbytes);
    { ggml_sycl_pipe_wait_scope ws_; u0.wait(); }
    { ggml_sycl_pipe_wait_scope ws_; u1.wait(); }
}

// N-way (N > 2) all-reduce through pinned host memory, generalising the N=2 paths:
//   1. (BF16 path only) each card compresses its FP32 partial into an outbox
//   2. each card downloads its partial (or outbox) into its own pinned host buffer; host waits for all
//   3. each card uploads the other N-1 host buffers into inbox slots on its own queue; host waits
//      for all (the host buffers are reused next call, so no upload may still be reading them)
//   4. each card sums the N partials in card order 0..N-1 into its own tensor
// Per card that is 1 download + (N-1) uploads of nbytes (BF16: half), with no CPU arithmetic.
// Summing in a fixed order gives identical results on every card for F32 and F16. On the BF16 path a
// card adds its own partial in FP32 and the others as BF16 (same as the N=2 BF16 path), so the cards
// can differ in the low bits there.
template <typename T_own, typename T_in, bool bf16_in>
static void ggml_sycl_comm_nway_sum(queue_ptr q, T_own * own, const T_in * inbox, int64_t nelem, int n, int self) {
    q->parallel_for(sycl::range<1>(nelem), [=](sycl::id<1> id) {
        const int64_t i = id[0];
        float acc = 0.0f;
        for (int c = 0; c < n; ++c) {
            float v;
            if (c == self) {
                v = (float) own[i];
            } else {
                const int64_t slot = c < self ? c : c - 1;
                const T_in x = inbox[slot * nelem + i];
                if constexpr (bf16_in) {
                    v = sycl::bit_cast<float>(((uint32_t) x) << 16);
                } else {
                    v = (float) x;
                }
            }
            acc += v;
        }
        own[i] = (T_own) acc;
    });
}

static bool ggml_sycl_comm_allreduce_nway(ggml_backend_sycl_comm_context * comm, ggml_tensor ** tensors) {
    const int n = (int) comm->backends.size();
    const ggml_type type = tensors[0]->type;
    if (type != GGML_TYPE_F32 && type != GGML_TYPE_F16) {
        return false;
    }
    for (int j = 0; j < n; ++j) {
        if (tensors[j]->type != type || !ggml_is_contiguous(tensors[j]) ||
            ggml_nelements(tensors[j]) != ggml_nelements(tensors[0])) {
            return false;
        }
    }
    const int64_t nelem = ggml_nelements(tensors[0]);
    if (nelem == 0) {
        return true;
    }

    static constexpr int64_t BF16_THRESHOLD = 32768; // same as the N=2 path
    const bool   use_bf16 = type == GGML_TYPE_F32 && nelem >= BF16_THRESHOLD;
    const size_t xbytes   = (size_t) nelem * (type == GGML_TYPE_F32 && !use_bf16 ? 4 : 2);

    std::vector<queue_ptr> q(n);
    for (int j = 0; j < n; ++j) {
        q[j] = ((ggml_backend_sycl_context *) comm->backends[j]->context)->stream();
    }

    // device scratch per card: [outbox (BF16 only)] + (N-1) inbox slots. 4*(N-1)*nelem bytes covers
    // F32 ((N-1)*4), F16 ((N-1)*2) and BF16 (N*2) for N >= 2.
    if (comm->buf_nelem < nelem) {
        for (int j = 0; j < n; ++j) {
            comm->bufs[j]->realloc((size_t) nelem * 4 * (n - 1));
        }
        comm->buf_nelem = nelem;
    }
    if (comm->host_bytes < xbytes) {
        for (int j = 0; j < n; ++j) {
            ggml_sycl_pipe_wait_scope ws_;
            q[j]->wait();
        }
        comm->free_host();
        for (int j = 0; j < n; ++j) {
            comm->host_out[j] = sycl::malloc_host<uint8_t>(xbytes, q[j]->get_context());
            GGML_ASSERT(comm->host_out[j]);
        }
        comm->host_bytes = xbytes;
    }

    // Large BF16 tensors (prompt batches): reduce-scatter + all-gather. Card j sums only chunk j
    // (1/N of the elements) and then shares it, so each card uploads 2*(N-1)/N tensors' worth of BF16
    // instead of N-1, at the cost of one more host wait. Chunk j is summed on card j alone, so the other
    // cards receive it as BF16 while card j keeps FP32 (low-bit differences, as on the BF16 path above).
    // GGML_SYCL_COMM_RS=0 keeps the all-to-all path for every size.
    static const bool rs_enabled = [] {
        const char * s = getenv("GGML_SYCL_COMM_RS");
        return s == nullptr || atoi(s) != 0;
    }();
    if (use_bf16 && rs_enabled) {
        auto cbeg = [&](int c) { return (int64_t) c * nelem / n; };
        std::vector<sycl::event> evs;
        evs.reserve((size_t) n * (n - 1));
        auto wait_all = [&]() {
            for (auto & e : evs) {
                ggml_sycl_pipe_wait_scope ws_;
                e.wait();
            }
            evs.clear();
        };
        // device scratch on card j: [outbox: nelem bf16][inbox: (N-1) slots of ceil(nelem/N) bf16][red: ceil(nelem/N) bf16]
        const int64_t cmax = (nelem + n - 1) / n;
        std::vector<uint16_t *> outbox(n), inb(n), red(n);
        for (int j = 0; j < n; ++j) {
            uint16_t * b = (uint16_t *) comm->bufs[j]->get();
            outbox[j] = b;
            inb[j]    = b + nelem;
            red[j]    = inb[j] + (size_t) (n - 1) * cmax;
            const float * part = (const float *) tensors[j]->data;
            uint16_t *    ob   = outbox[j];
            q[j]->parallel_for(sycl::range<1>(nelem), [=](sycl::id<1> i) {
                ob[i] = (uint16_t) (sycl::bit_cast<uint32_t>(part[i]) >> 16);
            });
            evs.push_back(q[j]->memcpy(comm->host_out[j], ob, xbytes));
        }
        wait_all();

        // reduce-scatter: card j gathers chunk j of every other card's partial
        for (int j = 0; j < n; ++j) {
            const int64_t b0 = cbeg(j), len = cbeg(j + 1) - b0;
            for (int c = 0; c < n; ++c) {
                if (c == j || len == 0) {
                    continue;
                }
                const int slot = c < j ? c : c - 1;
                evs.push_back(q[j]->memcpy(inb[j] + (size_t) slot * cmax,
                                           (const uint16_t *) comm->host_out[c] + b0, len * sizeof(uint16_t)));
            }
        }
        wait_all();

        // sum chunk j in card order, keep FP32 locally, publish it as BF16 in host_out[j] (same region it came from;
        // every upload reading host_out has finished)
        for (int j = 0; j < n; ++j) {
            const int64_t b0 = cbeg(j), len = cbeg(j + 1) - b0;
            if (len == 0) {
                continue;
            }
            float *          own = (float *) tensors[j]->data + b0;
            const uint16_t * in  = inb[j];
            uint16_t *       rd  = red[j];
            const int        self = j;
            const int        nn   = n;
            const int64_t    cm   = cmax;
            q[j]->parallel_for(sycl::range<1>(len), [=](sycl::id<1> id) {
                const int64_t i = id[0];
                float acc = 0.0f;
                for (int c = 0; c < nn; ++c) {
                    if (c == self) {
                        acc += own[i];
                    } else {
                        const int64_t slot = c < self ? c : c - 1;
                        acc += sycl::bit_cast<float>(((uint32_t) in[slot * cm + i]) << 16);
                    }
                }
                own[i] = acc;
                rd[i]  = (uint16_t) (sycl::bit_cast<uint32_t>(acc) >> 16);
            });
            evs.push_back(q[j]->memcpy((uint16_t *) comm->host_out[j] + b0, rd, len * sizeof(uint16_t)));
        }
        wait_all();

        // all-gather: card j fetches every other summed chunk and expands it into its own tensor
        for (int j = 0; j < n; ++j) {
            for (int c = 0; c < n; ++c) {
                const int64_t b0 = cbeg(c), len = cbeg(c + 1) - b0;
                if (c == j || len == 0) {
                    continue;
                }
                const int slot = c < j ? c : c - 1;
                evs.push_back(q[j]->memcpy(inb[j] + (size_t) slot * cmax,
                                           (const uint16_t *) comm->host_out[c] + b0, len * sizeof(uint16_t)));
            }
        }
        wait_all();
        for (int j = 0; j < n; ++j) {
            float *          dst = (float *) tensors[j]->data;
            const uint16_t * in  = inb[j];
            const int        self = j;
            const int        nn   = n;
            const int64_t    cm   = cmax;
            const int64_t    ne   = nelem;
            // one pass over the N-1 foreign chunks; element i of the tensor belongs to chunk c = owner(i)
            q[j]->parallel_for(sycl::range<1>(nelem), [=](sycl::id<1> id) {
                const int64_t i = id[0];
                int c = (int) ((i * nn) / ne);
                while (c + 1 < nn && (int64_t) (c + 1) * ne / nn <= i) { ++c; }
                while (c > 0 && (int64_t) c * ne / nn > i) { --c; }
                if (c == self) {
                    return;
                }
                const int64_t slot = c < self ? c : c - 1;
                const int64_t off  = i - (int64_t) c * ne / nn;
                dst[i] = sycl::bit_cast<float>(((uint32_t) in[slot * cm + off]) << 16);
            });
        }
        return true;
    }

    std::vector<uint8_t *> inbox(n);
    std::vector<const void *> send(n);
    for (int j = 0; j < n; ++j) {
        uint8_t * buf = comm->bufs[j]->get();
        if (use_bf16) {
            uint16_t *    outbox = (uint16_t *) buf;
            const float * part   = (const float *) tensors[j]->data;
            q[j]->parallel_for(sycl::range<1>(nelem), [=](sycl::id<1> i) {
                outbox[i] = (uint16_t) (sycl::bit_cast<uint32_t>(part[i]) >> 16);
            });
            send[j]  = outbox;
            inbox[j] = buf + xbytes;
        } else {
            send[j]  = tensors[j]->data;
            inbox[j] = buf;
        }
    }

    std::vector<sycl::event> ev;
    ev.reserve((size_t) n * (n - 1));
    for (int j = 0; j < n; ++j) {
        ev.push_back(q[j]->memcpy(comm->host_out[j], send[j], xbytes));
    }
    for (auto & e : ev) {
        ggml_sycl_pipe_wait_scope ws_;
        e.wait();
    }
    ev.clear();

    for (int j = 0; j < n; ++j) {
        for (int c = 0; c < n; ++c) {
            if (c == j) {
                continue;
            }
            const int slot = c < j ? c : c - 1;
            ev.push_back(q[j]->memcpy(inbox[j] + (size_t) slot * xbytes, comm->host_out[c], xbytes));
        }
    }
    for (auto & e : ev) {
        ggml_sycl_pipe_wait_scope ws_;
        e.wait();
    }

    for (int j = 0; j < n; ++j) {
        if (type == GGML_TYPE_F16) {
            ggml_sycl_comm_nway_sum<sycl::half, sycl::half, false>(
                q[j], (sycl::half *) tensors[j]->data, (const sycl::half *) inbox[j], nelem, n, j);
        } else if (use_bf16) {
            ggml_sycl_comm_nway_sum<float, uint16_t, true>(
                q[j], (float *) tensors[j]->data, (const uint16_t *) inbox[j], nelem, n, j);
        } else {
            ggml_sycl_comm_nway_sum<float, float, false>(
                q[j], (float *) tensors[j]->data, (const float *) inbox[j], nelem, n, j);
        }
    }
    return true;
}

void * ggml_backend_sycl_comm_init(ggml_backend_t * backends, size_t n_backends) try {
    for (size_t i = 0; i < n_backends; ++i) {
        if (!ggml_backend_is_sycl(backends[i])) {
            return nullptr;
        }
    }

    // N=2: dual-GPU paths below. N=3..GGML_SYCL_MAX_DEVICES: host-staged N-way path
    // (GGML_SYCL_COMM_NWAY=0 turns it off and leaves N>2 to the meta-backend fallback).
    // Returning null makes the meta-backend skip this backend-specific allreduce entirely.
    static const bool nway_enabled = [] {
        const char * s = getenv("GGML_SYCL_COMM_NWAY");
        return s == nullptr || atoi(s) != 0;
    }();
    if (n_backends < 2 || n_backends > GGML_SYCL_MAX_DEVICES || (n_backends > 2 && !nway_enabled)) {
        return nullptr;
    }

    auto * ctx = new ggml_backend_sycl_comm_context;
    ctx->backends.assign(backends, backends + n_backends);
    ctx->host_out.assign(n_backends, nullptr);
    for (size_t i = 0; i < n_backends; ++i) {
        auto * sctx = (ggml_backend_sycl_context *) backends[i]->context;
        ctx->bufs.push_back(std::make_unique<ggml_sycl_pool_alloc<uint8_t>>(sctx->pool()));
    }
    return ctx;
}
catch (const sycl::exception &) { return nullptr; }
catch (...)                     { return nullptr; }

void ggml_backend_sycl_comm_free(void * comm_ctx_v) {
    auto * comm_ctx = static_cast<ggml_backend_sycl_comm_context *>(comm_ctx_v);
    if (comm_ctx == nullptr) {
        return;
    }

    // Sync both per-device queues so the pool_alloc destructors don't
    // return memory still in use by the last kernel.
    try {
        for (ggml_backend_t b : comm_ctx->backends) {
            auto * sctx = (ggml_backend_sycl_context *) b->context;
            ggml_sycl_pipe_wait_scope ws_;
            sctx->stream()->wait();
        }
        comm_ctx->free_host();
    } catch (...) { /* best effort during shutdown */ }

    delete comm_ctx;
}

bool ggml_backend_sycl_comm_allreduce_tensor(void * comm_ctx_v, struct ggml_tensor ** tensors) try {
    if (comm_ctx_v == nullptr) {
        return false;
    }

    auto * comm_ctx = static_cast<ggml_backend_sycl_comm_context *>(comm_ctx_v);
    const size_t n_backends = comm_ctx->backends.size();

    if (n_backends > 2) {
        return ggml_sycl_comm_allreduce_nway(comm_ctx, tensors);
    }
    // Fast path: N=2, F32/F16, contiguous, matching shapes.
    if (n_backends != 2) {
        return false;
    }
    // Accept F32 or F16 inputs natively (types must match). F16 takes the
    // direct 2-byte memcpy + add path below; other types return false so the
    // meta-backend uses its generic all-reduce.
    if (tensors[0]->type != tensors[1]->type) {
        return false;
    }
    if (tensors[0]->type != GGML_TYPE_F32 && tensors[0]->type != GGML_TYPE_F16) {
        return false;
    }
    if (!ggml_is_contiguous(tensors[0]) || !ggml_is_contiguous(tensors[1])) {
        return false;
    }
    if (ggml_nelements(tensors[0]) != ggml_nelements(tensors[1])) {
        return false;
    }

    const int64_t nelem  = ggml_nelements(tensors[0]);
    const size_t  nbytes = ggml_nbytes(tensors[0]);
    if (nelem == 0) {
        return true;
    }

    auto * ctx0 = (ggml_backend_sycl_context *) comm_ctx->backends[0]->context;
    auto * ctx1 = (ggml_backend_sycl_context *) comm_ctx->backends[1]->context;
    queue_ptr q0 = ctx0->stream();
    queue_ptr q1 = ctx1->stream();

    // Grow per-device byte buffers if needed (4 * nelem bytes each).
    if (comm_ctx->buf_nelem < nelem) {
        comm_ctx->bufs[0]->realloc(nelem * 4);
        comm_ctx->bufs[1]->realloc(nelem * 4);
        comm_ctx->buf_nelem = nelem;
    }
    uint8_t * buf0 = comm_ctx->bufs[0]->get();
    uint8_t * buf1 = comm_ctx->bufs[1]->get();

    // F16 native path: direct 2-byte cross-device copy + add, skipping the
    // F32 round-trip the meta-backend fallback would force. The devices are in separate SYCL
    // contexts (a raw peer-USM q->memcpy would be a silent no-op), so copies go through host memory.
    if (tensors[0]->type == GGML_TYPE_F16) {
        sycl::half * f16_out0 = (sycl::half *) tensors[0]->data;
        sycl::half * f16_out1 = (sycl::half *) tensors[1]->data;
        sycl::half * f16_tmp0 = (sycl::half *) buf0;
        sycl::half * f16_tmp1 = (sycl::half *) buf1;

        ggml_sycl_comm_exchange(comm_ctx, q0, q1, f16_tmp0, tensors[1]->data, f16_tmp1, tensors[0]->data, nbytes);

        q0->submit([&](sycl::handler & h) {
            h.parallel_for(sycl::range<1>(nelem), [=](sycl::id<1> i) {
                f16_out0[i] = (sycl::half) ((float) f16_out0[i] + (float) f16_tmp0[i]);
            });
        });
        q1->submit([&](sycl::handler & h) {
            h.parallel_for(sycl::range<1>(nelem), [=](sycl::id<1> i) {
                f16_out1[i] = (sycl::half) ((float) f16_out1[i] + (float) f16_tmp1[i]);
            });
        });
        return true;
    }

    float * out0 = (float *) tensors[0]->data;
    float * out1 = (float *) tensors[1]->data;

    // BF16 threshold: above this, the PCIe savings from halving the
    // cross-device bytes outweigh the 2 extra compress kernels.
    // Below: stay on the FP32 fast path.  Threshold mirrors the CUDA
    // NCCL allreduce pattern for n_backends=2.
    static constexpr int64_t BF16_THRESHOLD = 32768;

    if (nelem < BF16_THRESHOLD) {
        // FP32 small path: 4 SYCL submissions per allreduce.
        float * tmp0 = (float *) buf0;
        float * tmp1 = (float *) buf1;

        // separate SYCL contexts: a raw peer-USM q->memcpy is a silent no-op, so stage through host memory
        ggml_sycl_comm_exchange(comm_ctx, q0, q1, tmp0, tensors[1]->data, tmp1, tensors[0]->data, nbytes);

        q0->submit([&](sycl::handler & h) {
            h.parallel_for(sycl::range<1>(nelem), [=](sycl::id<1> i) {
                out0[i] += tmp0[i];
            });
        });
        q1->submit([&](sycl::handler & h) {
            h.parallel_for(sycl::range<1>(nelem), [=](sycl::id<1> i) {
                out1[i] += tmp1[i];
            });
        });
        return true;
    }

    // BF16 large path: 6 SYCL submissions per allreduce, but the
    // cross-device memcpy is HALF the bytes. Pure bit-shift
    // conversion (no rounding) — matches ggml's truncating fp32->bf16.
    uint16_t * outbox0 = (uint16_t *) buf0;
    uint16_t * inbox0  = outbox0 + nelem;
    uint16_t * outbox1 = (uint16_t *) buf1;
    uint16_t * inbox1  = outbox1 + nelem;

    // Phase A: compress each device's local partial in parallel.
    q0->parallel_for(sycl::range<1>(nelem), [=](sycl::id<1> i) {
        outbox0[i] = (uint16_t) (sycl::bit_cast<uint32_t>(out0[i]) >> 16);
    });

    q1->parallel_for(sycl::range<1>(nelem), [=](sycl::id<1> i) {
        outbox1[i] = (uint16_t) (sycl::bit_cast<uint32_t>(out1[i]) >> 16);
    });

    // Phase B: exchange the compressed bytes (in-order queues keep this after the compress kernels).
    const size_t bf16_bytes = nelem * sizeof(uint16_t);
    ggml_sycl_comm_exchange(comm_ctx, q0, q1, inbox0, outbox1, inbox1, outbox0, bf16_bytes);

    // Phase C: decompress + add into local FP32 partial.
    q0->submit([&](sycl::handler & h) {
        h.parallel_for(sycl::range<1>(nelem), [=](sycl::id<1> i) {
            out0[i] += sycl::bit_cast<float>(((uint32_t) inbox0[i]) << 16);
        });
    });

    q1->submit([&](sycl::handler & h) {
        h.parallel_for(sycl::range<1>(nelem), [=](sycl::id<1> i) {
            out1[i] += sycl::bit_cast<float>(((uint32_t) inbox1[i]) << 16);
        });
    });

    return true;
}
catch (const sycl::exception &) { return false; }
catch (...)                     { return false; }

static void *ggml_backend_sycl_reg_get_proc_address(ggml_backend_reg_t reg, const char *name) {
    GGML_UNUSED(reg);

    if (strcmp(name, "ggml_backend_split_buffer_type") == 0) {
        return (void *)ggml_backend_sycl_split_buffer_type;
    }

    // Tensor parallelism (--split-mode tensor) entry points.
    if (strcmp(name, "ggml_backend_comm_init") == 0) {
        return (void *)ggml_backend_sycl_comm_init;
    }
    if (strcmp(name, "ggml_backend_comm_free") == 0) {
        return (void *)ggml_backend_sycl_comm_free;
    }
    if (strcmp(name, "ggml_backend_comm_allreduce_tensor") == 0) {
        return (void *)ggml_backend_sycl_comm_allreduce_tensor;
    }

    // SYCL doesn't support registering host memory, left here for reference
    // "ggml_backend_register_host_buffer"
    // "ggml_backend_unregister_host_buffer"
    GGML_UNUSED(name);
    return nullptr;
}

static const ggml_backend_reg_i ggml_backend_sycl_reg_interface = {
    /* .get_name          = */ ggml_backend_sycl_reg_get_name,
    /* .get_device_count  = */ ggml_backend_sycl_reg_get_device_count,
    /* .get_device        = */ ggml_backend_sycl_reg_get_device,
    /* .get_proc_address  = */ ggml_backend_sycl_reg_get_proc_address,
};


// backend registry

ggml_backend_reg_t ggml_backend_sycl_reg() {
    GGML_SYCL_DEBUG("[SYCL] call ggml_backend_sycl_reg\n");
    static ggml_backend_reg reg;
    static bool initialized = false;

    {
        static std::mutex mutex;
        std::lock_guard<std::mutex> lock(mutex);
        if (!initialized) {
            ggml_check_sycl();
            ggml_backend_sycl_reg_context * ctx = new ggml_backend_sycl_reg_context;
            const int min_batch_size = getenv("GGML_OP_OFFLOAD_MIN_BATCH") ? atoi(getenv("GGML_OP_OFFLOAD_MIN_BATCH")) : 32;

            for (int i = 0; i < ggml_sycl_info().device_count; i++) {
                ggml_backend_sycl_device_context * dev_ctx = new ggml_backend_sycl_device_context;
                dev_ctx->device = i;
                dev_ctx->name = GGML_SYCL_NAME + std::to_string(i);

                ggml_sycl_set_device(i);

                dpct::device_info prop;
                SYCL_CHECK(CHECK_TRY_ERROR(dpct::get_device_info(
                    prop, dpct::dev_mgr::instance().get_device(i))));

                dev_ctx->description = prop.get_name();
                dev_ctx->op_offload_min_batch_size = min_batch_size;

                ggml_backend_dev_t dev = new ggml_backend_device {
                    /* .iface       = */ ggml_backend_sycl_device_interface,
                    /* .reg         = */ &reg,
                    /* .context     = */ dev_ctx
                };
                ctx->devices.push_back(dev);
            }

            reg = ggml_backend_reg {
                /* .api_version = */ GGML_BACKEND_API_VERSION,
                /* .iface       = */ ggml_backend_sycl_reg_interface,
                /* .context     = */ ctx
            };
        }

        initialized = true;
    }

    return &reg;
}

ggml_backend_t ggml_backend_sycl_init(int device) {
    GGML_SYCL_DEBUG("[SYCL] call ggml_backend_sycl_init\n");
    ggml_check_sycl();

    check_allow_gpu_index(device);

    ggml_backend_sycl_context * ctx = new ggml_backend_sycl_context(device);
    if (ctx == nullptr) {
        GGML_LOG_ERROR("%s: error: failed to allocate context\n", __func__);
        return nullptr;
    };

    ggml_backend_t sycl_backend = new ggml_backend {
        /* .guid    = */ ggml_backend_sycl_guid(),
        /* .iface   = */ ggml_backend_sycl_interface,
        /* .device  = */ ggml_backend_reg_dev_get(ggml_backend_sycl_reg(), device),
        /* .context = */ ctx
    };

    return sycl_backend;
}

GGML_BACKEND_DL_IMPL(ggml_backend_sycl_reg)

