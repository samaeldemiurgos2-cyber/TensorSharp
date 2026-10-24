// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// The backend side of MiniMax-M3's tensor parallelism (see
// ggml_ops_minimax_m3_tp.h): the meta device over the first N CUDA GPUs, the
// graph check that keeps CPU-fallback ops away from it, and cache-row copies.
#include "ggml_ops_minimax_m3_tp.h"
#include "ggml_ops_internal.h"

#include <algorithm>
#include <cinttypes>
#include <cstdio>
#include <cstring>

namespace tsg_mm3_tp
{
    ggml_backend_t open(int tp, userdata * ud, devices * out, std::string * why)
    {
        char buf[256];
        auto fail = [&](const char * text) -> ggml_backend_t { if (why) *why = text; return nullptr; };
        if (tp < 2 || tp > MAX_DEVICES)
        {
            snprintf(buf, sizeof(buf), "--tp %d: tensor parallelism spans 2 to %d GPUs", tp, MAX_DEVICES);
            return fail(buf);
        }
        // The first `tp` CUDA GPUs in index order, opened only through the meta
        // device so each gets one stream.
        devices d;
        std::vector<int> cuda_index;
        int cuda_seen = 0;
        for (size_t i = 0; i < ggml_backend_dev_count() && d.n < tp; i++)
        {
            ggml_backend_dev_t dev = ggml_backend_dev_get(i);
            ggml_backend_reg_t reg = ggml_backend_dev_backend_reg(dev);
            const char * reg_name = reg ? ggml_backend_reg_name(reg) : nullptr;
            const enum ggml_backend_dev_type dt = ggml_backend_dev_type(dev);
            if (!reg_name || strcmp(reg_name, "CUDA") != 0) continue;
            if (dt != GGML_BACKEND_DEVICE_TYPE_GPU && dt != GGML_BACKEND_DEVICE_TYPE_IGPU) continue;
            d.devs[d.n++] = dev;
            cuda_index.push_back(cuda_seen++);
        }
        if (d.n < tp)
        {
            snprintf(buf, sizeof(buf), "--tp %d needs %d CUDA GPUs; %d are visible", tp, tp, d.n);
            return fail(buf);
        }
        for (int j = 1; j < d.n; j++)
            if (strcmp(ggml_backend_dev_description(d.devs[j]), ggml_backend_dev_description(d.devs[0])) != 0)
            {
                snprintf(buf, sizeof(buf), "--tp needs identical GPUs; GPU %d is %s and GPU 0 is %s", j,
                         ggml_backend_dev_description(d.devs[j]), ggml_backend_dev_description(d.devs[0]));
                return fail(buf);
            }
#if defined(TSG_GGML_USE_CUDA)
        // Before the meta backend creates its communicator.
        tsg::tp_cuda_prepare_collective(cuda_index.data(), d.n);
#endif
        static std::mutex meta_mu;   // ggml's meta device cache is not thread-safe
        std::lock_guard<std::mutex> lk(meta_mu);
        d.meta = ggml_backend_meta_device(d.devs, (size_t) d.n, split_state_cb, ud);
        ggml_backend_t be = d.meta ? ggml_backend_dev_init(d.meta, nullptr) : nullptr;
        if (!be)
        {
            snprintf(buf, sizeof(buf), "creating the tensor-parallel backend over %d GPUs failed", d.n);
            return fail(buf);
        }
        *out = d;
        return be;
    }

    void memory(const devices & d, size_t * free_min, size_t * total_min)
    {
        *free_min = 0;
        *total_min = 0;
        for (int j = 0; j < d.n; j++)
        {
            size_t f = 0, t = 0;
            ggml_backend_dev_memory(d.devs[j], &f, &t);
            if (j == 0 || f < *free_min) *free_min = f;
            if (j == 0 || t < *total_min) *total_min = t;
        }
    }

    bool check_graph(ggml_backend_t meta, ggml_cgraph * gf, const ggml_tensor * host_table, std::string * why)
    {
        for (int i = 0; i < ggml_graph_n_nodes(gf); i++)
        {
            ggml_tensor * node = ggml_graph_node(gf, i);
            if (node->op == GGML_OP_NONE || node->op == GGML_OP_VIEW || node->op == GGML_OP_RESHAPE ||
                node->op == GGML_OP_PERMUTE || node->op == GGML_OP_TRANSPOSE) continue;
            bool host_side = false;
            for (int k = 0; k < GGML_MAX_SRC && !host_side; k++)
                if (const ggml_tensor * src = node->src[k]; src && src == host_table) host_side = true;
            if (host_side) continue;
            // ARANGE has no source to take a split state from (the meta backend
            // aborts on it) even though every GPU supports it.
            if (ggml_nelements(node) == 0 || node->op == GGML_OP_ARANGE || !ggml_backend_supports_op(meta, node))
            {
                char buf[512];
                snprintf(buf, sizeof(buf), "%s (%s, [%" PRId64 ", %" PRId64 ", %" PRId64 ", %" PRId64 "]) cannot run on the "
                         "tensor-parallel GPUs at this shape, and a tensor-parallel layer cannot fall back to the CPU",
                         ggml_op_desc(node), node->name, node->ne[0], node->ne[1], node->ne[2], node->ne[3]);
                if (why) *why = buf;
                return false;
            }
        }
        return true;
    }

    bool copy_rows(ggml_backend_t meta, const std::vector<row_copy> & copies)
    {
        const size_t n_nodes = copies.size() * 4 + 8;
        ggml_init_params p = { n_nodes * ggml_tensor_overhead() + ggml_graph_overhead_custom(n_nodes, false), nullptr, true };
        ggml_context * c = ggml_init(p);
        if (!c) return false;
        ggml_cgraph * gf = ggml_new_graph_custom(c, n_nodes, false);
        for (const row_copy & rc : copies)
        {
            if (!rc.src || !rc.dst) continue;
            const int64_t rows = std::min<int64_t>(rc.rows, std::min<int64_t>(rc.src->ne[1], rc.dst->ne[1]));
            if (rows <= 0) continue;
            ggml_tensor * vs = ggml_view_2d(c, rc.src, rc.src->ne[0], rows, rc.src->nb[1], 0);
            ggml_tensor * vd = ggml_view_2d(c, rc.dst, rc.dst->ne[0], rows, rc.dst->nb[1], 0);
            ggml_build_forward_expand(gf, ggml_cpy(c, vs, vd));
        }
        bool ok = true;
        if (ggml_graph_n_nodes(gf) > 0)
        {
            for (int i = 0; i < ggml_graph_n_nodes(gf); i++)
            {
                ggml_tensor * node = ggml_graph_node(gf, i);
                for (int k = 0; k < GGML_MAX_SRC; k++)
                    if (ggml_tensor * src = node->src[k]; src && src->view_src && !src->buffer) ggml_backend_view_init(src);
                if (node->view_src && !node->buffer) ggml_backend_view_init(node);
            }
            ok = ggml_backend_graph_compute(meta, gf) == GGML_STATUS_SUCCESS;
        }
        ggml_free(c);
        return ok;
    }
}
