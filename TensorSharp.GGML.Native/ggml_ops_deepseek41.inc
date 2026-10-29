// DeepSeek V4.1 graph methods. Included inside graph_builder to share the
// existing ggml scheduler, MoE kernels, weight placement and sequence slots.
// Equations follow deepseek-ai/DeepSeek-V4.1-Flash/inference/model.py.

ggml_tensor * v41_shared_mask = nullptr;
ggml_tensor * v41_shared_top = nullptr;
ggml_tensor * v41_candidates = nullptr;
ggml_tensor * v41_candidate_scores = nullptr;
ggml_tensor * v41_candidate_top = nullptr;

ggml_tensor * v41_quant(ggml_tensor * x, int mode, int dev)
{
    x = ggml_cont(ctx, x);
    return fused_node(TSG_DSV41_FUSED_QUANT, GGML_TYPE_F32,
        x->ne[0], x->ne[1], x->ne[2], x->ne[3], { x }, dev, mode);
}

// Rotate only the last n_rot components; the preceding components are MLA
// content dimensions. V4.1 does not normalize the projected query heads.
ggml_tensor * v41_rope(ggml_tensor * x, ggml_tensor * pos, int il, bool inverse = false)
{
    const int64_t head = x->ne[0], nhead = x->ne[1], n = x->ne[2];
    const int64_t nope = head - hp.n_rot;
    ggml_tensor * content = ggml_view_3d(ctx, x, nope, nhead, n, x->nb[1], x->nb[2], 0);
    ggml_tensor * tail = ggml_view_3d(ctx, x, hp.n_rot, nhead, n, x->nb[1], x->nb[2], nope * x->nb[0]);
    float base, scale, ext, factor, fast, slow;
    int orig;
    rope_params(il, base, scale, ext, factor, fast, slow, orig);
    tail = inverse
        ? ggml_rope_ext_back(ctx, tail, pos, nullptr, hp.n_rot, GGML_ROPE_TYPE_NORMAL, orig, base, scale, ext, factor, fast, slow)
        : ggml_rope_ext(ctx, tail, pos, nullptr, hp.n_rot, GGML_ROPE_TYPE_NORMAL, orig, base, scale, ext, factor, fast, slow);
    return ggml_concat(ctx, content, tail, 0);
}

// Shared-cache mirrors (TS_DSV41_KV_MIRRORS). A kv source writes its new
// compressed and indexer rows into each mirror as well; those writes are
// emitted just before the first layer the mirror's GPU builds, so they join
// that GPU's split instead of cutting the source GPU's. Only the new rows cross
// GPUs, and a reader's mirror holds the source's bytes.
struct v41_mirror_write { ggml_tensor * dst; ggml_tensor * rows; ggml_tensor * idxs; int dev; };
std::vector<v41_mirror_write> v41_mirror_pending;

void v41_queue_mirror_writes(const dsv4_layer & L, const dsv4_slot_layer & C, ggml_tensor * index_rows,
                             ggml_tensor * kv_rows, const plan_inputs * per_device)
{
    for (int d = 0; d < m.n_gpu; ++d)
    {
        if (!(L.mirror_devices >> d & 1) || !C.csa_k_mirror[d] || !per_device[d].write_idxs) continue;
        v41_mirror_pending.push_back({ C.lid_k_mirror[d], index_rows, per_device[d].write_idxs, d });
        v41_mirror_pending.push_back({ C.csa_k_mirror[d], kv_rows, per_device[d].write_idxs, d });
    }
}

// dev < 0 emits every pending write.
void v41_flush_mirror_writes(int dev)
{
    for (auto it = v41_mirror_pending.begin(); it != v41_mirror_pending.end();)
    {
        if (dev >= 0 && it->dev != dev) { ++it; continue; }
        ggml_tensor * write = ggml_set_rows(ctx, it->dst, it->rows, it->idxs);
        pin(write, it->dev);
        ggml_build_forward_expand(gf, write);
        it = v41_mirror_pending.erase(it);
    }
}

// The compressed / indexer cache of `source` that a layer on `dev` reads.
static ggml_tensor * v41_shared_csa(const dsv4_slot_layer & source, int dev)
{
    return dev >= 0 && dev <= MAX_GPUS && source.csa_k_mirror[dev] ? source.csa_k_mirror[dev] : source.csa_k;
}
static ggml_tensor * v41_shared_lid(const dsv4_slot_layer & source, int dev)
{
    return dev >= 0 && dev <= MAX_GPUS && source.lid_k_mirror[dev] ? source.lid_k_mirror[dev] : source.lid_k;
}

// Gathered-decode K: [raw_rows window rows | the top-k selected compressed
// rows]. The compressed rows are gathered where the shared cache lives. When
// that is another GPU, only those top-k rows cross over and are appended on the
// consumer: assembling K on the source GPU instead moved the consumer's whole
// raw ring there and the assembled K back, ~3 MiB per layer and token staged
// through the host on a layer-split box without P2P. K's bytes are identical.
ggml_tensor * v41_gathered_k(ggml_tensor * raw, int64_t raw_rows, ggml_tensor * shared_kv, ggml_tensor * top,
                             int source_dev, int dev)
{
    const int64_t head = hp.n_embd_head;
    if (source_dev == dev)
        return fused_node(TSG_DSV4_FUSED_KGATHER, GGML_TYPE_F16, head, 1, raw_rows + hp.indexer_top_k, 1,
            { raw, shared_kv, top }, dev, (int) raw_rows);
    ggml_tensor * selected = fused_node(TSG_DSV4_FUSED_KGATHER, GGML_TYPE_F16, head, 1, hp.indexer_top_k, 1,
        { shared_kv, shared_kv, top }, source_dev, 0);
    // C.raw_k is [head, ring]; a compacted window is already [head, 1, rows].
    if (raw->ne[1] != 1) raw = ggml_reshape_3d(ctx, raw, head, 1, raw->ne[1]);
    GGML_ASSERT(raw->ne[2] == raw_rows);
    ggml_tensor * k = ggml_concat(ctx, raw, selected, 2);
    pin(k, dev);
    return k;
}

ggml_tensor * build_engram(int il, ggml_tensor * x)
{
    const auto & L = m.layers[il];
    if (L.engram_index < 0) return x;
    const int64_t dim = hp.n_embd, hc = hp.hc_mult;
    ggml_tensor * lookup = res.inp.engram[L.engram_index];
    if (m.engram_on_device)
    {
        // The table is resident on this layer's device: gather the quantized
        // rows here. The id array is laid out token-major (index = token *
        // hash_columns + column), so get_rows returns [head_dim, hash_columns *
        // nt] in exactly the order the host path staged its F32 rows. The
        // reshape below is therefore a relabelling, not a copy.
        const int64_t head_dim = m.engram.head_dim, cols = m.engram.hash_columns();
        lookup = ggml_get_rows(ctx, L.engram_embd, res.inp.engram_ids[L.engram_index]);
        // Pin the gather to the device that holds the table. Device weight
        // buffers are not marked USAGE_WEIGHTS, so the scheduler's "run the op
        // where the weight lives" rule does not apply here, and a 30 GiB table
        // must never become a cross-backend copy.
        pin(lookup, L.device);
        lookup = ggml_reshape_2d(ctx, lookup, cols * head_dim, nt);
        if (res.image_tokens)
        {
            // Visual positions consume no Engram embedding. The host path
            // wrote zero rows for them; zero the gathered rows the same way,
            // so a garbage-free but arbitrary row cannot reach the gate.
            ggml_tensor * text_mask = ggml_scale_bias(ctx,
                ggml_cast(ctx, res.inp.image_types[L.device], GGML_TYPE_F32), -1.0f, 1.0f);
            lookup = ggml_mul(ctx, lookup, ggml_reshape_2d(ctx, text_mask, 1, nt));
        }
    }
    ggml_tensor * kv = ggml_mul_mat(ctx, L.engram_wkv, lookup);
    trace_v41(kv, il, "engram_kv");
    ggml_tensor * key = ggml_cont(ctx, ggml_view_2d(ctx, kv, dim * hc, nt, kv->nb[1], 0));
    key = ggml_reshape_3d(ctx, key, dim, hc, nt);
    ggml_tensor * value = ggml_view_3d(ctx, kv, dim, 1, nt, dim * sizeof(float), kv->nb[1], dim * hc * sizeof(float));
    // Q2_K releases quantize the learned elementwise weights too. get_rows
    // converts the four small rows to F32 on the owning GPU.
    ggml_tensor * qw = ggml_get_rows(ctx, L.engram_q, res.inp.engram_rows);
    ggml_tensor * q = ggml_mul(ctx, ggml_rms_norm(ctx, x, hp.rms_eps), qw);
    key = ggml_rms_norm(ctx, key, hp.rms_eps);
    // Without engram_k, engram_q already holds q_weight * k_weight.
    if (L.engram_k) key = ggml_mul(ctx, key, ggml_get_rows(ctx, L.engram_k, res.inp.engram_rows));
    ggml_tensor * dot = ggml_scale(ctx, ggml_sum_rows(ctx, ggml_mul(ctx, q, key)), 1.0f / sqrtf((float) dim));
    trace_v41(dot, il, "engram_dot");
    ggml_tensor * magnitude = ggml_sqrt(ctx, ggml_clamp(ctx, ggml_abs(ctx, dot), 1e-6f, INFINITY));
    ggml_tensor * sign = ggml_scale_bias(ctx, ggml_step(ctx, ggml_scale(ctx, dot, -1.0f)), -2.0f, 1.0f);
    ggml_tensor * gate = ggml_sigmoid(ctx, ggml_mul(ctx, sign, magnitude));
    if (res.image_tokens)
    {
        // Every visual span position, including START/NEWLINE/END, suppresses
        // Engram. Its negative history marker also blocks later text lookback.
        ggml_tensor * text_mask = ggml_scale_bias(ctx,
            ggml_cast(ctx, res.inp.image_types[L.device], GGML_TYPE_F32), -1.0f, 1.0f);
        gate = ggml_mul(ctx, gate, ggml_reshape_3d(ctx, text_mask, 1, 1, nt));
    }
    trace_v41(gate, il, "engram_gate");
    value = ggml_repeat_4d(ctx, value, dim, hc, nt, 1);
    return ggml_add(ctx, x, ggml_mul(ctx, value, gate));
}

// The returned latent is normalized but not rotated: the indexer's K
// projection consumes this version, before the main compressed cache is quantized.
ggml_tensor * v41_compress(int il, ggml_tensor * cur, const plan_inputs & pi, const comp_plan & plan)
{
    const auto & L = m.layers[il];
    const auto & C = m.active_slot->layers[il];
    const int ratio = hp.compress_ratios[il];
    const int64_t head = hp.n_embd_head;
    ggml_tensor * kv = ggml_mul_mat(ctx, L.attn_comp_wkv, cur);
    if (ratio == 1) return rms(kv, L.attn_comp_norm);
    ggml_tensor * score = ggml_mul_mat(ctx, L.attn_comp_wgate, cur);
    ggml_tensor * latent = nullptr;
    if (pi.write_idxs)
    {
        const int64_t n = plan.state_write_idxs.size();
        ggml_tensor * all_kv = ggml_concat(ctx, C.comp_state_kv, kv, 1);
        ggml_tensor * all_score = ggml_concat(ctx, C.comp_state_score, score, 1);
        ggml_tensor * values = ggml_reshape_3d(ctx, ggml_get_rows(ctx, all_kv, pi.read_idxs), head, ratio, n);
        ggml_tensor * scores = ggml_reshape_3d(ctx, ggml_get_rows(ctx, all_score, pi.read_idxs), head, ratio, n);
        values = ggml_cont(ctx, ggml_permute(ctx, values, 1, 0, 2, 3));
        scores = ggml_cont(ctx, ggml_permute(ctx, scores, 1, 0, 2, 3));
        latent = ggml_sum_rows(ctx, ggml_mul(ctx, values, ggml_soft_max(ctx, scores)));
        latent = ggml_cont(ctx, ggml_permute(ctx, latent, 1, 0, 2, 3));
        latent = rms(ggml_reshape_2d(ctx, latent, head, n), L.attn_comp_norm);
        // Read the old tail before overwriting it with this ubatch's tail.
        ggml_build_forward_expand(gf, latent);
    }
    ggml_build_forward_expand(gf, ggml_set_rows(ctx, C.comp_state_kv, ggml_get_rows(ctx, kv, pi.persist_src), pi.persist_dst));
    ggml_build_forward_expand(gf, ggml_set_rows(ctx, C.comp_state_score, ggml_get_rows(ctx, score, pi.persist_src), pi.persist_dst));
    return latent;
}

// A kv source's own writes: the compressed and indexer cache rows its
// attention input produces, plus their mirrors. YOCO runs this for every
// position of the ubatch before the replay layers narrow to the kept tail.
bool v41_skip_source_writes = false;

void v41_source_writes(int il, ggml_tensor * cur)
{
    const auto & L = m.layers[il];
    const auto & C = m.active_slot->layers[il];
    const int dev = L.device, ratio = hp.compress_ratios[il];
    const int64_t head = hp.n_embd_head;
    const auto & pi = ratio == 2 ? res.inp.csa[dev] : res.inp.hca[dev];
    const auto & plan = ratio == 2 ? res.plan_csa : res.plan_hca;
    ggml_tensor * latent = v41_compress(il, cur, pi, plan);
    if (!latent) return;
    trace_v41(latent, il, "latent");
    const int64_t n = latent->ne[1];
    ggml_tensor * ik = rms(ggml_mul_mat(ctx, L.indexer_k, latent), L.indexer_k_norm);
    ik = v41_quant(v41_rope(ggml_reshape_3d(ctx, ik, hp.indexer_head_size, 1, n), pi.write_pos, il), 1, dev);
    ggml_tensor * index_rows = ggml_reshape_2d(ctx, ik, hp.indexer_head_size, n);
    ggml_build_forward_expand(gf, ggml_set_rows(ctx, C.lid_k, index_rows, pi.write_idxs));
    latent = v41_quant(v41_rope(ggml_reshape_3d(ctx, latent, head, 1, n), pi.write_pos, il), 2, dev);
    ggml_tensor * kv_rows = ggml_reshape_2d(ctx, latent, head, n);
    ggml_build_forward_expand(gf, ggml_set_rows(ctx, C.csa_k, kv_rows, pi.write_idxs));
    v41_queue_mirror_writes(L, C, index_rows, kv_rows, ratio == 2 ? res.inp.csa : res.inp.hca);
}

// YOCO: the last `kept` positions of a tensor whose outermost token axis has nt.
ggml_tensor * yoco_tail(ggml_tensor * t, int64_t kept)
{
    int axis = 3;
    while (axis > 0 && t->ne[axis] != nt) --axis;
    if (t->ne[axis] != nt) throw std::runtime_error("V4.1 YOCO: tensor has no token axis");
    int64_t ne[4] = { t->ne[0], t->ne[1], t->ne[2], t->ne[3] };
    ne[axis] = kept;
    return ggml_view_4d(ctx, t, ne[0], ne[1], ne[2], ne[3], t->nb[1], t->nb[2], t->nb[3],
        (size_t) (nt - kept) * t->nb[axis]);
}

// YOCO: from here on the graph covers only the ubatch's last `kept` tokens.
// Every per-token input becomes its tail; the window mask becomes the replay's.
// The host still fills the full-width input tensors, which the tails view, so
// yoco_restore() puts them back in res.inp once the graph is built.
graph_inputs yoco_saved_inputs;
bool yoco_narrowed = false;

void yoco_narrow(int64_t kept)
{
    graph_inputs & inp = res.inp;
    yoco_saved_inputs = inp;
    yoco_narrowed = true;
    const int64_t skip = nt - kept;
    auto tail_1d = [&](ggml_tensor * t) { return t ? ggml_view_1d(ctx, t, kept, (size_t) skip * t->nb[0]) : t; };
    for (int d = 0; d <= m.n_gpu; d++)
    {
        inp.pos[d] = tail_1d(inp.pos[d]);
        inp.tokens[d] = tail_1d(inp.tokens[d]);
        inp.raw_idxs[d] = tail_1d(inp.raw_idxs[d]);
        if (inp.raw_mask_yoco[d]) inp.raw_mask[d] = inp.raw_mask_yoco[d];
        for (plan_inputs * pi : { &inp.csa[d], &inp.hca[d], &inp.lid[d] })
            if (pi->kq_mask)
                pi->kq_mask = ggml_view_2d(ctx, pi->kq_mask, pi->kq_mask->ne[0], kept, pi->kq_mask->nb[1],
                    (size_t) skip * pi->kq_mask->nb[1]);
    }
    p0 += skip;
    nt = kept;
}

void yoco_restore()
{
    if (yoco_narrowed) res.inp = yoco_saved_inputs;
    yoco_narrowed = false;
}

ggml_tensor * build_attention_v41(int il, ggml_tensor * cur, ggml_tensor * pos)
{
    const auto & L = m.layers[il];
    const auto & C = m.active_slot->layers[il];
    const int dev = L.device, ratio = hp.compress_ratios[il];
    const int64_t head = hp.n_embd_head;
    trace_v41(cur, il, "attn_input");
    ggml_tensor * qr = rms(ggml_mul_mat(ctx, L.wq_a, cur), L.attn_q_a_norm);
    trace_v41(qr, il, "qr");
    ggml_tensor * q = v41_rope(ggml_reshape_3d(ctx, ggml_mul_mat(ctx, L.wq_b, qr), head, hp.n_head, nt), pos, il);
    trace_v41(q, il, "query");
    ggml_tensor * kv = rms(ggml_mul_mat(ctx, L.wkv, cur), L.attn_kv_norm);
    kv = v41_quant(v41_rope(ggml_reshape_3d(ctx, kv, head, 1, nt), pos, il), 0, dev);
    trace_v41(kv, il, "raw_kv");
    ggml_build_forward_expand(gf, ggml_set_rows(ctx, C.raw_k, ggml_reshape_2d(ctx, kv, head, nt), res.inp.raw_idxs[dev]));
    ggml_tensor * k = ggml_reshape_3d(ctx, C.raw_k, head, 1, m.ring_raw);
    ggml_tensor * mask = res.inp.raw_mask[dev];

    if (ratio)
    {
        const auto & source = m.active_slot->layers[L.kv_source];
        const auto & pi = ratio == 2 ? res.inp.csa[dev] : res.inp.hca[dev];
        const auto & plan = ratio == 2 ? res.plan_csa : res.plan_hca;
        if (L.kv_source == il && !v41_skip_source_writes) v41_source_writes(il, cur);
        if (L.index_source == il && plan.n_kv <= hp.indexer_top_k)
        {
            // Every reachable compressed row is selected. Keep producing K
            // caches, but omit query projections and top-k until selection matters.
            v41_shared_mask = pi.kq_mask;
            v41_shared_top = nullptr;
        }
        else if (L.index_source == il)
        {
            ggml_tensor * iq = ggml_reshape_3d(ctx, ggml_mul_mat(ctx, L.indexer_attn_q_b, qr), hp.indexer_head_size, hp.indexer_n_head, nt);
            iq = v41_quant(v41_rope(iq, pos, il), 1, dev);
            ggml_tensor * iw = ggml_scale(ctx, ggml_mul_mat(ctx, L.indexer_proj, cur),
                1.0f / sqrtf((float) (hp.indexer_head_size * hp.indexer_n_head)));
            ggml_tensor * shared_lid = v41_shared_lid(source, dev);
            ggml_tensor * ik = ggml_view_2d(ctx, shared_lid, hp.indexer_head_size, plan.n_kv, shared_lid->nb[1], 0);
            ik = ggml_reshape_3d(ctx, ik, hp.indexer_head_size, 1, plan.n_kv);
            ggml_tensor * imask = pi.kq_mask;
            if (v41_candidates && il > hp.candidate_source)
                imask = ggml_add(ctx, imask, v41_candidates);
            ggml_tensor * scores = ggml_lightning_indexer(ctx, iq, ik, iw, imask);
            if (il == hp.candidate_source && plan.n_kv > (int64_t) hp.candidate_topk * hp.candidate_block)
            {
                ggml_tensor * blocks = fused_node(TSG_DSV41_CANDIDATE_SCORES, GGML_TYPE_F32,
                    (plan.n_kv + hp.candidate_block - 1) / hp.candidate_block, nt, 1, 1,
                    { scores, pos }, dev, hp.candidate_block);
                // Selections use ggml_argsort_top_k, not ggml_top_k: ggml-cuda's radix top-k picks
                // among tied scores and orders its output by atomic arrival, so the kept set and the
                // gathered decode order changed from run to run (and between TP nodes, which each run
                // every selection) at long context. The sort is deterministic.
                ggml_tensor * selected = ggml_cont(ctx, ggml_argsort_top_k(ctx, blocks, (int) std::min<int64_t>(hp.candidate_topk, blocks->ne[0])));
                v41_candidate_scores = blocks;
                v41_candidate_top = selected;
                v41_candidates = fused_node(TSG_DSV41_CANDIDATE_MASK, GGML_TYPE_F16,
                    plan.n_kv, nt, 1, 1, { blocks, selected }, dev, hp.candidate_block);
            }
            trace_v41(scores, il, "index_scores");
            ggml_tensor * top = ggml_cont(ctx, ggml_argsort_top_k(ctx, scores, (int) std::min<int64_t>(hp.indexer_top_k, plan.n_kv)));
            trace_v41(top, il, "index_top");
            v41_shared_top = top;
            // Gathered decode consumes the selected row IDs directly. The
            // full-width mask is needed only by masked-dense attention.
            v41_shared_mask = res.inp.gather_mask[dev] ? nullptr : build_top_k_mask(pi.kq_mask, top);
        }
        if (res.inp.gather_mask[dev])
        {
            if (!v41_shared_top || v41_shared_top->ne[0] != hp.indexer_top_k)
                throw std::runtime_error("V4.1 index source has not published a complete sparse selection");
            // Writes to source.csa_k were expanded before index selection.
            // Every selected row is visible under dsv41_use_gather's gate,
            // including candidate-pruned layers, so the compact tail is all zero.
            // Gather where the large shared cache resides. Gathering on the
            // consumer would copy the entire cache across GPUs first.
            const int source_dev = m.layers[L.kv_source].device;
            ggml_tensor * raw = C.raw_k;
            int64_t raw_rows = m.ring_raw;
            if (res.inp.raw_read_idxs[dev])
            {
                // KGATHER's i0=0 mode gathers every output row from src[1].
                // Compact on the raw cache's own GPU first: reading fewer rows
                // inside the source-GPU kernel alone would still make the ggml
                // scheduler transfer the complete C.raw_k input beforehand.
                // Masked duplicate rows pad the final K width to CUDA's
                // head-512 attention stride without changing visible rows.
                raw_rows = res.inp.raw_read_idxs[dev]->ne[0];
                raw = fused_node(TSG_DSV4_FUSED_KGATHER, GGML_TYPE_F16,
                    head, 1, raw_rows, 1,
                    { C.raw_k, C.raw_k, res.inp.raw_read_idxs[dev] }, dev, 0);
            }
            const bool local = v41_shared_csa(source, dev) != source.csa_k;
            k = v41_gathered_k(raw, raw_rows, v41_shared_csa(source, dev), v41_shared_top, local ? dev : source_dev, dev);
            mask = res.inp.gather_mask[dev];
        }
        else
        {
            if (!v41_shared_mask) throw std::runtime_error("V4.1 index source has not published its selection");
            ggml_tensor * shared_csa = v41_shared_csa(source, dev);
            ggml_tensor * compressed = ggml_view_2d(ctx, shared_csa, head, plan.n_kv, shared_csa->nb[1], 0);
            compressed = ggml_reshape_3d(ctx, compressed, head, 1, plan.n_kv);
            k = ggml_concat(ctx, k, compressed, 2);
            mask = ggml_concat(ctx, mask, v41_shared_mask, 0);
        }
    }
    // The CPU F16 attention kernel also narrows Q and its running numerator
    // to F16. Keep the trained cache quantization, but accumulate in F32.
    if (ggml_backend_is_cpu(m.backends[dev])) k = ggml_cast(ctx, k, GGML_TYPE_F32);
    ggml_tensor * out = attn_mha(q, k, mask, L.attn_sinks, 1.0f / sqrtf((float) head), dev);
    out = v41_rope(ggml_reshape_3d(ctx, out, head, hp.n_head, nt), pos, il, true);
    const int64_t group_dim = hp.n_head / hp.o_groups * head;
    ggml_tensor * grouped = ggml_permute(ctx, ggml_reshape_3d(ctx, out, group_dim, hp.o_groups, nt), 0, 2, 1, 3);
    ggml_tensor * oa = ggml_mul_mat(ctx, ggml_reshape_3d(ctx, L.wo_a, L.wo_a->ne[0], hp.o_lora_rank, hp.o_groups), grouped);
    oa = ggml_cont_2d(ctx, ggml_permute(ctx, oa, 0, 2, 1, 3), hp.o_lora_rank * hp.o_groups, nt);
    ggml_tensor * result = ggml_mul_mat(ctx, L.wo_b, oa);
    trace_v41(result, il, "attn_output");
    return result;
}

// ---------------------------------------------------------------------------
// Token-batched multi-slot decode (V4.1)
//
// One decode token for each of N sequence slots in a single graph. The weights
// are what makes a decode step expensive -- ~9.8 GiB is read per token at
// Q4_K_M, and the step is also dominated by ~2,200 small kernels -- and both
// costs are paid ONCE here instead of once per sequence. Only the parts that
// touch a sequence's own state fork per slot: the sliding-window ring, the
// compressor state, the indexer selection and the attention itself.
//
// Everything a slot forks reads the SHARED batched projections, so the
// compressor index inputs address this step's column for their slot; the host
// fill offsets them (see dsv4_forward_batched_decode).
// ---------------------------------------------------------------------------

// Published per slot by the index-source layer and consumed by the layers that
// share its cache, exactly as v41_shared_top/mask/candidates do for one slot.
ggml_tensor * v41_bd_shared_mask[DSV4_MAX_BATCHED_SLOTS] = {};
ggml_tensor * v41_bd_shared_top[DSV4_MAX_BATCHED_SLOTS] = {};
ggml_tensor * v41_bd_candidates[DSV4_MAX_BATCHED_SLOTS] = {};

ggml_tensor * build_attention_v41_batched(int il, ggml_tensor * cur, ggml_tensor * inp_pos)
{
    const auto & L = m.layers[il];
    const int dev = L.device, ratio = hp.compress_ratios[il];
    const int64_t head = hp.n_embd_head;
    const int64_t N = nt;

    // ---- dense prologue, batched over the N slots' tokens ----
    ggml_tensor * qr = rms(ggml_mul_mat(ctx, L.wq_a, cur), L.attn_q_a_norm);
    ggml_tensor * q_all = v41_rope(
        ggml_reshape_3d(ctx, ggml_mul_mat(ctx, L.wq_b, qr), head, hp.n_head, N), inp_pos, il);
    ggml_tensor * kv_all = rms(ggml_mul_mat(ctx, L.wkv, cur), L.attn_kv_norm);
    kv_all = v41_quant(v41_rope(ggml_reshape_3d(ctx, kv_all, head, 1, N), inp_pos, il), 0, dev);
    ggml_tensor * kv_flat = ggml_reshape_2d(ctx, kv_all, head, N);

    // Compressor and indexer-query projections are weight-heavy too, so they
    // are shared as well; only their state reads/writes fork below.
    ggml_tensor * comp_kv = nullptr, * comp_score = nullptr, * comp_latent = nullptr;
    if (ratio && L.kv_source == il)
    {
        comp_kv = ggml_mul_mat(ctx, L.attn_comp_wkv, cur);
        if (ratio == 1) comp_latent = rms(comp_kv, L.attn_comp_norm);
        else comp_score = ggml_mul_mat(ctx, L.attn_comp_wgate, cur);
    }
    ggml_tensor * iq_all = nullptr, * iw_all = nullptr;
    const bool index_here = ratio && L.index_source == il;
    if (index_here)
    {
        iq_all = ggml_reshape_3d(ctx, ggml_mul_mat(ctx, L.indexer_attn_q_b, qr),
            hp.indexer_head_size, hp.indexer_n_head, N);
        iq_all = v41_quant(v41_rope(iq_all, inp_pos, il), 1, dev);
        iw_all = ggml_scale(ctx, ggml_mul_mat(ctx, L.indexer_proj, cur),
            1.0f / sqrtf((float) (hp.indexer_head_size * hp.indexer_n_head)));
    }

    ggml_tensor * attn_cat = nullptr;
    for (int64_t i = 0; i < N; i++)
    {
        bd_slot_state & B = res.bd[i];
        const dsv4_slot_layer & C = m.slots.at(B.slot_id)->layers[il];
        ggml_tensor * pos_i = ggml_view_1d(ctx, inp_pos, 1, i * sizeof(int32_t));
        const plan_inputs & pi = ratio == 1 ? B.v41_hca[dev] : B.v41_csa[dev];
        const comp_plan & plan = ratio == 1 ? B.plan_hca : B.plan_csa;

        // sliding-window ring write for this slot's single token. A single
        // column of a contiguous [head, N] tensor is itself contiguous, so
        // this is a relabelling, not a copy.
        ggml_tensor * kv_i = ggml_view_2d(ctx, kv_flat, head, 1, kv_flat->nb[1], i * kv_flat->nb[1]);
        ggml_build_forward_expand(gf, ggml_set_rows(ctx, C.raw_k, kv_i, B.raw_idxs[dev]));
        ggml_tensor * k = ggml_reshape_3d(ctx, C.raw_k, head, 1, m.ring_raw);
        ggml_tensor * mask = B.raw_mask[dev];

        if (ratio)
        {
            const dsv4_slot_layer & source = m.slots.at(B.slot_id)->layers[L.kv_source];
            if (L.kv_source == il)
            {
                ggml_tensor * latent = nullptr;
                if (ratio == 1)
                {
                    latent = ggml_cont(ctx, ggml_view_2d(ctx, comp_latent, head, 1,
                        comp_latent->nb[1], i * comp_latent->nb[1]));
                }
                else
                {
                    // The same pooled-compression arithmetic v41_compress runs
                    // for one sequence. Narrowing the shared projections to this
                    // slot's single column first keeps the plan's index space
                    // identical to the single-sequence one: the concatenated row
                    // space is still [state ring | zero row | one ubatch column].
                    ggml_tensor * kv_col = ggml_cont(ctx,
                        ggml_view_2d(ctx, comp_kv, head, 1, comp_kv->nb[1], i * comp_kv->nb[1]));
                    ggml_tensor * score_col = ggml_cont(ctx,
                        ggml_view_2d(ctx, comp_score, head, 1, comp_score->nb[1], i * comp_score->nb[1]));
                    if (pi.write_idxs)
                    {
                        const int64_t n = plan.state_write_idxs.size();
                        ggml_tensor * all_kv = ggml_concat(ctx, C.comp_state_kv, kv_col, 1);
                        ggml_tensor * all_score = ggml_concat(ctx, C.comp_state_score, score_col, 1);
                        ggml_tensor * values = ggml_reshape_3d(ctx, ggml_get_rows(ctx, all_kv, pi.read_idxs), head, ratio, n);
                        ggml_tensor * scores = ggml_reshape_3d(ctx, ggml_get_rows(ctx, all_score, pi.read_idxs), head, ratio, n);
                        values = ggml_cont(ctx, ggml_permute(ctx, values, 1, 0, 2, 3));
                        scores = ggml_cont(ctx, ggml_permute(ctx, scores, 1, 0, 2, 3));
                        latent = ggml_sum_rows(ctx, ggml_mul(ctx, values, ggml_soft_max(ctx, scores)));
                        latent = ggml_cont(ctx, ggml_permute(ctx, latent, 1, 0, 2, 3));
                        latent = rms(ggml_reshape_2d(ctx, latent, head, n), L.attn_comp_norm);
                        // Read the old tail before overwriting it below.
                        ggml_build_forward_expand(gf, latent);
                    }
                    ggml_build_forward_expand(gf, ggml_set_rows(ctx, C.comp_state_kv,
                        ggml_get_rows(ctx, kv_col, pi.persist_src), pi.persist_dst));
                    ggml_build_forward_expand(gf, ggml_set_rows(ctx, C.comp_state_score,
                        ggml_get_rows(ctx, score_col, pi.persist_src), pi.persist_dst));
                }
                if (latent)
                {
                    const int64_t n = latent->ne[1];
                    ggml_tensor * ik = rms(ggml_mul_mat(ctx, L.indexer_k, latent), L.indexer_k_norm);
                    ik = v41_quant(v41_rope(ggml_reshape_3d(ctx, ik, hp.indexer_head_size, 1, n), pi.write_pos, il), 1, dev);
                    ggml_tensor * index_rows = ggml_reshape_2d(ctx, ik, hp.indexer_head_size, n);
                    ggml_build_forward_expand(gf, ggml_set_rows(ctx, C.lid_k, index_rows, pi.write_idxs));
                    ggml_tensor * ck = v41_quant(v41_rope(ggml_reshape_3d(ctx, latent, head, 1, n), pi.write_pos, il), 2, dev);
                    ggml_tensor * kv_rows = ggml_reshape_2d(ctx, ck, head, n);
                    ggml_build_forward_expand(gf, ggml_set_rows(ctx, C.csa_k, kv_rows, pi.write_idxs));
                    v41_queue_mirror_writes(L, C, index_rows, kv_rows, ratio == 1 ? B.v41_hca : B.v41_csa);
                }
            }

            if (index_here && plan.n_kv <= hp.indexer_top_k)
            {
                v41_bd_shared_mask[i] = pi.kq_mask;
                v41_bd_shared_top[i] = nullptr;
            }
            else if (index_here)
            {
                ggml_tensor * iq = ggml_view_3d(ctx, iq_all, hp.indexer_head_size, hp.indexer_n_head, 1,
                    iq_all->nb[1], iq_all->nb[2], i * iq_all->nb[2]);
                ggml_tensor * iw = ggml_view_2d(ctx, iw_all, iw_all->ne[0], 1, iw_all->nb[1], i * iw_all->nb[1]);
                ggml_tensor * shared_lid = v41_shared_lid(source, dev);
                ggml_tensor * ik = ggml_view_2d(ctx, shared_lid, hp.indexer_head_size, plan.n_kv, shared_lid->nb[1], 0);
                ik = ggml_reshape_3d(ctx, ik, hp.indexer_head_size, 1, plan.n_kv);
                ggml_tensor * imask = pi.kq_mask;
                if (v41_bd_candidates[i] && il > hp.candidate_source)
                    imask = ggml_add(ctx, imask, v41_bd_candidates[i]);
                ggml_tensor * scores = ggml_lightning_indexer(ctx, iq, ik, iw, imask);
                if (il == hp.candidate_source && plan.n_kv > (int64_t) hp.candidate_topk * hp.candidate_block)
                {
                    ggml_tensor * blocks = fused_node(TSG_DSV41_CANDIDATE_SCORES, GGML_TYPE_F32,
                        (plan.n_kv + hp.candidate_block - 1) / hp.candidate_block, 1, 1, 1,
                        { scores, pos_i }, dev, hp.candidate_block);
                    ggml_tensor * chosen = ggml_cont(ctx, ggml_argsort_top_k(ctx, blocks, (int) std::min<int64_t>(hp.candidate_topk, blocks->ne[0])));
                    v41_bd_candidates[i] = fused_node(TSG_DSV41_CANDIDATE_MASK, GGML_TYPE_F16,
                        plan.n_kv, 1, 1, 1, { blocks, chosen }, dev, hp.candidate_block);
                }
                ggml_tensor * top = ggml_cont(ctx, ggml_argsort_top_k(ctx, scores, (int) std::min<int64_t>(hp.indexer_top_k, plan.n_kv)));
                v41_bd_shared_top[i] = top;
                v41_bd_shared_mask[i] = B.gather_mask[dev] ? nullptr : build_top_k_mask(pi.kq_mask, top);
            }

            if (B.gather_mask[dev])
            {
                if (!v41_bd_shared_top[i] || v41_bd_shared_top[i]->ne[0] != hp.indexer_top_k)
                    throw std::runtime_error("V4.1 index source has not published a complete sparse selection");
                const int source_dev = m.layers[L.kv_source].device;
                ggml_tensor * raw = C.raw_k;
                int64_t raw_rows = m.ring_raw;
                if (B.raw_read_idxs[dev])
                {
                    raw_rows = B.raw_read_idxs[dev]->ne[0];
                    raw = fused_node(TSG_DSV4_FUSED_KGATHER, GGML_TYPE_F16,
                        head, 1, raw_rows, 1,
                        { C.raw_k, C.raw_k, B.raw_read_idxs[dev] }, dev, 0);
                }
                const bool local = v41_shared_csa(source, dev) != source.csa_k;
                k = v41_gathered_k(raw, raw_rows, v41_shared_csa(source, dev), v41_bd_shared_top[i],
                    local ? dev : source_dev, dev);
                mask = B.gather_mask[dev];
            }
            else
            {
                if (!v41_bd_shared_mask[i]) throw std::runtime_error("V4.1 index source has not published its selection");
                ggml_tensor * shared_csa = v41_shared_csa(source, dev);
                ggml_tensor * compressed = ggml_view_2d(ctx, shared_csa, head, plan.n_kv, shared_csa->nb[1], 0);
                compressed = ggml_reshape_3d(ctx, compressed, head, 1, plan.n_kv);
                k = ggml_concat(ctx, k, compressed, 2);
                mask = ggml_concat(ctx, mask, v41_bd_shared_mask[i], 0);
            }
        }

        ggml_tensor * q = ggml_view_3d(ctx, q_all, head, hp.n_head, 1, q_all->nb[1], q_all->nb[2], i * q_all->nb[2]);
        if (ggml_backend_is_cpu(m.backends[dev])) k = ggml_cast(ctx, k, GGML_TYPE_F32);
        ggml_tensor * out = attn_mha(q, k, mask, L.attn_sinks, 1.0f / sqrtf((float) head), dev);
        attn_cat = attn_cat ? ggml_concat(ctx, attn_cat, out, 1) : out;   // [head*n_head, N]
    }

    // ---- batched epilogue: inverse rope + grouped LoRA out (weights once) ----
    ggml_tensor * out = v41_rope(ggml_reshape_3d(ctx, attn_cat, head, hp.n_head, N), inp_pos, il, true);
    const int64_t group_dim = hp.n_head / hp.o_groups * head;
    ggml_tensor * grouped = ggml_permute(ctx, ggml_reshape_3d(ctx, out, group_dim, hp.o_groups, N), 0, 2, 1, 3);
    ggml_tensor * oa = ggml_mul_mat(ctx, ggml_reshape_3d(ctx, L.wo_a, L.wo_a->ne[0], hp.o_lora_rank, hp.o_groups), grouped);
    oa = ggml_cont_2d(ctx, ggml_permute(ctx, oa, 0, 2, 1, 3), hp.o_lora_rank * hp.o_groups, N);
    return ggml_mul_mat(ctx, L.wo_b, oa);
}
