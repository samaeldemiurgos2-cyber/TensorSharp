// Included inside graph_builder. Each graph contains only this node's layers.
// The boundary is architectural state, not just the residual: V4.1 also
// publishes delayed gates, sparse selections and a candidate mask.
const comp_plan & pipeline_plan(int layer) const
{
    return hp.compress_ratios[layer] == 1 ? res.plan_hca : res.plan_csa;
}

int pipeline_index_before(int boundary) const
{
    int source = -1;
    for (int il : hp.index_sources) if (il < boundary) source = std::max(source, il);
    return source;
}

ggml_tensor * pipeline_input(ggml_type type, int64_t a, int64_t b, int64_t c, int dev)
{
    auto * t = ggml_new_tensor_3d(ctx, type, a, b, c);
    ggml_set_input(t);
    ggml_format_name(t, "pipeline_input.%zu", res.pipeline_in.size());
    pin(t, dev);
    // Some boundaries precede a new index source, which supersedes the
    // previous sparse selection. Retain every agreed wire input regardless.
    ggml_build_forward_expand(gf, t);
    res.pipeline_in.push_back(t);
    return t;
}

void pipeline_begin(ggml_tensor * & hidden, ggml_tensor * & gates)
{
    const int dev = m.layers[m.layer_begin].device;
    hidden = pipeline_input(GGML_TYPE_F32, hp.n_embd, hp.hc_mult, nt, dev);
    if (!hp.v41) return;
    gates = pipeline_input(GGML_TYPE_F32, hp.hc_mult, nt, 1, dev);
    const int index = pipeline_index_before(m.layer_begin);
    if (index >= 0)
    {
        const auto & plan = pipeline_plan(index);
        const auto & inputs = hp.compress_ratios[index] == 1 ? res.inp.hca[dev] : res.inp.csa[dev];
        if (plan.n_kv > hp.indexer_top_k)
        {
            v41_shared_top = pipeline_input(GGML_TYPE_I32, hp.indexer_top_k, nt, 1, dev);
            v41_shared_mask = res.inp.gather_mask[dev] ? nullptr : build_top_k_mask(inputs.kq_mask, v41_shared_top);
        }
        else v41_shared_mask = inputs.kq_mask;
    }
    if (hp.candidate_source >= 0 && hp.candidate_source < m.layer_begin &&
        pipeline_plan(hp.candidate_source).n_kv > (int64_t) hp.candidate_topk * hp.candidate_block)
    {
        const int64_t width = pipeline_plan(hp.candidate_source).n_kv;
        const int64_t blocks = (width + hp.candidate_block - 1) / hp.candidate_block;
        // A full candidate mask scales with context length per token. Carry
        // its pooled scores and selected blocks, then recreate the exact mask
        // on the receiving GPU, including -infinity validity checks.
        v41_candidate_scores = pipeline_input(GGML_TYPE_F32, blocks, nt, 1, dev);
        v41_candidate_top = pipeline_input(GGML_TYPE_I32, std::min<int64_t>(hp.candidate_topk, blocks), nt, 1, dev);
        v41_candidates = fused_node(TSG_DSV41_CANDIDATE_MASK, GGML_TYPE_F16,
            width, nt, 1, 1, {v41_candidate_scores, v41_candidate_top}, dev, hp.candidate_block);
    }
}

void pipeline_output(ggml_tensor * t)
{
    if (!t) throw std::runtime_error("A distributed layer boundary is missing model state");
    t = ggml_cont(ctx, t);
    ggml_set_output(t);
    ggml_build_forward_expand(gf, t);
    res.pipeline_out.push_back(t);
}

void pipeline_end(ggml_tensor * hidden, ggml_tensor * gates)
{
    pipeline_output(hidden);
    if (!hp.v41) return;
    pipeline_output(gates);
    const int index = pipeline_index_before(m.layer_end);
    if (index >= 0 && pipeline_plan(index).n_kv > hp.indexer_top_k)
        pipeline_output(v41_shared_top);
    if (hp.candidate_source >= 0 && hp.candidate_source < m.layer_end &&
        pipeline_plan(hp.candidate_source).n_kv > (int64_t) hp.candidate_topk * hp.candidate_block)
    {
        pipeline_output(v41_candidate_scores);
        pipeline_output(v41_candidate_top);
    }
}
